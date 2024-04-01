import logging
import math
import types

import toml
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import AgsDatasetInfo
from dataset.pl_dataset_module import AgsDataModule
from lora.lora_modules import LoraLinear, update_lora_importance_alpha_require_grad
from models.model_info import AgsModelInfo
from models.modeling_opt_lora import (
    OPTLoraForCausalLM,
    OPTLoraForQuestionAnswering,
    OPTLoraForSequenceClassification,
    OPTLoraDecoderLayer,
)
from pl_model_wrapper.base import PlWrapperBase
from tools.trainable_param_printer import print_trainable_parameters

logger = logging.getLogger(__name__)


def synflow_test(
    pl_model: PlWrapperBase,
    model_info: AgsModelInfo,  # dataclass of model's task type and name
    data_module: AgsDataModule,  # for preparing and loading datasets for pl trainer
    dataset_info: AgsDatasetInfo,  # dataclass including e.g. number of classes for the pl model wrapper
    task,  # to decide the pl model wrapper of which type should be used
    optimizer,  # optimizer for pl trainer
    learning_rate,  # lr for optimizer. lr_scheduler is default as CosineAnnealingLR
    weight_decay,  # weight_decay for optimizer
    lr_scheduler,  # for building lr scheduler
    eta_min,  # for building lr scheduler
    pl_trainer_args,  # args for pl trainer; include e.g. "max_epochs" for setting up lr_scheduler
    auto_requeue,  # for setting up SLURMEnvironment, environment for distributed launch
    save_path,  # path for saving checkpoints
    save_time,  # for result toml filename
    load_name,  # path to the saved checkpoint
    load_type,  # model checkpoint's type: ['pt', 'pl']
    resume_training,  # whether resume zero-proxy trained model from the checkpoint
    **kwargs,
):
    logger.warning("Running SYNFLOW test")

    model = pl_model.model

    trainable_params = []
    if model_info.is_lora:
        trainable_params.append("lora_")
    if model_info.is_ags:
        trainable_params.append("proj_")
    if len(trainable_params) > 0:
        for name, param in model.named_parameters():
            if name.startswith("model") or name.startswith("roberta"):
                param.requires_grad = False
                for trainable_param in trainable_params:
                    if trainable_param in name:
                        param.requires_grad = True
                        break
            else:
                param.requires_grad = True
    update_lora_importance_alpha_require_grad(model, require_grad=False)
    print_trainable_parameters(model)

    def get_unshuffled_train_dataloader(datamodule: AgsDataModule):
        if datamodule.training_dataset is None:
            raise RuntimeError("The training dataset is not available.")
        data_collator = None
        if datamodule.dataset_info.data_collator_cls is not None:
            data_collator = datamodule.dataset_info.data_collator_cls(
                tokenizer=datamodule.tokenizer
            )
        return DataLoader(
            datamodule.training_dataset,
            batch_size=datamodule.batch_size,
            shuffle=False,
            num_workers=datamodule.num_workers,
            collate_fn=data_collator,
        )

    data_module.prepare_data()
    data_module.setup()
    dataloader = get_unshuffled_train_dataloader(data_module)

    # SYNFLOW
    assert (
        type(model) is OPTLoraForCausalLM
        or type(model) is OPTLoraForSequenceClassification
        or type(model) is OPTLoraForQuestionAnswering
    )
    model: OPTLoraForCausalLM | OPTLoraForSequenceClassification | OPTLoraForQuestionAnswering
    model.to("cuda")

    # convert params to their abs, keep sign for converting it back
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(model)

    # compute gradients with input of ones
    model.zero_grad()
    example_input = next(iter(dataloader))
    input_dim = list(example_input["input_ids"].shape) + [model.model.decoder.embed_tokens.weight.shape[1]]
    inputs = torch.ones(input_dim).float().to("cuda")
    attention_mask = example_input["attention_mask"]
    token_type_ids = example_input.get("token_type_ids", None)
    labels = example_input["labels"]
    if isinstance(inputs, list):
        inputs = torch.stack(inputs)
    if isinstance(attention_mask, list):
        attention_mask = torch.stack(attention_mask)
    if isinstance(token_type_ids, list):
        token_type_ids = torch.stack(token_type_ids)
    if isinstance(labels, list):
        labels = torch.stack(labels)

    if token_type_ids is not None:
        output = model.forward(
            inputs_embeds=inputs.to("cuda"),
            attention_mask=attention_mask.to("cuda"),
            token_type_ids=token_type_ids.to("cuda"),
            labels=labels.to("cuda"),
        )
    else:
        output = model.forward(
            inputs_embeds=inputs.to("cuda"),
            attention_mask=attention_mask.to("cuda"),
            labels=labels.to("cuda"),
        )
    torch.sum(output["loss"]).backward()

    # calculate score of every lora module
    grads_abs = {}
    for decoder_layer in model.model.decoder.layers:
        decoder_layer: OPTLoraDecoderLayer
        layer_id = decoder_layer.layer_id
        lora_modules: dict[str, LoraLinear] = {
            "q_proj": decoder_layer.self_attn.q_proj,
            "k_proj": decoder_layer.self_attn.k_proj,
            "v_proj": decoder_layer.self_attn.v_proj,
            "out_proj": decoder_layer.self_attn.out_proj,
            "fc1": decoder_layer.fc1,
            "fc2": decoder_layer.fc2,
        }
        for proj_name, lora in lora_modules.items():
            if (
                lora.active_adapter not in lora.lora_A.keys()
                or lora.disable_adapters
                or lora.r[lora.active_adapter] == 0
            ):
                continue

            grad_lora = (
                torch.sum(torch.abs(
                    lora.lora_A[lora.active_adapter].weight * lora.lora_A[lora.active_adapter].weight.grad
                ))
                + torch.sum(torch.abs(
                    lora.lora_B[lora.active_adapter].weight * lora.lora_B[lora.active_adapter].weight.grad
                ))
            ).item()

            if f"layer_{layer_id}" not in grads_abs:
                grads_abs[f"layer_{layer_id}"] = {}
            grads_abs[f"layer_{layer_id}"][proj_name] = grad_lora

    log_path = f"{save_path}/synflow_{save_time}.toml"
    with open(log_path, "w+") as fout:
        toml.dump(grads_abs, fout)
    logger.info("Result saved as toml")

    # apply signs of all params
    nonlinearize(model, signs)

    logger.warning("SYNFLOW test done")
