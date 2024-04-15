import logging
import math

import toml
import torch
from torch.utils.data import DataLoader

from dataset import AgsDatasetInfo
from dataset.pl_dataset_module import AgsDataModule
from lora.lora_modules import LoraLinear, update_lora_importance_alpha_require_grad, reset_lora
from models.model_info import AgsModelInfo
from models.modeling_opt_lora import (
    OPTLoraForCausalLM,
    OPTLoraForQuestionAnswering,
    OPTLoraForSequenceClassification,
    OPTLoraDecoderLayer,
)
from pl_model_wrapper.base import PlWrapperBase
from projectors.shortcut_modules import update_ags_importance_beta_require_grad, reset_shortcut
from tools.trainable_param_printer import print_trainable_parameters

logger = logging.getLogger(__name__)


TEST_BATCH = 32


def grad_norm_test(
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
    limit_test_batches,  # number of test batches limit for the zero-proxy test
    **kwargs,
):
    logger.warning("Running GRAD_NORM importance test")

    model = pl_model.model

    trainable_params = []
    if model_info.is_lora:
        reset_lora(model)
        trainable_params.append("lora_")
    if model_info.is_ags:
        reset_shortcut(model)
        trainable_params.append("proj_")
    #     trainable_params.append("shortcut_ln_")

    if len(trainable_params) > 0:
        for name, param in model.named_parameters():
            # print(name)
            if name.startswith("model") or name.startswith("roberta"):
                param.requires_grad = False
                for trainable_param in trainable_params:
                    if trainable_param in name:
                        param.requires_grad = True
                        break
            else:
                param.requires_grad = True

    update_lora_importance_alpha_require_grad(model, require_grad=False)
    update_ags_importance_beta_require_grad(model, require_grad=False)
    # update_ags_ln_require_grad(model, require_grad=False)
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

    # GRAD NORM
    assert (
        type(model) is OPTLoraForCausalLM
        or type(model) is OPTLoraForSequenceClassification
        or type(model) is OPTLoraForQuestionAnswering
    )
    model: OPTLoraForCausalLM | OPTLoraForSequenceClassification | OPTLoraForQuestionAnswering

    # compute gradients
    if type(limit_test_batches) is float:
        limit_batch_num = math.ceil(len(dataloader) * limit_test_batches)
        if limit_batch_num != len(dataloader) * limit_test_batches:
            logger.warning(
                "More data batches than the provided test ratio limit are used"
            )
    else:
        limit_batch_num = limit_test_batches
    pl_model.to("cuda")
    pl_model.zero_grad()
    msg = ""
    for i, batch in enumerate(dataloader):
        if i >= limit_batch_num:
            break
        print(" " * len(msg), end="\r")
        msg = f">>> Testing on training batch {i+1} / {limit_batch_num}"
        print(msg, end="\r")
        batch = data_module.transfer_batch_to_device(batch, torch.device("cuda"), 0)
        loss = pl_model.training_step(batch=batch, batch_idx=i)
        loss.backward()
    print()

    # calculate score of every lora module
    grads_norm = {
        "limit_test_num": limit_batch_num * dataloader.batch_size,
    }
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
                lora.lora_A[lora.active_adapter].weight.grad.norm()
                + lora.lora_B[lora.active_adapter].weight.grad.norm()
            ).item()

            if f"layer_{layer_id}" not in grads_norm:
                grads_norm[f"layer_{layer_id}"] = {}
            grads_norm[f"layer_{layer_id}"][proj_name] = grad_lora

    log_path = f"{save_path}/grad-norm_{save_time}.toml"
    with open(log_path, "w+") as fout:
        toml.dump(grads_norm, fout)
    logger.info("Result saved as toml")

    logger.warning("GRAD_NORM test done")
