import copy
import logging
import os
import pickle
import time
import types

import toml
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from lightning_fabric.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
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
from tools.checkpoint_load import load_model_chkpt
import pl_model_wrapper
from tools.trainable_param_printer import print_trainable_parameters

logger = logging.getLogger(__name__)


TEST_BATCH = 32


def snip_test(
    model: torch.nn.Module | torch.fx.GraphModule,
    tokenizer,
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
    load_name,  # path to the saved checkpoint
    load_type,  # model checkpoint's type: ['pt', 'pl']
    resume_training,  # whether resume zero-proxy trained model from the checkpoint
    limit_test_num,  # number of data row limit for the zero-proxy test
):
    t = time.strftime("%H-%M")

    logger.warning("Running SNIP importance test")

    if save_path is not None:  # if save_path is None, model won't be saved
        # setup callbacks
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # TensorBoard logger
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_path, name="logs")
        pl_trainer_args["logger"] = [tb_logger]

    wrapper_pl_model: pl.LightningModule = pl_model_wrapper.get_model_wrapper(
        model_info, task
    )

    # load model state from checkpoint
    if load_name is not None:
        model = load_model_chkpt(load_name, load_type=load_type, model=model)
        logger.warning(
            f"Restore model state from pl checkpoint {load_name}. Entered model hyperparameter configuration ignored."
        )

    pl_model: pl.LightningModule = wrapper_pl_model(
        model,
        dataset_info=dataset_info,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,  # for building lr scheduler
        eta_min=eta_min,  # for building lr scheduler
        epochs=pl_trainer_args["max_epochs"],
        optimizer=optimizer,
    )

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

    def get_alpha_test_dataloader(datamodule: AgsDataModule):
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
    dataloader = get_alpha_test_dataloader(data_module)

    # SNIP
    assert (
        limit_test_num % dataloader.batch_size == 0
    ), f"Test number limit must be dividable by batch size. Got test number limit {limit_test_num}, batch size {dataloader.batch_size}."
    assert (
        type(model) is OPTLoraForCausalLM
        or type(model) is OPTLoraForSequenceClassification
        or type(model) is OPTLoraForQuestionAnswering
    )
    model: OPTLoraForCausalLM | OPTLoraForSequenceClassification | OPTLoraForQuestionAnswering

    for name, param in model.named_parameters():
        param.requires_grad = False

    # add weight masks
    def lora_forward(self, x):
        if self.active_adapter not in self.lora_A.keys():
            res = F.linear(
                x, self.weight if not self.fan_in_fan_out else self.weight.T, self.bias
            )
        elif self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            res = F.linear(
                x, self.weight if not self.fan_in_fan_out else self.weight.T, self.bias
            )
        else:
            self.unmerge()
            res = F.linear(
                x, self.weight if not self.fan_in_fan_out else self.weight.T, self.bias
            )
            res = (
                res
                + (
                    F.linear(
                        F.linear(
                            self.lora_dropout[self.active_adapter](x),
                            self.lora_A[self.active_adapter].weight
                            * self.weight_mask_A,
                        ),
                        self.lora_B[self.active_adapter].weight * self.weight_mask_B,
                    )
                )
                * self.scaling[self.active_adapter]
            )
        # res = res.to(input_dtype)
        return res

    for decoder_layer in reversed(model.model.decoder.layers):
        decoder_layer: OPTLoraDecoderLayer
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

            lora_A: nn.Linear = lora.lora_A[lora.active_adapter]
            lora_A.weight.requires_grad = False
            lora.weight_mask_A = nn.Parameter(torch.ones_like(lora_A.weight))
            lora_B: nn.Linear = lora.lora_B[lora.active_adapter]
            lora_B.weight.requires_grad = False
            lora.weight_mask_B = nn.Parameter(torch.ones_like(lora_B.weight))

            lora.forward = types.MethodType(lora_forward, lora)

    # compute gradients
    pl_model.zero_grad()
    msg = ""
    limit_batch_num = limit_test_num // dataloader.batch_size
    for i, batch in enumerate(dataloader):
        if i >= limit_batch_num:
            break
        print(" " * len(msg), end="\r")
        msg = f"Testing on training batch {i+1} / {limit_batch_num}"
        print(msg, end="\r")
        loss = pl_model.training_step(batch=batch, batch_idx=i)
        loss.backward()
    print()

    # calculate score of every lora module
    grads_abs = {
        "limit_test_num": limit_test_num,
    }
    for decoder_layer in reversed(model.model.decoder.layers):
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
                torch.sum(torch.abs(lora.weight_mask_A.grad))
                + torch.sum(torch.abs(lora.weight_mask_B))
            ).item()

            if f"layer_{layer_id}" not in grads_abs:
                grads_abs[f"layer{layer_id}"] = {}
            grads_abs[f"layer{layer_id}"][proj_name] = grad_lora

    log_path = f"{save_path}/snip_{t}.toml"
    with open(log_path, "w+") as fout:
        toml.dump(grads_abs, fout)
    logger.info("Result saved as toml")

    logger.warning("SNIP test done")
