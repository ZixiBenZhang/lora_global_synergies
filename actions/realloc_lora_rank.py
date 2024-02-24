import copy
import logging
import math
import os
import pickle
import time

import toml
import torch
import pytorch_lightning as pl
from lightning_fabric.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from dataset import AgsDatasetInfo
from lora.lora_modules import LoraLinear, mark_only_lora_as_trainable, update_lora_importance_alpha_require_grad
from models.model_info import AgsModelInfo
from models.modeling_opt_lora import (
    OPTLoraForCausalLM,
    OPTLoraForQuestionAnswering,
    OPTLoraForSequenceClassification, OPTLoraDecoderLayer,
)
from projectors.shortcut_modules import mark_ags_as_trainable
from tools.checkpoint_load import load_model_chkpt
import pl_model_wrapper
from tools.trainable_param_printer import print_trainable_parameters

logger = logging.getLogger(__name__)

LORA_NAME_MAP = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "out_proj": "o_proj",
    "fc1": "w1",
    "fc2": "w2",
}

QUANTILE = 0.75


def reallocate_lora_rank(
    model: torch.nn.Module | torch.fx.GraphModule,
    tokenizer,
    model_info: AgsModelInfo,  # dataclass of model's task type and name
    data_module: pl.LightningDataModule,  # for preparing and loading datasets for pl trainer
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
    metric_reduction_tolerance,  # for calculating alpha threshold
):
    t = time.strftime("%m-%d")

    assert (
            type(model) is OPTLoraForCausalLM
            or type(model) is OPTLoraForSequenceClassification
            or type(model) is OPTLoraForQuestionAnswering
    )
    model: OPTLoraForCausalLM | OPTLoraForSequenceClassification | OPTLoraForQuestionAnswering

    with open(
        ".toml",
        "r",
    ) as f:
        alpha_dict = toml.load(f)

    alpha_dict: dict[str, dict[str, float]] = {k: v for k, v in alpha_dict.items() if "layer" in k}

    alpha_threshold = torch.quantile(
        torch.tensor([d.values() for d in alpha_dict.values()]),
        QUANTILE,
        interpolation="lower",
    )
    budget = math.floor(sum([len(d.keys()) for d in alpha_dict.values()]) * (1 - QUANTILE))

    # Constant new rank
    new_r = round(8 / (1 - QUANTILE))

    lora_new_config: dict[str, str | dict[str, int | float | str | dict[str, int | float | str]]] = {
        "granularity": "layer",
        "default": {
            "r": 0,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "adapter_name": f"ags-layer-importance-{dataset_info.name}",
            "init_lora_weghts": True,
            "fan_in_fan_out": False,
            "disable_adapter": True,
        },
    }

    with torch.no_grad():
        # prioritise later layers
        for decoder_layer in reversed(model.model.decoder.layers):
            if budget <= 0:
                break

            decoder_layer: OPTLoraDecoderLayer
            layer_id = decoder_layer.layer_id
            if f"layer_{layer_id}" not in alpha_dict:
                continue

            lora_modules: dict[str, LoraLinear] = {
                "q_proj": decoder_layer.self_attn.q_proj,
                "k_proj": decoder_layer.self_attn.k_proj,
                "v_proj": decoder_layer.self_attn.v_proj,
                "out_proj": decoder_layer.self_attn.out_proj,
                "fc1": decoder_layer.fc1,
                "fc2": decoder_layer.fc2,
            }

            for proj_name, lora in lora_modules.items():
                if budget <= 0:
                    break

                if proj_name not in alpha_dict[f"layer_{layer_id}"]:
                    continue

                alpha = alpha_dict[f"layer_{layer_id}"][proj_name]
                if alpha >= alpha_threshold:
                    lora_new_config[f"model_layer_{layer_id}"][LORA_NAME_MAP[proj_name]] = {
                        "r": new_r,
                        "lora_alpha": 16,
                        "lora_dropout": 0.0,
                        "adapter_name": f"ags-layer-importance-{dataset_info.name}",
                        "disable_adapter": False,
                    }
                    budget -= 1

        i = 0
        while os.path.isfile(os.path.join(save_path, f"version_{i}.toml")):
            i += 1
        config_path = os.path.join(save_path, f"version_{i}.toml")
        with open(config_path, "w+") as fout:
            toml.dump(lora_new_config, fout)
        logger.info(f"New lora config saved to {config_path}")
