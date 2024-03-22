import logging

import toml

from dataset import AgsDatasetInfo
from dataset.pl_dataset_module import AgsDataModule
from lora.lora_modules import LoraLinear
from models.model_info import AgsModelInfo
from models.modeling_opt_lora import (
    OPTLoraForCausalLM,
    OPTLoraForQuestionAnswering,
    OPTLoraForSequenceClassification,
    OPTLoraDecoderLayer,
)
from pl_model_wrapper.base import PlWrapperBase

logger = logging.getLogger(__name__)


def constant_test(
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
):
    logger.warning("Running CONSTANT test")

    model = pl_model.model

    # CONSTANT score
    assert (
        type(model) is OPTLoraForCausalLM
        or type(model) is OPTLoraForSequenceClassification
        or type(model) is OPTLoraForQuestionAnswering
    )
    model: OPTLoraForCausalLM | OPTLoraForSequenceClassification | OPTLoraForQuestionAnswering

    # calculate score of every lora module
    score = {}
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

            if f"layer_{layer_id}" not in score:
                score[f"layer{layer_id}"] = {}
            score[f"layer{layer_id}"][proj_name] = 1

    log_path = f"{save_path}/const_{save_time}.toml"
    with open(log_path, "w+") as fout:
        toml.dump(score, fout)
    logger.info("Result saved as toml")

    logger.warning("CONSTANT test done")
