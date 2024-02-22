import logging
import os
import pickle
import time

import toml
import torch
import pytorch_lightning as pl
from lightning_fabric.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from lora.lora_modules import LoraLinear
from models.modeling_opt_lora import (
    OPTLoraForCausalLM,
    OPTLoraForQuestionAnswering,
    OPTLoraForSequenceClassification, OPTLoraDecoderLayer,
)
from tools.checkpoint_load import load_model_chkpt
import pl_model_wrapper

logger = logging.getLogger(__name__)


def alpha_importance_test(
    model: torch.nn.Module | torch.fx.GraphModule,
    tokenizer,
    model_info,  # dataclass of model's task type and name
    data_module: pl.LightningDataModule,  # for preparing and loading datasets for pl trainer
    dataset_info,  # dataclass including e.g. number of classes for the pl model wrapper
    task,  # to decide the pl model wrapper of which type should be used
    optimizer,  # optimizer for pl trainer
    learning_rate,  # lr for optimizer. lr_scheduler is default as CosineAnnealingLR
    weight_decay,  # weight_decay for optimizer
    pl_trainer_args,  # args for pl trainer; include e.g. "max_epochs" for setting up lr_scheduler
    auto_requeue,  # for setting up SLURMEnvironment, environment for distributed launch
    save_path,  # path for saving checkpoints
    load_name,  # path to the saved checkpoint
    load_type,  # model checkpoint's type: ['pt', 'pl']
):
    t = time.strftime("%H-%M")

    if save_path is not None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # tb_logger = TensorBoardLogger(save_dir=save_path, name="logs_test")
        # csv_logger = CSVLogger(save_dir=save_path, name="alpha_importance_logs")
        pl_trainer_args["callbacks"] = []
        pl_trainer_args["logger"] = []

    if auto_requeue is not None:
        plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]
    else:
        plugins = None
    pl_trainer_args["plugins"] = plugins

    wrapper_pl_model: pl.LightningModule = pl_model_wrapper.get_model_wrapper(
        model_info, task
    )

    # load model from pl checkpoint
    if load_name is None:
        raise ValueError(
            "Path to checkpoint required for resuming training. Please use --load PATH."
        )
    model = load_model_chkpt(load_name, load_type=load_type, model=model)
    # if load_type != "pl":
    #     raise ValueError("Load-type pl is required for resuming training. Please use --load-type pl.")
    logger.warning(
        f"Restore model state from pl checkpoint {load_name}. Entered hyperparameter configuration ignored."
    )

    pl_model = wrapper_pl_model.load_from_checkpoint(load_name, model=model)

    # Run each test only on one minibatch
    trainer = pl.Trainer(**pl_trainer_args, limit_val_batches=1)

    logger.warning(
        f"Using perplexity as the testing metrics. Currently only causal LM models supported."
    )
    assert task == "causal_language_modeling"

    original_val_metrics = trainer.validate(pl_model, datamodule=data_module)[0]
    original_perplexity = original_val_metrics["val_perplexity_epoch"]

    threshold_perplexity = original_perplexity + 0.005 * original_perplexity

    # Result format: {layer_idx: {proj: alpha}}
    res_val: dict[str, str | float | dict[str, float]] = {
        "task": task,
        "metrics": "perplexity",
        "original_perplexity": original_perplexity,
    }

    # FOR RESUMING ALPHA TESTING
    resume_layer_id = -1
    resume_toml = ""

    with torch.no_grad():
        assert (
            type(model) is OPTLoraForCausalLM
            # or type(model) is OPTLoraForSequenceClassification
            # or type(model) is OPTLoraForQuestionAnswering
        )
        model: OPTLoraForCausalLM  # | OPTLoraForSequenceClassification | OPTLoraForQuestionAnswering

        # Resume alpha testing
        if resume_layer_id > 0:
            with open(resume_toml, "r") as f:
                res_val = toml.load(f)
            logger.warning(f"Resuming alpha testing from layer {resume_layer_id} based on {resume_toml}")

        def save_toml(res: dict):
            log_path = f"{save_path}/alpha-importance_{t}.toml"
            with open(log_path, "w+") as fout:
                toml.dump(res, fout)
            logger.info("Result saved as toml")

        for decoder_layer in model.model.decoder.layers:
            decoder_layer: OPTLoraDecoderLayer
            layer_id = decoder_layer.layer_id

            if layer_id < resume_layer_id:
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
                adapter_name = lora.active_adapter
                alpha_res = 1.0
                logger.info(f">>> Testing layer {layer_id} projection {proj_name} <<<")

                alpha = 0.5
                lora.importance_alpha[adapter_name] = alpha
                val_metrics = trainer.validate(pl_model, datamodule=data_module)[0]
                perplexity = val_metrics["val_perplexity_epoch"]

                if perplexity >= threshold_perplexity:
                    while alpha < 1.0:
                        alpha += 0.1
                        lora.importance_alpha[adapter_name] = alpha
                        val_metrics = trainer.validate(pl_model, datamodule=data_module)[0]
                        perplexity = val_metrics["val_perplexity_epoch"]
                        if perplexity < threshold_perplexity:
                            alpha_res = alpha - 0.1
                            break
                else:
                    while alpha > 0.0:
                        alpha -= 0.1
                        lora.importance_alpha[adapter_name] = alpha
                        val_metrics = trainer.validate(pl_model, datamodule=data_module)[0]
                        perplexity = val_metrics["val_perplexity_epoch"]
                        if perplexity >= threshold_perplexity:
                            alpha_res = alpha
                            break

                res_val[f"layer_{layer_id}"][proj_name] = alpha_res
                save_toml(res_val)

        save_toml(res_val)
