import logging
import os
import time

import torch
import pytorch_lightning as pl
from lightning_fabric.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import TensorBoardLogger

from lora.lora_modules import (
    update_lora_importance_alpha_require_grad,
    reset_lora,
)
from models.model_info import AgsModelInfo
from pl_callbacks.val_callback import MMLUValidationCallback
from projectors.shortcut_modules import (
    update_ags_importance_beta_require_grad,
    update_ags_ln_require_grad,
    reset_shortcut,
)
from tools.checkpoint_load import load_model_chkpt
import pl_model_wrapper
from pl_callbacks.metrics_callback import ValidationMetricsCallback
from tools.mmlu_load import setup_mmlu
from tools.trainable_param_printer import print_trainable_parameters

logger = logging.getLogger(__name__)


def train(
    model: torch.nn.Module | torch.fx.GraphModule,
    tokenizer,
    model_info: AgsModelInfo,  # dataclass of model's task type and name
    data_module: pl.LightningDataModule,  # for preparing and loading datasets for pl trainer
    dataset_info,  # dataclass including e.g. number of classes for the pl model wrapper
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
    resume_training,  # whether resume full training from the checkpoint
    ags_config_paths,  # for logging in Tensorboard
    seed,  # for logging in Tensorboard
    mmlu_mode,  # zero-shot/few-shot for MMLU in validation
    mmlu_args,  # arguments for MMLUValidationCallback
):
    t = time.strftime("%H-%M")

    if save_path is not None:  # if save_path is None, model won't be saved
        # setup callbacks
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # Saving checkpoints
        task_metric = {
            "classification": ("val_acc_epoch", "max"),
            "summarization": ("val_rouge_epoch", "max"),
            "causal_language_modeling": ("val_perplexity_epoch", "min"),
        }
        best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_path,
            filename=f"best_chkpt-{mmlu_mode}" + "-{epoch}-" + t,
            save_top_k=1 if mmlu_mode is None else -1,
            monitor=task_metric[task][0],
            mode=task_metric[task][1],
            # save_last=True,
        )
        latest_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_path,
            filename=f"last_chkpt-{mmlu_mode}" + "-{epoch}-" + t,
            # save_last=True,
        )
        # Monitoring lr for the lr_scheduler
        lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
        # TensorBoard logger
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_path, name="logs")
        tb_logger.log_hyperparams(ags_config_paths)
        tb_logger.log_hyperparams({"seed": seed})
        # csv_logger = pl.loggers.CSVLogger(save_dir=save_path, name="csv_logs")
        # wandb_logger = pl.loggers.WandbLogger(save_dir=save_path, name="wandb_logs")
        pl_trainer_args["callbacks"] = [
            # best_checkpoint_callback,
            # latest_checkpoint_callback,
            lr_monitor_callback,
        ]
        pl_trainer_args["logger"] = [tb_logger]

    # MMLU validation callback
    mmlu_val = None
    if mmlu_mode is not None:
        mmlu_val_callback = MMLUValidationCallback(
            **mmlu_args, few_shot=(mmlu_mode == "fs")
        )
        pl_trainer_args["callbacks"].insert(0, mmlu_val_callback)

        mmlu_val_getter, _ = setup_mmlu(
            **mmlu_args, few_shot=(mmlu_mode == "fs")
        )
        mmlu_val = mmlu_val_getter()

    # Validation metrics history, for hyperparameter search
    val_history = ValidationMetricsCallback()
    pl_trainer_args["callbacks"].append(val_history)

    if auto_requeue is not None:
        plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]
    else:
        plugins = None
    pl_trainer_args["plugins"] = plugins

    wrapper_pl_model: pl.LightningModule = pl_model_wrapper.get_model_wrapper(
        model_info, task
    )

    if resume_training:
        # resume full training from pl checkpoint
        if load_name is None:
            raise ValueError(
                "Path to checkpoint required for resuming training. Please use --load PATH."
            )
        # model = load_model_chkpt(load_name, load_type=load_type, model=model)

        if load_type != "pl":
            raise ValueError(
                "Load-type pl is required for resuming full training state. Please use --load-type pl."
            )
        logger.warning(
            f"Resume full training state from pl checkpoint {load_name}. Entered hyperparameters and configuration ignored."
        )

        trainable_params = []
        if model_info.is_lora:
            # reset_lora(model)
            trainable_params.append("lora_")
        if model_info.is_ags:
            # reset_shortcut(model)
            trainable_params.append("proj_")
            # trainable_params.append("shortcut_ln_")

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
        update_ags_importance_beta_require_grad(model, require_grad=False)
        # update_ags_ln_require_grad(model, require_grad=False)
        print_trainable_parameters(model)

        pl_model = wrapper_pl_model.load_from_checkpoint(load_name, model=model)

        logger.warning(f"Resuming hyperparameters: {pl_model.hparams}")

        trainer = pl.Trainer(**pl_trainer_args)
        trainer.fit(
            pl_model,
            datamodule=data_module,
            ckpt_path=load_name,
        )
    else:
        # load model state checkpoint
        if load_name is not None:
            model = load_model_chkpt(load_name, load_type=load_type, model=model)

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

        trainer = pl.Trainer(
            **pl_trainer_args, limit_train_batches=0.05, enable_checkpointing=False
        )
        trainer.fit(pl_model, datamodule=data_module)

    # mmlu_val_getter, _ = setup_mmlu(**mmlu_args, few_shot=True)
    # mmlu_val = mmlu_val_getter()
    trainer.test(pl_model, dataloaders=mmlu_val)
