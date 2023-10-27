import logging
import os
import pickle

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from datasets import load_dataset, load_metric
from pytorch_lightning.loggers import TensorBoardLogger

from tools.checkpoint_load import load_model_chkpt
import pl_model_wrapper
from metrics_callback import ValidationMetricsCallback


logger = logging.getLogger(__name__)


def train(
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
    if save_path is not None:  # if save_path is None, model won't be saved
        # setup callbacks
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # Saving checkpoints
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_path,
            filename="best_chkpt",
            save_top_k=1,
            monitor="val_loss_epoch",
            mode="min",
            save_last=True,
        )
        # Monitoring lr for the lr_scheduler
        lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
        # TensorBoard logger
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_path, name="logs")
        pl_trainer_args["callbacks"] = [checkpoint_callback, lr_monitor_callback]
        pl_trainer_args["logger"] = tb_logger

    # Validation metrics history
    val_history = ValidationMetricsCallback()
    pl_trainer_args["callbacks"].append(val_history)

    # TODO: setup environment plugins if necessary
    plugins = None
    pl_trainer_args["plugins"] = plugins

    wrapper_pl_model = pl_model_wrapper.get_model_wrapper(model_info, task)

    # load model checkpoint
    if load_name is not None:
        model = load_model_chkpt(load_name, load_type=load_type, model=model)

    pl_model: pl.LightningModule = wrapper_pl_model(
        model,
        dataset_info=dataset_info,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=pl_trainer_args["max_epochs"],
        optimizer=optimizer,
    )

    trainer = pl.Trainer(**pl_trainer_args)
    trainer.fit(pl_model, datamodule=data_module)

    # TODO: save the trained model graph if there are architectural changes.
    # NOTE: This is important if the model was previously transformed with architectural
    # changes. The state dictionary that's saved by PyTorch Lightning wouldn't work.

    # match task:
    #     case "classification":
    #         val_metric = val_history.val_history_metrics["val_acc_epoch"]
    #         best_perf = max(val_metric)
    #     case "summarization":
    #         val_metric = trainer.callback_metrics["val_rouge_epoch"]
    #         best_perf = max(val_metric)
    #     case _:
    #         val_metric = trainer.callback_metrics["val_acc_epoch"]
    #         best_perf = max(val_metric)
    val_metric = val_history.val_history_metrics["val_loss_epoch"]
    best_perf = min(val_metric)

    return best_perf
