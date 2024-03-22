import logging
import time

import torch
import pytorch_lightning as pl

from dataset import AgsDatasetInfo
from dataset.pl_dataset_module import AgsDataModule
from importance_testing import get_importance_method
from models.model_info import AgsModelInfo
from tools.checkpoint_load import load_model_chkpt
import pl_model_wrapper

logger = logging.getLogger(__name__)


TEST_BATCH = 32


def pretrain_alloc(
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
    importance_test_name: str,  # name (metric) of the importance test
    importance_test_args: dict,  # arguments for importance tests
):
    t = time.strftime("%H-%M")

    logger.info(f"Running importance test")

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

    # IMPORTANCE TESTING
    importance_method = get_importance_method(importance_test_name)

    importance_method(
        pl_model,
        model_info,
        data_module,
        dataset_info,
        task,
        optimizer,
        learning_rate,
        weight_decay,
        lr_scheduler,
        eta_min,
        pl_trainer_args,
        auto_requeue,
        save_path,
        t,
        load_name,
        load_type,
        resume_training,
        **importance_test_args,
    )

    logger.info(f"Importance test done")
