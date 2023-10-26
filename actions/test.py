import logging
import os
import pickle

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from tools.checkpoint_load import load_model_chkpt
import pl_model_wrapper

logger = logging.getLogger(__name__)


def test(
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
    if save_path is not None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        tb_logger = TensorBoardLogger(save_dir=save_path, name="logs_test")
        pl_trainer_args["callbacks"] = []
        pl_trainer_args["logger"] = tb_logger

    # TODO: environment plugins

    wrapper_pl_model = pl_model_wrapper.get_model_wrapper(model_info, task)

    if load_name is not None:
        model = load_model_chkpt(load_name, load_type=load_type, model=model)

    plt_model = wrapper_pl_model(
        model,
        dataset_info=dataset_info,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer=optimizer,
    )

    trainer = pl.Trainer(**pl_trainer_args)

    if dataset_info.test_split_available:
        # Testing
        trainer.test(plt_model, datamodule=data_module)
    elif dataset_info.pred_split_available:
        # Predicting, save to predicted_result.pkl
        predicted_results = trainer.predict(plt_model, datamodule=data_module)
        pred_save_name = os.path.join(save_path, "predicted_result.pkl")
        with open(pred_save_name, "wb") as f:
            pickle.dump(predicted_results, f)
        logger.info(f"Predicted results is saved to {pred_save_name}")
    else:
        raise ValueError(
            f"Test or pred split not available for dataset {data_module.name}"
        )
