import logging
import os
import pickle
import time

import pytorch_lightning as pl
import torch
from lightning_fabric.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import TensorBoardLogger

import pl_model_wrapper
from lora.lora_modules import LoraLinear
from pl_callbacks.val_callback import MMLUValidationCallback
from tools.checkpoint_load import load_model_chkpt
from tools.mmlu_load import setup_mmlu

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
    ags_config_paths,  # for logging in Tensorboard
    seed,  # for logging in Tensorboard
    mmlu_mode,  # zero-shot/few-shot for MMLU in validation
    mmlu_args,  # arguments for MMLUValidationCallback
):
    t = time.strftime("%H-%M")

    if save_path is not None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        tb_logger = TensorBoardLogger(save_dir=save_path, name="logs_test")
        tb_logger.log_hyperparams(ags_config_paths)
        tb_logger.log_hyperparams({"seed": seed})
        pl_trainer_args["callbacks"] = []
        pl_trainer_args["logger"] = tb_logger

    # MMLU validation callback
    mmlu_test = None
    if mmlu_mode is not None:
        # mmlu_val_callback = MMLUValidationCallback(
        #     **mmlu_args, few_shot=(mmlu_mode == "fs")
        # )
        # pl_trainer_args["callbacks"].insert(0, mmlu_val_callback)

        _, mmlu_test_getter = setup_mmlu(**mmlu_args, few_shot=(mmlu_mode == "fs"))
        mmlu_test = mmlu_test_getter()

    if auto_requeue is not None:
        plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]
    else:
        plugins = None
    pl_trainer_args["plugins"] = plugins

    wrapper_pl_model: pl.LightningModule = pl_model_wrapper.get_model_wrapper(
        model_info, task if mmlu_mode is None else task + "-mmlu"
    )

    # load model from pl checkpoint
    if load_name is None:
        # raise ValueError(
        #     "Path to checkpoint required for resuming training. Please use --load PATH."
        # )
        pl_model: pl.LightningModule = wrapper_pl_model(
            model,
            dataset_info=dataset_info,
        ) if mmlu_mode is None else wrapper_pl_model(
            model,
            dataset_info=dataset_info,
            tokenizer=tokenizer,
        )
    else:
        if load_type == "pl":
            model = load_model_chkpt(load_name, load_type=load_type, model=model)
            logger.warning(
                f"Running test from pl checkpoint {load_name}. Entered hyperparameter configuration ignored."
            )
            pl_model: pl.LightningModule = wrapper_pl_model(
                model,
                dataset_info=dataset_info,
                tokenizer=tokenizer,
            )
            logger.warning(f"Resuming hyperparameters: {pl_model.hparams}")
        else:
            pl_model: pl.LightningModule = wrapper_pl_model(
                model,
                dataset_info=dataset_info,
            ) if mmlu_mode is None else wrapper_pl_model(
                model,
                dataset_info=dataset_info,
                tokenizer=tokenizer,
            )

    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            print(name, module.disable_adapters)
            if not module.disable_adapters:
                print(module.lora_B[module.active_adapter].weight)

    trainer = pl.Trainer(**pl_trainer_args)

    if mmlu_mode is not None:
        # Testing MMLU
        trainer.test(pl_model, dataloaders=mmlu_test)
    elif dataset_info.test_split_available:
        # Testing
        trainer.test(pl_model, datamodule=data_module)
    elif dataset_info.pred_split_available:
        # Predicting, save to predicted_result.pkl
        predicted_results = trainer.predict(pl_model, datamodule=data_module)
        pred_save_name = os.path.join(save_path, "predicted_result.pkl")
        with open(pred_save_name, "wb") as f:
            pickle.dump(predicted_results, f)
        logger.info(f"Predicted results is saved to {pred_save_name}")
    else:
        raise ValueError(
            f"Test or pred split not available for dataset {data_module.name}"
        )
