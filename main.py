import logging
import os
import sys
from pprint import pprint

import toml
import torch
import transformers
import datasets
from datasets import load_dataset
import evaluate
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics.text.rouge import ROUGEScore
import optuna
from transformers import AutoTokenizer

import importance_testing
from dataset import get_nlp_dataset_split, get_config_names
from loading.argparser_ags import get_arg_parser, CLI_DEFAULTS
from loading.config_load import post_parse_load_config
from loading.setup_model_and_dataset import setup_model_and_dataset
from loading.setup_folders import setup_folder
import actions
from projectors.shortcut_modules import ShortcutFromIdentity, ShortcutFromZeros


def main():
    logger = logging.getLogger("main")

    parser = get_arg_parser()
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    # if args.to_debug:
    #     sys.excepthook = self._excepthook
    #     self.logger.setLevel(logging.DEBUG)
    #     self.logger.debug("Enabled debug mode.")
    match args.log_level:
        case "debug":
            transformers.logging.set_verbosity_debug()
            datasets.logging.set_verbosity_debug()
            optuna.logging.set_verbosity(optuna.logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        case "info":
            transformers.logging.set_verbosity_warning()
            datasets.logging.set_verbosity_warning()
            # mute optuna's logger by default since it's too verbose
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            logger.setLevel(logging.INFO)
        case "warning":
            transformers.logging.set_verbosity_warning()
            datasets.logging.set_verbosity_warning()
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            logger.setLevel(logging.WARNING)
        case "error":
            transformers.logging.set_verbosity_error()
            datasets.logging.set_verbosity_error()
            optuna.logging.set_verbosity(optuna.logging.ERROR)
            logger.setLevel(logging.ERROR)
        case "critical":
            transformers.logging.set_verbosity(transformers.logging.CRITICAL)
            datasets.logging.set_verbosity(datasets.logging.CRITICAL)
            optuna.logging.set_verbosity(optuna.logging.CRITICAL)
            logger.setLevel(logging.CRITICAL)

    args = post_parse_load_config(args, CLI_DEFAULTS)
    if not args.model or not args.dataset:
        raise ValueError("No model and/or dataset provided! These are required.")

    if args.model is None or args.dataset is None:
        raise ValueError("No model and/or dataset provided.")

    model, model_info, tokenizer, data_module, dataset_info = setup_model_and_dataset(
        args
    )

    output_dir = setup_folder(args)

    match args.action:
        case "train":
            logger.info(f"Training model {args.model!r}...")

            pl_trainer_args = {
                "max_epochs": args.max_epochs,
                "max_steps": args.max_steps,
                "devices": args.num_devices,
                "num_nodes": args.num_nodes,
                "accelerator": args.accelerator,
                "strategy": args.strategy,
                "fast_dev_run": args.to_debug,
                "accumulate_grad_batches": args.accumulate_grad_batches,
                "log_every_n_steps": args.log_every_n_steps,
                "lora_config": args.lora_config,
                "shortcut_config": args.shortcut_config,
            }

            # Load from a checkpoint!
            load_name = None
            load_types = ["pt", "pl"]
            if args.load_name is not None and args.load_type in load_types:
                load_name = args.load_name

            train_params = {
                "model": model,
                "tokenizer": tokenizer,
                "model_info": model_info,
                "data_module": data_module,
                "dataset_info": dataset_info,
                "task": args.task,
                "optimizer": args.training_optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "lr_scheduler": args.lr_scheduler,
                "eta_min": args.eta_min,
                "pl_trainer_args": pl_trainer_args,
                "auto_requeue": args.is_to_auto_requeue,
                "save_path": os.path.join(output_dir, "training_ckpts"),
                "load_name": load_name,
                "load_type": args.load_type,
                "resume_training": args.resume_training,
            }

            logger.info(f"##### WEIGHT DECAY ##### {args.weight_decay}")

            actions.train(**train_params)
            logger.info("Training is completed")

        case "test":
            logger.info(f"Testing model {args.model!r}...")

            pl_trainer_args = {
                "devices": args.num_devices,
                "num_nodes": args.num_nodes,
                "accelerator": args.accelerator,
                "strategy": args.strategy,
            }

            # The checkpoint must be present, except when the model is pretrained.
            if args.load_name is None and not args.is_pretrained:
                raise ValueError("expected checkpoint via --load, got None")

            test_params = {
                "model": model,
                "tokenizer": tokenizer,
                "model_info": model_info,
                "data_module": data_module,
                "dataset_info": dataset_info,
                "task": args.task,
                "optimizer": args.training_optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "pl_trainer_args": pl_trainer_args,
                "auto_requeue": args.is_to_auto_requeue,
                "save_path": os.path.join(output_dir, "checkpoints"),
                "load_name": args.load_name,
                "load_type": args.load_type,
                "alpha": args.alpha,
            }

            actions.test(**test_params)
            logger.info("Testing is completed")

        case "imp-test":
            logger.info(f"Conducting importance test on model {args.model!r}...")

            pl_trainer_args = {
                "max_epochs": args.max_epochs,
                "max_steps": args.max_steps,
                "devices": args.num_devices,
                "num_nodes": args.num_nodes,
                "accelerator": args.accelerator,
                "strategy": args.strategy,
                "fast_dev_run": args.to_debug,
                "accumulate_grad_batches": args.accumulate_grad_batches,
                "log_every_n_steps": args.log_every_n_steps,
            }

            # Load from a checkpoint!
            load_name = None
            load_types = ["pt", "pl"]
            if args.load_name is not None and args.load_type in load_types:
                load_name = args.load_name

            importance_test_args = {
                "limit_test_batches": args.imp_limit_test_batches,
                "metric_reduction_tolerance": args.metric_red_tolerance,
                "limit_zero_proxy_train_batches": args.alpha_limit_zptrain_batches,
                "alpha": args.alpha,
            }

            test_params = {
                "model": model,
                "tokenizer": tokenizer,
                "model_info": model_info,
                "data_module": data_module,
                "dataset_info": dataset_info,
                "task": args.task,
                "optimizer": args.training_optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "lr_scheduler": args.lr_scheduler,
                "eta_min": args.eta_min,
                "pl_trainer_args": pl_trainer_args,
                "auto_requeue": args.is_to_auto_requeue,
                "save_path": os.path.join(output_dir, "importance_ckpts"),
                "load_name": load_name,
                "load_type": args.load_type,
                "resume_training": args.resume_training,
                "importance_test_name": args.importance_test_name,
                "importance_test_args": importance_test_args,
            }

            actions.pretrain_alloc(**test_params)
            logger.info("Importance test is completed")

        case "train-dyrealloc":
            logger.info(f"Dynamic-LoRA-reallocation training model {args.model!r}...")

            pl_trainer_args = {
                "max_epochs": args.max_epochs,
                "max_steps": args.max_steps,
                "devices": args.num_devices,
                "num_nodes": args.num_nodes,
                "accelerator": args.accelerator,
                "strategy": args.strategy,
                "fast_dev_run": args.to_debug,
                "accumulate_grad_batches": args.accumulate_grad_batches,
                "log_every_n_steps": args.log_every_n_steps,
            }

            # Load from a checkpoint!
            load_name = None
            load_types = ["pt", "pl"]
            if args.load_name is not None and args.load_type in load_types:
                load_name = args.load_name

            importance_test_args = {
                "limit_test_batches": args.imp_limit_test_batches,
                "metric_reduction_tolerance": args.metric_red_tolerance,
                "limit_zero_proxy_train_batches": args.alpha_limit_zptrain_batches,
                "alpha": args.alpha,
            }

            dyrealloc_args = {
                "realloc_N": args.realloc_N,
                "turn_on_percentile": args.turn_on_percentile,
                "ags_mode": args.dyrealloc_ags_mode,
            }

            train_dyrealloc_params = {
                "model": model,
                "tokenizer": tokenizer,
                "model_info": model_info,
                "data_module": data_module,
                "dataset_info": dataset_info,
                "task": args.task,
                "optimizer": args.training_optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "lr_scheduler": args.lr_scheduler,
                "eta_min": args.eta_min,
                "pl_trainer_args": pl_trainer_args,
                "auto_requeue": args.is_to_auto_requeue,
                "save_path": os.path.join(output_dir, "dyrealloc_ckpts"),
                "load_name": load_name,
                "load_type": args.load_type,
                "resume_training": args.resume_training,
                "dynamic_reallocation_args": dyrealloc_args,
                "importance_test_name": args.importance_test_name,
                "importance_test_args": importance_test_args,
            }

            logger.info(f"##### WEIGHT DECAY ##### {args.weight_decay}")

            actions.train_dynamic_reallocation(**train_dyrealloc_params)
            logger.info("Dynamic-LoRA-reallocation training is completed")


def t():
    s = ShortcutFromZeros(
        in_out_features=5,
        config={
            "r": 2,
            "shortcut_alpha": 2,
            "proj_dropout": 0.0,
            "projector_name": "test",
            "disable_projector": False,
        },
    )
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    t = s(x)
    y = torch.sum(t + x)
    y.backward()
    print(t, y)
    print(x.grad)

    # print(datasets.get_dataset_split_names("tatsu-lab/alpaca", None))

    # with open("ags_output/opt_lora_classification_mrpc_2024-02-01/checkpoints/logs_test/importance_08-22.toml", "r") as f:
    #     data = toml.load(f)
    # for mat, res in data.items():
    #     if res["acc_reduction"] != 0:
    #         print(mat, res)


if __name__ == "__main__":
    main()
