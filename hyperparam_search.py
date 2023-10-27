import logging
import os

import torch
import transformers
import datasets
import pytorch_lightning as pl
import optuna

from loading.argparser_ags import get_arg_parser, CLI_DEFAULTS
from loading.config_load import post_parse_load_config
from loading.setup_model_and_dataset import setup_model_and_dataset
from loading.setup_folders import setup_folder
import actions


def search_lr_objective(trial: optuna.Trial):
    lr_suggested = trial.suggest_float("lr", 1e-8, 1e-5)

    logger = logging.getLogger("main")

    parser = get_arg_parser()
    args = parser.parse_args()

    args.__setattr__("project_dir", "./ags_lr_search")
    args.__setattr__("learning_rate", lr_suggested)

    pl.seed_everything(args.seed)

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
                "pl_trainer_args": pl_trainer_args,
                "auto_requeue": args.is_to_auto_requeue,
                "save_path": os.path.join(output_dir, "training_ckpts"),
                "load_name": load_name,
                "load_type": args.load_type,
            }

            logger.info(f"##### WEIGHT DECAY ##### {args.weight_decay}")

            best_perf = actions.train(**train_params)
            logger.info("Training is completed")

            return best_perf

        case _:
            raise ValueError("Invalid action. Optuna hyperparameter only support 'train' action.")


def search():
    search_space = {"lr": [1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8]}
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(search_lr_objective)


if __name__ == "__main__":
    search()
