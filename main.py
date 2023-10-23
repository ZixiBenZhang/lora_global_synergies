import logging
import os
import sys

import transformers
import datasets
from datasets import load_dataset
import evaluate
import pytorch_lightning as pl
from torchmetrics.text.rouge import ROUGEScore
import optuna

from loading.argparser_ags import get_arg_parser, CLI_DEFAULTS
from loading.config_load import post_parse_load_config
from loading.setup_model_and_dataset import setup_model_and_dataset
from loading.setup_folders import setup_folder
import actions


def main():
    logger = logging.getLogger("main")

    parser = get_arg_parser()
    args = parser.parse_args(sys.argv)

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

    # TODO: load & apply fine-grained LoRA configuration
    args = post_parse_load_config(args, CLI_DEFAULTS)
    if not args.model or not args.dataset:
        raise ValueError("No model and/or dataset provided! These are required.")

    # TODO: hf type checkpoint loading??

    if args.model is None or args.dataset is None:
        raise ValueError("No model and/or dataset provided.")

    model, model_info, tokenizer, data_module, dataset_info = setup_model_and_dataset(args)

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
                "precision": args.trainer_precision,
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
            }

            actions.test(**test_params)
            logger.info("Testing is completed")


def t():
    datasets_ = load_dataset("glue", "mnli")
    print(type(datasets_))
    rouge = ROUGEScore(rouge_keys=('rouge1', 'rouge2', 'rougeL'))
    print(type(rouge))


if __name__ == "__main__":
    main()
