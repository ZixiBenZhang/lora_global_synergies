import argparse
import os.path
from functools import partial

ACTIONS = ["train", "test"]
TASKS = ["cls"]

CLI_DEFAULTS = {
    # Main program arguments
    # NOTE: The following two are required if a configuration file isn't specified.
    "model": None,
    "dataset": None,
    # General options
    "config": None,
    "task": TASKS[0],
    "load_name": None,
    "load_type": LOAD_TYPE[2],
    "batch_size": 128,
    "to_debug": False,
    "log_level": LOG_LEVELS[1],
    "seed": 0,
    # Trainer options
    "training_optimizer": OPTIMIZERS[0],
    "trainer_precision": TRAINER_PRECISION[1],
    "learning_rate": 1e-5,
    "weight_decay": 0,
    "max_epochs": 20,
    "max_steps": -1,
    "accumulate_grad_batches": 1,
    "log_every_n_steps": 50,
    # Runtime environment options
    "num_workers": os.cpu_count(),
    "num_devices": 1,
    "num_nodes": 1,
    "accelerator": ACCELERATORS[0],
    "strategy": STRATEGIES[0],
    "is_to_auto_requeue": False,
    "github_ci": False,
    "disable_dataset_cache": False,
    # Hardware generation options
    "target": "xcu250-figd2104-2L-e",
    "num_targets": 100,
    # Language model options
    "is_pretrained": False,
    "max_token_len": 512,
    # Project options,
    "project_dir": os.path.join(ROOT, "mase_output"),
    "project": None,
}


def get_arg_parser():
    parser = argparse.ArgumentParser(description="")

    # TODO: use HF arg parser??

    # args for main.py
    main_group = parser.add_argument_group("main arguments")
    main_group.add_argument(
        "action",
        choices=ACTIONS,
        help=f"action to perform. One of {'(' + '|'.join(ACTIONS) + ')'}",
        metavar="ACTION",
    )
    main_group.add_argument(
        "model",
        nargs="?",
        default=None,
        help="name of a supported model. Required if configuration file NOT provided.",
    )
    main_group.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="name of a supported dataset. Required if configuration file NOT provided.",
    )

    # General options
    general_group = parser.add_argument_group("general options")
    general_group.add_argument(
        "--config",
        dest="config",
        type=_valid_filepath,
        help="path to a configuration file in the TOML format.",
        metavar="PATH",
    )
    general_group.add_argument(
        "--task",
        dest="task",
        choices=TASKS,
        help=f"task to perform. One of {'(' + '|'.join(TASKS) + ')'}",
        metavar="TASK",
    )
    general_group.add_argument(
        "--load",
        dest="load_name",
    )
    general_group.add_argument(
        "--load-type",
        dest="load_type",
    )
    general_group.add_argument(
        "--batch-size",
        dest="batch_size",
    )
    general_group.add_argument(
        "--seed",
        dest="seed",
    )
    # debug mode, for detailed logging
    general_group.add_argument(
        "--log-level",
        dest="log_level",
    )

    # Trainer options
    trainer_group = parser.add_argument_group("trainer options")
    trainer_group.add_argument(
        "--optimizer",
        dest="optimizer",
    )
    trainer_group.add_argument(
        "--learning-rate",
        dest="learning_rate",
    )
    trainer_group.add_argument(
        "--weight-decay",
        dest="weight_decay",
    )
    trainer_group.add_argument(
        "--max-epochs",
        dest="max_epochs",
    )
    trainer_group.add_argument(
        "--max-steps",
        dest="max_steps",
    )
    trainer_group.add_argument(
        "--accumulate-grad-batches",
        dest="accumulate_grad_batches",
        type=int,
        help="number of batches to accumulate gradients. (default: %(default)s)",
        metavar="NUM",
    )
    trainer_group.add_argument(
        "--log-every-n-steps",
        dest="log_every_n_steps",
        type=_positive_int,
        help="log every n steps. No logs if num_batches < log_every_n_steps. (default: %(default)s))",
        metavar="NUM",
    )

    # TODO: Runtime environment group

    # Language model options
    lm_group = parser.add_argument_group(title="language model options")
    lm_group.add_argument(
        "--pretrained",
        action="store_true",
        dest="is_pretrained",
        help="load pretrained checkpoint from HuggingFace/Torchvision when initialising models. (default: %(default)s)",
    )
    lm_group.add_argument(
        "--max-token-len",
        dest="max_token_len",
        type=_positive_int,
        help="maximum number of tokens. A negative value will use tokenizer.model_max_length. (default: %(default)s)",
        metavar="NUM",
    )

    # Project-level options
    project_group = parser.add_argument_group(title="project options")
    project_group.add_argument(
        "--project-dir",
        dest="project_dir",
        type=partial(_valid_dir_path, create_dir=True),
        help="directory to save the project to. (default: %(default)s)",
        metavar="DIR",
    )
    project_group.add_argument(
        "--project",
        dest="project",
        help="""
                    name of the project.
                    (default: {MODEL-NAME}_{TASK-TYPE}_{DATASET-NAME}_{TIMESTAMP})
                """,
        metavar="NAME",
    )

    parser.set_defaults(**CLI_DEFAULTS)
    return parser


def _valid_filepath(path: str):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("file not found")
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"expected path to file, got {path!r}")
    return os.path.abspath(path)


def _valid_dir_path(path: str, create_dir: bool):
    if os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"expected path to directory, got file {path!r}")
    if (not os.path.exists(path)) and (not create_dir):
        raise argparse.ArgumentTypeError(f"directory not found")
    if (not os.path.exists(path)) and create_dir:
        os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def _valid_dir_or_file_path(path: str):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("file or directory not found")
    return os.path.abspath(path)


def _positive_int(s: str) -> int | None:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentError(f"expected integer, got {s!r}")
    if v <= 0:
        return None
    return v
