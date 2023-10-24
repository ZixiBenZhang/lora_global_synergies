import argparse
import os.path
from functools import partial
from pathlib import Path


ROOT = Path(__file__).parent.parent.absolute()
ACTIONS = ["train", "test"]
TASKS = ["classification", "causal_language_modeling", "summarization"]
LOAD_TYPE = [
    "pt",  # PyTorch module state dictionary
    "pl",  # PyTorch Lightning checkpoint
    "hf",  # HuggingFace's checkpoint directory saved by 'save_pretrained'
]
OPTIMIZERS = ["adam", "sgd", "adamw"]
LOG_LEVELS = ["debug", "info", "warning", "error", "critical"]
STRATEGIES = [
    "ddp",
    "ddp_find_unused_parameters_true",
    # "fsdp",
    # "fsdp_native",
    # "fsdp_custom",
    # "deepspeed_stage_3_offload",
]
ACCELERATORS = ["auto", "cpu", "gpu"]

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
    # "trainer_precision": TRAINER_PRECISION[1],
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
    parser = argparse.ArgumentParser(description="", add_help=False)

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
        type=_valid_dir_or_file_path,
        help="path to load the model from. (default: %(default)s)",
        metavar="PATH",
    )
    general_group.add_argument(
        "--load-type",
        dest="load_type",
        choices=LOAD_TYPE,
        help=f"""
                    the type of checkpoint to be loaded; it's disregarded if --load is NOT
                    specified. It is designed to and must be used in tandem with --load.
                    One of {'(' + '|'.join(LOAD_TYPE) + ')'} (default: %(default)s)
                """,
        metavar="LOAD_TYPE",
    )
    general_group.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="batch size for training and evaluation. (default: %(default)s)",
        metavar="NUM",
    )
    general_group.add_argument(
        "--seed",
        dest="seed",
        type=int,
        help="""
                    seed for random number generators set via Pytorch Lightning's
                    seed_everything function. (default: %(default)s)
                """,
        metavar="NUM",
    )
    # debug mode, for detailed logging
    general_group.add_argument(
        "--log-level",
        dest="log_level",
        choices=LOG_LEVELS,
        help=f"""
                    verbosity level of the logger; it's only effective when --debug flag is
                    NOT passed in. One of {'(' + '|'.join(LOG_LEVELS) + ')'}
                    (default: %(default)s)
                """,
        metavar="",
    )

    # Trainer options
    trainer_group = parser.add_argument_group("trainer options")
    trainer_group.add_argument(
        "--optimizer",
        dest="optimizer",
        choices=OPTIMIZERS,
        help=f"""
                    name of supported optimiser for training. One of
                    {'(' + '|'.join(OPTIMIZERS) + ')'} (default: %(default)s)
                """,
        metavar="TYPE",
    )
    trainer_group.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        help="initial learning rate for training. (default: %(default)s)",
        metavar="NUM",
    )
    trainer_group.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        help="initial learning rate for training. (default: %(default)s)",
        metavar="NUM",
    )
    trainer_group.add_argument(
        "--max-epochs",
        dest="max_epochs",
        type=int,
        help="maximum number of epochs for training. (default: %(default)s)",
        metavar="NUM",
    )
    trainer_group.add_argument(
        "--max-steps",
        dest="max_steps",
        type=_positive_int,
        help="""
                    maximum number of steps for training. A negative value disables this
                    option. (default: %(default)s)
                """,
        metavar="NUM",
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

    # Runtime environment options
    runtime_group = parser.add_argument_group("runtime environment options")
    runtime_group.add_argument(
        "--cpu",
        "--num-workers",
        dest="num_proc",
        type=_positive_int,
        help="""
                    number of CPU workers; the default varies across systems and is set to
                    os.cpu_count(). (default: %(default)s)
                """,
        metavar="NUM",
    )
    runtime_group.add_argument(
        "--gpu",
        "--num-devices",
        dest="num_devices",
        type=_positive_int,
        help="number of GPU devices. (default: %(default)s)",
        metavar="NUM",
    )
    runtime_group.add_argument(
        "--nodes",
        dest="num_nodes",
        type=int,
        help="number of nodes. (default: %(default)s)",
        metavar="NUM",
    )
    runtime_group.add_argument(
        "--accelerator",
        dest="accelerator",
        choices=ACCELERATORS,
        help=f"""
                    type of accelerator for training. One of
                    {'(' + '|'.join(ACCELERATORS) + ')'} (default: %(default)s)
                """,
        metavar="TYPE",
    )
    runtime_group.add_argument(
        "--strategy",
        dest="strategy",
        choices=STRATEGIES,
        help=f"""
                    type of strategy for training. One of
                    {'(' + '|'.join(STRATEGIES) + ')'} (default: %(default)s)
                """,
        metavar="TYPE",
    )
    runtime_group.add_argument(
        "--disable-dataset-cache",
        dest="disable_dataset_cache",
        action="store_true",
        help="""
                    disable caching of datasets. (default: %(default)s)
                """,
    )
    # TODO: args for runtime environment SLURM and GitHub CI
    # runtime_group.add_argument(
    #     "--auto-requeue",
    #     dest="is_to_auto_requeue",
    #     action="store_true",
    #     help="""
    #                 enable automatic job resubmission on SLURM managed cluster. (default:
    #                 %(default)s)
    #             """,
    # )
    # runtime_group.add_argument(
    #     "--github-ci",
    #     action="store_true",
    #     dest="github_ci",
    #     help="""
    #                 set the execution environment to GitHub's CI pipeline; it's used in the
    #                 MASE verilog emitter transform pass to skip simulations.
    #                 (default: %(default)s)
    #                 """,
    # )

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
        raise argparse.ArgumentError(None, f"expected integer, got {s!r}")
    if v <= 0:
        return None
    return v
