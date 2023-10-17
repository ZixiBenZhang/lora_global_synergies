import logging
import sys

import datasets
import pytorch_lightning as pl
from datasets import load_dataset

from loading.argparser_lora import get_arg_parser
from loading.model_and_dataset_loader import setup_model_and_dataset


def main():
    logger = logging.getLogger("main")

    parser = get_arg_parser()
    args = parser.parse_args(sys.argv)

    pl.seed_everything(args.seed)

    # TODO: setup log level

    # TODO: turn config path to dict to attributes of args

    if args.model is None or args.dataset is None:
        raise ValueError("No model and/or dataset provided.")

    model, model_info, tokenizer, data_module, dataset_info = setup_model_and_dataset(args)


    # TODO: setup save paths

    # Todo: apply config
    model_info = None  # for choosing the pl wrapper
    dataset_info = None  # for creating pl model from the pl wrapper

    save_path =
    checkpoint =

    # TODO: AutoConfig
    # TODO: AutoTokenizer??
    # TODO: AutoModel




def t():
    datasets_ = load_dataset("glue", "mnli")
    print(type(datasets_))
    print(datasets_['train'].features)
    print(datasets_['train'][0])
    info = datasets.get_dataset_config_info("glue", "mnli")
    print(info.features)

    info = datasets.get_dataset_infos("xsum")
    print(info)
    # info["num_classes"] = 2
    # print("******\n", info)
    # print(info['default'])


if __name__ == "__main__":
    t()
