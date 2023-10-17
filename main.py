import logging
import sys

import datasets
import pytorch_lightning as pl
from datasets import load_dataset

# from argparser_lora import get_arg_parser
# from dataset.dataset_loader import setup_dataset


# def main():
#     logger = logging.getLogger("main")
#
#     parser = get_arg_parser()
#     args = parser.parse_args(sys.argv)
#
#     pl.seed_everything(args.seed)
#
#     # TODO: setup log level
#
#     # TODO: turn config path to dict to attributes of args
#
#     if args.model is None or args.dataset is None:
#         raise ValueError("No model and/or dataset provided.")
#
#     data_module, dataset_info = setup_dataset(args)
#
#     model, model_info, tokenizer = setup_model(args, )
#
#     # TODO: setup save paths
#
#     # Todo: apply config
#     model_info = None  # for choosing the pl wrapper
#     dataset_info = None  # for creating pl model from the pl wrapper
#
#     save_path =
#     checkpoint =
#
#     # TODO: AutoConfig
#     # TODO: AutoTokenizer??
#     # TODO: AutoModel




def t():
    datasets_ = load_dataset("xsum")
    print(type(datasets_))
    print(datasets_.shape)
    print(datasets_["test"].column_names)


if __name__ == "__main__":
    t()
