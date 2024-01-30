import os

import datasets
from datasets import DatasetDict, DatasetInfo


def download_datasets():
    config_names = datasets.get_dataset_config_names("glue")
    for c in config_names:
        dataset: DatasetDict = datasets.load_dataset("glue", c, save_infos=True)
        # info = datasets.get_dataset_config_info("glue", c)

    dataset = datasets.load_dataset("xsum", save_infos=True)
    # info = datasets.get_dataset_infos("xsum")["default"]

    config_names = datasets.get_dataset_config_names("super_glue")
    for c in config_names:
        dataset: DatasetDict = datasets.load_dataset("super_glue", c, save_infos=True)
        # info = datasets.get_dataset_config_info("super_glue", c)


def save_dataset_info():
    config_names = datasets.get_dataset_config_names("glue")
    for c in config_names:
        info = datasets.get_dataset_config_info("glue", c)
        save_path = f"/home/zz458/.cache/huggingface/datasets/dataset_info/{c}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Created {save_path}")
        info.write_to_directory(save_path, pretty_print=True)

    info: DatasetInfo = datasets.get_dataset_infos("xsum")["default"]
    save_path = "/home/zz458/.cache/huggingface/datasets/dataset_info/xsum/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created {save_path}")
    info.write_to_directory(save_path, pretty_print=True)

    config_names = datasets.get_dataset_config_names("super_glue")
    for c in config_names:
        info = datasets.get_dataset_config_info("super_glue", c)
        save_path = f"/home/zz458/.cache/huggingface/datasets/dataset_info/{c}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Created {save_path}")
        info.write_to_directory(save_path, pretty_print=True)


if __name__ == "__main__":
    save_dataset_info()
