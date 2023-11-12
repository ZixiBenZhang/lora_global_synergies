import datasets
from datasets import DatasetDict

if __name__ == "__main__":
    config_names = datasets.get_dataset_config_names("glue")
    for c in config_names:
        dataset: DatasetDict = datasets.load_dataset("glue", c)
        info = datasets.get_dataset_config_info("glue", c)

    dataset = datasets.load_dataset("xsum")
    info = datasets.get_dataset_infos("xsum")["default"]

    config_names = datasets.get_dataset_config_names("super_glue")
    for c in config_names:
        dataset: DatasetDict = datasets.load_dataset("super_glue", c)
        info = datasets.get_dataset_config_info("super_glue", c)
