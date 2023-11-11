import datasets
from datasets import DatasetDict

if __name__ == "__main__":
    config_names = datasets.get_dataset_config_names("glue")
    for c in config_names:
        dataset: DatasetDict = datasets.load_dataset("glue", c)
        # dataset.save_to_disk(f"../saved_hf_datasets/glue/{c}")
