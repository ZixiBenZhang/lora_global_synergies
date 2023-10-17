import torch
import pytorch_lightning as pl
import datasets
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

task_to_keys = {
    "ax": ("premise", "hypothesis"),
    "cola": ("sentence",),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence",),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),

    "xsum": ("document", "summary"),

    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    # TODO: decide how COPA, WiC are passed in to model
    # "copa": ,
    # "wic": ,
}


# adapter global synergies data module
class AgsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_len: int,  # for tokenizing
        num_proc: int = None,
        load_from_cache_file: bool = True,
    ):
        super().__init__()

        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_proc = num_proc
        self.load_from_cache_file = load_from_cache_file

        self.training_dataset = None
        self.validation_dataset = None
        self.testing_dataset = None
        self.prediction_dataset = None

        self.text_column_names = task_to_keys[self.dataset_name]

        self.dataset_info = self._get_dataset_info()

    def _get_dataset_info(self):
        if self.dataset_name in datasets.get_dataset_config_names("glue"):
            return datasets.get_dataset_config_info("glue", self.dataset_name)
        return datasets.get_dataset_infos(self.dataset_name)

    # def _get_split(self, dataset: DatasetDict, split: str) -> Dataset:
    #     split_dataset = None
    #     for i, split in enumerate(splits):
    #         split_dataset = dataset[split]
    #         if i == 0:
    #             train_dataset = dataset_
    #         else:
    #             train_dataset = datasets.concatenate_datasets([train_dataset, dataset_])

    # Called on rank 0
    def prepare_data(self) -> None:
        # Downloads dataset splits, but doesn't assign to attributes

        # Accept datasets not in the project plan
        # configs = datasets.get_dataset_config_names(self.dataset_name)
        # if configs == ['default']:
        #     path = self.dataset_name
        #     name = None
        # else:
        #     # Default format: dataset.config
        #     path = self.dataset_name.split('.')[0]
        #     name = self.dataset_name.split('.')[1]
        #     assert name in configs, \
        #         f"Dataset {path} has no config {name}. " \
        #         f"Please ensure the config exists in the dataset. Use format 'DATASET.CONFIG'."

        # Accept only datasets in the project plan
        assert self.dataset_name in task_to_keys.keys()
        if self.dataset_name in datasets.get_dataset_config_names("glue"):
            path = "glue"
            name = self.dataset_name
        elif self.dataset_name in datasets.get_dataset_config_names("super_glue"):
            path = "super_glue"
            name = self.dataset_name
        elif self.dataset_name == "xsum":
            path = self.dataset_name
            name = None
        else:
            raise ValueError(
                f"Dataset {self.dataset_name} not supported. Please use one of [{'|'.join(task_to_keys.keys())}]"
            )

        train_splits = [n for n in datasets.get_dataset_split_names(path, name) if 'train' in n]
        val_splits = [n for n in datasets.get_dataset_split_names(path, name) if 'validation' in n]
        test_splits = [n for n in datasets.get_dataset_split_names(path, name) if 'test' in n]
        pred_splits = [n for n in datasets.get_dataset_split_names(path, name) if 'pred' in n]

        dataset_ = load_dataset(path, name=name, num_proc=self.num_proc)

        if len(train_splits) > 0:
            _train_dataset = datasets.concatenate_datasets([dataset_[split] for split in train_splits])
        if len(val_splits) > 0:
            _val_dataset = datasets.concatenate_datasets([dataset_[split] for split in val_splits])
        if len(test_splits) > 0:
            _test_dataset = datasets.concatenate_datasets([dataset_[split] for split in test_splits])
        if len(pred_splits) > 0:
            _pred_dataset = datasets.concatenate_datasets([dataset_[split] for split in pred_splits])

    # Called on all ranks
    def setup(self, stage: str = None) -> None:
        # Accept only datasets in the project plan
        assert self.dataset_name in task_to_keys.keys()
        if self.dataset_name in datasets.get_dataset_config_names("glue"):
            path = "glue"
            name = self.dataset_name
        elif self.dataset_name in datasets.get_dataset_config_names("super_glue"):
            path = "super_glue"
            name = self.dataset_name
        elif self.dataset_name == "xsum":
            path = self.dataset_name
            name = None
        else:
            raise ValueError(
                f"Dataset {self.dataset_name} not supported. Please use one of [{'|'.join(task_to_keys.keys())}]"
            )

        train_splits = [n for n in datasets.get_dataset_split_names(path, name) if 'train' in n]
        val_splits = [n for n in datasets.get_dataset_split_names(path, name) if 'validation' in n]
        test_splits = [n for n in datasets.get_dataset_split_names(path, name) if 'test' in n]
        pred_splits = [n for n in datasets.get_dataset_split_names(path, name) if 'pred' in n]

        dataset_ = load_dataset(path, name=name, num_proc=self.num_proc)

        if self.max_token_len > self.tokenizer.model_max_length:
            # Todo: logger warning
            pass
            # logger.warning(
            #     f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            #     f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            # )
        block_size = min(self.max_token_len, self.tokenizer.model_max_length)

        def preprocess_function(examples):
            # Tokenize the texts
            result = self.tokenizer(
                self.text_column_names,
                max_length=block_size,
                # Currently disabled padding & truncation
                padding=False,
                truncation=False,
            )
            # Todo: label_to_ids for ordered labels??
            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [(label_to_id[label] if label != -1 else -1) for label in examples["label"]]
            return result

        dataset_ = dataset_.map(
            preprocess_function,
            batched=True,
            num_proc=self.num_proc,
            load_from_cache_file=self.load_from_cache_file,
            desc="Running tokenizer on dataset",
        )

        if stage in ["fit", None]:
            self.training_dataset = None if len(train_splits) == 0 \
                else datasets.concatenate_datasets([dataset_[split] for split in train_splits])
        if stage in ["fit", "validate", None]:
            self.validation_dataset = None if len(val_splits) == 0 \
                else datasets.concatenate_datasets([dataset_[split] for split in val_splits])
        if stage in ["test", None]:
            self.testing_dataset = None if len(test_splits) == 0 \
                else datasets.concatenate_datasets([dataset_[split] for split in test_splits])
        if stage in ["predict", None]:
            self.prediction_dataset = None if len(pred_splits) == 0 \
                else datasets.concatenate_datasets([dataset_[split] for split in pred_splits])

    def train_dataloader(self) -> DataLoader:
        if self.training_dataset is None:
            raise RuntimeError("The training dataset is not available.")
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_proc,
        )

    def val_dataloader(self) -> DataLoader:
        if self.validation_dataset is None:
            raise RuntimeError("The validation dataset is not available.")
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_proc,
        )

    def test_dataloader(self) -> DataLoader:
        if self.testing_dataset is None:
            raise RuntimeError("The test dataset is not available.")
        return DataLoader(
            self.testing_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_proc,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.prediction_dataset is None:
            raise RuntimeError("The prediction dataset is not available.")
        return DataLoader(
            self.prediction_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_proc,
        )
