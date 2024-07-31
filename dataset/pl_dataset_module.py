import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from dataset import (
    get_nlp_dataset_split,
    get_config_names,
    get_split_names,
    get_dataset_info,
)

# Only using the task names
task_to_keys = {
    "cola": ("sentence",),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence",),
    "wnli": ("sentence1", "sentence2"),
    "xsum": ("document", "summary"),
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "copa": (),
    "wic": (),
    "alpaca": (),
    "alpaca-cleaned": (),
    "lambada": (),
}


# Adapter global synergies data module, accept GLUE, XSum, SuperGLUE.
class AgsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_len: int,  # for tokenizing
        num_workers: int = None,
        load_from_cache_file: bool = True,
        load_from_saved_path: str = None,
    ):
        super().__init__()

        self.dataset_name = dataset_name.lower()
        # Accept only datasets in the project plan
        # assert self.dataset_name in task_to_keys.keys()

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.load_from_cache_file = load_from_cache_file
        self.load_from_saved_path = load_from_saved_path
        self.dataset_info = get_dataset_info(dataset_name)

        self.training_dataset = None
        self.validation_dataset = None
        self.testing_dataset = None
        self.prediction_dataset = None

        self.save_hyperparameters(
            ignore=[
                "training_dataset",
                "validation_dataset",
                "testing_dataset",
                "prediction_dataset",
            ]
        )

    # Called on rank 0
    def prepare_data(self) -> None:
        # Accept only datasets in the project plan
        # assert self.dataset_name in task_to_keys.keys()
        if self.dataset_name in get_config_names("glue", self.load_from_saved_path):
            path = "glue"
            name = self.dataset_name
        elif self.dataset_name in get_config_names(
            "super_glue", self.load_from_saved_path
        ):
            path = "super_glue"
            name = self.dataset_name
        else:
            path = self.dataset_name
            name = None
        # else:
        #     raise ValueError(
        #         f"Dataset {self.dataset_name} not supported. Please use one of [{'|'.join(task_to_keys.keys())}]"
        #     )

        train_split_names = [
            n
            for n in get_split_names(path, name, self.load_from_saved_path)
            if "train" in n
        ]
        val_split_names = [
            n
            for n in get_split_names(path, name, self.load_from_saved_path)
            if "validation" in n
        ]
        test_split_names = [
            n
            for n in get_split_names(path, name, self.load_from_saved_path)
            if "test" in n
        ]
        pred_split_names = [
            n
            for n in get_split_names(path, name, self.load_from_saved_path)
            if "pred" in n
        ]

        _training_dataset = (
            None
            if len(train_split_names) == 0
            else [
                get_nlp_dataset_split(
                    name=name if name is not None else path,
                    split=split_name,
                    tokenizer=self.tokenizer,
                    max_token_len=self.max_token_len,
                    num_workers=self.num_workers,
                    load_from_cache_file=self.load_from_cache_file,
                    load_from_saved_path=self.load_from_saved_path,
                    auto_setup=False,
                )
                for split_name in train_split_names
            ]
        )
        _validation_dataset = (
            None
            if len(val_split_names) == 0
            else [
                get_nlp_dataset_split(
                    name=name if name is not None else path,
                    split=split_name,
                    tokenizer=self.tokenizer,
                    max_token_len=self.max_token_len,
                    num_workers=self.num_workers,
                    load_from_cache_file=self.load_from_cache_file,
                    load_from_saved_path=self.load_from_saved_path,
                    auto_setup=False,
                )
                for split_name in val_split_names
            ]
        )
        _testing_dataset = (
            None
            if len(test_split_names) == 0
            else [
                get_nlp_dataset_split(
                    name=name if name is not None else path,
                    split=split_name,
                    tokenizer=self.tokenizer,
                    max_token_len=self.max_token_len,
                    num_workers=self.num_workers,
                    load_from_cache_file=self.load_from_cache_file,
                    load_from_saved_path=self.load_from_saved_path,
                    auto_setup=False,
                )
                for split_name in test_split_names
            ]
        )
        _prediction_dataset = (
            None
            if len(pred_split_names) == 0
            else [
                get_nlp_dataset_split(
                    name=name if name is not None else path,
                    split=split_name,
                    tokenizer=self.tokenizer,
                    max_token_len=self.max_token_len,
                    num_workers=self.num_workers,
                    load_from_cache_file=self.load_from_cache_file,
                    load_from_saved_path=self.load_from_saved_path,
                    auto_setup=False,
                )
                for split_name in pred_split_names
            ]
        )

    # Called on all ranks
    def setup(self, stage: str = None) -> None:
        # Accept only datasets in the project plan
        # assert self.dataset_name in task_to_keys.keys()
        if self.dataset_name in get_config_names("glue", self.load_from_saved_path):
            path = "glue"
            name = self.dataset_name
        elif self.dataset_name in get_config_names(
            "super_glue", self.load_from_saved_path
        ):
            path = "super_glue"
            name = self.dataset_name
        else:
            path = self.dataset_name
            name = None
        # else:
        #     raise ValueError(
        #         f"Dataset {self.dataset_name} not supported. Please use one of [{'|'.join(task_to_keys.keys())}]"
        #     )

        train_split_names = [
            n
            for n in get_split_names(path, name, self.load_from_saved_path)
            if "train" in n
        ]
        val_split_names = [
            n
            for n in get_split_names(path, name, self.load_from_saved_path)
            if "validation" in n
        ]
        test_split_names = [
            n
            for n in get_split_names(path, name, self.load_from_saved_path)
            if "test" in n
        ]
        pred_split_names = [
            n
            for n in get_split_names(path, name, self.load_from_saved_path)
            if "pred" in n
        ]

        # print(
        #     f">>> Loaded splits:\n"
        #     f"  Train splits: {train_split_names}\n"
        #     f"  Validation splits: {val_split_names}\n"
        #     f"  Test splits: {test_split_names}\n"
        #     f"  Prediction splits: {pred_split_names}"
        # )

        if stage in ["fit", None]:
            self.training_dataset = (
                None
                if len(train_split_names) == 0
                else torch.utils.data.ConcatDataset(
                    [
                        get_nlp_dataset_split(
                            name=name if name is not None else path,
                            split=split_name,
                            tokenizer=self.tokenizer,
                            max_token_len=self.max_token_len,
                            num_workers=self.num_workers,
                            load_from_cache_file=self.load_from_cache_file,
                            load_from_saved_path=self.load_from_saved_path,
                            auto_setup=True,
                        )
                        for split_name in train_split_names
                    ]
                )
            )
        if stage in ["fit", "validate", None]:
            self.validation_dataset = (
                None
                if len(val_split_names) == 0
                else torch.utils.data.ConcatDataset(
                    [
                        get_nlp_dataset_split(
                            name=name if name is not None else path,
                            split=split_name,
                            tokenizer=self.tokenizer,
                            max_token_len=self.max_token_len,
                            num_workers=self.num_workers,
                            load_from_cache_file=self.load_from_cache_file,
                            load_from_saved_path=self.load_from_saved_path,
                            auto_setup=True,
                        )
                        for split_name in val_split_names
                    ]
                )
            )
        if stage in ["test", None]:
            self.testing_dataset = (
                None
                if len(test_split_names) == 0
                else torch.utils.data.ConcatDataset(
                    [
                        get_nlp_dataset_split(
                            name=name if name is not None else path,
                            split=split_name,
                            tokenizer=self.tokenizer,
                            max_token_len=self.max_token_len,
                            num_workers=self.num_workers,
                            load_from_cache_file=self.load_from_cache_file,
                            load_from_saved_path=self.load_from_saved_path,
                            auto_setup=True,
                        )
                        for split_name in test_split_names
                    ]
                )
            )
        if stage in ["predict", None]:
            self.prediction_dataset = (
                None
                if len(pred_split_names) == 0
                else torch.utils.data.ConcatDataset(
                    [
                        get_nlp_dataset_split(
                            name=name if name is not None else path,
                            split=split_name,
                            tokenizer=self.tokenizer,
                            max_token_len=self.max_token_len,
                            num_workers=self.num_workers,
                            load_from_cache_file=self.load_from_cache_file,
                            load_from_saved_path=self.load_from_saved_path,
                            auto_setup=True,
                        )
                        for split_name in pred_split_names
                    ]
                )
            )

    def train_dataloader(self) -> DataLoader:
        if self.training_dataset is None:
            raise RuntimeError("The training dataset is not available.")
        data_collator = None
        if self.dataset_info.data_collator_cls is not None:
            data_collator = self.dataset_info.data_collator_cls(
                tokenizer=self.tokenizer
            )
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=data_collator,
        )

    def val_dataloader(self) -> DataLoader:
        if self.testing_dataset is None:
            raise RuntimeError("The validation dataset is not available.")
        data_collator = None
        if self.dataset_info.data_collator_cls is not None:
            data_collator = self.dataset_info.data_collator_cls(
                tokenizer=self.tokenizer
            )
        return DataLoader(
            self.testing_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=data_collator,
        )

    def test_dataloader(self) -> DataLoader:
        if self.testing_dataset is None:
            raise RuntimeError("The test dataset is not available.")
        data_collator = None
        if self.dataset_info.data_collator_cls is not None:
            data_collator = self.dataset_info.data_collator_cls(
                tokenizer=self.tokenizer
            )
        return DataLoader(
            self.testing_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=data_collator,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.prediction_dataset is None:
            raise RuntimeError("The prediction dataset is not available.")
        data_collator = None
        if self.dataset_info.data_collator_cls is not None:
            data_collator = self.dataset_info.data_collator_cls(
                tokenizer=self.tokenizer
            )
        return DataLoader(
            self.prediction_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=data_collator,
        )
