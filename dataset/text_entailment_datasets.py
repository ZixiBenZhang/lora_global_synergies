import torch
from torch.utils.data import Dataset
import datasets


class TextEntailmentDatasetBase(Dataset):
    info = None

    special_token_mapping: dict[str, str] = None

    sent1_col_name = None
    sent2_col_name = None
    label_col_name = None

    def __init__(
        self,
        tokenizer,
        max_token_len: int,
        num_workers: int,
        load_from_cache_file: bool = True,
        auto_setup: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.load_from_cache_file = load_from_cache_file
        self.data_ = None

        if self.special_token_mapping is not None:
            self.tokenizer.add_special_tokens(self.special_token_mapping)

        if auto_setup:
            self.prepare_data()
            self.setup()

    def _download_dataset(self) -> datasets.DatasetDict:
        raise NotImplementedError

    def prepare_data(self):
        self._download_dataset()

    def setup(self):
        self.data_ = self._download_dataset()

    def __len__(self):
        if self.data_ is None:
            raise ValueError(
                "Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset."
            )
        return len(self.data_)

    def __getitem__(self, index):
        if self.data_ is None:
            raise ValueError(
                "Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset."
            )
        data_row = self.data_[index]
        question = data_row[self.sent1_col_name]
        answer = data_row[self.sent2_col_name]
        labels = data_row[self.label_col_name]
        encoding = self.tokenizer(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        input_dict = dict(
            question=question,
            answer=answer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=torch.tensor([labels]),
        )
        if "token_type_ids" in self.tokenizer.model_input_names:
            input_dict["token_type_ids"] = encoding["token_type_ids"].flatten()

        return input_dict


class TextEntailmentDatasetQNLI(TextEntailmentDatasetBase):
    sent1_col_name = "question"
    sent2_col_name = "sentence"
    label_col_name = "label"

    def _download_dataset(self) -> datasets.DatasetDict:
        dataset_dict = datasets.load_dataset("glue", "qnli")
        return dataset_dict


class TextEntailmentDatasetWNLI(TextEntailmentDatasetBase):
    sent1_col_name = "sentence1"
    sent2_col_name = "sentence2"
    label_col_name = "label"

    def _download_dataset(self) -> datasets.DatasetDict:
        dataset_dict = datasets.load_dataset("glue", "wnli")
        return dataset_dict


class TextEntailmentDatasetMNLI(TextEntailmentDatasetBase):
    sent1_col_name = "premise"
    sent2_col_name = "hypothesis"
    label_col_name = "label"

    def _download_dataset(self) -> datasets.DatasetDict:
        dataset_dict = datasets.load_dataset("glue", "mnli")
        return dataset_dict


class TextEntailmentDatasetRTE(TextEntailmentDatasetBase):
    sent1_col_name = "sentence1"
    sent2_col_name = "sentence2"
    label_col_name = "label"

    def _download_dataset(self) -> datasets.DatasetDict:
        dataset_dict = datasets.load_dataset("glue", "rte")
        return dataset_dict


class TextEntailmentDatasetQQP(TextEntailmentDatasetBase):
    sent1_col_name = "question1"
    sent2_col_name = "question2"
    label_col_name = "label"

    def _download_dataset(self) -> datasets.DatasetDict:
        dataset_dict = datasets.load_dataset("glue", "qqp")
        return dataset_dict


class TextEntailmentDatasetMRPC(TextEntailmentDatasetBase):
    sent1_col_name = "sentence1"
    sent2_col_name = "sentence2"
    label_col_name = "label"

    def _download_dataset(self) -> datasets.DatasetDict:
        dataset_dict = datasets.load_dataset("glue", "mrpc")
        return dataset_dict


class TextEntailmentDatasetSTSB(TextEntailmentDatasetBase):
    sent1_col_name = "sentence1"
    sent2_col_name = "sentence2"
    label_col_name = "label"

    def _download_dataset(self) -> datasets.DatasetDict:
        dataset_dict = datasets.load_dataset("glue", "stsb")
        return dataset_dict


class TextEntailmentDatasetBoolQ(TextEntailmentDatasetBase):
    """
    Subset of SuperGLUE
    """

    sent1_col_name = "passage"
    sent2_col_name = "question"
    label_col_name = "label"

    def _download_dataset(self) -> datasets.DatasetDict:
        dataset_dict = datasets.load_dataset("super_glue", "boolq")
        return dataset_dict
