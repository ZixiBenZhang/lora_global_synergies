import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from dataset.pl_dataset_module import AgsDataModule
from loading.tokenizer_loader import get_hf_model_tokenizer


def test_AgsDataModule_setup():
    ags_data_module = AgsDataModule(
        dataset_name="cb",
        batch_size=8,
        tokenizer=get_hf_model_tokenizer("roberta-large"),
        max_token_len=512,
        num_workers=12,
    )

    ags_data_module.prepare_data()
    ags_data_module.setup()

    assert isinstance(ags_data_module.training_dataset, Dataset)
    assert isinstance(ags_data_module.validation_dataset, Dataset)
    assert isinstance(ags_data_module.testing_dataset, Dataset)
    assert ags_data_module.prediction_dataset is None


def test_AgsDataModule_tokenization():
    ags_data_module = AgsDataModule(
        dataset_name="cb",
        batch_size=3,
        tokenizer=get_hf_model_tokenizer("roberta-large"),
        max_token_len=278,
        num_workers=10,
    )

    ags_data_module.prepare_data()
    ags_data_module.setup("validate")
    val_data_loader = ags_data_module.val_dataloader()
    example_data = None
    for data in val_data_loader:
        example_data = data
        break

    assert isinstance(val_data_loader, DataLoader)
    assert example_data["input_ids"].size() == torch.Size([3, 278])
    assert example_data["attention_mask"].size() == torch.Size([3, 278])
    assert torch.all(example_data["labels"] == torch.tensor([[1], [2], [0]]))


def test_AgsDataModule_tokenization_invalid_split():
    ags_data_module = AgsDataModule(
        dataset_name="alpaca",
        batch_size=3,
        tokenizer=get_hf_model_tokenizer("roberta-large"),
        max_token_len=278,
        num_workers=10,
    )

    ags_data_module.prepare_data()
    ags_data_module.setup()

    with pytest.raises(RuntimeError):
        _ = ags_data_module.test_dataloader()
