from dataclasses import dataclass
from enum import Enum


class DatasetSource(Enum):
    """
    The source of the dataset, must be one of the following:
    - MANUAL: manual dataset from MASE
    - HF_DATASETS: dataset from HuggingFace datasets
    - OTHERS: other datasets
    """

    MANUAL = "manual"
    HF_DATASETS = "hf_datasets"
    OTHERS = "others"


class DatasetSplit(Enum):
    """
    The split of the dataset, must be one of the following:
    - TRAIN: training split
    - VALIDATION: validation split
    - TEST: test split
    - PRED: prediction split
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PRED = "pred"


@dataclass
class AgsDatasetInfo:
    """
    The dataset info for AGS.
    """

    name: str

    # dataset source
    dataset_source: DatasetSource

    # available splits
    available_splits: tuple[DatasetSplit]

    # preprocess on split will preprocess all the splits in the dataset
    preprocess_one_split_for_all: bool = True

    data_collator_cls: type = None

    # tasks
    # cls
    sequence_classification: bool = False
    # lm
    causal_LM: bool = False
    # tran
    seq2seqLM: bool = False

    # classification fields
    num_classes: int = None
    num_features: int = None

    def __post_init__(self):
        self.dataset_source = (
            DatasetSource(self.dataset_source)
            if isinstance(self.dataset_source, str)
            else self.dataset_source
        )
        self.available_splits = tuple(
            DatasetSplit(split) if isinstance(split, str) else split
            for split in self.available_splits
        )
        self._entries = {
            "name",
            "dataset_source",
            "available_splits",
            "sequence_classification",
            "causal_LM",
            "seq2seqLM",
            "num_classes",
            "num_features",
        }

    @property
    def train_split_available(self):
        return DatasetSplit.TRAIN in self.available_splits

    @property
    def validation_split_available(self):
        return DatasetSplit.VALIDATION in self.available_splits

    @property
    def test_split_available(self):
        return DatasetSplit.TEST in self.available_splits

    @property
    def pred_split_available(self):
        return DatasetSplit.PRED in self.available_splits


def add_dataset_info(
        name: str,
        dataset_source: DatasetSource,
        available_splits: tuple[DatasetSplit],
        sequence_classification: bool = False,
        causal_LM: bool = False,
        seq2seqLM: bool = False,
        num_classes: int = None,
        num_features: int = None,
        data_collator_cls=None,
):
    """
    a decorator (factory) for adding dataset info to a dataset class

    Args:
        name (str): the name of the dataset
        dataset_source (DatasetSource): the source of the dataset, must be one of "manual", "hf_datasets", "torchvision", "others"
        available_splits (tuple[DatasetSplit]): a tuple of the available splits of the dataset, the split must be one of "train", "valid", "test", "pred"
        sequence_classification (bool, optional): whether the dataset is for sequence classification. Defaults to False.
        causal_LM (bool, optional): whether the dataset is for causal language modeling. Defaults to False.
        seq2seqLM (bool, optional): whether the dataset is for sequence-to-sequence language modeling. Defaults to False.
        num_classes (int, optional): the number of classes of the dataset. Defaults to None.
        num_features (int, optional): Specifies the number of features in the dataset. This is particularly relevant for physical classification tasks that involve input feature vectors. Defaults to None.
        data_collator_cls (optional): Specifies a callable class for merging samples into minibatches. Defaults to None.
    Returns:
        type: the dataset class with dataset info
    """

    def _add_dataset_info_to_cls(cls: type):
        cls.info = AgsDatasetInfo(
            name=name,
            dataset_source=dataset_source,
            available_splits=available_splits,
            sequence_classification=sequence_classification,
            causal_LM=causal_LM,
            seq2seqLM=seq2seqLM,
            num_classes=num_classes,
            num_features=num_features,
            data_collator_cls=data_collator_cls,
        )

        return cls

    return _add_dataset_info_to_cls
