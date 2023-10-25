from .text_entailment_datasets import *
from .sentiment_analysis_datasets import *


def get_nlp_dataset_split(
    name: str,
    split: str,
    tokenizer,
    max_token_len: int,
    num_workers: int,
    load_from_cache_file: bool = True,
    auto_setup: bool = True,
) -> Dataset:
    match name:
        case "sst2":
            dataset_cls = SentimentalAnalysisDatasetSST2
        case "cola":
            dataset_cls = SentimentalAnalysisDatasetCoLa
        case "mnli":
            dataset_cls = TextEntailmentDatasetMNLI
        case "qnli":
            dataset_cls = TextEntailmentDatasetQNLI
        case "wnli":
            dataset_cls = TextEntailmentDatasetWNLI
        case "rte":
            dataset_cls = TextEntailmentDatasetRTE
        case "stsb":
            dataset_cls = TextEntailmentDatasetSTSB
        case "qqp":
            dataset_cls = TextEntailmentDatasetQQP
        case "mrpc":
            dataset_cls = TextEntailmentDatasetMRPC
        case "boolq":
            dataset_cls = TextEntailmentDatasetBoolQ
        # TODO: CB, COPA, WiC
        case _:
            raise ValueError(f"Unknown dataset {name}, or not supported yet.")

    dataset = dataset_cls(
        split,
        tokenizer,
        max_token_len,
        num_workers,
        load_from_cache_file,
        auto_setup,
    )
    return dataset
