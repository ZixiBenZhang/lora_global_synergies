from .language_modeling_datasets import LanguageModelingDatasetAlpacaCleaned, LanguageModelingDatasetAlpaca
from .text_entailment_datasets import *
from .sentiment_analysis_datasets import *


def get_nlp_dataset_split(
    name: str,
    split: str,
    tokenizer,
    max_token_len: int,
    num_workers: int,
    load_from_cache_file: bool = True,
    load_from_saved_path: str = None,
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
        case "alpaca":
            dataset_cls = LanguageModelingDatasetAlpaca
        case "alpaca-cleaned":
            dataset_cls = LanguageModelingDatasetAlpacaCleaned
        # TODO: Lambada
        case _:
            raise ValueError(f"Unknown dataset {name}, or not supported yet.")

    dataset = dataset_cls(
        split,
        tokenizer,
        max_token_len,
        num_workers,
        load_from_cache_file,
        load_from_saved_path,
        auto_setup,
    )
    return dataset


def get_config_names(path: str, load_from_saved_path: str = None) -> list:
    # Hard-coded because getting from HF may error
    match path:
        case "glue":
            return [
                "cola",
                "sst2",
                "mrpc",
                "qqp",
                "stsb",
                "mnli",
                "mnli_mismatched",
                "mnli_matched",
                "qnli",
                "rte",
                "wnli",
                "ax",
            ]
        case "super_glue":
            return [
                "axb",
                "axg",
                "boolq",
                "cb",
                "copa",
                "multirc",
                "record",
                "rte",
                "wic",
                "wsc",
                "wsc.fixed",
            ]
        case _:
            if load_from_saved_path is None:
                return datasets.get_dataset_config_names(path)
            else:
                raise ValueError(f"Unsupported dataset path: {path}")


def get_split_names(
    path: str, name: str = None, load_from_saved_path: str = None
) -> list:
    # Hard-coded because getting from HF may error
    if path == "glue":
        match name:
            case "ax":
                return ["test"]
            case "cola":
                return ["train", "validation", "test"]
            case "mnli":
                return [
                    "train",
                    "validation_matched",
                    "validation_mismatched",
                    "test_matched",
                    "test_mismatched",
                ]
            case "mnli_matched":
                return ["validation", "test"]
            case "mnli_mismatched":
                return ["validation", "test"]
            case "mrpc":
                return ["train", "validation", "test"]
            case "qnli":
                return ["train", "validation", "test"]
            case "qqp":
                return ["train", "validation", "test"]
            case "rte":
                return ["train", "validation", "test"]
            case "sst2":
                return ["train", "validation", "test"]
            case "stsb":
                return ["train", "validation", "test"]
            case "wnli":
                return ["train", "validation", "test"]
            case _:
                raise ValueError(
                    f"Invalid combination of dataset path and name: {path}, {name}."
                )
    elif path == "xsum":
        return ["train", "validation", "test"]
    elif path == "super_glue":
        match name:
            case "axb":
                return ["test"]
            case "axb":
                return ["test"]
            case "boolq":
                return ["train", "validation", "test"]
            case "cb":
                return ["train", "validation", "test"]
            case "copa":
                return ["train", "validation", "test"]
            case "multirc":
                return ["train", "validation", "test"]
            case "record":
                return ["train", "validation", "test"]
            case "rte":
                return ["train", "validation", "test"]
            case "wic":
                return ["train", "validation", "test"]
            case "wsc":
                return ["train", "validation", "test"]
            case "wsc.fixed":
                return ["train", "validation", "test"]
            case _:
                raise ValueError(
                    f"Invalid combination of d ataset path and name: {path}, {name}."
                )
    elif path == "tatsu-lab/alpaca":
        return ["train"]
    elif path == "yahma/alpaca-cleaned":
        return ["train"]
    elif path == "lambada":
        return ["train", "validation", "test"]
    else:
        if load_from_saved_path is None:
            return datasets.get_dataset_split_names(path, name)
        else:
            raise ValueError(f"Unsupported dataset path: {path}")
