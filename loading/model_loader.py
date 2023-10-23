from os import PathLike

import datasets
from datasets import DatasetInfo
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification, \
    AutoModelForSeq2SeqLM

from models.model_info import get_model_info, ModelSource, MANUAL_MODELS


def get_model(
    name: str,
    task: str,
    dataset_info: DatasetInfo,
    pretrained: bool,
    checkpoint: str | PathLike = None,
    lora_config: dict = None,
) -> PreTrainedModel:
    model_info = get_model_info(name)
    model_kwargs = {
        "name": name,
        "task": task,
        "dataset_info": dataset_info,
        "pretrained": pretrained,
        "checkpoint": checkpoint,
    }
    if model_info.is_lora:
        model_kwargs['lora_config'] = lora_config

    match model_info.model_source:
        case ModelSource.HF_TRANSFORMERS:
            model = get_hf_model(**model_kwargs)
        case ModelSource.MANUAL:
            model = get_manual_model(**model_kwargs)
        case _:
            raise ValueError(f"Model source {model_info.model_source} not supported.")

    return model


def get_hf_model(
    name: str,
    task: str,
    dataset_info: datasets.DatasetInfo,
    pretrained: bool,
    checkpoint: str | PathLike = None,
) -> PreTrainedModel:
    model_info = get_model_info(name)

    match task:
        case "causal_language_modeling":
            if not model_info.causal_LM:
                raise ValueError(f"Task {task} is not supported for {name}")
            if pretrained:
                model = AutoModelForCausalLM.from_pretrained(name if checkpoint is None else checkpoint)
            else:
                config = AutoConfig.from_pretrained(name if checkpoint is None else checkpoint)
                model = AutoModelForCausalLM.from_config(config)
        case "classification":
            if not model_info.sequence_classification:
                raise ValueError(f"Task {task} is not supported for {name}")
            assert 'label' in dataset_info.features.keys()
            config = AutoConfig.from_pretrained(
                name if checkpoint is None else checkpoint,
                num_labels=dataset_info.features['label'].num_classes,
            )
            if pretrained:
                model = AutoModelForSequenceClassification.from_pretrained(
                    name if checkpoint is None else checkpoint, config=config
                )
            else:
                model = AutoModelForSequenceClassification.from_config(config)
        case "summarization":
            if not model_info.seq2seqLM:
                raise ValueError(f"Task {task} is not supported for {name}")
            if pretrained:
                model = AutoModelForSeq2SeqLM.from_pretrained(name if checkpoint is None else checkpoint)
            else:
                config = AutoConfig.from_pretrained(name if checkpoint is None else checkpoint)
                model = AutoModelForSeq2SeqLM.from_config(config)
        case _:
            raise ValueError(f"Task {task} is not supported for {name}")

    return model


def get_manual_model(
    name: str,
    task: str,
    dataset_info: datasets.DatasetInfo,
    pretrained: bool,
    checkpoint: str | PathLike,
    lora_config: dict = None,
) -> PreTrainedModel:
    model_info = get_model_info(name)

    match task:
        case "classification":
            assert (
                model_info.sequence_classification
            ), f"Task {task} is not supported for {name}"
            assert 'label' in dataset_info.features.keys()
            num_classes = dataset_info.features['label'].num_classes
            if lora_config is not None:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                    checkpoint,
                    lora_config=lora_config,
                    num_labels=num_classes,
                )
            else:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                    checkpoint,
                    num_labels=num_classes,
                )
            model_cls = MANUAL_MODELS[name]["sequence_classification"]
        case "causal_language_modeling":
            assert model_info.causal_LM, f"Task {task} is not supported for {name}"
            if lora_config is not None:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                    checkpoint,
                    lora_config=lora_config,
                )
            else:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(checkpoint)
            model_cls = MANUAL_MODELS[name]["causal_LM"]
        case "summarization":
            assert model_info.seq2seqLM, f"Task {task} is not supported for {name}"
            if lora_config is not None:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                    checkpoint,
                    lora_config=lora_config,
                )
            else:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(checkpoint)
            model_cls = MANUAL_MODELS[name]["seq2seqLM"]
        case _:
            raise ValueError(f"Task {task} is not supported for {name}")
    if pretrained:
        model = model_cls.from_pretrained(checkpoint, config=config)
    else:
        model = model_cls(config)

    return model
