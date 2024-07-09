import logging
from os import PathLike
from transformers import AutoTokenizer

from models.model_info import get_model_info, ModelSource, MANUAL_MODELS

logger = logging.getLogger(__name__)


def get_tokenizer(name: str, checkpoint: str | PathLike = None):
    model_info = get_model_info(name)

    match model_info.model_source:
        case ModelSource.HF_TRANSFORMERS:
            tokenizer = get_hf_model_tokenizer(name, checkpoint)
        case ModelSource.MANUAL:
            tokenizer = get_manual_model_tokenizer(name, checkpoint)
        case _:
            raise ValueError(
                f"Tokenizer for model source {model_info.model_source} not supported"
            )
    return tokenizer


def get_hf_model_tokenizer(name: str, checkpoint: str | PathLike = None):
    tokenizer = AutoTokenizer.from_pretrained(name if checkpoint is None else checkpoint)
    if "llama" in name:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Setting pad_token as eos_token")
    return tokenizer


def get_manual_model_tokenizer(name: str, checkpoint: str | PathLike = None):
    if name not in MANUAL_MODELS:
        raise ValueError(f"Manual model {name} is not supported")
    tokenizer = MANUAL_MODELS[name]["tokenizer_cls"].from_pretrained(
        name if checkpoint is None else checkpoint
    )
    if "llama" in name:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Setting pad_token as eos_token")
    return tokenizer
