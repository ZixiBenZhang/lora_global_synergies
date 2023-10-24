from dataclasses import dataclass
from enum import Enum

from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    OPTConfig,
    GPT2Tokenizer,
    LlamaConfig,
    LlamaTokenizer,
    LlamaForSequenceClassification,
    LlamaForCausalLM,
    OPTForSequenceClassification,
    OPTForCausalLM,
)


class ModelSource(Enum):
    """
    The source of the model, must be one of the following:
    - HF: HuggingFace
    - MANUAL: manually implemented
    """

    HF_TRANSFORMERS = "hf_transformers"
    MANUAL = "manual"


@dataclass
class AgsModelInfo:
    name: str
    model_source: ModelSource
    task_type: str = "nlp"
    # NLP models
    sequence_classification: bool = False
    seq2seqLM: bool = False
    causal_LM: bool = False
    # Manual models
    is_lora: bool = False

    def __post_init__(self):
        self.model_source = (
            ModelSource(self.model_source)
            if isinstance(self.model_source, str)
            else self.model_source
        )
        assert self.sequence_classification or self.seq2seqLM or self.causal_LM
        if self.is_lora:
            assert (
                self.model_source == ModelSource.MANUAL
            ), "LoRA model must be a manual model."


HF_NLP_MODELS = {
    "roberta-base": {
        "config_cls": RobertaConfig,
        "tokenizer_cls": RobertaTokenizer,
        "info": AgsModelInfo(
            "roberta-base",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
        ),
    },
    "roberta-large": {
        "config_cls": RobertaConfig,
        "tokenizer_cls": RobertaTokenizer,
        "info": AgsModelInfo(
            "roberta-large",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
        ),
    },
    "facebook/opt-125m": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-125m",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "facebook/opt-350m": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-350m",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "facebook/opt-1.3b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-1.3b",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "facebook/opt-2.7b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-2.7b",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "facebook/opt-6.7b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-6.7b",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "facebook/opt-13b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-13b",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "facebook/opt-30b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-30b",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "facebook/opt-66b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-66b",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    # TODO: Llama & Vicuna model info
}

MANUAL_MODELS = {
    "llama_plain": {
        "config_cls": LlamaConfig,
        "tokenizer_cls": LlamaTokenizer,
        "info": AgsModelInfo(
            "llama_plain",
            model_source="manual",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
        "sequence_classification": LlamaForSequenceClassification,
        "causal_LM": LlamaForCausalLM,
    },
    "opt_plain": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "opt_plain",
            model_source="manual",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
        "sequence_classification": OPTForSequenceClassification,
        "causal_LM": OPTForCausalLM,
    },
    # TODO: LoRA model info
}


def get_model_info(name: str) -> AgsModelInfo:
    if name in MANUAL_MODELS:
        return MANUAL_MODELS[name]["info"]
    elif name in HF_NLP_MODELS:
        return HF_NLP_MODELS[name]["info"]
    else:
        raise ValueError(f"Model {name} not found.")
