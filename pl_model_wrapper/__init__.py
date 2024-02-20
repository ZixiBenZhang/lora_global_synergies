from models.model_info import AgsModelInfo
from .classification_wrapper import NLPClassificationModelWrapper
from .lm_wrapper import NLPLanguageModelingModelWrapper
from .summarization_wrapper import NLPSummarizationModelWrapper


def get_model_wrapper(model_info: AgsModelInfo, task: str):
    assert model_info.task_type == "nlp"
    match task:
        case "classification":
            return NLPClassificationModelWrapper
        case "summarization":
            return NLPSummarizationModelWrapper
        case "causal_language_modeling":
            return NLPLanguageModelingModelWrapper
        case _:
            raise ValueError(f"Task {task} is not supported for {model_info.name}.")
