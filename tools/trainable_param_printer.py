import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def print_trainable_parameters(model: nn.Module):
    trainable_param = 0
    total_param = 0
    for name, param in model.named_parameters():
        total_param += param.numel()
        if param.requires_grad:
            trainable_param += param.numel()
    logger.info(
        f"Trainable param number: {trainable_param} || "
        f"All param number: {total_param} || "
        f"Trainable %: {100 * trainable_param / total_param:.2f}"
    )
