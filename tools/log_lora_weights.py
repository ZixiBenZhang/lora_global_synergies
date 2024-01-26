import csv
import os
import re
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedModel

from models.modeling_opt_lora import (
    OPTLoraPreTrainedModel,
    OPTLoraForCausalLM,
    OPTLoraForSequenceClassification,
    OPTLoraForQuestionAnswering,
)
from models.modeling_roberta_lora import RobertaLoraPreTrainedModel
from tools.get_unevenness import compute_unevenness_metrics

"""
LoRA weight logging: once per epoch, ~ succinct model weights checkpoint
"""

"""
LoRA weights (per-epoch) format:
res = {
    epoch: current_epoch
    layers.{i}.residual1: {
        svdvals_layers.{i}.residual1: [svd1,svd2,...]
        uneven_{metric_name}_layers.{i}.residual1: metric_val 
        ...
    }
    layers.{i}.residual2: {...}
    ...
}
"""

"""
Visualisation plan:
- Choose certain epochs to visualise e.g. 5, 10, ..., final epoch
- A figure per metric per shortcut 
- Each figure: layer depth vs. metric
"""


def log_layer_lora_svd(
    model: PreTrainedModel,
    log_dir: str = None,
    current_epoch: int = None,
) -> dict[str, Tensor | float]:
    singular_uneven: dict[str, Any]
    if isinstance(model, OPTLoraPreTrainedModel):
        # print(f"Epoch {self.current_epoch} getting singular values...")
        singular_uneven = get_opt_layer_lora_svd(model)
        # print(singular_uneven)
    elif isinstance(model, RobertaLoraPreTrainedModel):
        singular_uneven = get_roberta_layer_lora_svd(model)
    else:
        raise ValueError(
            f"Model {model.__class__.__name__} not supported for logging shortcut singular values"
        )

    _header = list(singular_uneven.keys())
    _header.insert(0, "epoch")
    singular_uneven["epoch"] = current_epoch

    # TODO: log by YAML & Wandb

    # Log to csv
    # filename = f"{log_dir}/svd.csv"
    # if os.path.isfile(filename):
    #     with open(filename, "a+", encoding="UTF8", newline="") as f:
    #         dict_writer = csv.DictWriter(f, fieldnames=_header)
    #         dict_writer.writerow(singular_uneven)
    # else:
    #     with open(filename, "a+", encoding="UTF8", newline="") as f:
    #         dict_writer = csv.DictWriter(f, fieldnames=_header)
    #         dict_writer.writeheader()
    #         dict_writer.writerow(singular_uneven)

    return singular_uneven


def get_opt_layer_lora_svd(model: OPTLoraPreTrainedModel) -> dict[str, Tensor | float]:
    assert (
        type(model) is OPTLoraForCausalLM
        or type(model) is OPTLoraForSequenceClassification
        or type(model) is OPTLoraForQuestionAnswering
    )
    model: OPTLoraForCausalLM | OPTLoraForSequenceClassification | OPTLoraForQuestionAnswering
    lora_weights = {}
    res = {}
    num_heads: int = model.model.decoder.layers[0].self_attn.num_heads
    head_dim: int = model.model.decoder.layers[0].self_attn.head_dim
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            continue

        mat_name: str = re.findall(
            r"layers\.\d+\..*\.lora_[A-B]\..*\.weight",
            name,
        )[0]
        lora_weights[mat_name] = param.data

        if "lora_A" in mat_name:
            corr_name: str = mat_name.replace("lora_A", "lora_B")
            if corr_name not in lora_weights:
                continue
            delta_w = torch.matmul(
                lora_weights[corr_name],
                lora_weights[mat_name],  # B (d_out, r) * A (r, d_in)
            )  # shape: (out_features, in_features)
        else:
            corr_name: str = mat_name.replace("lora_B", "lora_A")
            if corr_name not in lora_weights:
                continue
            delta_w = torch.matmul(
                lora_weights[mat_name],
                lora_weights[corr_name],  # B (d_out, r) * A (r, d_in)
            )  # shape: (out_features, in_features)

        if "q_proj" in mat_name or "k_proj" in mat_name or "v_proj" in mat_name:
            # Split by heads
            delta_w = delta_w.view(
                num_heads, head_dim, -1
            )  # shape: (num_heads, d_head, d_model)

        singulars: Tensor = torch.linalg.svdvals(
            delta_w
        )  # shape: (num_heads, d_singulars) or (d_singulars)
        unevenness = compute_unevenness_metrics(singulars)

        # layer.[i].[self_attn?].[proj_name].lora_[A-B]
        lora_name: str = re.findall(r"layers\.\d+\..*\.lora_[A-B]", mat_name)[0]
        if "q_proj" in mat_name or "k_proj" in mat_name or "v_proj" in mat_name:
            for head_id in range(num_heads):
                head_name = f"{lora_name}.head.{head_id}"
                res[f"svdvals_{head_name}"] = singulars[head_id]
                for metric_name, values in unevenness.items():
                    res[f"uneven_{metric_name}_{head_name}"] = values[head_id]
        else:
            res[f"svdvals_{lora_name}"] = singulars.tolist()
            for metric_name, value in unevenness.items():
                res[f"uneven_{metric_name}_{lora_name}"] = value

    return res


def get_roberta_layer_lora_svd(model: RobertaLoraPreTrainedModel) -> dict[str, Tensor]:
    raise NotImplementedError
