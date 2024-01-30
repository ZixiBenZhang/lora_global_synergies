import csv
import os
import re
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedModel

from tools.get_unevenness import compute_unevenness_metrics

"""
Shortcut weight logging: once per epoch, ~ succinct model weights checkpoint
"""

"""
Shortcut weights (per-epoch) format:
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


# TODO: update, as log_lora_weights.py
def log_layer_res_shortcut_svd(
    model: PreTrainedModel, current_epoch: int, log_dir
) -> dict[str, Tensor | float]:
    singular_uneven: dict[str, Any]
    if "OPT" in model.__class__.__name__:
        # print(f"Epoch {self.current_epoch} getting singular values...")
        singular_uneven = get_opt_layer_res_shortcut_svd(model)
        # print(singular_uneven)
    elif "Roberta" in model.__class__.__name__:
        singular_uneven = get_roberta_layer_res_shortcut_svd(model)
    else:
        # TODO: accommodate more models
        raise ValueError(
            f"Model {model.__class__.__name__} not supported for logging shortcut singular values"
        )
    _header = list(singular_uneven.keys())
    _header.insert(0, "epoch")
    singular_uneven["epoch"] = current_epoch

    # TODO: log by YAML

    filename = f"{log_dir}/svd.csv"
    if os.path.isfile(filename):
        with open(filename, "a+", encoding="UTF8", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=_header)
            dict_writer.writerow(singular_uneven)
    else:
        with open(filename, "a+", encoding="UTF8", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=_header)
            dict_writer.writeheader()
            dict_writer.writerow(singular_uneven)

    return singular_uneven


def get_opt_layer_res_shortcut_svd(model: PreTrainedModel) -> dict[str, Tensor | float]:
    shortcut_weights = {}
    res = {}
    for name, param in model.named_parameters():
        if "proj_A" not in name and "proj_B" not in name:
            continue

        mat_name: str = re.findall(
            r"layers\.\d+\.residual[1-2]\.proj_[A-B]\.ags-layer-res-network-opt\.weight",
            name,
        )[0]
        shortcut_weights[mat_name] = param.data

        if "proj_A" in mat_name:
            corr_name: str = mat_name.replace("proj_A", "proj_B")
            if corr_name not in shortcut_weights:
                continue
            singulars: Tensor = torch.linalg.svdvals(
                torch.matmul(shortcut_weights[mat_name], shortcut_weights[corr_name])
            )
        else:
            corr_name: str = mat_name.replace("proj_B", "proj_A")
            if corr_name not in shortcut_weights:
                continue
            singulars: Tensor = torch.linalg.svdvals(
                torch.matmul(shortcut_weights[corr_name], shortcut_weights[mat_name])
            )

        shortcut_name: str = re.findall(r"layers\.\d+\.residual[1-2]", mat_name)[0]
        res[f"svdvals_{shortcut_name}"] = singulars.tolist()

        unevenness = compute_unevenness_metrics(singulars)
        for metric_name, value in unevenness.items():
            res[f"uneven_{metric_name}_{shortcut_name}"] = value

    return res


def get_roberta_layer_res_shortcut_svd(model: PreTrainedModel) -> dict[str, Tensor]:
    raise NotImplementedError
