import re

import scipy.stats
from scipy.stats import entropy
import torch
from torch import Tensor
from transformers import PreTrainedModel


def get_opt_layer_res_shortcut_svd(model: PreTrainedModel) -> dict[str, Tensor]:
    shortcut_weights = {}
    res = {}
    for name, param in model.named_parameters():
        if "ags-layer-res-network-opt" not in name:
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
            shortcut_name: str = re.findall(r"layer\.\d+\.residual[1-2]", mat_name)[0]
            singulars = torch.linalg.svdvals(
                torch.matmul(shortcut_weights[mat_name], shortcut_weights[corr_name])
            )
        else:
            corr_name: str = mat_name.replace("proj_B", "proj_A")
            if corr_name not in shortcut_weights:
                continue
            shortcut_name: str = re.findall(r"layer\.\d+\.residual[1-2]", mat_name)[0]
            singulars = torch.linalg.svdvals(
                torch.matmul(shortcut_weights[corr_name], shortcut_weights[mat_name])
            )
        res[f"svdvals_{shortcut_name}_epoch"] = singulars

        unevenness = compute_unevenness_metrics(singulars)
        for metric_name, value in unevenness:
            res[f"uneven_{metric_name}_{shortcut_name}_epoch"] = value

    return res


def get_roberta_layer_res_shortcut_svd(model: PreTrainedModel) -> dict[str, Tensor]:
    raise NotImplementedError


def compute_unevenness_metrics(singulars: Tensor) -> dict[str, float]:
    cv = torch.var(singulars) / torch.mean(singulars)
    max_deviation = (torch.max(singulars) - torch.min(singulars)) / torch.max(singulars)
    mean_deviation = (
        (torch.max(singulars) - torch.min(singulars)) / 2 / torch.mean(singulars)
    )
    deviation = (torch.max(singulars) - torch.min(singulars)) / torch.mean(singulars)
    _normalised: Tensor = singulars / torch.sum(singulars)
    shannon_entropy = entropy(_normalised, base=2)
    _uniform: Tensor = torch.full(_normalised.size(), 1)
    kl_div = entropy(_uniform / torch.sum(_uniform), _normalised, base=2)
    return {
        "coefficient_of_variation": cv,
        "max_deviation": max_deviation,
        "mean_deviation": mean_deviation,
        "deviation": deviation,
        "entropy": shannon_entropy,
        "kl_divergence": kl_div,
    }
