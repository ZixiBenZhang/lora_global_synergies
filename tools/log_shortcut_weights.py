import re

import torch
from torch import Tensor
from transformers import PreTrainedModel


def get_opt_layer_res_shortcut_svd(model: PreTrainedModel) -> dict[str, Tensor]:
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
            singulars = torch.linalg.svdvals(
                torch.matmul(shortcut_weights[mat_name], shortcut_weights[corr_name])
            )
        else:
            corr_name: str = mat_name.replace("proj_B", "proj_A")
            if corr_name not in shortcut_weights:
                continue
            singulars = torch.linalg.svdvals(
                torch.matmul(shortcut_weights[corr_name], shortcut_weights[mat_name])
            )

        shortcut_name: str = re.findall(r"layers\.\d+\.residual[1-2]", mat_name)[0]
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
    shannon_entropy = entropy(_normalised)
    _uniform: Tensor = torch.full(_normalised.size(), 1)
    kl_div = kl_divergence(_uniform / torch.sum(_uniform), _normalised)
    return {
        "coefficient_of_variation": cv,
        "max_deviation": max_deviation,
        "mean_deviation": mean_deviation,
        "deviation": deviation,
        "entropy": shannon_entropy,
        "kl_divergence": kl_div,
    }


EPSILON = 1e-7


def entropy(distribution: Tensor) -> float:
    assert len(distribution.size()) == 1, "Input distribution needs to be 1D tensor"
    assert (
        1 - EPSILON <= torch.sum(distribution) <= 1 + EPSILON
    ), "Input distribution must sum to 1"
    return torch.sum(-distribution * torch.log2(distribution)).item()


def kl_divergence(distribution1: Tensor, distribution2: Tensor) -> float:
    assert (
        len(distribution1.size()) == 1 and len(distribution2.size()) == 1
    ), "Input distributions need to be 1D tensor"
    assert (
        1 - EPSILON <= torch.sum(distribution1) <= 1 + EPSILON
        and 1 - EPSILON <= torch.sum(distribution2) <= 1 + EPSILON
    ), "Input distributions must sum to 1"
    return torch.sum(distribution1 * torch.log2(distribution1 / distribution2)).item()
