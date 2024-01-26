import torch
from torch import Tensor


def compute_unevenness_metrics(singulars: Tensor) -> dict[str, float | list[float]]:
    cv = torch.var(singulars, dim=-1) / torch.mean(singulars, dim=-1)
    max_deviation = (
        torch.max(singulars, dim=-1) - torch.min(singulars, dim=-1)
    ) / torch.max(singulars, dim=-1)
    mean_deviation = (
        (torch.max(singulars, dim=-1) - torch.min(singulars, dim=-1))
        / 2
        / torch.mean(singulars, dim=-1)
    )
    deviation = (
        torch.max(singulars, dim=-1) - torch.min(singulars, dim=-1)
    ) / torch.mean(singulars, dim=-1)
    _normalised: Tensor = singulars / torch.sum(singulars, dim=-1).unsqueeze(-1)
    shannon_entropy = entropy(_normalised)
    _uniform: Tensor = torch.full(_normalised.size(), 1 / _normalised.size()[-1]).to(
        _normalised.device
    )
    kl_div = kl_divergence(_uniform, _normalised)
    return {
        "coefficient_of_variation": cv,
        "max_deviation": max_deviation,
        "mean_deviation": mean_deviation,
        "deviation": deviation,
        "entropy": shannon_entropy,
        "kl_divergence": kl_div,
    }


_EPSILON = 1e-7


def entropy(distribution: Tensor) -> float | list[float]:
    assert torch.all(
        1 - _EPSILON <= torch.sum(distribution, dim=-1) <= 1 + _EPSILON
    ), "Input distribution(s) must sum to 1"
    return torch.sum(-distribution * torch.log2(distribution), dim=-1).tolist()


def kl_divergence(distribution1: Tensor, distribution2: Tensor) -> float | list[float]:
    if len(distribution1.size()) == 1 and len(distribution2.size()) == 1:
        assert torch.all(
            1 - _EPSILON <= torch.sum(distribution1, dim=-1) <= 1 + _EPSILON
            and 1 - _EPSILON <= torch.sum(distribution2, dim=-1) <= 1 + _EPSILON
        ), "Input distributions must sum to 1"
        return torch.sum(
            distribution1 * torch.log2(distribution1 / distribution2), dim=-1
        ).tolist()
