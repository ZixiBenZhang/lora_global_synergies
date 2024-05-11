import copy

import torch
import torch.nn.functional as F

from lora.lora_modules import LoraLinear


def test_LoraLinear_init():
    lora = LoraLinear(
        in_features=5,
        out_features=6,
        config={
            "r": 3,
            "lora_alpha": 5,
            "lora_dropout": 0.0,
            "adapter_name": "test_lora",
            "disable_adapter": False,
        },
        bias=False,
    )

    assert lora.active_adapter == "test_lora"
    assert torch.all(lora.lora_B["test_lora"].weight == 0.0)
    assert lora.weight.size() == torch.Size([6, 5])
    assert lora.lora_A["test_lora"].weight.size() == torch.Size([3, 5])
    assert lora.lora_B["test_lora"].weight.size() == torch.Size([6, 3])
    assert lora.scaling["test_lora"] == 5 / 3
    assert lora.importance_alpha == 1.0
    assert lora.merged == False
    assert lora.bias is None


def test_LoraLinear_get_delta_w():
    lora = LoraLinear(
        in_features=5,
        out_features=6,
        config={
            "r": 3,
            "lora_alpha": 5,
            "lora_dropout": 0.0,
            "adapter_name": "test_lora",
            "disable_adapter": False,
        },
        bias=False,
    )
    torch.nn.init.kaiming_normal(lora.lora_B["test_lora"].weight)

    assert torch.allclose(
        lora.get_delta_w("test_lora"),
        5 / 3 * lora.lora_B["test_lora"].weight @ lora.lora_A["test_lora"].weight,
    )


def test_LoraLinear_merge():
    lora = LoraLinear(
        in_features=5,
        out_features=6,
        config={
            "r": 3,
            "lora_alpha": 5,
            "lora_dropout": 0.0,
            "adapter_name": "test_lora",
            "disable_adapter": False,
        },
        bias=False,
    )
    torch.nn.init.kaiming_normal(lora.lora_B["test_lora"].weight)

    w = copy.deepcopy(lora.weight)
    lora.merge()

    assert torch.allclose(
        w + 5 / 3 * lora.lora_B["test_lora"].weight @ lora.lora_A["test_lora"].weight,
        lora.weight,
    )


def test_LoraLinear_unmerge():
    lora = LoraLinear(
        in_features=5,
        out_features=6,
        config={
            "r": 3,
            "lora_alpha": 5,
            "lora_dropout": 0.0,
            "adapter_name": "test_lora",
            "disable_adapter": False,
        },
        bias=False,
    )
    torch.nn.init.kaiming_normal(lora.lora_B["test_lora"].weight)

    w = copy.deepcopy(lora.weight)
    lora.merge()
    lora.unmerge()

    assert torch.allclose(w, lora.weight)


def test_LoraLinear_unmerged_forward():
    lora = LoraLinear(
        in_features=5,
        out_features=6,
        config={
            "r": 3,
            "lora_alpha": 5,
            "lora_dropout": 0.0,
            "adapter_name": "test_lora",
            "disable_adapter": False,
        },
        bias=False,
    )
    torch.nn.init.kaiming_normal(lora.lora_B["test_lora"].weight)
    x = torch.rand(2, 5)

    res = lora.forward(x)

    assert torch.allclose(
        res,
        F.linear(x, lora.weight)
        + 5
        / 3
        * F.linear(
            F.linear(x, lora.lora_A["test_lora"].weight),
            lora.lora_B["test_lora"].weight,
        ),
    )


def test_LoraLinear_merged_forward():
    lora = LoraLinear(
        in_features=5,
        out_features=6,
        config={
            "r": 3,
            "lora_alpha": 5,
            "lora_dropout": 0.0,
            "adapter_name": "test_lora",
            "disable_adapter": False,
        },
        bias=False,
    )
    torch.nn.init.kaiming_normal(lora.lora_B["test_lora"].weight)
    x = torch.rand(2, 5)

    w = copy.deepcopy(lora.weight)
    lora.merge()
    res = lora.forward(x)

    assert torch.allclose(
        res,
        F.linear(x, w)
        + 5
        / 3
        * F.linear(
            F.linear(x, lora.lora_A["test_lora"].weight),
            lora.lora_B["test_lora"].weight,
        ),
    )


def test_LoraLinear_invalid_module_forward():
    lora = LoraLinear(
        in_features=5,
        out_features=6,
        config={
            "r": 0,
            "lora_alpha": 5,
            "lora_dropout": 0.0,
            "adapter_name": "test_lora",
            "disable_adapter": False,
        },
        bias=False,
    )
    x = torch.rand(2, 5)

    res = lora.forward(x)

    assert torch.allclose(
        res,
        F.linear(x, lora.weight),
    )


def test_LoraLinear_disabled_module_forward():
    lora = LoraLinear(
        in_features=5,
        out_features=6,
        config={
            "r": 3,
            "lora_alpha": 5,
            "lora_dropout": 0.0,
            "adapter_name": "test_lora",
            "disable_adapter": True,
        },
        bias=False,
    )
    torch.nn.init.kaiming_normal(lora.lora_B["test_lora"].weight)
    x = torch.rand(2, 5)

    w = copy.deepcopy(lora.weight)
    lora.merge()
    res = lora.forward(x)

    assert torch.allclose(
        res,
        F.linear(x, w),
    )


def test_LoraLinear_set_importance_alpha():
    lora = LoraLinear(
        in_features=5,
        out_features=6,
        config={
            "r": 3,
            "lora_alpha": 5,
            "lora_dropout": 0.0,
            "adapter_name": "test_lora",
            "disable_adapter": True,
        },
        bias=False,
    )

    lora.set_importance_alpha(0.6)

    assert lora.importance_alpha == 0.6


def test_LoraLinear_set_importance_alpha_forward():
    lora = LoraLinear(
        in_features=5,
        out_features=6,
        config={
            "r": 3,
            "lora_alpha": 5,
            "lora_dropout": 0.0,
            "adapter_name": "test_lora",
            "disable_adapter": False,
        },
        bias=False,
    )
    torch.nn.init.kaiming_normal(lora.lora_B["test_lora"].weight)
    x = torch.rand(2, 5)

    lora.set_importance_alpha(0.6)
    res = lora.forward(x)

    assert torch.allclose(
        res,
        (
            F.linear(x, lora.weight)
            + 5
            / 3
            * F.linear(
                F.linear(x, lora.lora_A["test_lora"].weight),
                lora.lora_B["test_lora"].weight,
            )
        )
        * 0.6,
    )
