import copy

import torch
import torch.nn.functional as F

from lora.lora_modules import LoraLinear
from projectors.shortcut_modules import ShortcutFromIdentity, ShortcutFromZeros


def test_ShortcutFromIdentity_init():
    sc = ShortcutFromIdentity(
        in_out_features=5,
        config={
            "r": 3,
            "shortcut_alpha": 5,
            "proj_dropout": 0.0,
            "projector_name": "test_shortcut",
            "disable_projector": False,
        },
    )

    assert sc.active_projector == "test_shortcut"
    assert torch.all(sc.proj_B["test_shortcut"].weight == 0.0)
    assert torch.all(sc.weight == torch.eye(5))
    assert sc.proj_A["test_shortcut"].weight.size() == torch.Size([3, 5])
    assert sc.proj_B["test_shortcut"].weight.size() == torch.Size([5, 3])
    assert sc.scaling["test_shortcut"] == 5 / 3
    assert sc.importance_beta == 1.0
    assert sc.merged == False
    assert sc.bias is None


def test_ShortcutFromZeros():
    sc = ShortcutFromZeros(
        in_out_features=5,
        config={
            "r": 3,
            "shortcut_alpha": 5,
            "proj_dropout": 0.0,
            "projector_name": "test_shortcut",
            "disable_projector": False,
        },
    )

    assert sc.active_projector == "test_shortcut"
    assert torch.all(sc.proj_B["test_shortcut"].weight == 0.0)
    assert torch.all(sc.weight == torch.zeros(5))
    assert sc.proj_A["test_shortcut"].weight.size() == torch.Size([3, 5])
    assert sc.proj_B["test_shortcut"].weight.size() == torch.Size([5, 3])
    assert sc.scaling["test_shortcut"] == 5 / 3
    assert sc.importance_beta == 1.0
    assert sc.merged == False
    assert sc.bias is None


def test_ShortcutFromIdentity_forward():
    sc = ShortcutFromIdentity(
        in_out_features=5,
        config={
            "r": 3,
            "shortcut_alpha": 5,
            "proj_dropout": 0.0,
            "projector_name": "test_shortcut",
            "disable_projector": False,
        },
    )
    torch.nn.init.kaiming_normal(sc.proj_B["test_shortcut"].weight)
    x = torch.rand(2, 5)

    res = sc.forward(x)

    assert torch.allclose(
        res,
        x
        + 5
        / 3
        * F.linear(
            F.linear(x, sc.proj_A["test_shortcut"].weight),
            sc.proj_B["test_shortcut"].weight,
        ),
    )


def test_ShortcutFromZeros_forward():
    sc = ShortcutFromZeros(
        in_out_features=5,
        config={
            "r": 3,
            "shortcut_alpha": 5,
            "proj_dropout": 0.0,
            "projector_name": "test_shortcut",
            "disable_projector": False,
        },
    )
    torch.nn.init.kaiming_normal(sc.proj_B["test_shortcut"].weight)
    x = torch.rand(2, 5)

    res = sc.forward(x)

    assert torch.allclose(
        res,
        5
        / 3
        * F.linear(
            F.linear(x, sc.proj_A["test_shortcut"].weight),
            sc.proj_B["test_shortcut"].weight,
        ),
    )
