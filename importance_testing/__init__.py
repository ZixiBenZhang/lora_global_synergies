from typing import Callable

from .alpha_test import alpha_importance_test
from .constant import constant_test
from .grad_norm import grad_norm_test
from .snip import snip_test
from .synflow import synflow_test


def get_importance_method(importance_test_name: str) -> Callable:
    match importance_test_name:
        case "alpha_test":
            return alpha_importance_test
        case "constant":
            return constant_test
        case "grad_norm":
            return grad_norm_test
        case "snip":
            return snip_test
        case "synflow":
            return synflow_test
