import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation


class Adapter(nn.Module):
    def __init__(
            self,
            model_dim: int,  # model dimension
            d: int,  # adapter bottleneck dimension
            activation: str,  # activation function
    ):
        super().__init__()
        self.down_proj = nn.Linear(model_dim, d)
        self.act = get_activation(activation)
        self.up_proj = nn.Linear(d, model_dim)

    def reset_parameters(self) -> None:
        # TODO: matrices initialisation
        pass

    def forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor = 0,  # residual=0 when no residual shortcut within adapter (e.g. PA)
    ) -> torch.Tensor:
        res = self.down_proj(x)
        res = self.act(res)
        res = self.up_proj(res)
        return res + residual

