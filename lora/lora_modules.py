import torch
import torch.nn as nn
import torch.nn.functional as F

from lora.lora_base import LoRALayer


class LoraLinear(nn.Linear, LoRALayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout_p: float = 0.,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r, lora_alpha, lora_dropout_p, merge_weights)

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((in_features, r)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((r, out_features)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False  # since pre-trained W is always frozen
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.r > 0:
            # Initialise A as random Gaussian, B as zeros
            nn.init.normal_(self.lora_A)
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            orig_res = F.linear(x, self.weight, self.bias)
            delta_res = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
            return orig_res + delta_res
        else:
            return F.linear(x, self.weight, self.bias)

    def train(self, mode: bool = True) -> None:  # switch on/off training mode
        nn.Linear.train(self, mode)
        if mode:
            # unmerge W and A@B during training
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= self.lora_A @ self.lora_B * self.scaling
                self.merged = False
        else:
            # merge W and A@B during evaluation
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += self.lora_A @ self.lora_B * self.scaling
                self.merged = True
