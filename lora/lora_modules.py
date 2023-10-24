import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        **kwargs,
    ):
        # params are stored as {adapter_config_name: param_value}
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})

        self.disable_adapters = False
        self.merged = False

    def set_adapter(
        self, adapter_name, r, lora_alpha, lora_dropout_p, init_lora_weights
    ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout_p > 0:
            dropout_layer = nn.Dropout(p=lora_dropout_p)
        else:
            dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: dropout_layer}))

        if r > 0:
            self.lora_A.update(
                nn.ModuleDict(
                    {adapter_name: nn.Linear(self.in_features, r, bias=False)}
                )
            )
            self.lora_B.update(
                nn.ModuleDict(
                    {adapter_name: nn.Linear(r, self.out_features, bias=False)}
                )
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)


class LoraLinear(nn.Linear, LoRALayer):
    # nn.Linear with LoRA
    def __init__(
        self, in_features: int, out_features: int, config: dict = None, **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, in_features, out_features)
        self.weight.requires_grad = False

        self.config = config
        r, lora_alpha, lora_dropout_p, adapter_name = (
            config["r"],
            config["lora_alpha"],
            float(config["lora_dropout"]),
            config["adapter_name"],
        )

        init_lora_weights = kwargs.pop("init_lora_weights", default=True)
        self.fan_in_fan_out = config.get("fan_in_fan_out", default=False)
        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.set_adapter(adapter_name, r, lora_alpha, lora_dropout_p, init_lora_weights)
        self.active_adapter = adapter_name

    def get_delta_w(self, adapter_name):
        # Linear's tensor is out_features rows * in_features columns as default
        prod = self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight
        if self.fan_in_fan_out:
            prod = prod.T
        return prod * self.scaling[adapter_name]

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += self.get_delta_w(self.active_adapter)
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= self.get_delta_w(self.active_adapter)
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            res = F.linear(
                x, self.weight if not self.fan_in_fan_out else self.weight.T, self.bias
            )
        elif self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            res = F.linear(
                x, self.weight if not self.fan_in_fan_out else self.weight.T, self.bias
            )
        elif self.r[self.active_adapter] > 0 and not self.merged:
            res = F.linear(
                x, self.weight if not self.fan_in_fan_out else self.weight.T, self.bias
            )
            # x = x.to(self.lora_A[self.active_adapter].weight.dtype)
            res += (
                self.lora_B[self.active_adapter](
                    self.lora_A[self.active_adapter](
                        self.lora_dropout[self.active_adapter](x)
                    )
                )
            ) * self.scaling[self.active_adapter]
        else:
            res = F.linear(
                x, self.weight if not self.fan_in_fan_out else self.weight.T, self.bias
            )
        # res = res.to(input_dtype)
        return res
