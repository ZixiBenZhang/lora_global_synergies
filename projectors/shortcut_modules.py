import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankProjectorLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        **kwargs,
    ):
        # params are stored as {projector_config_name: param_value}
        self.r = {}
        self.shortcut_alpha = {}
        self.scaling = {}
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

        self.proj_dropout = nn.ModuleDict({})
        self.proj_A = nn.ModuleDict({})
        self.proj_B = nn.ModuleDict({})

        self.disable_projectors = False
        self.merged = False

    def set_projector(
        self, projector_name, r, shortcut_alpha, proj_dropout_p, init_proj_weights
    ):
        self.r[projector_name] = r
        self.shortcut_alpha[projector_name] = shortcut_alpha

        if proj_dropout_p > 0:
            dropout_layer = nn.Dropout(p=proj_dropout_p)
        else:
            dropout_layer = nn.Identity()
        self.proj_dropout.update(nn.ModuleDict({projector_name: dropout_layer}))

        if r > 0:
            self.proj_A.update(
                nn.ModuleDict(
                    {projector_name: nn.Linear(self.in_features, r, bias=False)}
                )
            )
            self.proj_B.update(
                nn.ModuleDict(
                    {projector_name: nn.Linear(r, self.out_features, bias=False)}
                )
            )
            self.scaling[projector_name] = shortcut_alpha / r
        if init_proj_weights:
            self.reset_proj_parameters(projector_name)

    def reset_proj_parameters(self, projector_name):
        if projector_name in self.proj_A.keys():
            nn.init.kaiming_uniform_(self.proj_A[projector_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj_B[projector_name].weight)


class ShortcutBase(nn.Linear, LowRankProjectorLayer):
    def __init__(self, in_out_features: int, config: dict = None, **kwargs):
        nn.Linear.__init__(self, in_out_features, in_out_features, bias=False, **kwargs)
        LowRankProjectorLayer.__init__(self, in_out_features, in_out_features)
        self.weight.requires_grad = False

        self.config = config
        (
            r,
            shortcut_alpha,
            proj_dropout_p,
            projector_name,
            disable_projector,
            importance_beta,
        ) = (
            config["r"],
            config["shortcut_alpha"],
            float(config["proj_dropout"]),
            config["projector_name"],
            config["disable_projector"],
            config.get("importance_beta", 1.0),
        )
        init_proj_weights = config.get("init_proj_weights", True)
        self.disable_projectors = disable_projector
        self.fan_in_fan_out = config.get("fan_in_fan_out", False)
        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.T

        self.set_projector(
            projector_name, r, shortcut_alpha, proj_dropout_p, init_proj_weights
        )
        self.active_projector = projector_name

        self.importance_beta = torch.tensor(importance_beta, requires_grad=False)

    def set_importance_beta(self, beta: float):
        assert type(beta) is float
        self.importance_beta = torch.tensor(
            beta, requires_grad=self.importance_beta.requires_grad
        )

    def _force_init_weight(self):
        pass

    def get_delta_w(self, projector_name):
        # Linear's tensor is out_features rows * in_features columns as default
        prod = self.proj_B[projector_name].weight @ self.proj_A[projector_name].weight
        if self.fan_in_fan_out:
            prod = prod.T
        return prod * self.scaling[projector_name]

    def merge(self):
        self._force_init_weight()
        if self.active_projector not in self.proj_A.keys():
            return
        if self.merged:
            return
        if self.r[self.active_projector] > 0:
            self.weight.data += self.get_delta_w(self.active_projector)
            self.merged = True

    def unmerge(self):
        self._force_init_weight()
        if self.active_projector not in self.proj_A.keys():
            return
        if not self.merged:
            return
        if self.r[self.active_projector] > 0:
            self.weight.data -= self.get_delta_w(self.active_projector)
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._force_init_weight()
        # input_dtype = x.dtype
        if (
            self.active_projector not in self.proj_A.keys()
        ):  # active_projector wouldn't be in proj_A.keys() if r==0
            res = F.linear(
                x, self.weight if not self.fan_in_fan_out else self.weight.T, self.bias
            )
        elif self.disable_projectors:
            res = F.linear(
                x, self.weight if not self.fan_in_fan_out else self.weight.T, self.bias
            )
        elif self.r[self.active_projector] > 0 and not self.merged:
            # Projector dropout used
            res = F.linear(
                x, self.weight if not self.fan_in_fan_out else self.weight.T, self.bias
            )
            # x = x.to(self.proj_A[self.active_projector].weight.dtype)
            # Beta only applied to (delta W)
            res = (
                res
                + self.proj_B[self.active_projector](
                    self.proj_A[self.active_projector](
                        self.proj_dropout[self.active_projector](x)
                    )
                )
                * self.scaling[self.active_projector]
                * self.importance_beta
            )
        else:
            if self.importance_beta != 1.0 and self.r[self.active_projector] > 0:
                self.unmerge()
                # Projector dropout used
                res = F.linear(
                    x,
                    self.weight if not self.fan_in_fan_out else self.weight.T,
                    self.bias,
                )
                # x = x.to(self.proj_A[self.active_projector].weight.dtype)
                # Beta only applied to (delta W)
                res = (
                    res
                    + self.proj_B[self.active_projector](
                        self.proj_A[self.active_projector](
                            self.proj_dropout[self.active_projector](x)
                        )
                    )
                    * self.scaling[self.active_projector]
                    * self.importance_beta
                )
            else:
                # Projector dropout unused
                res = F.linear(
                    x,
                    self.weight if not self.fan_in_fan_out else self.weight.T,
                    self.bias,
                )
        # res = res.to(input_dtype)
        return res


class ShortcutFromIdentity(ShortcutBase):
    def __init__(self, in_out_features: int, config: dict = None, **kwargs):
        super().__init__(in_out_features, config, **kwargs)

        # Identity layer
        nn.init.eye_(self.weight)

    def _force_init_weight(self):
        nn.init.eye_(self.weight)

    # def merge(self):
    #     if self.active_projector not in self.proj_A.keys():
    #         return
    #     if self.merged:
    #         return
    #     if self.r[self.active_projector] > 0:
    #         self.weight.data = torch.eye(self.weight.data.size(0)) + self.get_delta_w(self.active_projector)
    #         self.merged = True
    #
    # def unmerge(self):
    #     if self.active_projector not in self.proj_A.keys():
    #         return
    #     if not self.merged:
    #         return
    #     if self.r[self.active_projector] > 0:
    #         self.weight.data -= self.get_delta_w(self.active_projector)
    #         self.merged = False
    #
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # input_dtype = x.dtype
    #     if (
    #             self.active_projector not in self.proj_A.keys()
    #     ):  # active_projector wouldn't be in proj_A.keys() if r==0
    #         res = x
    #     elif self.disable_projectors:
    #         res = x
    #     elif self.r[self.active_projector] > 0 and not self.merged:
    #         # Projector dropout used
    #         res = x
    #         # x = x.to(self.proj_A[self.active_projector].weight.dtype)
    #         # Beta only applied to (delta W)
    #         res = (
    #             res
    #             + self.proj_B[self.active_projector](
    #                 self.proj_A[self.active_projector](
    #                     self.proj_dropout[self.active_projector](x)
    #                 )
    #             )
    #             * self.scaling[self.active_projector]
    #             * self.importance_beta
    #         )
    #     else:
    #         if self.importance_beta != 1.0 and self.r[self.active_projector] > 0:
    #             self.unmerge()
    #             # Projector dropout used
    #             res = x
    #             # x = x.to(self.proj_A[self.active_projector].weight.dtype)
    #             # Beta only applied to (delta W)
    #             res = (
    #                 res
    #                 + self.proj_B[self.active_projector](
    #                     self.proj_A[self.active_projector](
    #                         self.proj_dropout[self.active_projector](x)
    #                     )
    #                 )
    #                 * self.scaling[self.active_projector]
    #                 * self.importance_beta
    #             )
    #         else:
    #             # Projector dropout unused
    #             res = F.linear(
    #                 x,
    #                 self.weight if not self.fan_in_fan_out else self.weight.T,
    #                 self.bias,
    #             )
    #     # res = res.to(input_dtype)
    #     return res


class ShortcutFromZeros(ShortcutBase):
    def __init__(self, in_out_features: int, config: dict = None, **kwargs):
        super().__init__(in_out_features, config, **kwargs)

        # Zeros layer
        nn.init.zeros_(self.weight)

    def _force_init_weight(self):
        nn.init.zeros_(self.weight)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # input_dtype = x.dtype
    #     if (
    #         self.active_projector not in self.proj_A.keys()
    #     ):  # active_projector wouldn't be in proj_A.keys() if r==0
    #         res = torch.zeros_like(x)
    #     elif self.disable_projectors:
    #         res = torch.zeros_like(x)
    #     elif self.r[self.active_projector] > 0 and not self.merged:
    #         # Projector dropout used
    #         # x = x.to(self.proj_A[self.active_projector].weight.dtype)
    #         res = self.proj_B[self.active_projector](
    #             self.proj_A[self.active_projector](
    #                 self.proj_dropout[self.active_projector](x)
    #             )
    #         )
    #     else:
    #         # Projector dropout unused
    #         res = F.linear(
    #             x, self.weight if not self.fan_in_fan_out else self.weight.T, self.bias
    #         )
    #     # res = res.to(input_dtype)
    #     # Only effective during beta importance testing
    #     res = res * self.importance_beta
    #     return res


def mark_ags_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "proj_" in n:
            p.requires_grad = True


def update_ags_importance_beta_require_grad(model: nn.Module, require_grad: bool):
    for n, p in model.named_parameters():
        if "importance_beta" in n:
            p.requires_grad = require_grad


def update_ags_ln_require_grad(model: nn.Module, require_grad: bool):
    for n, p in model.named_parameters():
        if "shortcut_ln_" in n:
            p.requires_grad = require_grad
