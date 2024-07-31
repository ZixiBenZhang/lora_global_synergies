import logging
import math
import time
import types
from typing import Any, Optional, Callable

import numpy as np
import pytorch_lightning as pl
import toml
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from dataset.pl_dataset_module import AgsDataModule
from lora.lora_modules import LoraLinear
from models.modeling_llama_lora_ags import (
    LlamaLoraAgsForCausalLM,
    LlamaLoraAgsForSequenceClassification,
    LlamaLoraAgsForQuestionAnswering,
    LlamaLoraAgsDecoderLayer,
)
from models.modeling_qwen2_lora_ags import Qwen2LoraAgsDecoderLayer, Qwen2LoraAgsForCausalLM, \
    Qwen2LoraAgsForSequenceClassification
from pl_model_wrapper.base import PlWrapperBase
from projectors.shortcut_modules import ShortcutBase

logger = logging.getLogger(__name__)

LORA_NAME_HASH = {
    "q_proj": 0,
    "k_proj": 1,
    "v_proj": 2,
    "out_proj": 3,
    "up_proj": 4,
    "gate_proj": 5,
    "down_proj": 6,
    "residual_1": 7,
    "residual_2": 8,
    "shortcut_sa": 9,
    "shortcut_ffn": 10,
}


class DynamicLoraReallocationForLlamaCallback(pl.Callback):
    def __init__(
        self,
        importance_test_name: str,
        N: int | float,
        data_module: AgsDataModule,
        alpha_trainer_args: pl.Trainer,
        alpha_pl_module: PlWrapperBase,
        task: str,
        metric_reduction_tolerance: float,
        turn_on_percentile: float = 0.25,
        limit_test_batches: Optional[int | float] = None,
        save_path: str = None,
        ags_mode: str = None,
    ):
        """
        :param importance_test_name: importance test name (metric) to be used
        :param N: every N steps conduct alpha testing and reallocate lora ranks
        :param data_module: for loading batches for alpha testing
        :param alpha_trainer_args: for building the pl trainer to conduct alpha testing
        :param alpha_pl_module: copy of pl_module, in order to separate its update to trainer
        :param task: task type for determining threshold comparison
        :param metric_reduction_tolerance: for computing the threshold for alpha testing
        :param turn_on_percentile: percentage of lora modules to be activated by the reallocation
        :param limit_test_batches: number of batches used in alpha testing
        :param save_path: file path for saving reallocation history
        :param ags_mode: mode of dynamic reallocation on AGS model shortcuts, one of [None, "combined", "separated"]
        """
        super().__init__()

        self.data_module = data_module
        self.alpha_trainer_args = alpha_trainer_args
        self.alpha_trainer: pl.Trainer = None
        self.alpha_pl_module = alpha_pl_module
        self.train_set_len = None
        self.val_set_len = None
        self.num_devices = None

        assert task in ["classification", "summarization", "causal_language_modeling"]
        self.task = task

        assert importance_test_name in [
            "constant",
            "grad_norm",
            "snip",
            "synflow",
            "fisher",
            "jacob_cov",
            "alpha_test",
        ]
        self.importance_test_name = importance_test_name
        self.importance_test = self._get_importance_test()
        self.ags_mode = ags_mode
        if self.ags_mode is not None and self.ags_mode != "off":
            self.ags_importance_test = self._get_ags_importance_test()
        else:
            self.ags_importance_test = None

        self.N = N
        self.limit_test_batches = limit_test_batches

        self.metric_reduction_tolerance = metric_reduction_tolerance
        self.turn_on_percentile = turn_on_percentile

        self.reallocation_history: list[dict[str, int | list]] = []
        t = time.strftime("%H-%M-%S")
        self.history_save_path = f"{save_path}/reallocation_history_{self.importance_test_name.replace('_', '-')}_{t}.toml"
        self.frequency_save_path = f"{save_path}/reallocation_frequency_{self.importance_test_name.replace('_', '-')}_{t}.toml"

        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            self.rng = torch.random.manual_seed(torch.seed())
        self.rng_state = self.rng.get_state()

    def setup(
        self, trainer: "pl.Trainer", pl_module: PlWrapperBase, stage: str
    ) -> None:
        self.train_set_len = math.ceil(
            len(self._get_train_dataloader()) / trainer.num_devices
        )
        self.val_set_len = math.ceil(
            len(self._get_val_dataloader()) / trainer.num_devices
        )
        if type(self.N) is int:
            # Num of batches between two reallocation
            self.N: int
        elif type(self.N) is float:
            # Percentage of training steps per epoch between two reallocation
            assert 0.0 < self.N <= 1.0, "N should be 0.0 < N <= 1.0"
            self.N: int = math.ceil(self.train_set_len * self.N)
        else:
            raise TypeError("N should be int or float between 0.0 and 1.0")

        if self.limit_test_batches is None:
            # Default: single-shot per epoch on the validation set
            self.limit_test_batches: int = math.ceil(
                round(self.val_set_len / self.N * 2)
            )
        elif type(self.limit_test_batches) is int:
            # Number of alpha test batches
            self.limit_test_batches: int
        elif type(self.limit_test_batches) is float:
            # Percentage of validation set
            self.limit_test_batches: int = (
                math.ceil(self.val_set_len * self.limit_test_batches) * 2
            )
        else:
            raise TypeError(
                "limit_test_batches should be None (assumed single-shot) or int or float between 0.0 and 1.0"
            )

        self.alpha_trainer = pl.Trainer(
            **self.alpha_trainer_args,
            limit_test_batches=self.limit_test_batches,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
        )
        self.num_devices = trainer.num_devices

    def _get_importance_test(self, test_name=None) -> Callable:
        if test_name is None:
            test_name = self.importance_test_name
        match test_name:
            case "alpha_test":
                raise NotImplementedError
            case "constant":
                return self._const_test
            case "grad_norm":
                return self._grad_norm_test
            case "snip":
                return self._snip_test
            case "synflow":
                return self._synflow_test
            case "fisher":
                raise NotImplementedError
            case "jacob_cov":
                raise NotImplementedError
            case _:
                raise ValueError(
                    f"Unsupported importance test {self.importance_test_name}"
                )

    def _get_ags_importance_test(self, test_name=None) -> Callable:
        if test_name is None:
            test_name = self.importance_test_name
        match test_name:
            case "alpha_test":
                pass
            case "constant":
                return self._const_ags_test
            case "grad_norm":
                return self._grad_norm_ags_test
            case "snip":
                pass
            case "synflow":
                pass
            case "fisher":
                raise NotImplementedError
            case "jacob_cov":
                raise NotImplementedError
            case _:
                raise ValueError(
                    f"Unsupported ags importance test {self.importance_test_name}"
                )

    def _get_train_dataloader(self) -> DataLoader:
        return self.data_module.train_dataloader()

    def _get_val_dataloader(self) -> DataLoader:
        return self.data_module.val_dataloader()

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: PlWrapperBase,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.N > 0:
            return
        self.reallocation(trainer, pl_module, batch, batch_idx)

    def reallocation(
        self,
        trainer: pl.Trainer,
        pl_module: PlWrapperBase,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if torch.cuda.current_device() == 0:
            logger.warning(
                f"\n\n>>>>> Running reallocation on epoch {pl_module.current_epoch}, step {batch_idx} <<<<<\n"
            )

        # Turn on all lora modules
        with torch.no_grad():
            model = self.alpha_pl_module.model
            # assert (
            #     type(model) is OPTLoraForCausalLM
            #     or type(model) is OPTLoraForSequenceClassification
            #     or type(model) is OPTLoraForQuestionAnswering
            #     or type(model) is OPTLoraAgsForCausalLM
            #     or type(model) is OPTLoraAgsForSequenceClassification
            #     or type(model) is OPTLoraAgsForQuestionAnswering
            # )
            # model: OPTLoraForCausalLM | OPTLoraForSequenceClassification | OPTLoraForQuestionAnswering | OPTLoraAgsForCausalLM | OPTLoraAgsForSequenceClassification | OPTLoraAgsForQuestionAnswering

            # for decoder_layer in reversed(model.model.decoder.layers):
            for decoder_layer in reversed(model.model.layers):  # Llama2
                # decoder_layer: OPTLoraDecoderLayer
                lora_modules: dict[str, LoraLinear] = {
                    "q_proj": decoder_layer.self_attn.q_proj,
                    "k_proj": decoder_layer.self_attn.k_proj,
                    "v_proj": decoder_layer.self_attn.v_proj,
                    "o_proj": decoder_layer.self_attn.o_proj,
                    "up_proj": decoder_layer.mlp.up_proj,
                    "gate_proj": decoder_layer.mlp.gate_proj,
                    "down_proj": decoder_layer.mlp.down_proj,
                }
                for proj_name, lora in lora_modules.items():
                    if (
                        lora.active_adapter not in lora.lora_A.keys()
                        or lora.r[lora.active_adapter] == 0
                    ):
                        continue
                    lora.disable_adapters = False

                if isinstance(decoder_layer, (LlamaLoraAgsDecoderLayer, Qwen2LoraAgsDecoderLayer)):
                    decoder_layer: LlamaLoraAgsDecoderLayer | Qwen2LoraAgsDecoderLayer
                    shortcut_modules: dict[str, ShortcutBase] = {
                        "residual_1": decoder_layer.residual_1,
                        "residual_2": decoder_layer.residual_2,
                        "shortcut_sa": decoder_layer.shortcut_sa,
                        "shortcut_ffn": decoder_layer.shortcut_ffn,
                    }
                    for proj_name, shortcut in shortcut_modules.items():
                        if (
                            shortcut is None
                            or shortcut.active_projector not in shortcut.proj_A.keys()
                            or shortcut.r[shortcut.active_projector] == 0
                        ):
                            continue
                        shortcut.disable_projectors = False

        # Get alpha importance of lora modules
        # format: {layer_idx: {proj: alpha}}
        # Force CONSTANT for first half epochs
        importance_test = self._get_importance_test(
            test_name="constant"
            if pl_module.current_epoch < 2
            else None
        )
        res_val: dict[int, dict[str, float]] = importance_test(
            trainer,
            pl_module,
            batch,
            batch_idx,
        )
        if self.ags_mode is not None and self.ags_mode != "off":
            ags_importance_test = self._get_ags_importance_test(
                test_name="constant"
                if pl_module.current_epoch < 2
                else None
            )
            ags_res_val: None | dict[int, dict[str, float]] = ags_importance_test(
                trainer,
                pl_module,
                batch,
                batch_idx,
            )
        else:
            ags_res_val = None

        with torch.no_grad():
            if self.ags_mode == "combined":
                # Decide which modules to keep
                alpha_list = np.concatenate(
                    [
                        [
                            (layer_idx, LORA_NAME_HASH[proj_name], v)
                            for proj_name, v in d.items()
                        ]
                        for layer_idx, d in res_val.items()
                    ],
                    axis=0,
                )
                original_lora_module_num = len(alpha_list)
                ags_alpha_list = np.concatenate(
                    [
                        [
                            (layer_idx, LORA_NAME_HASH[proj_name], v)
                            for proj_name, v in d.items()
                        ]
                        for layer_idx, d in ags_res_val.items()
                    ],
                    axis=0,
                )
                original_ags_module_num = len(ags_alpha_list)

                budget = math.floor(
                    self.turn_on_percentile * original_lora_module_num
                ) + round(self.turn_on_percentile * original_ags_module_num)
                alpha_list = np.concatenate([alpha_list, ags_alpha_list], axis=0)
                idx = alpha_list[:, 2].argsort()
                alpha_threshold = alpha_list[idx[-budget], 2]
                if sum(alpha_list[:, 2] == alpha_threshold) > 1:
                    # Uniformly break tie
                    greater = alpha_list[alpha_list[:, 2] > alpha_threshold, :2]
                    tie = alpha_list[alpha_list[:, 2] == alpha_threshold, :2]
                    self.rng.set_state(self.rng_state)
                    tie_idx = torch.randperm(len(tie), generator=self.rng)[
                        : (budget - len(greater))
                    ].numpy()
                    self.rng_state = self.rng.get_state()
                    turn_on = np.concatenate([tie[tie_idx], greater], axis=0)
                else:
                    idx = idx[-budget:]
                    turn_on = alpha_list[idx, :2]
                turn_on = turn_on.astype(int).tolist()
                assert len(turn_on) == budget

                reallocation: list[list[int]] = alpha_list.tolist()
                reallocation = [
                    [
                        str(int(layer_idx)),
                        list(LORA_NAME_HASH.keys())[int(proj_hash)],
                        str(alpha),
                        str(([layer_idx, proj_hash] in turn_on)),
                    ]
                    for layer_idx, proj_hash, alpha in reallocation
                ]
                self.reallocation_history.append(
                    {
                        "epoch": pl_module.current_epoch,
                        "step": batch_idx,
                        "turn_on": reallocation,
                    }
                )

                # Turn on/off lora modules
                model = self.alpha_pl_module.model
                # assert (
                #     type(model) is OPTLoraAgsForCausalLM
                #     or type(model) is OPTLoraAgsForSequenceClassification
                #     or type(model) is OPTLoraAgsForQuestionAnswering
                # )
                # model: OPTLoraAgsForCausalLM | OPTLoraAgsForSequenceClassification | OPTLoraAgsForQuestionAnswering

                # for decoder_layer in reversed(model.model.decoder.layers):
                for decoder_layer in reversed(model.model.layers):  # Llama2
                    # decoder_layer: OPTLoraAgsDecoderLayer
                    layer_idx = decoder_layer.layer_idx

                    lora_modules: dict[str, LoraLinear] = {
                        "q_proj": decoder_layer.self_attn.q_proj,
                        "k_proj": decoder_layer.self_attn.k_proj,
                        "v_proj": decoder_layer.self_attn.v_proj,
                        "o_proj": decoder_layer.self_attn.o_proj,
                        "up_proj": decoder_layer.mlp.up_proj,
                        "gate_proj": decoder_layer.mlp.gate_proj,
                        "down_proj": decoder_layer.mlp.down_proj,
                    }
                    for proj_name, lora in lora_modules.items():
                        if (
                            lora.active_adapter not in lora.lora_A.keys()
                            or lora.r[lora.active_adapter] == 0
                        ):
                            continue
                        proj_hash = LORA_NAME_HASH[proj_name]
                        lora.disable_adapters = [layer_idx, proj_hash] not in turn_on

                    shortcut_modules: dict[str, ShortcutBase] = {
                        "residual_1": decoder_layer.residual_1,
                        "residual_2": decoder_layer.residual_2,
                        "shortcut_sa": decoder_layer.shortcut_sa,
                        "shortcut_ffn": decoder_layer.shortcut_ffn,
                    }
                    for proj_name, shortcut in shortcut_modules.items():
                        if (
                            shortcut is None
                            or shortcut.active_projector not in shortcut.proj_A.keys()
                            or shortcut.r[shortcut.active_projector] == 0
                        ):
                            continue
                        proj_hash = LORA_NAME_HASH[proj_name]
                        shortcut.disable_projectors = [
                            layer_idx,
                            proj_hash,
                        ] not in turn_on

            elif self.ags_mode == "separated":
                # Decide which modules to keep
                alpha_list = np.concatenate(
                    [
                        [
                            (layer_idx, LORA_NAME_HASH[proj_name], v)
                            for proj_name, v in d.items()
                        ]
                        for layer_idx, d in res_val.items()
                    ],
                    axis=0,
                )
                original_lora_module_num = len(alpha_list)
                ags_alpha_list = np.concatenate(
                    [
                        [
                            (layer_idx, LORA_NAME_HASH[proj_name], v)
                            for proj_name, v in d.items()
                        ]
                        for layer_idx, d in ags_res_val.items()
                    ],
                    axis=0,
                )
                original_ags_module_num = len(ags_alpha_list)

                budget = math.floor(self.turn_on_percentile * original_lora_module_num)
                idx = alpha_list[:, 2].argsort()
                alpha_threshold = alpha_list[idx[-budget], 2]
                ags_budget = round(self.turn_on_percentile * original_ags_module_num)
                ags_idx = ags_alpha_list[:, 2].argsort()
                ags_alpha_threshold = ags_alpha_list[ags_idx[-ags_budget], 2]

                if sum(alpha_list[:, 2] == alpha_threshold) > 1:
                    # Uniformly break tie
                    greater = alpha_list[alpha_list[:, 2] > alpha_threshold, :2]
                    tie = alpha_list[alpha_list[:, 2] == alpha_threshold, :2]
                    self.rng.set_state(self.rng_state)
                    tie_idx = torch.randperm(len(tie), generator=self.rng)[
                        : (budget - len(greater))
                    ].numpy()
                    self.rng_state = self.rng.get_state()
                    turn_on = np.concatenate([tie[tie_idx], greater], axis=0)
                else:
                    idx = idx[-budget:]
                    turn_on = alpha_list[idx, :2]
                if sum(ags_alpha_list[:, 2] == ags_alpha_threshold) > 1:
                    # Uniformly break tie
                    greater = ags_alpha_list[
                        ags_alpha_list[:, 2] > ags_alpha_threshold, :2
                    ]
                    tie = ags_alpha_list[
                        ags_alpha_list[:, 2] == ags_alpha_threshold, :2
                    ]
                    self.rng.set_state(self.rng_state)
                    tie_idx = torch.randperm(len(tie), generator=self.rng)[
                        : (ags_budget - len(greater))
                    ].numpy()
                    self.rng_state = self.rng.get_state()
                    ags_turn_on = np.concatenate([tie[tie_idx], greater], axis=0)
                else:
                    ags_idx = ags_idx[-ags_budget:]
                    ags_turn_on = ags_alpha_list[ags_idx, :2]

                turn_on = np.concatenate([turn_on, ags_turn_on], axis=0)
                turn_on = turn_on.astype(int).tolist()
                assert len(turn_on) == budget + ags_budget

                reallocation: list[list[int]] = np.concatenate(
                    [alpha_list, ags_alpha_list], axis=0
                ).tolist()
                reallocation = [
                    [
                        str(int(layer_idx)),
                        list(LORA_NAME_HASH.keys())[int(proj_hash)],
                        str(alpha),
                        str(([layer_idx, proj_hash] in turn_on)),
                    ]
                    for layer_idx, proj_hash, alpha in reallocation
                ]
                self.reallocation_history.append(
                    {
                        "epoch": pl_module.current_epoch,
                        "step": batch_idx,
                        "turn_on": reallocation,
                    }
                )

                # Turn on/off lora modules
                model = self.alpha_pl_module.model
                assert isinstance(model, (
                    LlamaLoraAgsForCausalLM,
                    LlamaLoraAgsForSequenceClassification,
                    LlamaLoraAgsForQuestionAnswering,
                    Qwen2LoraAgsForCausalLM,
                    Qwen2LoraAgsForSequenceClassification,
                ))

                model: LlamaLoraAgsForCausalLM | LlamaLoraAgsForSequenceClassification | LlamaLoraAgsForQuestionAnswering | Qwen2LoraAgsForCausalLM | Qwen2LoraAgsForSequenceClassification

                # for decoder_layer in reversed(model.model.decoder.layers):
                for decoder_layer in reversed(model.model.layers):  # Llama2
                    layer_idx = decoder_layer.layer_idx

                    lora_modules: dict[str, LoraLinear] = {
                        "q_proj": decoder_layer.self_attn.q_proj,
                        "k_proj": decoder_layer.self_attn.k_proj,
                        "v_proj": decoder_layer.self_attn.v_proj,
                        "o_proj": decoder_layer.self_attn.o_proj,
                        "up_proj": decoder_layer.mlp.up_proj,
                        "gate_proj": decoder_layer.mlp.gate_proj,
                        "down_proj": decoder_layer.mlp.down_proj,
                    }
                    for proj_name, lora in lora_modules.items():
                        if (
                            lora.active_adapter not in lora.lora_A.keys()
                            or lora.r[lora.active_adapter] == 0
                        ):
                            continue
                        proj_hash = LORA_NAME_HASH[proj_name]
                        lora.disable_adapters = [layer_idx, proj_hash] not in turn_on

                    shortcut_modules: dict[str, ShortcutBase] = {
                        "residual_1": decoder_layer.residual_1,
                        "residual_2": decoder_layer.residual_2,
                        "shortcut_sa": decoder_layer.shortcut_sa,
                        "shortcut_ffn": decoder_layer.shortcut_ffn,
                    }
                    for proj_name, shortcut in shortcut_modules.items():
                        if (
                            shortcut is None
                            or shortcut.active_projector not in shortcut.proj_A.keys()
                            or shortcut.r[shortcut.active_projector] == 0
                        ):
                            continue
                        proj_hash = LORA_NAME_HASH[proj_name]
                        shortcut.disable_projectors = [
                            layer_idx,
                            proj_hash,
                        ] not in turn_on

            else:
                # Decide which modules to keep
                alpha_list = np.concatenate(
                    [
                        [
                            (layer_idx, LORA_NAME_HASH[proj_name], v)
                            for proj_name, v in d.items()
                        ]
                        for layer_idx, d in res_val.items()
                    ],
                    axis=0,
                )
                original_lora_module_num = len(alpha_list)
                budget = math.floor(self.turn_on_percentile * original_lora_module_num)
                idx = alpha_list[:, 2].argsort()
                alpha_threshold = alpha_list[idx[-budget], 2]
                if sum(alpha_list[:, 2] == alpha_threshold) > 1:
                    # Uniformly break tie
                    greater = alpha_list[alpha_list[:, 2] > alpha_threshold, :2]
                    tie = alpha_list[alpha_list[:, 2] == alpha_threshold, :2]
                    self.rng.set_state(self.rng_state)
                    tie_idx = torch.randperm(len(tie), generator=self.rng)[
                        : (budget - len(greater))
                    ].numpy()
                    self.rng_state = self.rng.get_state()
                    turn_on = np.concatenate([tie[tie_idx], greater], axis=0)
                else:
                    idx = idx[-budget:]
                    turn_on = alpha_list[idx, :2]
                turn_on = turn_on.astype(int).tolist()
                assert len(turn_on) == budget

                reallocation: list[list[int]] = alpha_list.tolist()
                reallocation = [
                    [
                        str(int(layer_idx)),
                        list(LORA_NAME_HASH.keys())[int(proj_hash)],
                        str(alpha),
                        str(([layer_idx, proj_hash] in turn_on)),
                    ]
                    for layer_idx, proj_hash, alpha in reallocation
                ]
                self.reallocation_history.append(
                    {
                        "epoch": pl_module.current_epoch,
                        "step": batch_idx,
                        "turn_on": reallocation,
                    }
                )

                # Turn on/off lora modules
                model = self.alpha_pl_module.model
                assert isinstance(model, (
                    LlamaLoraAgsForCausalLM,
                    LlamaLoraAgsForSequenceClassification,
                    LlamaLoraAgsForQuestionAnswering,
                    Qwen2LoraAgsForCausalLM,
                    Qwen2LoraAgsForSequenceClassification,
                ))

                model: LlamaLoraAgsForCausalLM | LlamaLoraAgsForSequenceClassification | LlamaLoraAgsForQuestionAnswering | Qwen2LoraAgsForCausalLM | Qwen2LoraAgsForSequenceClassification

                # for decoder_layer in reversed(model.model.decoder.layers):
                for decoder_layer in reversed(model.model.layers):
                    # decoder_layer: OPTLoraDecoderLayer
                    layer_idx = decoder_layer.layer_idx
                    lora_modules: dict[str, LoraLinear] = {
                        "q_proj": decoder_layer.self_attn.q_proj,
                        "k_proj": decoder_layer.self_attn.k_proj,
                        "v_proj": decoder_layer.self_attn.v_proj,
                        "o_proj": decoder_layer.self_attn.o_proj,
                        "up_proj": decoder_layer.mlp.up_proj,
                        "gate_proj": decoder_layer.mlp.gate_proj,
                        "down_proj": decoder_layer.mlp.down_proj,
                    }

                    for proj_name, lora in lora_modules.items():
                        if (
                            lora.active_adapter not in lora.lora_A.keys()
                            or lora.r[lora.active_adapter] == 0
                        ):
                            continue
                        proj_hash = LORA_NAME_HASH[proj_name]
                        lora.disable_adapters = [layer_idx, proj_hash] not in turn_on

            self.save_reallocation_history()

        if torch.cuda.current_device() == 0:
            logger.warning(
                f"\n>>>>> Finish reallocation on epoch {pl_module.current_epoch}, step {batch_idx} <<<<<\n"
            )

    def _const_test(
        self,
        trainer: pl.Trainer,
        pl_module: PlWrapperBase,
        batch: Any,
        batch_idx: int,
    ) -> dict[int, dict[str, float]]:
        model = self.alpha_pl_module.model
        assert isinstance(model, (
            LlamaLoraAgsForCausalLM,
            LlamaLoraAgsForSequenceClassification,
            LlamaLoraAgsForQuestionAnswering,
            Qwen2LoraAgsForCausalLM,
            Qwen2LoraAgsForSequenceClassification,
        ))

        model: LlamaLoraAgsForCausalLM | LlamaLoraAgsForSequenceClassification | LlamaLoraAgsForQuestionAnswering | Qwen2LoraAgsForCausalLM | Qwen2LoraAgsForSequenceClassification

        # constant score of every lora module
        with torch.no_grad():
            res_val = {}
            for decoder_layer in model.model.layers:
                decoder_layer: LlamaLoraAgsDecoderLayer | Qwen2LoraAgsDecoderLayer
                layer_idx = decoder_layer.layer_idx
                lora_modules: dict[str, LoraLinear] = {
                    "q_proj": decoder_layer.self_attn.q_proj,
                    "k_proj": decoder_layer.self_attn.k_proj,
                    "v_proj": decoder_layer.self_attn.v_proj,
                    "o_proj": decoder_layer.self_attn.o_proj,
                    "up_proj": decoder_layer.mlp.up_proj,
                    "gate_proj": decoder_layer.mlp.gate_proj,
                    "down_proj": decoder_layer.mlp.down_proj,
                }
                for proj_name, lora in lora_modules.items():
                    if (
                        lora.active_adapter not in lora.lora_A.keys()
                        or lora.r[lora.active_adapter] == 0
                    ):
                        continue

                    if layer_idx not in res_val:
                        res_val[layer_idx] = {}
                    res_val[layer_idx][proj_name] = 1.0

        return res_val

    def _const_ags_test(
        self,
        trainer: pl.Trainer,
        pl_module: PlWrapperBase,
        batch: Any,
        batch_idx: int,
    ) -> dict[int, dict[str, float]]:
        model = self.alpha_pl_module.model
        assert isinstance(model, (
            LlamaLoraAgsForCausalLM,
            LlamaLoraAgsForSequenceClassification,
            LlamaLoraAgsForQuestionAnswering,
            Qwen2LoraAgsForCausalLM,
            Qwen2LoraAgsForSequenceClassification,
        ))

        model: LlamaLoraAgsForCausalLM | LlamaLoraAgsForSequenceClassification | LlamaLoraAgsForQuestionAnswering | Qwen2LoraAgsForCausalLM | Qwen2LoraAgsForSequenceClassification

        # constant score of every lora module
        with torch.no_grad():
            res_val = {}
            for decoder_layer in model.model.layers:
                decoder_layer: LlamaLoraAgsDecoderLayer | Qwen2LoraAgsDecoderLayer
                layer_idx = decoder_layer.layer_idx
                shortcut_modules: dict[str, ShortcutBase] = {
                    "residual_1": decoder_layer.residual_1,
                    "residual_2": decoder_layer.residual_2,
                    "shortcut_sa": decoder_layer.shortcut_sa,
                    "shortcut_ffn": decoder_layer.shortcut_ffn,
                }
                for proj_name, shortcut in shortcut_modules.items():
                    if (
                            shortcut is None
                            or shortcut.active_projector not in shortcut.proj_A.keys()
                            or shortcut.r[shortcut.active_projector] == 0
                    ):
                        continue

                    if layer_idx not in res_val:
                        res_val[layer_idx] = {}
                    res_val[layer_idx][proj_name] = 1.0

        return res_val

    def _grad_norm_test(
        self,
        trainer: pl.Trainer,
        pl_module: PlWrapperBase,
        batch: Any,
        batch_idx: int,
    ) -> dict[int, dict[str, float]]:
        device = pl_module.model.device

        def get_unshuffled_train_dataloader(datamodule: AgsDataModule):
            if datamodule.training_dataset is None:
                raise RuntimeError("The training dataset is not available.")
            data_collator = None
            if datamodule.dataset_info.data_collator_cls is not None:
                data_collator = datamodule.dataset_info.data_collator_cls(
                    tokenizer=datamodule.tokenizer
                )
            return DataLoader(
                datamodule.training_dataset,
                batch_size=datamodule.batch_size
                * trainer.num_devices,  # use effective batch size
                shuffle=False,
                num_workers=datamodule.num_workers,
                collate_fn=data_collator,
            )

        dataloader = get_unshuffled_train_dataloader(self.data_module)

        # GRAD NORM
        model = self.alpha_pl_module.model
        # assert (
        #     type(model) is OPTLoraForCausalLM
        #     or type(model) is OPTLoraForSequenceClassification
        #     or type(model) is OPTLoraForQuestionAnswering
        #     or type(model) is OPTLoraAgsForCausalLM
        #     or type(model) is OPTLoraAgsForSequenceClassification
        #     or type(model) is OPTLoraAgsForQuestionAnswering
        # )
        # model: OPTLoraForCausalLM | OPTLoraForSequenceClassification | OPTLoraForQuestionAnswering | OPTLoraAgsForCausalLM | OPTLoraAgsForSequenceClassification | OPTLoraAgsForQuestionAnswering

        # compute gradients
        self.alpha_pl_module.to("cuda")
        self.alpha_pl_module.zero_grad()
        msg = ""
        for i, batch in enumerate(dataloader):
            if i >= self.limit_test_batches:
                break
            print(" " * len(msg), end="\r")
            msg = f">>> Testing on batch {i + 1} / {self.limit_test_batches}"
            print(msg, end="\r")
            batch = self.data_module.transfer_batch_to_device(
                batch, torch.device("cuda"), 0
            )
            loss = self.alpha_pl_module.training_step(batch=batch, batch_idx=i)
            # print(loss.device, loss)
            loss.backward()
        print()

        # calculate score of every lora module
        grads_norm = {}
        # for decoder_layer in model.model.decoder.layers:
        for decoder_layer in model.model.layers:  # Llama2
            # decoder_layer: OPTLoraDecoderLayer
            layer_idx = decoder_layer.layer_idx
            lora_modules: dict[str, LoraLinear] = {
                "q_proj": decoder_layer.self_attn.q_proj,
                "k_proj": decoder_layer.self_attn.k_proj,
                "v_proj": decoder_layer.self_attn.v_proj,
                "o_proj": decoder_layer.self_attn.o_proj,
                "up_proj": decoder_layer.mlp.up_proj,
                "gate_proj": decoder_layer.mlp.gate_proj,
                "down_proj": decoder_layer.mlp.down_proj,
            }
            for proj_name, lora in lora_modules.items():
                if (
                    lora.active_adapter not in lora.lora_A.keys()
                    or lora.r[lora.active_adapter] == 0
                ):
                    continue

                grad_lora = (
                    lora.lora_A[lora.active_adapter].weight.grad.norm()
                    + lora.lora_B[lora.active_adapter].weight.grad.norm()
                ).item()

                if layer_idx not in grads_norm:
                    grads_norm[layer_idx] = {}
                grads_norm[layer_idx][proj_name] = grad_lora

        # reset grads
        self.alpha_pl_module.zero_grad()

        pl_module.model.to(device)
        return grads_norm

    def _grad_norm_ags_test(
        self,
        trainer: pl.Trainer,
        pl_module: PlWrapperBase,
        batch: Any,
        batch_idx: int,
    ) -> dict[int, dict[str, float]]:
        device = pl_module.model.device

        def get_unshuffled_train_dataloader(datamodule: AgsDataModule):
            if datamodule.training_dataset is None:
                raise RuntimeError("The training dataset is not available.")
            data_collator = None
            if datamodule.dataset_info.data_collator_cls is not None:
                data_collator = datamodule.dataset_info.data_collator_cls(
                    tokenizer=datamodule.tokenizer
                )
            return DataLoader(
                datamodule.training_dataset,
                batch_size=datamodule.batch_size
                * trainer.num_devices,  # use effective batch size
                shuffle=False,
                num_workers=datamodule.num_workers,
                collate_fn=data_collator,
            )

        dataloader = get_unshuffled_train_dataloader(self.data_module)

        # GRAD NORM on shortcuts
        model = self.alpha_pl_module.model
        # assert (
        #     type(model) is OPTLoraAgsForCausalLM
        #     or type(model) is OPTLoraAgsForSequenceClassification
        #     or type(model) is OPTLoraAgsForQuestionAnswering
        # )
        # model: OPTLoraAgsForCausalLM | OPTLoraAgsForSequenceClassification | OPTLoraAgsForQuestionAnswering

        # compute gradients
        self.alpha_pl_module.to("cuda")
        self.alpha_pl_module.zero_grad()
        msg = ""
        for i, batch in enumerate(dataloader):
            if i >= self.limit_test_batches:
                break
            if torch.cuda.current_device() == 0:
                print(" " * len(msg), end="\r")
            msg = f">>> Testing on batch {i + 1} / {self.limit_test_batches}"
            if torch.cuda.current_device() == 0:
                print(msg, end="\r")
            batch = self.data_module.transfer_batch_to_device(
                batch, torch.device("cuda"), 0
            )
            loss = self.alpha_pl_module.training_step(batch=batch, batch_idx=i)
            # print(loss.device, loss)
            loss.backward()
        if torch.cuda.current_device() == 0:
            print()

        # calculate score of every shortcut module
        grads_norm = {}
        # for decoder_layer in model.model.decoder.layers:
        for decoder_layer in model.model.layers:
            # decoder_layer: OPTLoraAgsDecoderLayer
            layer_idx = decoder_layer.layer_idx
            shortcut_modules: dict[str, ShortcutBase] = {
                "residual_1": decoder_layer.residual_1,
                "residual_2": decoder_layer.residual_2,
                "shortcut_sa": decoder_layer.shortcut_sa,
                "shortcut_ffn": decoder_layer.shortcut_ffn,
            }
            for proj_name, shortcut in shortcut_modules.items():
                if (
                    shortcut is None
                    or shortcut.active_projector not in shortcut.proj_A.keys()
                    or shortcut.r[shortcut.active_projector] == 0
                ):
                    continue

                grad_shortcut = (
                    shortcut.proj_A[shortcut.active_projector].weight.grad.norm()
                    + shortcut.proj_B[shortcut.active_projector].weight.grad.norm()
                ).item()

                if layer_idx not in grads_norm:
                    grads_norm[layer_idx] = {}
                grads_norm[layer_idx][proj_name] = grad_shortcut

        # reset grads
        self.alpha_pl_module.zero_grad()

        pl_module.model.to(device)
        return grads_norm

    def _snip_test(
        self,
        trainer: pl.Trainer,
        pl_module: PlWrapperBase,
        batch: Any,
        batch_idx: int,
    ) -> dict[int, dict[str, float]]:
        device = pl_module.model.device

        def get_unshuffled_train_dataloader(datamodule: AgsDataModule):
            if datamodule.training_dataset is None:
                raise RuntimeError("The training dataset is not available.")
            data_collator = None
            if datamodule.dataset_info.data_collator_cls is not None:
                data_collator = datamodule.dataset_info.data_collator_cls(
                    tokenizer=datamodule.tokenizer
                )
            return DataLoader(
                datamodule.training_dataset,
                batch_size=datamodule.batch_size
                * trainer.num_devices,  # use effective batch size
                shuffle=False,
                num_workers=datamodule.num_workers,
                collate_fn=data_collator,
            )

        dataloader = get_unshuffled_train_dataloader(self.data_module)

        # SNIP
        model = self.alpha_pl_module.model
        assert isinstance(model, (
            LlamaLoraAgsForCausalLM,
            LlamaLoraAgsForSequenceClassification,
            LlamaLoraAgsForQuestionAnswering,
            Qwen2LoraAgsForCausalLM,
            Qwen2LoraAgsForSequenceClassification,
        ))

        model: LlamaLoraAgsForCausalLM | LlamaLoraAgsForSequenceClassification | LlamaLoraAgsForQuestionAnswering | Qwen2LoraAgsForCausalLM | Qwen2LoraAgsForSequenceClassification

        @torch.no_grad()
        def get_require_grad(net: nn.Module):
            require_grad = {}
            for name, param in net.named_parameters():
                require_grad[name] = param.requires_grad
            return require_grad

        @torch.no_grad()
        def set_require_grad(net: nn.Module, require_grad: dict[str, bool]):
            for name, param in net.named_parameters():
                if name in require_grad:
                    param.requires_grad_(require_grad[name])
                else:
                    param.requires_grad_(False)

        # keep requires_grad of all params
        original_require_grad = get_require_grad(model)

        for name, param in model.named_parameters():
            param.requires_grad = False

        # add weight masks
        def lora_forward(self, x):
            if self.active_adapter not in self.lora_A.keys():
                res = F.linear(
                    x,
                    self.weight if not self.fan_in_fan_out else self.weight.T,
                    self.bias,
                )
            elif self.disable_adapters:
                if self.r[self.active_adapter] > 0 and self.merged:
                    self.unmerge()
                res = F.linear(
                    x,
                    self.weight if not self.fan_in_fan_out else self.weight.T,
                    self.bias,
                )
            else:
                # weight mask activated
                self.unmerge()
                res = F.linear(
                    x,
                    self.weight if not self.fan_in_fan_out else self.weight.T,
                    self.bias,
                )
                res = (
                    res
                    + (
                        F.linear(
                            F.linear(
                                self.lora_dropout[self.active_adapter](x),
                                self.lora_A[self.active_adapter].weight
                                * self.weight_mask_A,
                            ),
                            self.lora_B[self.active_adapter].weight
                            * self.weight_mask_B,
                        )
                    )
                    * self.scaling[self.active_adapter]
                )
            # res = res.to(input_dtype)
            return res

        # keep original forward of all lora
        original_forward = {}

        for decoder_layer in reversed(model.model.layers):
            decoder_layer: LlamaLoraAgsDecoderLayer | Qwen2LoraAgsDecoderLayer
            lora_modules: dict[str, LoraLinear] = {
                "q_proj": decoder_layer.self_attn.q_proj,
                "k_proj": decoder_layer.self_attn.k_proj,
                "v_proj": decoder_layer.self_attn.v_proj,
                "o_proj": decoder_layer.self_attn.o_proj,
                "up_proj": decoder_layer.mlp.up_proj,
                "gate_proj": decoder_layer.mlp.gate_proj,
                "down_proj": decoder_layer.mlp.down_proj,
            }
            original_forward[decoder_layer.layer_idx] = {}
            for proj_name, lora in lora_modules.items():
                if (
                    lora.active_adapter not in lora.lora_A.keys()
                    or lora.r[lora.active_adapter] == 0
                ):
                    continue

                lora_A: nn.Linear = lora.lora_A[lora.active_adapter]
                lora_A.weight.requires_grad = False
                lora.weight_mask_A = nn.Parameter(torch.ones_like(lora_A.weight))
                lora_B: nn.Linear = lora.lora_B[lora.active_adapter]
                lora_B.weight.requires_grad = False
                lora.weight_mask_B = nn.Parameter(torch.ones_like(lora_B.weight))

                original_forward[decoder_layer.layer_idx][proj_name] = lora.forward
                lora.forward = types.MethodType(lora_forward, lora)

        # compute gradients
        self.alpha_pl_module.to("cuda")
        self.alpha_pl_module.zero_grad()
        msg = ""
        for i, batch in enumerate(dataloader):
            if i >= self.limit_test_batches:
                break
            print(" " * len(msg), end="\r")
            msg = f">>> Testing on batch {i + 1} / {self.limit_test_batches}"
            print(msg, end="\r")
            batch = self.data_module.transfer_batch_to_device(
                batch, torch.device("cuda"), 0
            )
            loss = self.alpha_pl_module.training_step(batch=batch, batch_idx=i)
            # print(loss.device, loss)
            loss.backward()
        print()

        # calculate score of every lora module
        grads_abs = {}
        for decoder_layer in model.model.layers:
            decoder_layer: LlamaLoraAgsDecoderLayer | Qwen2LoraAgsDecoderLayer
            layer_idx = decoder_layer.layer_idx
            lora_modules: dict[str, LoraLinear] = {
                "q_proj": decoder_layer.self_attn.q_proj,
                "k_proj": decoder_layer.self_attn.k_proj,
                "v_proj": decoder_layer.self_attn.v_proj,
                "o_proj": decoder_layer.self_attn.o_proj,
                "up_proj": decoder_layer.mlp.up_proj,
                "gate_proj": decoder_layer.mlp.gate_proj,
                "down_proj": decoder_layer.mlp.down_proj,
            }
            for proj_name, lora in lora_modules.items():
                if (
                    lora.active_adapter not in lora.lora_A.keys()
                    or lora.r[lora.active_adapter] == 0
                ):
                    continue

                grad_lora = (
                    torch.sum(torch.abs(lora.weight_mask_A.grad))
                    + torch.sum(torch.abs(lora.weight_mask_B.grad))
                ).item()

                if layer_idx not in grads_abs:
                    grads_abs[layer_idx] = {}
                grads_abs[layer_idx][proj_name] = grad_lora

        # recover requires_grad and forward, reset grads
        set_require_grad(model, original_require_grad)
        for decoder_layer in reversed(model.model.layers):
            decoder_layer: LlamaLoraAgsDecoderLayer | Qwen2LoraAgsDecoderLayer
            lora_modules: dict[str, LoraLinear] = {
                "q_proj": decoder_layer.self_attn.q_proj,
                "k_proj": decoder_layer.self_attn.k_proj,
                "v_proj": decoder_layer.self_attn.v_proj,
                "o_proj": decoder_layer.self_attn.o_proj,
                "up_proj": decoder_layer.mlp.up_proj,
                "gate_proj": decoder_layer.mlp.gate_proj,
                "down_proj": decoder_layer.mlp.down_proj,
            }
            for proj_name, lora in lora_modules.items():
                if (
                    lora.active_adapter not in lora.lora_A.keys()
                    or lora.disable_adapters
                    or lora.r[lora.active_adapter] == 0
                ):
                    continue

                # del lora.weight_mask_A
                # del lora.weight_mask_B
                lora.forward = original_forward[decoder_layer.layer_idx][proj_name]

        self.alpha_pl_module.zero_grad()

        pl_module.model.to(device)
        return grads_abs

    def _synflow_test(
        self,
        trainer: pl.Trainer,
        pl_module: PlWrapperBase,
        batch: Any,
        batch_idx: int,
    ) -> dict[int, dict[str, float]]:
        device = pl_module.model.device

        dataloader = self._get_train_dataloader()

        # SYNFLOW
        model = self.alpha_pl_module.model
        assert isinstance(model, (
            LlamaLoraAgsForCausalLM,
            LlamaLoraAgsForSequenceClassification,
            LlamaLoraAgsForQuestionAnswering,
            Qwen2LoraAgsForCausalLM,
            Qwen2LoraAgsForSequenceClassification,
        ))

        model: LlamaLoraAgsForCausalLM | LlamaLoraAgsForSequenceClassification | LlamaLoraAgsForQuestionAnswering | Qwen2LoraAgsForCausalLM | Qwen2LoraAgsForSequenceClassification

        # convert params to their abs, keep sign for converting it back
        @torch.no_grad()
        def linearize(net):
            signs = {}
            for name, param in net.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        # convert to orig values
        @torch.no_grad()
        def nonlinearize(net, signs):
            for name, param in net.state_dict().items():
                param.mul_(signs[name])

        # keep signs of all params
        signs = linearize(model)

        # compute gradients
        self.alpha_pl_module.to("cuda")
        self.alpha_pl_module.zero_grad()
        example_input = next(iter(dataloader))
        input_dim = list(example_input["input_ids"].shape) + [
            model.model.embed_tokens.weight.shape[1]
        ]
        inputs = torch.ones(input_dim).float().to("cuda")
        attention_mask = example_input["attention_mask"]
        token_type_ids = example_input.get("token_type_ids", None)
        labels = example_input["labels"]
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if isinstance(attention_mask, list):
            attention_mask = torch.stack(attention_mask)
        if isinstance(token_type_ids, list):
            token_type_ids = torch.stack(token_type_ids)
        if isinstance(labels, list):
            labels = torch.stack(labels)
        if token_type_ids is not None:
            output = model.forward(
                inputs_embeds=inputs.to("cuda"),
                attention_mask=attention_mask.to("cuda"),
                token_type_ids=token_type_ids.to("cuda"),
                labels=labels.to("cuda"),
            )
        else:
            output = model.forward(
                inputs_embeds=inputs.to("cuda"),
                attention_mask=attention_mask.to("cuda"),
                labels=labels.to("cuda"),
            )
        torch.sum(output["loss"]).backward()

        # calculate score of every lora module
        grads_abs = {}
        for decoder_layer in model.model.layers:
            decoder_layer: LlamaLoraAgsDecoderLayer | Qwen2LoraAgsDecoderLayer
            layer_idx = decoder_layer.layer_idx
            lora_modules: dict[str, LoraLinear] = {
                "q_proj": decoder_layer.self_attn.q_proj,
                "k_proj": decoder_layer.self_attn.k_proj,
                "v_proj": decoder_layer.self_attn.v_proj,
                "o_proj": decoder_layer.self_attn.o_proj,
                "up_proj": decoder_layer.mlp.up_proj,
                "gate_proj": decoder_layer.mlp.gate_proj,
                "down_proj": decoder_layer.mlp.down_proj,
            }
            for proj_name, lora in lora_modules.items():
                if (
                    lora.active_adapter not in lora.lora_A.keys()
                    or lora.r[lora.active_adapter] == 0
                ):
                    continue

                grad_lora = (
                    torch.sum(
                        torch.abs(
                            lora.lora_A[lora.active_adapter].weight
                            * lora.lora_A[lora.active_adapter].weight.grad
                        )
                    )
                    + torch.sum(
                        torch.abs(
                            lora.lora_B[lora.active_adapter].weight
                            * lora.lora_B[lora.active_adapter].weight.grad
                        )
                    )
                ).item()

                if layer_idx not in grads_abs:
                    grads_abs[layer_idx] = {}
                grads_abs[layer_idx][proj_name] = grad_lora

        # apply signs of all params
        nonlinearize(model, signs)

        # reset grads
        self.alpha_pl_module.zero_grad()

        pl_module.model.to(device)
        return grads_abs

    def save_reallocation_history(self):
        if torch.cuda.current_device() != 0:
            return
        logger.warning(f"Saving history on GPU {torch.cuda.current_device()}")
        # Calculate frequency each lora module has been turned on
        turned_on_freq: dict[str, int | dict[str, int]] = {
            "total_reallocation_number": len(self.reallocation_history)
        }
        # format: {dyrealloc_{i}: {epoch: epoch, step: step, turn_on: turn_on[]}
        history: dict[str, int | dict[str, int | list]] = {}
        for i, reallocation in enumerate(self.reallocation_history):
            history[f"dyrealloc_{i}"] = reallocation
            for lora_module in reallocation["turn_on"]:
                layer_idx, proj_name, _, turned_on = lora_module
                if turned_on == "True":
                    if f"layer_{layer_idx}" not in turned_on_freq:
                        turned_on_freq[f"layer_{layer_idx}"] = {}
                    if proj_name not in turned_on_freq[f"layer_{layer_idx}"]:
                        turned_on_freq[f"layer_{layer_idx}"][proj_name] = 1
                    else:
                        turned_on_freq[f"layer_{layer_idx}"][proj_name] += 1
                else:
                    if f"layer_{layer_idx}" not in turned_on_freq:
                        turned_on_freq[f"layer_{layer_idx}"] = {}
                    if proj_name not in turned_on_freq[f"layer_{layer_idx}"]:
                        turned_on_freq[f"layer_{layer_idx}"][proj_name] = 0

        with open(self.history_save_path, "w+") as fout:
            toml.dump(history, fout)
        with open(self.frequency_save_path, "w+") as fout:
            toml.dump(turned_on_freq, fout)
        logger.warning("Reallocation history and frequency saved as toml")
