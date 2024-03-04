import logging
import math
import time
from typing import Any, Optional

import numpy as np
import toml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset.pl_dataset_module import AgsDataModule
from lora.lora_modules import LoraLinear
from models.modeling_opt_lora import (
    OPTLoraForCausalLM,
    OPTLoraForSequenceClassification,
    OPTLoraForQuestionAnswering,
    OPTLoraDecoderLayer,
)
from pl_model_wrapper.base import PlWrapperBase

logger = logging.getLogger(__name__)

LORA_NAME_HASH = {
    "q_proj": 0,
    "k_proj": 1,
    "v_proj": 2,
    "out_proj": 3,
    "fc1": 4,
    "fc2": 5,
}
ALPHA_UB = 8


class DynamicLoraReallocationCallback(pl.Callback):
    def __init__(
        self,
        N: int | float,
        data_module: AgsDataModule,
        alpha_trainer_args: pl.Trainer,
        task: str,
        metric_reduction_tolerance: float,
        turn_on_percentile: float = 0.25,
        limit_test_batches: Optional[int | float] = None,
        save_path: str = None,
    ):
        """
        :param N: every N steps conduct alpha testing and reallocate lora ranks
        :param data_module: for loading batches for alpha testing
        :param alpha_trainer_args: for building the pl trainer to conduct alpha testing
        :param task: task type for determining threshold comparison
        :param metric_reduction_tolerance: for computing the threshold for alpha testing
        :param turn_on_percentile: percentage of lora modules to be activated by the reallocation
        :param limit_test_batches: number of batches used in alpha testing
        :param save_path: file path for saving reallocation history
        """
        super().__init__()

        self.data_module = data_module
        self.alpha_trainer_args = alpha_trainer_args
        self.alpha_trainer = None

        assert task in ["classification", "summarization", "causal_language_modeling"]
        self.task = task

        self.N = N
        self.limit_test_batches = limit_test_batches

        self.metric_reduction_tolerance = metric_reduction_tolerance
        self.turn_on_percentile = turn_on_percentile

        self.reallocation_history: list[dict[str, int | list]] = []
        t = time.strftime("%H-%M")
        self.history_save_path = f"{save_path}/reallocation_history_{t}.toml"
        self.frequency_save_path = f"{save_path}/reallocation_frequency_{t}.toml"

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if type(self.N) is int:
            # Num of batches between two reallocation
            self.N: int
        elif type(self.N) is float:
            # Percentage of training steps per epoch between two reallocation
            assert 0.0 < self.N <= 1.0, "N should be 0.0 < N <= 1.0"
            self.N: int = round(len(self._get_train_dataloader()) * self.N)
        else:
            raise TypeError("N should be int or float between 0.0 and 1.0")

        if self.limit_test_batches is None:
            # Default: single-shot per epoch on the validation set
            self.limit_test_batches: int = round(len(self._get_val_dataloader()) / (self.N//2))
        elif type(self.limit_test_batches) is int:
            # Number of alpha test batches
            self.limit_test_batches: int
        elif type(self.limit_test_batches) is float:
            # Percentage of validation set
            self.limit_test_batches: int = round(len(self._get_val_dataloader()) * self.limit_test_batches) * 2
        else:
            raise TypeError(
                "limit_test_batches should be None (assumed single-shot) or int or float between 0.0 and 1.0"
            )

        self.alpha_trainer = pl.Trainer(
            **self.alpha_trainer_args,
            limit_test_batches=self.limit_test_batches,
        )

    def get_alpha_testing_dataloader(self):
        return self._get_mixed_dataloader()

    def _get_train_dataloader(self) -> DataLoader:
        return self.data_module.train_dataloader()

    def _get_val_dataloader(self) -> DataLoader:
        return self.data_module.val_dataloader()

    def _get_mixed_dataloader(self) -> DataLoader:
        # 1:1 mixed training set & validation set
        assert type(self.data_module) is AgsDataModule
        self.data_module: AgsDataModule
        if self.data_module.training_dataset is None:
            raise RuntimeError("The training dataset is not available.")
        if self.data_module.validation_dataset is None:
            raise RuntimeError("The validation dataset is not available.")

        train_idx = torch.randperm(len(self.data_module.training_dataset))
        validation_idx = torch.randperm(len(self.data_module.val_dataloader()))
        if len(train_idx) >= len(validation_idx):
            train_idx = train_idx[:len(validation_idx)]
            interleave_idx = torch.stack([train_idx, validation_idx], dim=1).view(-1)
        else:
            interleave_idx = torch.cat(
                [
                    torch.stack([train_idx, validation_idx[:len(train_idx)]], dim=1).view(-1),
                    validation_idx[len(train_idx):],
                ]
            )
        interleave_idx = interleave_idx.tolist()

        data_collator = None
        if self.data_module.dataset_info.data_collator_cls is not None:
            data_collator = self.data_module.dataset_info.data_collator_cls(
                tokenizer=self.data_module.tokenizer
            )

        return DataLoader(
            torch.utils.data.Subset(
                torch.utils.data.ConcatDataset(
                    [self.data_module.training_dataset, self.data_module.validation_dataset]
                ),
                indices=interleave_idx,
            ),
            batch_size=self.data_module.batch_size,
            shuffle=False,
            num_workers=self.data_module.num_workers,
            collate_fn=data_collator,
        )

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: PlWrapperBase,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.N > 0:
            return

        logger.warning(f"\n>>>>> Running reallocation on epoch {pl_module.current_epoch}, step {batch_idx} <<<<<\n")

        device = pl_module.model.device

        dataloader = self.get_alpha_testing_dataloader()

        original_val_metrics = self.alpha_trainer.test(
            pl_module, dataloaders=dataloader, verbose=False
        )[0]

        def get_metric_name():
            match self.task:
                case "classification":
                    return "test_acc_epoch"
                case "summarization":
                    return "test_rouge_epoch"
                case "causal_language_modeling":
                    return "test_perplexity_epoch"
                case _:
                    raise ValueError(f"Unsupported task: {self.task}")

        def get_metric_threshold():
            original_metric = original_val_metrics[get_metric_name()]
            match self.task:
                case "classification":
                    # Accuracy
                    return (
                        original_metric
                        - original_metric * self.metric_reduction_tolerance
                    )
                case "summarization":
                    # Rouge score
                    return (
                        original_metric
                        - original_metric * self.metric_reduction_tolerance
                    )
                case "causal_language_modeling":
                    # Perplexity
                    return (
                        original_metric
                        + original_metric * self.metric_reduction_tolerance
                    )
                case _:
                    raise ValueError(f"Unsupported task: {self.task}")

        def check_exceed_threshold(val_metrics_dict):
            val_metric = val_metrics_dict[get_metric_name()]
            threshold = get_metric_threshold()
            match self.task:
                case "classification":
                    return val_metric < threshold
                case "summarization":
                    return val_metric < threshold
                case "causal_language_modeling":
                    return val_metric > threshold
                case _:
                    raise ValueError(f"Unsupported task: {self.task}")

        # Result format: {layer_idx: {proj: alpha}}
        res_val: dict[int, dict[str, float]] = {}

        with torch.no_grad():
            model = pl_module.model
            assert (
                type(model) is OPTLoraForCausalLM
                or type(model) is OPTLoraForSequenceClassification
                or type(model) is OPTLoraForQuestionAnswering
            )
            model: OPTLoraForCausalLM | OPTLoraForSequenceClassification | OPTLoraForQuestionAnswering

            # Get alpha importance for each module
            for decoder_layer in reversed(model.model.decoder.layers):
                decoder_layer: OPTLoraDecoderLayer
                layer_id = decoder_layer.layer_id
                lora_modules: dict[str, LoraLinear] = {
                    "q_proj": decoder_layer.self_attn.q_proj,
                    "k_proj": decoder_layer.self_attn.k_proj,
                    "v_proj": decoder_layer.self_attn.v_proj,
                    "out_proj": decoder_layer.self_attn.out_proj,
                    "fc1": decoder_layer.fc1,
                    "fc2": decoder_layer.fc2,
                }

                for proj_name, lora in lora_modules.items():
                    if (
                        lora.active_adapter not in lora.lora_A.keys()
                        or lora.r[lora.active_adapter] == 0
                    ):
                        continue

                    logger.warning(
                        f"Alpha testing layer {layer_id} projection {proj_name}",
                        # end="\r",
                    )
                    # msg_len = len(f">>> Alpha testing layer {layer_id} projection {proj_name}")

                    lb, rb = (0, ALPHA_UB)
                    while lb < rb:
                        alpha = (lb + rb) // 2
                        lora.importance_alpha = alpha / ALPHA_UB
                        val_metrics = self.alpha_trainer.test(
                            pl_module, dataloaders=dataloader, verbose=False
                        )[0]
                        if check_exceed_threshold(val_metrics):
                            lb = alpha + 1
                        else:
                            rb = alpha
                    alpha_res = rb

                    lora.importance_alpha = 1.0
                    if layer_id not in res_val:
                        res_val[layer_id] = {}
                    res_val[layer_id][proj_name] = alpha_res

                    logger.warning(f">>> Layer {layer_id} Projection {proj_name} Alpha {alpha_res}")

            # Decide which modules to keep
            alpha_list = np.concatenate(
                [
                    [
                        (layer_id, LORA_NAME_HASH[proj_name], v)
                        for proj_name, v in d.items()
                    ]
                    for layer_id, d in res_val.items()
                ],
                axis=0,
            )
            alpha_list = alpha_list[alpha_list[:, 0].argsort(kind="stable")]
            original_lora_module_num = len(alpha_list)
            budget = math.floor(self.turn_on_percentile * original_lora_module_num)
            idx = alpha_list[:, 2].argsort(kind="stable")
            alpha_threshold = alpha_list[idx[-budget], 2]
            if sum(alpha_list[:, 2] == alpha_threshold) > 1:
                # Uniformly break tie
                greater = alpha_list[alpha_list[:, 2] > alpha_threshold, :2]
                tie = alpha_list[alpha_list[:, 2] == alpha_threshold, :2]
                tie_idx = np.random.choice(len(tie), size=budget-len(greater), replace=False)
                turn_on = np.concatenate([tie[tie_idx], greater], axis=0)
            else:
                idx = idx[-budget:]
                turn_on = alpha_list[idx, :2].tolist()
            assert len(turn_on) == budget

            self.reallocation_history.append(
                {
                    "epoch": pl_module.current_epoch,
                    "step": batch_idx,
                    "turn_on": turn_on
                }
            )

            # Turn on/off lora modules
            for decoder_layer in reversed(model.model.decoder.layers):
                decoder_layer: OPTLoraDecoderLayer
                layer_id = decoder_layer.layer_id
                lora_modules: dict[str, LoraLinear] = {
                    "q_proj": decoder_layer.self_attn.q_proj,
                    "k_proj": decoder_layer.self_attn.k_proj,
                    "v_proj": decoder_layer.self_attn.v_proj,
                    "out_proj": decoder_layer.self_attn.out_proj,
                    "fc1": decoder_layer.fc1,
                    "fc2": decoder_layer.fc2,
                }

                for proj_name, lora in lora_modules.items():
                    if (
                        lora.active_adapter not in lora.lora_A.keys()
                        or lora.r[lora.active_adapter] == 0
                    ):
                        continue
                    proj_hash = LORA_NAME_HASH[proj_name]
                    lora.disable_adapters = [layer_id, proj_hash] in turn_on

            self.save_reallocation_history()
        pl_module.model.to(device)

        logger.warning(f"\n>>>>> Finish reallocation on epoch {pl_module.current_epoch}, step {batch_idx} <<<<<\n")

    def save_reallocation_history(self):
        # Calculate frequency each lora module has been turned on
        turned_on_freq: dict[str, int | dict[str, int]] = {
            "total_reallocation_number": len(self.reallocation_history)
        }
        # format: {dyrealloc_{i}: {epoch: epoch, step: step, turn_on: turn_on[]}
        history: dict[str, dict[str, int | list]] = {}
        for i, reallocation in enumerate(self.reallocation_history):
            history[f"dyrealloc_{i}"] = reallocation
            for lora_module in reallocation["turn_on"]:
                layer_id, proj_hash = lora_module
                proj_name = list(LORA_NAME_HASH.keys())[proj_hash]
                if f"layer_{layer_id}" not in turned_on_freq:
                    turned_on_freq[f"layer_{layer_id}"] = {}
                if proj_name not in turned_on_freq[f"layer_{layer_id}"]:
                    turned_on_freq[f"layer_{layer_id}"][proj_name] = 0
                else:
                    turned_on_freq[f"layer_{layer_id}"][proj_name] += 1

        with open(self.history_save_path, "w+") as fout:
            toml.dump(history, fout)
        with open(self.frequency_save_path, "w+") as fout:
            toml.dump(turned_on_freq, fout)
        logger.warning("Reallocation history and frequency saved as toml")
