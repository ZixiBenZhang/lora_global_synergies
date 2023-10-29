import torch
import pytorch_lightning as pl
from torch import Tensor


class ValidationMetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.val_history_metrics: dict[str, list[Tensor]] = {}

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        metrics = trainer.callback_metrics
        for k in metrics:
            if "val" not in k:
                continue
            if k in self.val_history_metrics:
                self.val_history_metrics[k].append(metrics[k])
            else:
                self.val_history_metrics[k] = [metrics[k]]
