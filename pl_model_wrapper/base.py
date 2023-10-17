import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, MeanMetric
import pytorch_lightning as pl
from datasets import DatasetInfo
from transformers import PreTrainedModel


class PlWrapperBase(pl.LightningModule):
    def __init__(
            self,
            model: PreTrainedModel,
            optimizer: str = None,
            learning_rate=5e-4,  # for building optimizer
            weight_decay=0.0,  # for building optimizer
            epochs=1,  # for building lr_scheduler
            dataset_info: DatasetInfo = None,  # for getting num_classes for calculating Accuracy
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.loss_fn = nn.CrossEntropyLoss()

        assert 'label' in dataset_info.features.keys()
        self.num_classes = dataset_info.features['label'].num_classes

        # train step metrics are logged in every step
        self.acc_train = Accuracy("multiclass", num_classes=self.num_classes)

        # validation metrics are logged when epoch ends
        self.acc_val = Accuracy("multiclass", num_classes=self.num_classes)
        self.loss_val = MeanMetric()

        # test metrics are logged when epoch ends
        self.acc_test = Accuracy("multiclass", num_classes=self.num_classes)
        self.loss_test = MeanMetric()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_end(self):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def on_test_end(self):
        pass

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer|dict[str, torch.optim.Optimizer|torch.optim.lr_scheduler.LRScheduler]:
        pass
