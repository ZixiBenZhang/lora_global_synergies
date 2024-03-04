import torch
import torch.nn as nn
from datasets import DatasetInfo
from torchmetrics import MeanMetric
from transformers import PreTrainedModel

from dataset import AgsDatasetInfo
from .base import PlWrapperBase


class NLPLanguageModelingModelWrapper(PlWrapperBase):
    def __init__(
        self,
        model: PreTrainedModel,
        optimizer: str = None,
        learning_rate=1e-4,  # for building optimizer
        weight_decay=0.0,  # for building optimizer
        lr_scheduler: str = "none",  # for building lr scheduler
        eta_min=0.0,  # for building lr scheduler
        epochs=200,  # for building lr_scheduler
        dataset_info: AgsDatasetInfo = None,  # for getting num_classes for calculating Accuracy
    ):
        super().__init__(
            model,
            optimizer,
            learning_rate,
            weight_decay,
            lr_scheduler,
            eta_min,
            epochs,
            dataset_info,
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch.
        """
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs["loss"]

        perplexity = torch.exp(loss)

        self.log("train_loss_step", loss, prog_bar=True)
        self.log(
            "train_perplexity_step",
            perplexity,
            prog_bar=True,
        )
        # todo: check whether reallocation updates are effective
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)
        print("DEBUG <<<<<<<<<<<<<<<<")

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs["loss"]

        self.loss_val(loss)

        return loss

    def on_validation_epoch_end(self):
        loss_epoch = self.loss_val.compute()
        perplexity_epoch = torch.exp(loss_epoch)
        self.log("val_loss_epoch", loss_epoch, prog_bar=True)
        self.log("val_perplexity_epoch", perplexity_epoch, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        self.loss_test(loss)

        return loss

    def on_test_epoch_end(self):
        loss_epoch = self.loss_test.compute()
        perplexity_epoch = torch.exp(loss_epoch)
        self.log("test_loss_epoch", loss_epoch, prog_bar=True)
        self.log("test_perplexity_epoch", perplexity_epoch, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.forward(input_ids, attention_mask, labels=None)
        outputs["batch_idx"] = batch_idx
        return outputs
