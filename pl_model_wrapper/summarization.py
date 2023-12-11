import torch

from metrics import MyRouge
from datasets import DatasetInfo
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import PlWrapperBase


# Seq2seq, metric as ROUGE
class NLPSummarizationModelWrapper(PlWrapperBase):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: str = None,
        learning_rate=1e-4,  # for building optimizer
        weight_decay=0.0,  # for building optimizer
        lr_scheduler: str = "none",  # for building lr scheduler
        eta_min=0.0,  # for building lr scheduler
        epochs=200,  # for building lr_scheduler
        dataset_info: DatasetInfo = None,  # for getting num_classes for calculating Accuracy
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
        self.tokenizer = tokenizer

        self.rouge_train = MyRouge(
            use_stemmer=False,  # use Porter Stemmer to remove morphological affixes, leaving only the word stem.
            tokenizer=tokenizer,
            rouge_keys=("rouge1", "rouge2", "rougeL"),
        )
        self.rouge_val = MyRouge(
            use_stemmer=False,  # use Porter Stemmer to remove morphological affixes, leaving only the word stem.
            tokenizer=tokenizer,
            rouge_keys=("rouge1", "rouge2", "rougeL"),
        )
        self.rouge_test = MyRouge(
            use_stemmer=False,  # use Porter Stemmer to remove morphological affixes, leaving only the word stem.
            tokenizer=tokenizer,
            rouge_keys=("rouge1", "rouge2", "rougeL"),
        )

    def forward(
        self,
        input_ids,  # ids: tokenizer(token) -> an id in its word list
        attention_mask=None,  # to prevent applying attention to padding characters
        # Todo: determine how to pass decoder_intput_ids & decoder_attention_mask OR sth else to seq2seq model
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        if isinstance(input_ids, list):
            input_ids = torch.stack(input_ids)
        if isinstance(attention_mask, list):
            attention_mask = torch.stack(attention_mask)
        if isinstance(decoder_input_ids, list):
            decoder_input_ids = torch.stack(decoder_input_ids)
        if isinstance(decoder_attention_mask, list):
            decoder_attention_mask = torch.stack(decoder_attention_mask)
        if isinstance(labels, list):
            labels = torch.stack(labels)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        return outputs

    def training_step(self, batch, batch_idx):
        x = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        labels = batch["labels"]

        outputs = self.forward(
            input_ids=x,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        loss = outputs["loss"]
        logits = outputs["logits"]
        _pred_logits, pred_ids = torch.max(logits, dim=1)
        y = labels[0] if len(labels) == 1 else labels.squeeze()

        self.rouge_train(pred_ids, y)

        self.log("train_loss_step", loss, prog_bar=True)
        self.log("train_rouge_step", self.rouge_train, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        labels = batch["labels"]

        outputs = self.forward(
            input_ids=x,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        loss = outputs["loss"]
        logits = outputs["logits"]
        _pred_logits, pred_ids = torch.max(logits, dim=1)
        y = labels[0] if len(labels) == 1 else labels.squeeze()

        self.rouge_val(pred_ids, y)
        self.loss_val(loss)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss_epoch", self.loss_val, prog_bar=True)
        self.log("val_rouge_epoch", self.rouge_val, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        labels = batch["labels"]

        outputs = self.forward(
            input_ids=x,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        loss = outputs["loss"]
        logits = outputs["logits"]
        _pred_logits, pred_ids = torch.max(logits, dim=1)
        y = labels[0] if len(labels) == 1 else labels.squeeze()

        self.rouge_test(pred_ids, y)
        self.loss_test(loss)

        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_loss_epoch", self.loss_test, prog_bar=True)
        self.log("test_rouge_epoch", self.rouge_test, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        labels = batch["labels"]

        outputs = self.forward(
            input_ids=x,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        logits = outputs["logits"]
        _pred_logits, pred_ids = torch.max(logits, dim=1)

        return {"batch_idx": batch_idx, "outputs": outputs, "pred_ids": pred_ids}
