import logging

import numpy as np
import torch
from torchmetrics import Accuracy, Metric
from transformers import PreTrainedModel, PreTrainedTokenizer

from dataset import AgsDatasetInfo
from . import NLPLanguageModelingModelWrapper


logger = logging.getLogger(__name__)


class NLPMMLULanguageModelingModelWrapper(NLPLanguageModelingModelWrapper):
    IGNORE_INDEX = -100
    SUBJECTS = {
        "abstract_algebra": "stem",
        "anatomy": "stem",
        "astronomy": "stem",
        "business_ethics": "other",
        "clinical_knowledge": "other",
        "college_biology": "stem",
        "college_chemistry": "stem",
        "college_computer_science": "stem",
        "college_mathematics": "stem",
        "college_medicine": "other",
        "college_physics": "stem",
        "computer_security": "stem",
        "conceptual_physics": "stem",
        "econometrics": "social_sciences",
        "electrical_engineering": "stem",
        "elementary_mathematics": "stem",
        "formal_logic": "humanities",
        "global_facts": "other",
        "high_school_biology": "stem",
        "high_school_chemistry": "stem",
        "high_school_computer_science": "stem",
        "high_school_european_history": "humanities",
        "high_school_geography": "social_sciences",
        "high_school_government_and_politics": "social_sciences",
        "high_school_macroeconomics": "social_sciences",
        "high_school_mathematics": "stem",
        "high_school_microeconomics": "social_sciences",
        "high_school_physics": "stem",
        "high_school_psychology": "social_sciences",
        "high_school_statistics": "stem",
        "high_school_us_history": "humanities",
        "high_school_world_history": "humanities",
        "human_aging": "other",
        "human_sexuality": "social_sciences",
        "international_law": "humanities",
        "jurisprudence": "humanities",
        "logical_fallacies": "humanities",
        "machine_learning": "stem",
        "management": "other",
        "marketing": "other",
        "medical_genetics": "other",
        "miscellaneous": "other",
        "moral_disputes": "humanities",
        "moral_scenarios": "humanities",
        "nutrition": "other",
        "philosophy": "humanities",
        "prehistory": "humanities",
        "professional_accounting": "other",
        "professional_law": "humanities",
        "professional_medicine": "other",
        "professional_psychology": "social_sciences",
        "public_relations": "social_sciences",
        "security_studies": "social_sciences",
        "sociology": "social_sciences",
        "us_foreign_policy": "social_sciences",
        "virology": "other",
        "world_religions": "humanities",
    }

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
        mmlu_tokenizer: PreTrainedTokenizer = None,  # for getting abcd_idx
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
        self.subjects = set(self.SUBJECTS.values())
        self.acc_val = Accuracy("multiclass", num_classes=4)
        self.acc_test = Accuracy("multiclass", num_classes=4)

        self.acc_hum_val = Accuracy("multiclass", num_classes=4)
        self.acc_ss_val = Accuracy("multiclass", num_classes=4)
        self.acc_sci_val = Accuracy("multiclass", num_classes=4)
        self.acc_other_val = Accuracy("multiclass", num_classes=4)
        self.acc_sub_val: dict[str, Metric] = {
            "humanities": self.acc_hum_val,
            "social_sciences": self.acc_ss_val,
            "stem": self.acc_sci_val,
            "other": self.acc_other_val,
        }

        self.acc_hum_test = Accuracy("multiclass", num_classes=4)
        self.acc_ss_test = Accuracy("multiclass", num_classes=4)
        self.acc_sci_test = Accuracy("multiclass", num_classes=4)
        self.acc_other_test = Accuracy("multiclass", num_classes=4)
        self.acc_sub_test: dict[str, Metric] = {
            "humanities": self.acc_hum_test,
            "social_sciences": self.acc_ss_test,
            "stem": self.acc_sci_test,
            "other": self.acc_other_test,
        }

        self.abcd_idx = (
            [
                tokenizer.encode(" A")[1],
                tokenizer.encode(" B")[1],
                tokenizer.encode(" C")[1],
                tokenizer.encode(" D")[1],
            ]
            if "Llama" in type(tokenizer).__name__
            else [
                tokenizer.encode(" A")[0],
                tokenizer.encode(" B")[0],
                tokenizer.encode(" C")[0],
                tokenizer.encode(" D")[0],
            ]
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
        # Train with original dataset
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

        return loss

    def validation_step(self, batch, batch_idx):
        # Validate with MMLU
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        logits = outputs["logits"]

        # loss: (float) batch_size * seq_len
        # logits: (float) batch_size * seq_len * vocab_size
        # labels: (int) batch_size * seq_len
        preds = []
        for i, logit in enumerate(logits):
            label_non_zero_ids = (labels[i] != self.IGNORE_INDEX).nonzero()
            if len(label_non_zero_ids) == 0:  # if answer was truncated
                # regard as wrong prediction
                preds.append(5)
                labels[i][0] = 4
                continue
            logit_abcd = logit[label_non_zero_ids[0][0] - 1][self.abcd_idx]
            preds.append(torch.argmax(logit_abcd).item())
        labels = labels[labels != self.IGNORE_INDEX].view(-1, 1)[:, 0]
        refs = [
            self.abcd_idx.index(label) if label != 4 else 4 for label in labels.tolist()
        ]

        sub = batch["subject"]
        subjects_output = {
            self.SUBJECTS[s]: {"refs": [], "preds": []} for s in set(sub)
        }
        for s, p, r in zip(sub, preds, refs):
            subjects_output[self.SUBJECTS[s]]["preds"].append(p)
            subjects_output[self.SUBJECTS[s]]["refs"].append(r)

        # Log general accuracy
        self.acc_val(torch.tensor(preds).detach(), torch.tensor(refs))

        # Log subject accuracies
        for s in subjects_output:
            self.acc_sub_val[s](
                torch.tensor(subjects_output[s]["preds"]).detach(),
                torch.tensor(subjects_output[s]["refs"]),
            )

        self.loss_val(loss.detach())

        return loss

    def on_validation_epoch_end(self):
        self.log("mmlu_val_loss", self.loss_val, prog_bar=True)
        self.log("mmlu_val_acc", self.acc_val, prog_bar=True)
        for s, acc_sub in self.acc_sub_val.items():
            self.log(f"mmlu_val_acc_{s}", acc_sub)

    def test_step(self, batch, batch_idx):
        # Test with MMLU
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        logits = outputs["logits"]

        # loss: (float) batch_size * seq_len
        # logits: (float) batch_size * seq_len * vocab_size
        # labels: (int) batch_size * seq_len
        preds = []
        for i, logit in enumerate(logits):
            label_non_zero_ids = (labels[i] != self.IGNORE_INDEX).nonzero()
            if len(label_non_zero_ids) == 0:  # if answer was truncated
                # regard as wrong prediction
                preds.append(5)
                labels[i][0] = 4
                continue
            logit_abcd = logit[label_non_zero_ids[0][0] - 1][self.abcd_idx]
            preds.append(torch.argmax(logit_abcd).item())
        labels = labels[labels != self.IGNORE_INDEX].view(-1, 1)[:, 0]
        refs = [
            self.abcd_idx.index(label) if label != 4 else 4 for label in labels.tolist()
        ]

        sub = batch["subject"]
        subjects_output = {
            self.SUBJECTS[s]: {"refs": [], "preds": []} for s in set(sub)
        }
        for s, p, r in zip(sub, preds, refs):
            subjects_output[self.SUBJECTS[s]]["preds"].append(p)
            subjects_output[self.SUBJECTS[s]]["refs"].append(r)

        # Log general accuracy
        self.acc_test(torch.tensor(preds).detach(), torch.tensor(refs))

        # Log subject accuracies
        for s in subjects_output:
            self.acc_sub_test[s](
                torch.tensor(subjects_output[s]["preds"]).detach(),
                torch.tensor(subjects_output[s]["refs"]),
            )

        self.loss_test(loss.detach())

        return loss

    def on_test_epoch_end(self):
        self.log("mmlu_test_loss", self.loss_test, prog_bar=True)
        self.log("mmlu_test_acc", self.acc_test, prog_bar=True)
        for s, acc_sub in self.acc_sub_test.items():
            self.log(f"mmlu_test_acc_{s}", acc_sub)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        # Predict with original dataset
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.forward(input_ids, attention_mask, labels=None)
        outputs["batch_idx"] = batch_idx
        return outputs
