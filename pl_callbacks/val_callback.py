import copy
from functools import partial

import datasets
import evaluate
import numpy as np
import pytorch_lightning as pl
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling

from dataset.language_modeling_datasets import DataCollatorForCausalLMAlpaca

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


class MMLUValidationCallback(pl.Callback):
    IGNORE_INDEX = -100

    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_len: int,  # for tokenizing
        num_workers: int = None,
        load_from_cache_file: bool = True,
        load_from_saved_path: str = None,
        few_shot: bool = True,
    ):
        self.few_shot = few_shot
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.load_from_cache_file = load_from_cache_file
        self.load_from_saved_path = load_from_saved_path

        self.mmlu_dataset = None

        self.abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]

    def _download_dataset(self):
        if not self.few_shot:
            self.mmlu_dataset = datasets.load_dataset(
                "json",
                data_files={
                    "validation": "data/mmlu/zero_shot_mmlu_val.json",
                    "test": "data/mmlu/zero_shot_mmlu_test.json",
                },
            )
        else:
            self.mmlu_dataset = datasets.load_dataset(
                "json",
                data_files={
                    "validation": "data/mmlu/five_shot_mmlu_val.json",
                    "test": "data/mmlu/five_shot_mmlu_test.json",
                },
            )

    @staticmethod
    def _preprocess(example, tokenizer, max_length, ignore_id):
        def _tokenize(text, tokenizer, max_length):
            return tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True,
            )

        prompt = example["input"]
        target = example["output"]

        prompt_tokenized = _tokenize(prompt, tokenizer, max_length)["input_ids"][0]
        target_tokenized = _tokenize(prompt + target, tokenizer, max_length)[
            "input_ids"
        ][0]
        input_ids = copy.deepcopy(target_tokenized)

        prompt_len = prompt_tokenized.ne(tokenizer.pad_token_id).sum().item()
        target_tokenized[:prompt_len] = ignore_id
        return dict(
            input_ids=input_ids,
            labels=target_tokenized,
        )

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        self._download_dataset()
        self.mmlu_dataset: DatasetDict = self.mmlu_dataset.map(
            function=partial(
                self._preprocess,
                tokenizer=self.tokenizer,
                max_length=self.max_token_len,
                ignore_id=self.IGNORE_INDEX,
            ),
            num_proc=self.num_workers,
            load_from_cache_file=True,
            desc="Preprocessing MMLU dataset",
        )

    def _val_dataloader(self):
        data_collator = DataCollatorForCausalLMAlpaca(
            tokenizer=self.tokenizer,
        )
        return DataLoader(
            self.mmlu_dataset["validation"].remove_columns(
                ["subject", "input", "output"]
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=data_collator,
        )

    def _test_dataloader(self):
        data_collator = DataCollatorForCausalLMAlpaca(
            tokenizer=self.tokenizer,
        )
        return DataLoader(
            self.mmlu_dataset["test"].remove_columns(["subject", "input", "output"]),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=data_collator,
        )

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if torch.cuda.current_device() != 0:
            return
        data_loader = self._val_dataloader()
        pl_module.model.eval()
        loss_mmlu = 0.0
        preds, refs = [], []
        for batch_idx, batch in enumerate(
            tqdm(data_loader, total=len(data_loader), desc="Validating MMLU")
        ):
            input_ids = batch["input_ids"].to(pl_module.model.device)
            attention_mask = batch["attention_mask"].to(pl_module.model.device)
            labels = batch["labels"].to(pl_module.model.device)

            outputs = pl_module.forward(input_ids, attention_mask, labels)
            loss, logits = outputs["loss"], outputs["logits"]

            # loss: (float) batch_size * seq_len
            # logits: (float) batch_size * seq_len * vocab_size
            # labels: (int) batch_size * seq_len
            for i, logit in enumerate(logits):
                label_non_zero_ids = (labels[i] != self.IGNORE_INDEX).nonzero()
                if len(label_non_zero_ids) == 0:  # answer was truncated
                    # regard as wrong prediction
                    preds.append(5)
                    labels[i][0] = 4
                    continue
                logit_abcd = logit[label_non_zero_ids[0][0] - 1][self.abcd_idx]
                preds.append(torch.argmax(logit_abcd).item())
            labels = labels[labels != self.IGNORE_INDEX].view(-1, 1)[:, 0]
            refs += [self.abcd_idx.index(label) if label != 4 else 4 for label in labels.tolist()]
            loss_mmlu += loss.item()

        results = {"mmlu_loss": loss_mmlu / len(data_loader)}

        subject = self.mmlu_dataset["validation"]["subject"]
        subjects = {SUBJECTS[s]: {"refs": [], "preds": []} for s in set(subject)}
        for s, p, r in zip(subject, preds, refs):
            subjects[SUBJECTS[s]]["preds"].append(p)
            subjects[SUBJECTS[s]]["refs"].append(r)

        accuracy = evaluate.load("accuracy")
        subject_scores = []
        for s in subjects:
            subject_score = accuracy.compute(
                references=subjects[s]["refs"], predictions=subjects[s]["preds"]
            )["accuracy"]
            results[f"mmlu_val_acc_{s}"] = subject_score
            subject_scores.append(subject_score)
        results[f"mmlu_val_acc"] = accuracy.compute(
            references=refs,
            predictions=preds,
        )["accuracy"]

        pl_module.log_dict(results)
