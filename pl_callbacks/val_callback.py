import copy
from functools import partial

import datasets
import evaluate
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling


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
        few_shot: bool = True
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
            self.mmlu_dataset = datasets.load_dataset("json", data_files={
                'validation': 'data/mmlu/zero_shot_mmlu_val.json',
                'test': 'data/mmlu/zero_shot_mmlu_test.json',
            })
        else:
            self.mmlu_dataset = datasets.load_dataset("json", data_files={
                'validation': 'data/mmlu/five_shot_mmlu_val.json',
                'test': 'data/mmlu/five_shot_mmlu_test.json',
            })

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

        prompt_tokenized = _tokenize(prompt + " ", tokenizer, max_length)["input_ids"][0]
        target_tokenized = _tokenize(prompt + " " + target, tokenizer, max_length)["input_ids"][0]
        input_ids = copy.deepcopy(target_tokenized)

        prompt_len = prompt_tokenized.ne(tokenizer.pad_token_id).sum().item()
        target_tokenized[:prompt_len] = ignore_id
        return dict(
            input_ids=input_ids,
            labels=target_tokenized,
        )

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self._download_dataset()
        self.mmlu_dataset = self.mmlu_dataset.map(
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
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        return DataLoader(
            self.mmlu_dataset["validation"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=data_collator,
        )

    def _test_dataloader(self):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        return DataLoader(
            self.mmlu_dataset["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=data_collator,
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        data_loader = self._val_dataloader()
        pl_module.model.eval()
        loss_mmlu = 0.0
        preds, refs = [], []
        for batch_idx, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
            # todo: debug: in Collator, input_ids is Tensor before pad_sequence but list when error raised...
            outputs = pl_module.predict_step(batch=batch, batch_idx=batch_idx)
            # loss: (float) batch_size * seq_len
            # logits: (float) batch_size * seq_len * vocab_size
            # labels: (int) batch_size * seq_len
            loss, logits, labels = outputs["loss"], outputs["logits"], batch["labels"]
            for i, logit in enumerate(logits):
                label_non_zero_id = (labels[i] != self.IGNORE_INDEX).nonzero()[0][0]
                logit_abcd = logit[label_non_zero_id - 1][self.abcd_idx]
                preds.append(torch.argmax(logit_abcd).item())
            # There are two tokens, the output, and eos token.
            # todo: to be tested, is view(-1,2) needed?
            labels = labels[labels != self.IGNORE_INDEX].view(-1, 2)[:, 0]
            refs += [self.abcd_idx.index(label) for label in labels.tolist()]
            loss_mmlu += loss.item()

        results = {'mmlu_loss': loss_mmlu / len(data_loader)}

        subject = self.mmlu_dataset['subject']
        subjects = {s: {'refs': [], 'preds': []} for s in set(subject)}
        for s, p, r in zip(subject, preds, refs):
            subjects[s]['preds'].append(p)
            subjects[s]['refs'].append(r)

        accuracy = evaluate.load("accuracy")
        subject_scores = []
        for subject in subjects:
            subject_score = accuracy.compute(
                references=subjects[subject]['refs'],
                predictions=subjects[subject]['preds']
            )['accuracy']
            results[f'mmlu_val_acc_{subject}'] = subject_score
            subject_scores.append(subject_score)
        results[f'mmlu_val_acc'] = np.mean(subject_scores)

        pl_module.log_dict(results)
