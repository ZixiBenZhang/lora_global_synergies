import copy
from functools import partial

import datasets
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from dataset.language_modeling_datasets import DataCollatorForCausalLMAlpaca, DataCollatorForCausalLMMMLU

IGNORE_INDEX = -100


def setup_mmlu(
    batch_size: int,
    tokenizer: PreTrainedTokenizer,
    max_token_len: int,
    num_workers: int = None,
    few_shot: bool = True,
    load_from_cache_file: bool = True,
    force_shuffle=None,
):
    # Load MMLU dataset
    if not few_shot:
        mmlu_dataset = datasets.load_dataset(
            "json",
            data_files={
                "validation": "data/mmlu/zero_shot_mmlu_val.json",
                "test": "data/mmlu/zero_shot_mmlu_test.json",
            },
        )
    else:
        mmlu_dataset = datasets.load_dataset(
            "json",
            data_files={
                "validation": "data/mmlu/five_shot_mmlu_val.json",
                "test": "data/mmlu/five_shot_mmlu_test.json",
            },
        )

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

    # Preprocess MMLU
    mmlu_dataset: DatasetDict = mmlu_dataset.map(
        function=partial(
            _preprocess,
            tokenizer=tokenizer,
            max_length=max_token_len,
            ignore_id=IGNORE_INDEX,
        ),
        num_proc=num_workers,
        load_from_cache_file=load_from_cache_file,
        desc="Preprocessing MMLU dataset",
    )

    def get_mmlu_val():
        data_collator = DataCollatorForCausalLMMMLU(
            tokenizer=tokenizer,
        )
        return DataLoader(
            mmlu_dataset["validation"].remove_columns(["input", "output"]),
            batch_size=batch_size,
            shuffle=False if force_shuffle is None or not force_shuffle else True,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

    def get_mmlu_test():
        data_collator = DataCollatorForCausalLMMMLU(
            tokenizer=tokenizer,
        )
        return DataLoader(
            mmlu_dataset["test"].remove_columns(["input", "output"]),
            batch_size=batch_size,
            shuffle=False if force_shuffle is None or not force_shuffle else True,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

    return get_mmlu_val, get_mmlu_test
