import copy
import logging
import math
import os
import pickle
import time

import numpy as np
import toml
import torch
import pytorch_lightning as pl


logger = logging.getLogger(__name__)

LORA_NAME_MAP = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "out_proj": "o_proj",
    "fc1": "w1",
    "fc2": "w2",
}
LORA_NAME_HASH = {
    "q_proj": 0,
    "k_proj": 1,
    "v_proj": 2,
    "out_proj": 3,
    "fc1": 4,
    "fc2": 5,
}

PERCENTILE = 0.25
ORIGINAL_R = 8
NEW_R = 8


def reallocate_lora_rank(filename, metric_name=None, metric_attr=None):
    t = time.strftime("%m-%d")

    with open(filename, "r") as f:
        alpha_dict = toml.load(f)

    dataset_name = alpha_dict.get("dataset", "mrpc")

    alpha_list = np.concatenate(
        [
            [
                (int(layer_name.split("_")[-1]), LORA_NAME_HASH[proj_name], v)
                for proj_name, v in d.items()
            ]
            for layer_name, d in alpha_dict.items()
            if "layer" in layer_name
        ],
        axis=0,
    )

    original_lora_module_num = len(alpha_list)
    budget = math.floor(PERCENTILE * original_lora_module_num)
    new_r = NEW_R
    idx = alpha_list[:, 2].argsort()
    alpha_threshold = alpha_list[idx[-budget], 2]
    if sum(alpha_list[:, 2] == alpha_threshold) > 1:
        # Uniformly break tie
        greater = alpha_list[alpha_list[:, 2] > alpha_threshold, :2]
        tie = alpha_list[alpha_list[:, 2] == alpha_threshold, :2]
        tie_idx = torch.randperm(len(tie))[: (budget - len(greater))]
        turn_on = np.concatenate([tie[tie_idx], greater], axis=0)
    else:
        idx = idx[-budget:]
        turn_on = alpha_list[idx, :2]
    turn_on = turn_on.astype(int).tolist()
    assert len(turn_on) == budget

    print(
        f"{metric_name if metric_name is not None else 'Metric'} threshold: {alpha_threshold}"
    )

    lora_new_config: dict[
        str, str | dict[str, int | float | str | dict[str, int | float | str]]
    ] = {
        "granularity": "layer",
        "default": {
            "r": 0,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "adapter_name": f"ags-imp-{dataset_name}",
            "init_lora_weghts": True,
            "fan_in_fan_out": False,
            "disable_adapter": True,
        },
    }

    for layer_id, proj_hash in turn_on:
        if f"model_layer_{layer_id}" not in lora_new_config:
            lora_new_config[f"model_layer_{layer_id}"] = {}
        lora_new_config[f"model_layer_{layer_id}"][
            LORA_NAME_MAP[list(LORA_NAME_HASH.keys())[proj_hash]]
        ] = {
            "r": new_r,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "adapter_name": f"ags-layer-importance-{dataset_name}",
            "disable_adapter": False,
        }

    if not os.path.isdir("../realloc-alpha"):
        os.mkdir("../realloc-alpha")

    i = 0
    if metric_name is None:
        prefix = f"../realloc-alpha/{dataset_name}-{t}"
    else:
        if metric_attr is None:
            prefix = f"../realloc-alpha/{dataset_name}-{metric_name}-{t}"
        else:
            prefix = f"../realloc-alpha/{dataset_name}-{metric_name}-{metric_attr}-{t}"
    while os.path.isfile(f"{prefix}-version_{i}.toml"):
        i += 1

    config_path = os.path.join(f"{prefix}-version_{i}.toml")
    with open(config_path, "w+") as fout:
        toml.dump(lora_new_config, fout)
    logger.info(f"New lora config saved to {config_path}")


if __name__ == "__main__":
    pl.seed_everything(0)
    directory = (
        "../ags_output/opt_lora_classification_mrpc_2024-04-01/importance_ckpts/"
    )
    for filename in os.listdir(directory):
        if filename.endswith(".toml"):
            metric = filename.split("_")[0]
            batches = filename.split("_")[-1]
            if batches.startswith("b") or batches.startswith("all"):
                batches = batches.split(".")[0]
            else:
                batches = None

            reallocate_lora_rank(os.path.join(directory, filename), metric, batches)
        else:
            continue
    # reallocate_lora_rank(
    #     "../ags_output/opt_lora_classification_mrpc_2024-04-01/importance_ckpts/grad-norm_12-48_b8.toml"
    # )
