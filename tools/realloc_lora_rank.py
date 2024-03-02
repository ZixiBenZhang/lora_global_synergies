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


def reallocate_lora_rank(filename):
    t = time.strftime("%m-%d")

    with open(filename, "r") as f:
        alpha_dict = toml.load(f)

    dataset_name = alpha_dict["dataset"]

    alpha_list = np.concatenate(
        [
            [(int(layer_name.split("_")[-1]), LORA_NAME_HASH[proj_name], v) for proj_name, v in d.items()]
            for layer_name, d in alpha_dict.items() if "layer" in layer_name
        ],
        axis=0,
    )
    alpha_list = alpha_list[alpha_list[:, 0].argsort(kind="stable")]

    # Constant new rank
    new_r = round(ORIGINAL_R / PERCENTILE)

    original_lora_module_num = len(alpha_list)

    budget = math.floor(PERCENTILE * original_lora_module_num)
    # prioritise later layers
    idx = alpha_list[:, 2].argsort(kind="stable")[-budget:]
    print(alpha_list[idx])
    turn_on = alpha_list[idx, :2]
    threshold = alpha_list[idx[0]]
    print(f"Alpha threshold: {threshold}")

    lora_new_config: dict[str, str | dict[str, int | float | str | dict[str, int | float | str]]] = {
        "granularity": "layer",
        "default": {
            "r": 0,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "adapter_name": f"ags-layer-importance-{dataset_name}",
            "init_lora_weghts": True,
            "fan_in_fan_out": False,
            "disable_adapter": True,
        },
    }

    for layer_id, proj_hash in turn_on:
        if f"model_layer_{layer_id}" not in lora_new_config:
            lora_new_config[f"model_layer_{layer_id}"] = {}
        lora_new_config[f"model_layer_{layer_id}"][LORA_NAME_MAP[list(LORA_NAME_HASH.keys())[proj_hash]]] = {
            "r": new_r,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "adapter_name": f"ags-layer-importance-{dataset_name}",
            "disable_adapter": False,
        }

    if not os.path.isdir("../realloc-alpha"):
        os.mkdir("../realloc-alpha")
    i = 0
    while os.path.isfile(f"../realloc-alpha/{dataset_name}-{t}-version_{i}.toml"):
        i += 1
    config_path = os.path.join(f"../realloc-alpha/{dataset_name}-{t}-version_{i}.toml")
    with open(config_path, "w+") as fout:
        toml.dump(lora_new_config, fout)
    logger.info(f"New lora config saved to {config_path}")


if __name__ == "__main__":
    reallocate_lora_rank("../ags_output/opt_lora_classification_sst2_2024-03-02/alpha_ckpts/alpha-importance_17-34.toml")
