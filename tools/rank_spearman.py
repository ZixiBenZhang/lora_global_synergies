import math

import numpy as np
import toml
import scipy.stats as stats


def rank_spearman(filename1, filename2):
    with open(filename1, "r") as f1:
        alpha_dict_1 = toml.load(f1)
    with open(filename2, "r") as f2:
        alpha_dict_2 = toml.load(f2)

    assert alpha_dict_1["dataset"] == alpha_dict_2["dataset"]

    alpha_list_1 = np.concatenate(
        [
            [v for proj_name, v in d.items()]
            for layer_name, d in alpha_dict_1.items()
            if "layer" in layer_name
        ],
        axis=0,
    )
    alpha_list_2 = np.concatenate(
        [
            [v for proj_name, v in d.items()]
            for layer_name, d in alpha_dict_2.items()
            if "layer" in layer_name
        ],
        axis=0,
    )
    res = stats.spearmanr(alpha_list_1, alpha_list_2)
    rho, p_value = (res.statistic, res.pvalue)
    print(
        f"Spearman correlation: {rho}\n"
        f"p-value for two rankings are not correlated: {p_value}"
    )


LORA_NAME_HASH = {
    "q_proj": 0,
    "k_proj": 1,
    "v_proj": 2,
    "out_proj": 3,
    "fc1": 4,
    "fc2": 5,
}
PERCENTILE = 0.25


def reallocation_interleave(filename1, filename2):
    with open(filename1, "r") as f1:
        alpha_dict_1 = toml.load(f1)
    with open(filename2, "r") as f2:
        alpha_dict_2 = toml.load(f2)

    assert alpha_dict_1["dataset"] == alpha_dict_2["dataset"]

    alpha_list_1 = np.concatenate(
        [
            [
                (int(layer_name.split("_")[-1]), LORA_NAME_HASH[proj_name], v)
                for proj_name, v in d.items()
            ]
            for layer_name, d in alpha_dict_1.items()
            if "layer" in layer_name
        ],
        axis=0,
    )
    alpha_list_1 = alpha_list_1[alpha_list_1[:, 0].argsort(kind="stable")]

    original_lora_module_num = len(alpha_list_1)
    budget = math.floor(PERCENTILE * original_lora_module_num)
    idx = alpha_list_1[:, 2].argsort(kind="stable")[-budget:]
    turn_on_1 = alpha_list_1[idx, :2].tolist()

    alpha_list_2 = np.concatenate(
        [
            [
                (int(layer_name.split("_")[-1]), LORA_NAME_HASH[proj_name], v)
                for proj_name, v in d.items()
            ]
            for layer_name, d in alpha_dict_2.items()
            if "layer" in layer_name
        ],
        axis=0,
    )
    alpha_list_2 = alpha_list_2[alpha_list_2[:, 0].argsort(kind="stable")]

    original_lora_module_num = len(alpha_list_2)
    budget = math.floor(PERCENTILE * original_lora_module_num)
    idx = alpha_list_2[:, 2].argsort(kind="stable")[-budget:]
    turn_on_2 = alpha_list_2[idx, :2].tolist()

    cnt = 0
    for entry in turn_on_1:
        if entry in turn_on_2:
            cnt += 1
            print(entry)

    print(cnt / len(turn_on_1))


if __name__ == "__main__":
    f1 = "../ags_output/opt_lora_classification_sst2_2024-02-29/alpha_ckpts/alpha-importance_11-30.toml"
    f2 = "../ags_output/opt_lora_classification_sst2_2024-03-02/alpha_ckpts/alpha-importance_15-24.toml"
    reallocation_interleave(f1, f2)
