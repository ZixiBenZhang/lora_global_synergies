import torch
import toml

from tools.config_load import convert_strna_to_none

'''
LoRA config format:
lora_config = {
    default: {r, lora_alpha, lora_dropout_p, adapter_name, disable_adapter},
    transformer_layer_{i}: {
        q_proj: {
            head_{j}: {r, lora_alpha, lora_dropout_p, adapter_name, disable_adapter},
            ... 
        },
        k_proj: {
            head_{j}: {r, lora_alpha, lora_dropout_p, adapter_name, disable_adapter},
            ...
        },
        v_proj: {
            head_{j}: {r, lora_alpha, lora_dropout_p, adapter_name, disable_adapter},
            ...
        },
        o_proj: {r, lora_alpha, lora_dropout_p, adapter_name, disable_adapter},
        w1: {r, lora_alpha, lora_dropout_p, adapter_name, disable_adapter},
        w2: {r, lora_alpha, lora_dropout_p, adapter_name, disable_adapter},
    },
    ...
}
'''

def parse_lora_config(config: str | dict, num_hidden_layers: int, num_heads: int | dict[str, int]) -> dict:
    assert isinstance(config, (str, dict)), "config must be a str path to toml or dict"
    if isinstance(config, str):
        config = toml.load(config)
    if isinstance(num_heads, int):
        num_heads = {proj: num_heads for proj in ["q", "k", "v"]}
    config = convert_strna_to_none(config)
    granularity = config.pop("granularity", "network")
    match granularity:
        case "network":
            # same lora config for all layers
            return parse_by_network(config, num_hidden_layers, num_heads)
        case "layer":
            # same lora config for all heads in a layer
            return parse_by_layer(config, num_hidden_layers, num_heads)
        case "head":
            # different lora config for different head
            return parse_by_head(config, num_hidden_layers, num_heads)
        case _:
            raise ValueError(f"Unsupported config granularity: {granularity}")


def parse_by_network(config: dict, num_hidden_layers: int, num_heads: dict[str, int]) -> dict:
    assert "default" in config, "Must provide default config"
    default_lc: dict = get_mat_config(config["default"])  # same config for QKV in all layers
    all_layers_lc: dict = config.get("all_layers", None)  # config QKV for all layers

    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = {}
        # Q, K, V heads
        for proj in ["q", "k", "v"]:
            for j in range(num_heads[proj]):
                p_config[layer_entry][f"{proj}_proj"][f"head_{j}"] = get_mat_config(
                    create_a_mat_config(all_layers_lc, default_lc)
                )
        # O projection
        p_config[layer_entry]["o_proj"] = get_mat_config(
            create_a_mat_config(all_layers_lc, default_lc)
        )
        # ffn
        p_config[layer_entry]["w1"] = get_mat_config(
            create_a_mat_config(all_layers_lc, default_lc)
        )
        p_config[layer_entry]["w2"] = get_mat_config(
            create_a_mat_config(all_layers_lc, default_lc)
        )
    p_config["default"] = default_lc
    return p_config


def parse_by_layer(config: dict, num_hidden_layers: int, num_heads: dict[str, int]) -> dict:
    pass


def parse_by_head(config: dict, num_hidden_layers: int, num_heads: dict[str, int]) -> dict:
    pass


def create_a_mat_config(mat_lc: dict = None, default_lc: dict = None) -> dict:
    if mat_lc is None and default_lc is None:
        raise ValueError("Must provide either mat_lc or default_lc")
    if default_lc is None:
        default_lc = {}
    return mat_lc if mat_lc is not None else default_lc


def get_mat_config(config: dict):
    # default for linear layer's matrix
    return {
        "r": config["r"],
        "lora_alpha": config["lora_alpha"],
        "lora_dropout": config["lora_dropout"],
        "adapter_name": config["adapter_name"],
        "disable_adapter": config["disable_adapter"],
    }
