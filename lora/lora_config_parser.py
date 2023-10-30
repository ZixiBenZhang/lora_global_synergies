import torch
import toml

from tools.config_load import convert_strna_to_none

"""
LoRA config format:
lora_config = {
    default: {r, lora_alpha, lora_dropout, adapter_name, disable_adapter},
    transformer_layer_{i}: {
        q_proj: {
            head_{j}: {r, lora_alpha, lora_dropout, adapter_name, disable_adapter},
            ... 
        },
        k_proj: {
            head_{j}: {r, lora_alpha, lora_dropout, adapter_name, disable_adapter},
            ...
        },
        v_proj: {
            head_{j}: {r, lora_alpha, lora_dropout, adapter_name, disable_adapter},
            ...
        },
        o_proj: {r, lora_alpha, lora_dropout, adapter_name, disable_adapter},
        w1: {r, lora_alpha, lora_dropout, adapter_name, disable_adapter},
        w2: {r, lora_alpha, lora_dropout, adapter_name, disable_adapter},
    },
    ...
}

Granularity:
Default: a mat config i.e. {r, lora_alpha, lora_dropout, adapter_name, disable_adapter}
Network: specify a mat config in "all_layers"
Layer-wise: specify a mat config for each (i) in "model_layer_{i}"
Head-wise:
    Default (Type-wise): 
        specify a mat config for each (proj_type [q_proj, k_proj, v_proj], j) in "{proj_type}_head{j}";
        specify a mat config for each (proj_type [o_proj, w1, w2]) in "{proj_type}"
    Head-wise:
        specify a mat config for each proj_type (i, [q_proj, k_proj, v_proj], j) in "model_layer_{i}_{proj_type}_head{j}";
        specify a mat config for each proj_type (i, [o_proj, w1, w2]) in "model_layer_{i}_{proj_type}"
"""


def parse_lora_config(
    config: str | dict, num_hidden_layers: int, num_heads: int | dict[str, int]
) -> dict:
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


def parse_by_network(
    config: dict, num_hidden_layers: int, num_heads: dict[str, int]
) -> dict:
    assert "default" in config, "Must provide default config"
    default_lc: dict = get_mat_config(
        config["default"]
    )  # same config for QKV in all layers
    all_layers_lc: dict = config.get("all_layers", None)  # config mat for all layers

    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = {}
        # Q, K, V heads
        for proj in ["q", "k", "v"]:
            for j in range(num_heads[proj]):
                head_entry = f"head_{j}"
                p_config[layer_entry][f"{proj}_proj"][head_entry] = create_a_mat_config(
                    all_layers_lc, default_lc
                )
        # O projection
        p_config[layer_entry]["o_proj"] = create_a_mat_config(all_layers_lc, default_lc)
        # ffn
        p_config[layer_entry]["w1"] = create_a_mat_config(all_layers_lc, default_lc)
        p_config[layer_entry]["w2"] = create_a_mat_config(all_layers_lc, default_lc)
    p_config["default"] = default_lc
    return p_config


def parse_by_layer(
    config: dict, num_hidden_layers: int, num_heads: dict[str, int]
) -> dict:
    assert "default" in config, "Must provide default config"
    default_lc: dict = get_mat_config(
        config["default"]
    )  # same config for QKV in all layers

    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        layer_lc: dict = config.get(layer_entry, None)  # config all mat for each layer
        p_config[layer_entry] = {}
        # Q, K, V heads
        for proj in ["q", "k", "v"]:
            for j in range(num_heads[proj]):
                head_entry = f"head_{j}"
                p_config[layer_entry][f"{proj}_proj"][head_entry] = create_a_mat_config(
                    layer_lc, default_lc
                )
        # O projection
        p_config[layer_entry]["o_proj"] = create_a_mat_config(layer_lc, default_lc)
        # ffn
        p_config[layer_entry]["w1"] = create_a_mat_config(layer_lc, default_lc)
        p_config[layer_entry]["w2"] = create_a_mat_config(layer_lc, default_lc)
    p_config["default"] = default_lc
    return p_config


def parse_by_head(
    config: dict, num_hidden_layers: int, num_heads: dict[str, int]
) -> dict:
    assert "default" in config, "Must provide default config"
    default_lc: dict = get_mat_config(
        config["default"]
    )  # same config for QKV in all layers

    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = {}
        # Q, K, V heads
        for proj in ["q", "k", "v"]:
            for j in range(num_heads[proj]):
                head_entry = f"head_{j}"
                head_default_lc: dict = config.get(
                    f"{proj}_proj_{head_entry}", None
                )  # config heads respectively for all layers
                head_lc: dict = config.get(
                    f"{layer_entry}_{proj}_proj_{head_entry}", None
                )  # config heads respectively for each layer
                p_config[layer_entry][f"{proj}_proj"][head_entry] = create_a_mat_config(
                    head_lc, create_a_mat_config(head_default_lc, default_lc)
                )
        # O projection
        o_default_lc: dict = config.get(f"o_proj", None)  # config o_proj for all layers
        o_lc: dict = config.get(
            f"{layer_entry}_o_proj", None
        )  # config o_proj respectively for each layer
        p_config[layer_entry]["o_proj"] = create_a_mat_config(
            o_lc, create_a_mat_config(o_default_lc, default_lc)
        )
        # ffn
        w1_default_lc: dict = config.get(f"w1", None)  # config o_proj for all layers
        w1_lc: dict = config.get(
            f"{layer_entry}_w1", None
        )  # config o_proj respectively for each layer
        p_config[layer_entry]["w1"] = create_a_mat_config(
            w1_lc, create_a_mat_config(w1_default_lc, default_lc)
        )
        w2_default_lc: dict = config.get(f"w2", None)  # config o_proj for all layers
        w2_lc: dict = config.get(
            f"{layer_entry}_w2", None
        )  # config o_proj respectively for each layer
        p_config[layer_entry]["w2"] = create_a_mat_config(
            w2_lc, create_a_mat_config(w2_default_lc, default_lc)
        )
    p_config["default"] = default_lc
    return p_config


def create_a_mat_config(mat_lc: dict = None, default_lc: dict = None) -> dict:
    if mat_lc is None and default_lc is None:
        raise ValueError("Must provide either mat_lc or default_lc")
    if default_lc is None:
        default_lc = {}
    return get_mat_config(mat_lc if mat_lc is not None else default_lc)


def get_mat_config(config: dict):
    # default for linear layer's matrix
    return {
        "r": config["r"],
        "lora_alpha": config["lora_alpha"],
        "lora_dropout": config["lora_dropout"],
        "adapter_name": config["adapter_name"],
        "disable_adapter": config["disable_adapter"],
    }
