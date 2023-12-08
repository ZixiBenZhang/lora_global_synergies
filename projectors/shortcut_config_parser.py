import torch
import toml

from loading.config_load import convert_str_na_to_none


"""
Shortcut config format:
shortcut_config = {
    layer_residual: {
        model_layer_{i}: {
            residual1: {r, proj_dropout, projector_name, disable_projector},
            residual2: {r, proj_dropout, projector_name, disable_projector},
        }
        ...
    }
    head_residual: {
        model_layer_{i}: {
            q_residual: {
                head_{j}: {r, proj_dropout, projector_name, disable_projector},
                ... 
            },
            k_residual: {
                head_{j}: {r, proj_dropout, projector_name, disable_projector},
                ...
            },
            v_residual: {
                head_{j}: {r, proj_dropout, projector_name, disable_projector},
                ...
            },
            o_residual: {r, proj_dropout, projector_name, disable_projector},
        },
        ...
    }
    cross_layer_shortcut: {
        TODO: should be specified only when active
    }
}
If shortcut disabled => [default] r=0, ...
>>>>>>>>>> Currently using layer_residual only

Granularity:
Default: a mat config (i.e. {r, proj_dropout, projector_name, disable_projector},) for all linear in whole network
Network: specify a proj_type-wise config in "all_layers".proj_type
Layer-wise: specify a layer-wise & proj_type-wise config for each (i) in "model_layer_{i}".proj_type
Head-wise: specify at least one of the following two
    Default (Type-wise): type-wise & head-wise config for every layer in network
        specify a mat config for each (proj_type [q_proj, k_proj, v_proj], j) in "headwise_default".proj_type."head{j}",
        specify a mat config for proj_type o_proj in "headwise_default".proj_type
    Head-wise: layer-wise & type-wise & head-wise
        specify a mat config for each proj_type (i, [q_proj, k_proj, v_proj], j) in "model_layer_{i}".proj_type."head{j}",
        specify a mat config for each i and proj_type o_proj in "model_layer_{i}".proj_type
        
layer_residual: network/layer-wise for each residual shortcut
head_residual: network/layer-wise/head-wise for each head's shortcut
"""
# TODO: insert init_proj_weights and fan_in_fan_out to shortcut_config??


def parse_shortcut_config(
    config: str | dict, num_hidden_layers: int, num_heads: int | dict[str, int]
) -> dict:
    assert isinstance(config, (str, dict)), "config must be a str path to toml or dict"
    if isinstance(config, str):
        config = toml.load(config)
    if isinstance(num_heads, int):
        num_heads = {proj: num_heads for proj in ["q", "k", "v"]}
    config: dict = convert_str_na_to_none(config)
    p_config = {}

    layer_res_config = config["layer_residual"]
    granularity = layer_res_config.pop("granularity", "network")
    p_config["layer_residual"] = parse_layer_res_config(layer_res_config, granularity, num_hidden_layers, num_heads)

    head_res_config = config["head_residual"]
    granularity = head_res_config.pop("granularity", "network")
    p_config["head_residual"] = parse_head_res_config(head_res_config, granularity, num_hidden_layers, num_heads)

    # TODO: parse cross-layer shortcut config

    return p_config


def parse_layer_res_config(
    config: dict, granularity, num_hidden_layers: int, num_heads: dict[str, int]
) -> dict:
    match granularity:
        case "network":
            # same config for all layers
            return _parse_layer_res_by_network(config, num_hidden_layers, num_heads)
        case "layer":
            # same config for all heads in a layer
            return _parse_layer_res_by_layer(config, num_hidden_layers, num_heads)
        case _:
            raise ValueError(f"Unsupported layer_residual config granularity: {granularity}")


def parse_head_res_config(
    config: dict, granularity, num_hidden_layers: int, num_heads: dict[str, int]
) -> dict:
    match granularity:
        case "network":
            # same config for all layers
            return _parse_head_res_by_network(config, num_hidden_layers, num_heads)
        case "layer":
            # same config for all heads in a layer
            return _parse_head_res_by_layer(config, num_hidden_layers, num_heads)
        case "head":
            # different config for different head
            return _parse_head_res_by_head(config, num_hidden_layers, num_heads)
        case _:
            raise ValueError(f"Unsupported head_residual config granularity: {granularity}")


def _parse_layer_res_by_network(config: dict, num_hidden_layers: int, num_heads: dict[str, int]) -> dict:
    assert "default" in config, "Must provide default config"
    default_sc: dict = get_mat_config(
        config["default"]
    )  # same config for QKV in all layers

    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = {}

        # Residual shortcut projections
        all_layer_residual1_sc: dict = config.get("all_layers", {}).get("residual1", None)
        p_config[layer_entry]["residual1"] = create_a_mat_config(all_layer_residual1_sc, default_sc)
        all_layer_residual2_sc: dict = config.get("all_layers", {}).get("residual2", None)
        p_config[layer_entry]["residual2"] = create_a_mat_config(all_layer_residual2_sc, default_sc)

    p_config["default"] = default_sc
    return p_config


def _parse_layer_res_by_layer(
    config: dict, num_hidden_layers: int, num_heads: dict[str, int]
) -> dict:
    assert "default" in config, "Must provide default config"
    default_sc: dict = get_mat_config(
        config["default"]
    )  # same config for QKV in all layers

    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = {}

        # Residual shortcut projections
        layer_residual1_sc: dict = config.get(layer_entry, {}).get("residual1", None)
        p_config[layer_entry]["residual1"] = create_a_mat_config(layer_residual1_sc, default_sc)
        layer_residual2_sc: dict = config.get(layer_entry, {}).get("residual2", None)
        p_config[layer_entry]["residual2"] = create_a_mat_config(layer_residual2_sc, default_sc)

    p_config["default"] = default_sc
    return p_config


def _parse_head_res_by_network(config: dict, num_hidden_layers: int, num_heads: dict[str, int]) -> dict:
    assert "default" in config, "Must provide default config"
    default_sc: dict = get_mat_config(
        config["default"]
    )  # same config for QKV in all layers

    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = {}

        # Q, K, V heads
        for proj in ["q", "k", "v"]:
            proj_entry = f"{proj}_proj"
            p_config[layer_entry][proj_entry] = {}
            # Get type-wise config that's same across layers & heads
            all_layer_proj_sc: dict = config.get("all_layers", {}).get(proj_entry, None)
            for j in range(num_heads[proj]):
                head_entry = f"head_{j}"
                p_config[layer_entry][proj_entry][head_entry] = create_a_mat_config(
                    all_layer_proj_sc, default_sc
                )

        # O projection
        all_layer_o_sc: dict = config.get("all_layers", {}).get("o_proj", None)
        p_config[layer_entry]["o_proj"] = create_a_mat_config(all_layer_o_sc, default_sc)

    p_config["default"] = default_sc
    return p_config


def _parse_head_res_by_layer(
    config: dict, num_hidden_layers: int, num_heads: dict[str, int]
) -> dict:
    assert "default" in config, "Must provide default config"
    default_sc: dict = get_mat_config(
        config["default"]
    )  # same config for QKV in all layers

    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = {}

        # Q, K, V heads
        for proj in ["q", "k", "v"]:
            proj_entry = f"{proj}_proj"
            p_config[layer_entry][proj_entry] = {}

            # Get layer-wise & type-wise config that's same across heads
            layer_proj_sc: dict = config.get(layer_entry, {}).get(proj_entry, None)

            for j in range(num_heads[proj]):
                head_entry = f"head_{j}"
                p_config[layer_entry][proj_entry][head_entry] = create_a_mat_config(
                    layer_proj_sc, default_sc
                )

        # O projection
        layer_o_sc: dict = config.get(layer_entry, {}).get("o_proj", None)
        p_config[layer_entry]["o_proj"] = create_a_mat_config(layer_o_sc, default_sc)

    p_config["default"] = default_sc
    return p_config


def _parse_head_res_by_head(
    config: dict, num_hidden_layers: int, num_heads: dict[str, int]
) -> dict:
    assert "default" in config, "Must provide default config"
    default_sc: dict = get_mat_config(
        config["default"]
    )  # same config for QKV in all layers

    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = {}

        # Q, K, V heads
        for proj in ["q", "k", "v"]:
            proj_entry = f"{proj}_proj"
            p_config[layer_entry][proj_entry] = {}
            for j in range(num_heads[proj]):
                head_entry = f"head_{j}"

                # Get type-wise & head-wise config that's same across layers
                head_default_sc: dict = config.get("headwise_default").get(proj_entry, {}).get(head_entry, None)
                # Get layer-wise & type-wise & head-wise config
                head_sc: dict = config.get(layer_entry, {}).get(proj_entry, {}).get(head_entry, None)

                p_config[layer_entry][proj_entry][head_entry] = create_a_mat_config(
                    head_sc, create_a_mat_config(head_default_sc, default_sc)
                )

        # O projection
        o_default_sc: dict = config.get("headwise_default").get(f"o_proj", None)
        o_sc: dict = config.get(layer_entry, {}).get(f"o_proj", None)
        p_config[layer_entry]["o_proj"] = create_a_mat_config(
            o_sc, create_a_mat_config(o_default_sc, default_sc)
        )

    p_config["default"] = default_sc
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
        "proj_dropout": config["proj_dropout"],
        "projector_name": config["projector_name"],
        "disable_projector": config["disable_projector"],
    }
