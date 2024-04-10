import torch
import toml

from loading.config_load import convert_str_na_to_none


"""
Shortcut config format:
shortcut_config = {
    model_layer_{i}: {
        residual1: {r, proj_dropout, projector_name, disable_projector},
        residual2: {r, proj_dropout, projector_name, disable_projector},
        shortcut1: {r, proj_dropout, projector_name, disable_projector},
        shortcut2: {r, proj_dropout, projector_name, disable_projector},
    }
    ...
}
If shortcut disabled => [default] r=0, ...

Granularity:
Default: a mat config (i.e. {r, proj_dropout, projector_name, disable_projector},) for all linear in whole network
Network: specify a proj_type-wise config in "all_layers".proj_type
Layer-wise: specify a layer-wise & proj_type-wise config for each (i) in "model_layer_{i}".proj_type        
"""
# TODO: insert init_proj_weights and fan_in_fan_out to shortcut_config??


def parse_shortcut_config(config: str | dict, num_hidden_layers: int) -> dict:
    assert isinstance(config, (str, dict)), "config must be a str path to toml or dict"
    if isinstance(config, str):
        config = toml.load(config)
    config: dict = convert_str_na_to_none(config)

    granularity = config.pop("granularity", "network")
    match granularity:
        case "network":
            return parse_by_network(config, num_hidden_layers)
        case "layer":
            return parse_by_layer(config, num_hidden_layers)
        case _:
            raise ValueError(f"Unsupported config granularity: {granularity}")


def parse_by_network(config: dict, num_hidden_layers: int) -> dict:
    assert "default" in config, "Must provide default config"
    default_sc: dict = get_mat_config(
        config["default"]
    )  # for each shortcut, same config across all layers

    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = {}

        # Residual shortcut projections
        all_residual1_sc: dict = config.get("all_layers", {}).get("residual1", None)
        p_config[layer_entry]["residual1"] = create_a_mat_config(
            all_residual1_sc, default_sc
        )
        all_residual2_sc: dict = config.get("all_layers", {}).get("residual2", None)
        p_config[layer_entry]["residual2"] = create_a_mat_config(
            all_residual2_sc, default_sc
        )
        all_shortcut1_sc: dict = config.get("all_layers", {}).get("shortcut1", None)
        p_config[layer_entry]["shortcut1"] = create_a_mat_config(
            all_shortcut1_sc, default_sc
        )
        all_shortcut2_sc: dict = config.get("all_layers", {}).get("shortcut2", None)
        p_config[layer_entry]["shortcut2"] = create_a_mat_config(
            all_shortcut2_sc, default_sc
        )

    p_config["default"] = default_sc
    return p_config


def parse_by_layer(config: dict, num_hidden_layers: int) -> dict:
    assert "default" in config, "Must provide default config"
    default_sc: dict = get_mat_config(
        config["default"]
    )  # for each shortcut, specified config in each layer

    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = {}

        # Residual shortcut projections
        residual1_sc: dict = config.get(layer_entry, {}).get("residual1", None)
        p_config[layer_entry]["residual1"] = create_a_mat_config(
            residual1_sc, default_sc
        )
        residual2_sc: dict = config.get(layer_entry, {}).get("residual2", None)
        p_config[layer_entry]["residual2"] = create_a_mat_config(
            residual2_sc, default_sc
        )
        shortcut1_sc: dict = config.get(layer_entry, {}).get("shortcut1", None)
        p_config[layer_entry]["shortcut1"] = create_a_mat_config(
            shortcut1_sc, default_sc
        )
        shortcut2_sc: dict = config.get(layer_entry, {}).get("shortcut2", None)
        p_config[layer_entry]["shortcut2"] = create_a_mat_config(
            shortcut2_sc, default_sc
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
        "shortcut_alpha": config["shortcut_alpha"],
        "proj_dropout": config["proj_dropout"],
        "projector_name": config["projector_name"],
        "disable_projector": config["disable_projector"],
        "importance_beta": config.get("importance_beta", 1.0),
    }
