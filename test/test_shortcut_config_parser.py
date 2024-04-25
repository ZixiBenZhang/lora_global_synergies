from projectors.shortcut_config_parser import parse_shortcut_config


def test_parse_shortcut_config_networkwise():
    layers = 2
    config = {
        "granularity": "network",
        "default": {
            "r": 3,
            "shortcut_alpha": 5,
            "proj_dropout": 0.0,
            "projector_name": "test",
            "disable_projector": False,
        },
        "all_layers": {
            "residual1": {
                "r": 6,
                "shortcut_alpha": 10,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 0.4,
            },
            "shortcut2": {
                "r": 9,
                "shortcut_alpha": 15,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
            },
        },
    }

    res = parse_shortcut_config(config, layers)

    assert res == {
        "default": {
            "r": 3,
            "shortcut_alpha": 5,
            "proj_dropout": 0.0,
            "projector_name": "test",
            "disable_projector": False,
            "importance_beta": 1.0,
        },
        "model_layer_0": {
            "residual1": {
                "r": 6,
                "shortcut_alpha": 10,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 0.4,
            },
            "residual2": {
                "r": 3,
                "shortcut_alpha": 5,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 1.0,
            },
            "shortcut1": {
                "r": 3,
                "shortcut_alpha": 5,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 1.0,
            },
            "shortcut2": {
                "r": 9,
                "shortcut_alpha": 15,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 1.0,
            },
        },
        "model_layer_1": {
            "residual1": {
                "r": 6,
                "shortcut_alpha": 10,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 0.4,
            },
            "residual2": {
                "r": 3,
                "shortcut_alpha": 5,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 1.0,
            },
            "shortcut1": {
                "r": 3,
                "shortcut_alpha": 5,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 1.0,
            },
            "shortcut2": {
                "r": 9,
                "shortcut_alpha": 15,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 1.0,
            },
        },
    }


def test_parse_shortcut_config_layerwise():
    layers = 2
    config = {
        "granularity": "layer",
        "default": {
            "r": 3,
            "shortcut_alpha": 5,
            "proj_dropout": 0.0,
            "projector_name": "test",
            "disable_projector": False,
        },
        "model_layer_0": {
            "residual1": {
                "r": 6,
                "shortcut_alpha": 10,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 0.4,
            },
            "shortcut2": {
                "r": 9,
                "shortcut_alpha": 15,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
            },
        },
    }

    res = parse_shortcut_config(config, layers)

    assert res == {
        "default": {
            "r": 3,
            "shortcut_alpha": 5,
            "proj_dropout": 0.0,
            "projector_name": "test",
            "disable_projector": False,
            "importance_beta": 1.0,
        },
        "model_layer_0": {
            "residual1": {
                "r": 6,
                "shortcut_alpha": 10,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 0.4,
            },
            "residual2": {
                "r": 3,
                "shortcut_alpha": 5,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 1.0,
            },
            "shortcut1": {
                "r": 3,
                "shortcut_alpha": 5,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 1.0,
            },
            "shortcut2": {
                "r": 9,
                "shortcut_alpha": 15,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 1.0,
            },
        },
        "model_layer_1": {
            "residual1": {
                "r": 3,
                "shortcut_alpha": 5,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 1.0,
            },
            "residual2": {
                "r": 3,
                "shortcut_alpha": 5,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 1.0,
            },
            "shortcut1": {
                "r": 3,
                "shortcut_alpha": 5,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 1.0,
            },
            "shortcut2": {
                "r": 3,
                "shortcut_alpha": 5,
                "proj_dropout": 0.0,
                "projector_name": "test",
                "disable_projector": False,
                "importance_beta": 1.0,
            },
        },
    }
