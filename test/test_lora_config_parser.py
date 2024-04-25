from lora.lora_config_parser import parse_lora_config


def test_parse_lora_config_networkwise():
    layers = 2
    heads = 1
    config = {
        "granularity": "network",
        "default": {
            "r": 3,
            "lora_alpha": 5,
            "lora_dropout": 0.0,
            "adapter_name": "test",
            "disable_adapter": False,
        },
        "all_layers": {
            "q_proj": {
                "r": 6,
                "lora_alpha": 10,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
                "importance_alpha": 0.4,
            },
            "w1": {
                "r": 9,
                "lora_alpha": 15,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
            },

        },
    }

    res = parse_lora_config(config, layers, heads)

    assert res == {
        "default": {
            "r": 3,
            "lora_alpha": 5,
            "lora_dropout": 0.0,
            "adapter_name": "test",
            "disable_adapter": False,
            "importance_alpha": 1.0,
        },
        "model_layer_0": {
            "q_proj": {
                "head_0": {
                    "r": 6,
                    "lora_alpha": 10,
                    "lora_dropout": 0.0,
                    "adapter_name": "test",
                    "disable_adapter": False,
                    "importance_alpha": 0.4,
                },
            },
            "k_proj": {
                "head_0": {
                    "r": 3,
                    "lora_alpha": 5,
                    "lora_dropout": 0.0,
                    "adapter_name": "test",
                    "disable_adapter": False,
                    "importance_alpha": 1.0,
                },
            },
            "v_proj": {
                "head_0": {
                    "r": 3,
                    "lora_alpha": 5,
                    "lora_dropout": 0.0,
                    "adapter_name": "test",
                    "disable_adapter": False,
                    "importance_alpha": 1.0,
                },
            },
            "o_proj": {
                "r": 3,
                "lora_alpha": 5,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
                "importance_alpha": 1.0,
            },
            "w1": {
                "r": 9,
                "lora_alpha": 15,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
                "importance_alpha": 1.0,
            },
            "w2": {
                "r": 3,
                "lora_alpha": 5,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
                "importance_alpha": 1.0,
            },
        },
        "model_layer_1": {
            "q_proj": {
                "head_0": {
                    "r": 6,
                    "lora_alpha": 10,
                    "lora_dropout": 0.0,
                    "adapter_name": "test",
                    "disable_adapter": False,
                    "importance_alpha": 0.4,
                },
            },
            "k_proj": {
                "head_0": {
                    "r": 3,
                    "lora_alpha": 5,
                    "lora_dropout": 0.0,
                    "adapter_name": "test",
                    "disable_adapter": False,
                    "importance_alpha": 1.0,
                },
            },
            "v_proj": {
                "head_0": {
                    "r": 3,
                    "lora_alpha": 5,
                    "lora_dropout": 0.0,
                    "adapter_name": "test",
                    "disable_adapter": False,
                    "importance_alpha": 1.0,
                },
            },
            "o_proj": {
                "r": 3,
                "lora_alpha": 5,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
                "importance_alpha": 1.0,
            },
            "w1": {
                "r": 9,
                "lora_alpha": 15,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
                "importance_alpha": 1.0,
            },
            "w2": {
                "r": 3,
                "lora_alpha": 5,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
                "importance_alpha": 1.0,
            },
        },
    }


def test_parse_lora_config_layerwise():
    layers = 2
    heads = 1
    config = {
        "granularity": "layer",
        "default": {
            "r": 3,
            "lora_alpha": 5,
            "lora_dropout": 0.0,
            "adapter_name": "test",
            "disable_adapter": False,
        },
        "model_layer_0": {
            "q_proj": {
                "r": 6,
                "lora_alpha": 10,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
                "importance_alpha": 0.4,
            },
            "w1": {
                "r": 9,
                "lora_alpha": 15,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
            },

        },
    }

    res = parse_lora_config(config, layers, heads)

    assert res == {
        "default": {
            "r": 3,
            "lora_alpha": 5,
            "lora_dropout": 0.0,
            "adapter_name": "test",
            "disable_adapter": False,
            "importance_alpha": 1.0,
        },
        "model_layer_0": {
            "q_proj": {
                "head_0": {
                    "r": 6,
                    "lora_alpha": 10,
                    "lora_dropout": 0.0,
                    "adapter_name": "test",
                    "disable_adapter": False,
                    "importance_alpha": 0.4,
                },
            },
            "k_proj": {
                "head_0": {
                    "r": 3,
                    "lora_alpha": 5,
                    "lora_dropout": 0.0,
                    "adapter_name": "test",
                    "disable_adapter": False,
                    "importance_alpha": 1.0,
                },
            },
            "v_proj": {
                "head_0": {
                    "r": 3,
                    "lora_alpha": 5,
                    "lora_dropout": 0.0,
                    "adapter_name": "test",
                    "disable_adapter": False,
                    "importance_alpha": 1.0,
                },
            },
            "o_proj": {
                "r": 3,
                "lora_alpha": 5,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
                "importance_alpha": 1.0,
            },
            "w1": {
                "r": 9,
                "lora_alpha": 15,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
                "importance_alpha": 1.0,
            },
            "w2": {
                "r": 3,
                "lora_alpha": 5,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
                "importance_alpha": 1.0,
            },
        },
        "model_layer_1": {
            "q_proj": {
                "head_0": {
                    "r": 3,
                    "lora_alpha": 5,
                    "lora_dropout": 0.0,
                    "adapter_name": "test",
                    "disable_adapter": False,
                    "importance_alpha": 1.0,
                },
            },
            "k_proj": {
                "head_0": {
                    "r": 3,
                    "lora_alpha": 5,
                    "lora_dropout": 0.0,
                    "adapter_name": "test",
                    "disable_adapter": False,
                    "importance_alpha": 1.0,
                },
            },
            "v_proj": {
                "head_0": {
                    "r": 3,
                    "lora_alpha": 5,
                    "lora_dropout": 0.0,
                    "adapter_name": "test",
                    "disable_adapter": False,
                    "importance_alpha": 1.0,
                },
            },
            "o_proj": {
                "r": 3,
                "lora_alpha": 5,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
                "importance_alpha": 1.0,
            },
            "w1": {
                "r": 3,
                "lora_alpha": 5,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
                "importance_alpha": 1.0,
            },
            "w2": {
                "r": 3,
                "lora_alpha": 5,
                "lora_dropout": 0.0,
                "adapter_name": "test",
                "disable_adapter": False,
                "importance_alpha": 1.0,
            },
        },
    }
