import toml


def one_third_layers_lora(mode, num_layers=24, model_name="opt350m"):
    assert mode in [1, 2, 3]
    r = 3 * 8
    adapter_name = f"ags-layer-top-{model_name}"
    # Only Q, V LoRA activated

    data = {
        "granularity": "layer",
        "default": {
            "r": 0,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "adapter_name": adapter_name,
            "init_lora_weghts": True,
            "fan_in_fan_out": False,
            "disable_adapter": False,
        }
    }
    for i in range(num_layers):
        if i not in range(round((mode - 1) / 3 * num_layers), round(mode / 3 * num_layers)):
            continue
        data = {
            **data,
            f"model_layer_{i}": {
                "q_proj": {
                    "r": r,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "adapter_name": adapter_name,
                    "disable_adapter": False,
                },
                "k_proj": {
                    "r": 0,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "adapter_name": adapter_name,
                    "disable_adapter": True,
                },
                "v_proj": {
                    "r": r,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "adapter_name": adapter_name,
                    "disable_adapter": False,
                },
                "o_proj": {
                    "r": 0,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "adapter_name": adapter_name,
                    "disable_adapter": True,
                },
                "w1": {
                    "r": 0,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "adapter_name": adapter_name,
                    "disable_adapter": True,
                },
                "w2": {
                    "r": 0,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "adapter_name": adapter_name,
                    "disable_adapter": True,
                },
            }
        }

    with open(f"../configs/lora/lora_layerwise{mode}.toml", "w+") as f:
        toml.dump(data, f)


if __name__ == "__main__":
    one_third_layers_lora(3)
