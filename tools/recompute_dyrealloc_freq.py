import toml


def recompute_freq(path, frequency_save_path):
    with open(path, "r") as f:
        realloc_hist = toml.load(f)

    turned_on_freq: dict[str, int | dict[str, int]] = {
        "total_reallocation_number": len(realloc_hist)
    }
    # format: {dyrealloc_{i}: {epoch: epoch, step: step, turn_on: turn_on[]}
    for i, reallocation in realloc_hist.items():
        turn_on = reallocation["turn_on"]
        for lora_module in turn_on:
            layer_id, proj_name, _, turned_on = lora_module
            if turned_on == "True":
                if f"layer_{layer_id}" not in turned_on_freq:
                    turned_on_freq[f"layer_{layer_id}"] = {}
                if proj_name not in turned_on_freq[f"layer_{layer_id}"]:
                    turned_on_freq[f"layer_{layer_id}"][proj_name] = 1
                else:
                    turned_on_freq[f"layer_{layer_id}"][proj_name] += 1
            else:
                if f"layer_{layer_id}" not in turned_on_freq:
                    turned_on_freq[f"layer_{layer_id}"] = {}
                if proj_name not in turned_on_freq[f"layer_{layer_id}"]:
                    turned_on_freq[f"layer_{layer_id}"][proj_name] = 0

    with open(frequency_save_path, "w+") as fout:
        toml.dump(turned_on_freq, fout)


if __name__ == "__main__":
    recompute_freq(
        "../ags_output/llama_lora_ags_causal_language_modeling_wikitext-2-v1_2024-07-30/dyrealloc_ckpts/reallocation_history_grad-norm_05-44-05.toml",
        "../ags_output/llama_lora_ags_causal_language_modeling_wikitext-2-v1_2024-07-30/dyrealloc_ckpts/reallocation_frequency_grad-norm_05-44-05_recomputed.toml",
    )
