import tomllib
import matplotlib.pyplot as plt


def draw_dyrealloc_trend(filenames):
    f, ax = plt.subplots(1, 1)
    f: plt.Figure
    f.set_size_inches(10, 5)
    plt.rcParams["figure.dpi"] = 3000

    for filename in filenames:
        with open(filename, "rb") as f:
            realloc_history = tomllib.load(f)

        res = []
        rounds = 50 if "sst2" in filename else 100
        for i in range(rounds):
            enable = realloc_history[f"dyrealloc_{i}"]["turn_on"]
            cnt = 0
            total = 0
            for module in enable:
                if module[3] is True:
                    total += 1
                if "_proj" in module[1] and module[3] is True:
                    cnt += 1
            res.append(cnt / total)

        ax.plot(res)
        ax.set_xlabel("search #")
        ax.set_ylabel("LoRA freq")
        # ax.set_ylim(top=100.0)
        # ax.grid(True, "major", linestyle="dotted")
    plt.show()


if __name__ == "__main__":
    draw_dyrealloc_trend(
        [
            # mrpc
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-03-59.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-03-29.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-02-55.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-03-29.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-03-59.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-04-25.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-05-39.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-06-24.toml",
            # mrpc r=32
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-06-35.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-02-12.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-03-28.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-02-12.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-06-35.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-04-52.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-11-42.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-09-44.toml",
            # alpaca
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_21-10-58.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_21-12-04.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-18/dyrealloc_ckpts/logs/N-0.2/reallocation_history_grad-norm_11-29-35.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-18/dyrealloc_ckpts/logs/N-0.2/reallocation_history_grad-norm_11-30-05.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_21-09-23.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_21-10-34.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_21-10-58.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_21-12-04.toml",
            # alpaca r=32
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_22-27-34.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_22-29-34.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_22-27-34.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_22-29-34.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_22-30-14.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_22-32-48.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_22-33-25.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_22-34-59.toml",
            # sst2
            "../ags_output/opt_lora_ags_classification_sst2_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-17-45.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-17-27.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-17/dyrealloc_ckpts/reallocation_history_grad-norm_12-47-03.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-17/dyrealloc_ckpts/reallocation_history_grad-norm_12-47-30.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-17-27.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-17-45.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-17-56.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-20-22.toml",
            # sst2 r=32
            "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-37-46.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-39-15.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-29-43.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-31-46.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-33-47.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-34-25.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-37-46.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_17-39-15.toml",
            # rte
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_20-59-20.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_19-16-49.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-17/dyrealloc_ckpts/reallocation_history_grad-norm_15-50-32.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-17/dyrealloc_ckpts/reallocation_history_grad-norm_15-50-49.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_17-34-14.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_19-16-49.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_20-59-20.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-28/dyrealloc_ckpts/reallocation_history_grad-norm_22-43-26.toml",
            # rte r=32
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_19-47-40.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_19-50-48.toml",
            #
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_19-44-58.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_19-45-57.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_19-47-40.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_19-48-20.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_19-49-06.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_19-50-48.toml",
        ]
    )
