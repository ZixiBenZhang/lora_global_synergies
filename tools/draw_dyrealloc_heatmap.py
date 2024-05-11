import numpy as np
import toml
import matplotlib.pyplot as plt
import seaborn as sns


def draw_dyrealloc_heatmap(filenames):
    def load_freq(fn):
        with open(fn, "r") as f:
            realloc_frequency = toml.load(f)

        total_num = realloc_frequency.pop("total_reallocation_number")
        data = np.asarray(
            [
                list(layer.values()) + ([0] if i == 0 else [])
                for i, layer in enumerate(realloc_frequency.values())
            ]
        )
        data = data.transpose()
        return dict(
            data=data,
            total_num=total_num,
            layers=len(realloc_frequency.keys()),
            proj_names=["Q-proj", "V-proj", "res-1", "res-2", "cross-in", "cross-cut"],
        )

    if not isinstance(filenames, list):
        data_dict = load_freq(filenames)
        ax = plt.subplot(
            # figsize=(10, 4),
        )
        sns.heatmap(
            data_dict["data"],
            vmin=0,
            vmax=data_dict["total_num"],
            cbar=True,
            square=True,
            yticklabels=data_dict["proj_names"],
            xticklabels=range(data_dict["layers"]),
            cmap="viridis",
            ax=ax,
        )
        ax.set(xlabel="layer", ylabel="frequency")
        plt.rcParams["figure.dpi"] = 3000
        plt.show()
    else:
        data_dicts = [load_freq(fn) for fn in filenames]
        for data_dict in data_dicts:
            print(f"{np.sum(data_dict['data'][:2])} / {np.sum(data_dict['data'])}")
        if len(data_dicts) > 2:
            return
        f = plt.figure(
            figsize=(12, 8),
        )
        sf = f.subfigures(nrows=1, ncols=1)
        axes = sf.subplots(
            nrows=len(data_dicts),
            ncols=1,
        )
        cax = sf.add_axes([0.91, 0.2, 0.03, 0.6])
        # f.tight_layout(rect=[0, 0, .9, 1])
        subplot_names = ["Combined Allocation", "Separated Allocation"]
        for i, (data_dict, ax) in enumerate(zip(data_dicts, axes)):
            sns.heatmap(
                data_dict["data"],
                vmin=0,
                vmax=data_dict["total_num"],
                cbar=(i == 0),
                cbar_ax=None if i > 0 else cax,
                # cbar_kws={"shrink": 0.1},
                square=True,
                linewidths=1,
                yticklabels=data_dict["proj_names"],
                xticklabels=range(data_dict["layers"]),
                cmap="viridis",
                ax=ax,
            )
            ax.set(xlabel="Layer", ylabel="Projection")
            ax.set_title(subplot_names[i])
            # ax.margins(y=0.1)
        plt.subplots_adjust(left=0.15, right=0.85, bottom=0.16, top=0.84)
        plt.rcParams["figure.dpi"] = 3000
        plt.show()


if __name__ == "__main__":
    draw_dyrealloc_heatmap(
        [
            # mrpc
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-03-59.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-03-29.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-02-55.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-03-29.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-03-59.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-04-25.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-05-39.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-06-24.toml",
            # mrpc r=32
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-06-35.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-02-12.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-03-28.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-02-12.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-06-35.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-04-52.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-11-42.toml",
            # "../ags_output/opt_lora_ags_classification_mrpc_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-09-44.toml",
            # alpaca
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_21-10-58.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_21-12-04.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-18/dyrealloc_ckpts/logs/N-0.2/reallocation_frequency_grad-norm_11-29-35.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-18/dyrealloc_ckpts/logs/N-0.2/reallocation_frequency_grad-norm_11-30-05.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_21-09-23.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_21-10-34.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_21-10-58.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_21-12-04.toml",
            # alpaca r=32
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_22-27-34.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_22-29-34.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_22-27-34.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_22-29-34.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_22-30-14.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_22-32-48.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_22-33-25.toml",
            # "../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_22-34-59.toml",
            # sst2
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-17-45.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-17-27.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-17/dyrealloc_ckpts/reallocation_frequency_grad-norm_12-47-03.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-17/dyrealloc_ckpts/reallocation_frequency_grad-norm_12-47-30.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-17-27.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-17-45.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-17-56.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-20-22.toml",
            # sst2 r=32
            # "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-37-46.toml",
            # "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-39-15.toml",
            "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-29-43.toml",
            "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-31-46.toml",
            "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-33-47.toml",
            "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-34-25.toml",
            "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-37-46.toml",
            "../ags_output/opt_lora_ags_classification_sst2_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-39-15.toml",
            # rte
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_20-59-20.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_19-16-49.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-17/dyrealloc_ckpts/reallocation_frequency_grad-norm_15-50-32.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-17/dyrealloc_ckpts/reallocation_frequency_grad-norm_15-50-49.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_17-34-14.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_19-16-49.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_20-59-20.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-04-28/dyrealloc_ckpts/reallocation_frequency_grad-norm_22-43-26.toml",
            # rte r=32
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_19-47-40.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_19-50-48.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_19-44-58.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_19-45-57.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_19-47-40.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_19-48-20.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_19-49-06.toml",
            # "../ags_output/opt_lora_ags_classification_rte_2024-05-01/dyrealloc_ckpts/reallocation_frequency_grad-norm_19-50-48.toml",
        ]
    )
