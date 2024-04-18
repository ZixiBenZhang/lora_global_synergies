import numpy as np
import toml
import matplotlib.pyplot as plt
import seaborn as sns


def draw_dyrealloc_heatmap(filenames):

    def load_freq(fn):
        with open(fn, "r") as f:
            realloc_frequency = toml.load(f)

        total_num = realloc_frequency.pop("total_reallocation_number")
        data = np.asarray([
            list(layer.values()) + ([0] if i == 0 else [])
            for i, layer in enumerate(realloc_frequency.values())
        ])
        data = data.transpose()
        return dict(
            data=data,
            total_num=total_num,
            layers=len(realloc_frequency.keys()),
            proj_names=list(list(realloc_frequency.values())[-1].keys()),
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
        plt.show()
    else:
        data_dicts = [load_freq(fn) for fn in filenames]
        # for data_dict in data_dicts:
        #     print(f"{np.sum(data_dict['data'][:2])} / {np.sum(data_dict['data'])}")
        # return
        f = plt.figure(
            figsize=(12, 8),
        )
        sf = f.subfigures(nrows=1, ncols=1)
        axes = sf.subplots(
            nrows=len(data_dicts),
            ncols=1,
        )
        cax = sf.add_axes([.91, .2, .03, .6])
        # f.tight_layout(rect=[0, 0, .9, 1])
        subplot_names = ["Combined Reallocation", "Separated Reallocation"]
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
            ax.set(xlabel="Layer", ylabel="Frequency")
            ax.set_title(subplot_names[i])
            # ax.margins(y=0.1)
        plt.subplots_adjust(left=0.15, right=0.85, bottom=0.16, top=0.84)
        plt.show()


if __name__ == "__main__":
    draw_dyrealloc_heatmap([
        "../ags_output/opt_lora_ags_classification_rte_2024-04-16/dyrealloc_ckpts/reallocation_frequency_grad-norm_23-22_version-0.toml",
        "../ags_output/opt_lora_ags_classification_rte_2024-04-16/dyrealloc_ckpts/reallocation_frequency_grad-norm_23-26_version-0.toml",
    ])
