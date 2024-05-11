import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator, LogFormatter
import numpy as np


def lora_r_curve(filename):
    res = {}
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                r = np.asarray(row[1:], dtype=int)
            else:
                res[row[0]] = np.asarray(row[1:], dtype=float) - float(row[4])

    f, ax = plt.subplots(1, 1)
    f: plt.Figure
    ax: plt.Axes
    f.set_size_inches(10, 4)
    plt.rcParams["figure.dpi"] = 3000
    x = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    c = ["tab:blue", "tab:green", "tab:orange"]
    x_delta = [-0.2, 0.0, 0.2]
    for i, (metric, y) in enumerate(res.items()):
        ax.plot(x + x_delta[i], y, "-o", color=c[i], label=metric)
        ax.bar(x + x_delta[i], y, width=0.2, color=c[i], alpha=0.5)
        for xx, yy in zip(x, y):
            ax.annotate(
                round(yy, 1),
                xy=(xx + x_delta[i] - 0.075, yy + 0.3 if yy >= 0 else yy - 0.45),
            )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=4,
        fancybox=True,
        shadow=True,
    )
    ax.set_xlabel("Rank")
    ax.set_ylabel("ACC difference")
    ax.set_ybound(-4, 4)
    ax.set_xticks(x, r)
    # ax.set_xscale("log", base=2)
    # ax.xaxis.set_major_formatter(LogFormatter(base=2))
    ax.grid(True, "major", axis="y", linestyle="dotted")
    plt.show()


if __name__ == "__main__":
    lora_r_curve("../lora-r-result.csv")
