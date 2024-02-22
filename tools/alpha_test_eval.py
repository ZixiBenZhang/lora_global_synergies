import toml


def analyse():
    with open(
        "../ags_output/opt_lora_classification_rte_2024-02-01/checkpoints/logs_test/importance_22-40.toml",
        "r",
    ) as f:
        data = toml.load(f)
    for mat, res in data.items():
        if res["acc_reduction"] != 0:
            print(mat, res)


if __name__ == "__main__":
    analyse()
