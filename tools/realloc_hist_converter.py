import argparse


def convert_heterogeneous_realloc_hist(path):
    with open(path, "r") as f:
        hist = f.readlines()

    new_hist = []

    for line in hist:
        if "turn_on" in line:
            new_line = ""
            cnt = 0
            prev = ""
            for i, c in enumerate(line):
                if c == "[":
                    cnt += 1
                elif c == "]":
                    cnt -= 1
                elif cnt == 2 and prev == " " and c != '"':
                    new_line = new_line + '"'
                elif cnt == 2 and c == "," and prev != '"':
                    new_line = new_line + '"'

                new_line = new_line + c
                prev = c
            line = new_line

        new_hist.append(line)

    with open(path[:-5] + "_converted" + path[-5:], "w+") as f:
        f.writelines(new_hist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", add_help=False)
    parser.add_argument(
        "--path",
        dest="path",
        default="../ags_output/opt_lora_ags_causal_language_modeling_alpaca_2024-05-01/dyrealloc_ckpts/reallocation_history_grad-norm_22-27-34.toml"
    )

    args = parser.parse_args()
    convert_heterogeneous_realloc_hist(args.path)
