#!/bin/bash

torchrun --nproc_per_node gpu main.py train --config configs/opt_plain_mnli.toml
# OPT125M + RTE, 40 cpu + 2 gpu: 4min to setup & tokenize data splits before running fit
