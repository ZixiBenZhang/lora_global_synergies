#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,3
torchrun --nproc_per_node 2 main.py train --config configs/opt_plain_sst2.toml \
--num-devices 2
# OPT125M + RTE, 40 cpu + 2 gpu: 4min to setup & tokenize data splits before running fit
