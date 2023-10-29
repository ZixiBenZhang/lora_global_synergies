#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,3
torchrun --nproc_per_node gpu main.py train --config configs/opt_plain_mnli.toml \
--num-devices NUM \
--load PATH \
--load-type pl \
--resume-training
# OPT125M + RTE, 40 cpu + 2 gpu: 4min to setup & tokenize data splits before running fit
