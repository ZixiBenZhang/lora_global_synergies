#!/bin/bash

torchrun --nproc_per_node gpu hyperparam_search.py train --config configs/opt_plain_rte.toml
# project_dir="./ags_lr_search", lr=lr_suggested
