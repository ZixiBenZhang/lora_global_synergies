#!/bin/bash

torchrun --nproc_per_node 2 main.py train \
--config configs/opt_plain.toml \
--load ./ags_output/facebook-opt-125m_classification_rte_2023-10-25/training_ckpts/best_chkpt-v2.ckpt \
--load-type pl
