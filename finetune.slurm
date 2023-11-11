#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu
#SBATCH --time=24:00:00
#SBATCH --account=ec249

source /work/ec249/ec249/zzhang/torch_lora/bin/activate
export HF_DATASETS_CACHE="/work/ec249/ec249/zzhang/saved_hf_datasets"
torchrun --nproc_per_node 4 main.py train --config configs/roberta-large_sst2.toml \
--learning-rate 1e-5 \
--dataset-saved-path /work/ec249/ec249/zzhang/saved_hf_datasets