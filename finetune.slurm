#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu
#SBATCH --time=48:00:00
#SBATCH --account=ec249

source /work/ec249/ec249/zzhang/torch_lora/bin/activate
export HF_DATASETS_CACHE="/work/ec249/ec249/zzhang/saved_hf_datasets"
python -m main train --config configs/opt_plain_qqp_350m.toml \
--dataset-saved-path /work/ec249/ec249/zzhang/saved_hf_datasets
