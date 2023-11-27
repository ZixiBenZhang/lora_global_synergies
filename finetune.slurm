#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu
#SBATCH --time=48:00:00
#SBATCH --account=ec249
#SBATCH --exclusive

source /work/ec249/ec249/zzhang/torch_lora/bin/activate
export HUGGINGFACE_HUB_CACHE="/work/ec249/ec249/zzhang/hf_cache/hub"
export HF_DATASETS_CACHE="/work/ec249/ec249/zzhang/hf_cache/datasets"
export HF_MODULES_CACHE="/work/ec249/ec249/zzhang/hf_cache/modules"
python -m main train --config configs/opt_plain_qqp_350m.toml
