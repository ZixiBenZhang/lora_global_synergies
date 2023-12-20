#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu
#SBATCH --time=48:00:00
#SBATCH --account=ec249
#SBATCH --exclusive
#SBATCH --hint=nomultithread

source /work/ec249/ec249/zzhang/torch_lora/bin/activate
export HUGGINGFACE_HUB_CACHE="/work/ec249/ec249/zzhang/hf_cache/hub"
export HF_DATASETS_CACHE="/work/ec249/ec249/zzhang/hf_cache/datasets"
export HF_MODULES_CACHE="/work/ec249/ec249/zzhang/hf_cache/modules"

unset SLURM_NTASKS
env srun python -m main train --config configs/opt_lora_qqp_350m.toml
