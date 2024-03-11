#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:3
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --account=su114-gpu

source ~/torch_lora/bin/activate

srun python -m main train --config configs/opt_lora_qqp_350m.toml --num-devices 4 --batch-size 2
