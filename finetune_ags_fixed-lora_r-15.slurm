#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:2
#SBATCH --partition=gpu
#SBATCH --time=40:00:00
#SBATCH --account=su114

source ~/torch/bin/activate

srun python -m main train --config ./configs/opt_ags_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-2.toml \
--shortcut-config ./configs/shortcut/ags_networkwise_r-15.toml \
--learning-rate 3e-4
