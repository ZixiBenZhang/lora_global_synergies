#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:2
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=su114

source ~/torch/bin/activate

srun python -m main train --config ./configs/gemma_lora_mnli_continue.toml \
--load ./ags_output/gemma_lora_classification_mnli_2024-04-22/training_ckpts/last_chkpt-v2.ckpt \
--learning-rate 5e-5
