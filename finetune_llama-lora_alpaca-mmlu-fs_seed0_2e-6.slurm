#!/bin/bash
#
#SBATCH -A MULLINS-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
##SBATCH --ntasks=56
#! How much memory in MB is required _per node_? Not setting this
#! as here will lead to a default amount per task.
#! Setting a larger amount per task increases the number of CPUs.
##SBATCH --mem=
#SBATCH --time=20:00:00

source ~/rds/hpc-work/torch/bin/activate
export HF_HOME=~/rds/hpc-work/lora_global_synergies/cache
huggingface-cli login --token $HUGGINGFACE_TOKEN

python -m main train --config ./configs/llama_lora_alpaca-mmlu-fs.toml --learning-rate 2e-6
