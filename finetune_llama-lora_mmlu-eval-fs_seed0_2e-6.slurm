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

python -m main test --config ./configs/llama_lora_mmlu-fs-eval.toml \
--load ./ags_output/llama_lora_causal_language_modeling_alpaca-mmlu_2024-07-28/training_ckpts/best_chkpt-fs-epoch=1-16-30.ckpt
