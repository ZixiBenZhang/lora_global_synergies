#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=24:00:00
#SBATCH --account=ec249

source /work/ec249/ec249/zzhang/torch_lora/bin/activate
torchrun --nproc_per_node 4 main.py train --config configs/roberta-large_sst2.toml --learning-rate 1e-5