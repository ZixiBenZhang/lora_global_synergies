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

srun python -m main train --config ./configs/opt_lora_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-1.toml \
--learning-rate 4e-4

srun python -m main train --config ./configs/opt_lora_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-2.toml \
--learning-rate 4e-4

srun python -m main train --config ./configs/opt_lora_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-3.toml \
--learning-rate 4e-4

srun python -m main train --config ./configs/opt_lora_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-4.toml \
--learning-rate 4e-4

srun python -m main train --config ./configs/opt_lora_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-5.toml \
--learning-rate 4e-4

srun python -m main train --config ./configs/opt_lora_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-6.toml \
--learning-rate 4e-4

srun python -m main train --config ./configs/opt_lora_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-7.toml \
--learning-rate 4e-4

srun python -m main train --config ./configs/opt_lora_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-8.toml \
--learning-rate 4e-4

srun python -m main train --config ./configs/opt_lora_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-9.toml \
--learning-rate 4e-4

srun python -m main train --config ./configs/opt_lora_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-10.toml \
--learning-rate 4e-4

srun python -m main train --config ./configs/opt_lora_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-12.toml \
--learning-rate 4e-4

srun python -m main train --config ./configs/opt_lora_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-24.toml \
--learning-rate 4e-4

srun python -m main train --config ./configs/opt_lora_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-32.toml \
--learning-rate 4e-4

srun python -m main train --config ./configs/opt_lora_rte_350m.toml \
--num-devices 2 --batch-size 4 \
--lora-config ./configs/lora/lora_networkwise_r-96.toml \
--learning-rate 4e-4
