#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:2
#SBATCH --partition=gpu
#SBATCH --time=40:00:00
#SBATCH --account=su114

source ~/torch/bin/activate

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--lora-config ./configs/lora/lora_networkwise_abla_q_r-8.toml

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--lora-config ./configs/lora/lora_networkwise_abla_k_r-8.toml

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--lora-config ./configs/lora/lora_networkwise_abla_v_r-8.toml

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--lora-config ./configs/lora/lora_networkwise_abla_o_r-8.toml

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--lora-config ./configs/lora/lora_networkwise_abla_w1_r-8.toml

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--lora-config ./configs/lora/lora_networkwise_abla_w2_r-8.toml

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--lora-config ./configs/lora/lora_networkwise_abla_q-k_r-4.toml

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--lora-config ./configs/lora/lora_networkwise_abla_q-v_r-4.toml

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--lora-config ./configs/lora/lora_networkwise_abla_w1-w2_r-4.toml

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--lora-config ./configs/lora/lora_networkwise_abla_q-k-v-o_r-2.toml

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--lora-config ./configs/lora/lora_networkwise_abla_q-v-w1-w2_r-2.toml
