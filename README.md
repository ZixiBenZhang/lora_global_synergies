# Unlocking the Global Synergies in Low-Rank Adapters

__Heterogeneous LoRA configuration search (HeteroLoRA)__ is a lightweight autonomous LoRA configuration 
search method. The search space is extended with shortcut connections. HeteroLoRA decides enable/disable 
of every LoRA and shortcut module statically (i.e. before the training) or dynamically (i.e. during the training).

This repository is a framework for LoRA training and HeteroLoRA.

## Getting started

Download dependencies listed in `python-requirements.txt`
which can be installed with pip:

> pip install --user -r python-requirements.txt

Four actions are supported: training, testing, static HeteroLoRA search, dynamic HeteroLoRA training.

### Training

You can fine-tune e.g. pre-trained OPT-350M on RTE by

> python -m main train \
> --model facebook/opt-350m \
> --dataset rte \
> --task classification \
> --optimizer adamw \
> --max-epochs 10 \
> --learning-rate 1e-5 \
> --batch-size 16 \
> --pretrained

Supported datasets are implemented in `dataset/`, which can be extended to support new datasets.

LoRA-adapted OPT, RoBERTa, and Gemma are supported. You can fine-tune them by

> python -m main train \
> --model opt_lora \
> --dataset rte \
> --task classification \
> --load facebook/opt-350m \
> --load-type hf \
> --optimizer adamw \
> --max-epochs 20 \
> --learning-rate 1e-4 \
> --batch-size 8 \
> --pretrained \
> --lora-config ./configs/lora/lora_networkwise.toml

where `--lora-config` takes a LoRA configuration file in TOML.

Shortcut-adapted OPT is supported. You can fine-tune it by

> python -m main train \
> --model opt_lora_ags \
> --dataset rte \
> --task classification \
> --load facebook/opt-350m \
> --load-type hf \
> --optimizer adamw \
> --max-epochs 20 \
> --learning-rate 1e-4 \
> --batch-size 8 \
> --pretrained \
> --lora-config ./configs/lora/lora_networkwise.toml \
> --shortcut-config ./configs/shortcut/ags_networkwise.toml

where ``--shortcut-config`` takes a shortcut configuration file in TOML.

You can extend LoRA and shortcut connections to new models in `model/`.

You can also pass in training configurations by TOML file using `--config`. 

### Testing

You can test e.g. pre-trained OPT-350M on RTE by

> python -m main test \
> --model facebook/opt-350m \
> --dataset rte \
> --task classification \
> --batch-size 16 \
> --pretrained

### Static HeteroLoRA

Five zero-cost saliency heuristics are supported: CONSTANT, SNIP, SYNFLOW, GRAD-NORM, ALPHA-TEST.

You can compute the saliency scores of e.g. GRAD-NORM with OPT-350M LoRA on RTE by

> python -m main imp-test \
> --model opt_lora \
> --dataset rte \
> --task classification \
> --load facebook/opt-350m \
> --load-type hf \
> --batch-size 8 \
> --pretrained \
> --lora-config ./configs/lora/lora_networkwise.toml \
> --importance-test-name grad_norm \
> --imp-limit-test-batches 32 

and then generate a heterogeneous LoRA configuration accordingly using `tools/realloc_lora_rank.py`.

### Dynamic HeteroLoRA

You can train e.g. shortcut-adapted OPT-350M on RTE with dynamic HeteroLoRA by

> python -m main train-dyrealloc \
> --model opt_lora_ags \
> --dataset rte \
> --task classification \
> --load facebook/opt-350m \
> --load-type hf \
> --optimizer adamw \
> --max-epochs 20 \
> --learning-rate 1e-4 \
> --batch-size 8 \
> --pretrained \
> --strategy ddp_find_unused_parameters_true \
> --lora-config ./configs/lora/lora_networkwise.toml \
> --shortcut-config ./configs/shortcut/ags_networkwise.toml \
> --importance-test-name grad_norm \
> --imp-limit-test-batches 32 \
> --realloc-N 0.2 \
> --turn-on-percentile 0.25 \
> --dyrealloc-ags-mode combined

which will run heterogeneous LoRA configuration search per 0.2 = 1/5 epoch,
based on GRAD-NORM evaluated on 32 training batches and ranking LoRA and shortcut modules together,
with each time 25% of the modules enabled.
