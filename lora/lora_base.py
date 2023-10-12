import torch
import torch.nn as nn


class LoRALayer:
    def __init__(
            self,
            lora_r: int,  # LoRA's rank
            lora_alpha: int,  # LoRA's scaling factor
            lora_dropout_p: float,  # dropout probability of LoRA's input
            merge_weights: bool = True,  # whether A@B should be merged into W during inference
    ):
        # r=0: no LoRA
        # merge_weights=False: train LoRA but doesn't apply to W during inference
        # merged=False: haven't added DeltaW:=A@B to W
        self.r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = (lambda x: x) if lora_dropout_p == 0 else nn.Dropout(p=lora_dropout_p)
        self.merge_weights = merge_weights
        self.merged = False
