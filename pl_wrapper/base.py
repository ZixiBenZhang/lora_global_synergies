import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR


class PlWrapperBase(pl.LightningModule):
    def __init__(self):
        super().__init__()

