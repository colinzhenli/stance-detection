import torch
import time
import torch.nn as nn
import numpy as np
import torchmetrics
import pytorch_lightning as pl


class Stat_Feat(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.relu = nn.functional.relu
        self.fc1 = nn.Linear(10001, 500)
        self.fc2 = nn.Linear(500, 50)
        self.dropout = nn.Dropout(p=0.4)


    def forward(self, stat_feat):
        x = self.dropout(stat_feat)
        x = self.relu(self.fc1(x)).clone()
        x = self.relu(self.fc2(x)).clone()
        return x
