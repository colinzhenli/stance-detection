import torch
import time
import torch.nn as nn
import numpy as np
import torchmetrics
import pytorch_lightning as pl
from minsu3d.evaluation.instance_segmentation import get_gt_instances, rle_encode
from minsu3d.evaluation.object_detection import get_gt_bbox
from minsu3d.common_ops.functions import hais_ops, common_ops
from minsu3d.loss import MaskScoringLoss, ScoreLoss
from minsu3d.loss.utils import get_segmented_scores
from minsu3d.model.module import TinyUnet
from minsu3d.evaluation.semantic_segmentation import *
from minsu3d.model.general_model import GeneralModel, clusters_voxelization, get_batch_offsets


class ObbPred(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        self.conv1 = nn.Conv3d(1, 2, 2, 3, 3)
        self.conv2 = nn.Conv3d(2, 4, 3, 3)
        self.conv3 = nn.Conv3d(4, 8, 2, 3, 3)
        self.conv4 = nn.Conv3d(8, 16, 2, 3, 3)


        self.pool1 = torch.nn.MaxPool3d(2)
        self.pool2 = torch.nn.MaxPool3d(2)
        
        n_sizes = self._get_conv_output(input_shape)


        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.accuracy = torchmetrics.Accuracy()


    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))


        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = nn.ReLU(self.conv1(x))
        x = self.pool1(nn.ReLU(self.conv2(x)))
        x = nn.ReLU(self.conv3(x))
        x = self.pool2(nn.ReLU(self.conv4(x)))
        return x
    
    # will be used during inference
    def forward(self, x):
       x = self._forward_features(x)
       x = x.view(x.size(0), -1)
       x = nn.ReLU(self.fc1(x))
       x = nn.ReLU(self.fc2(x))
       x = nn.log_softmax(self.fc3(x), dim=1)
       
       return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.nll_loss(logits, y)
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.nll_loss(logits, y)


        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.nll_loss(logits, y)
        
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

