from pyexpat import features
import torch
import time
import torch.nn as nn
import numpy as np
import torchmetrics
import pytorch_lightning as pl
from stance_detection.model.module import External_feat
from stance_detection.util.score import report_score, LABELS, score_submission, print_confusion_matrix


class Stance(pl.LightningModule):
    def __init__(self, model, data, optimizer, lr_decay, inference=None):
        super().__init__()
        self.save_hyperparameters()
        #def external feature
        self.external_feat = External_feat(self.hparams.data.feature_file)

    # def configure_optimizers(self):
    #     return init_optimizer(parameters=self.parameters(), **self.hparams.optimizer)

        # log hyperparameters
        
        self.conv1 = nn.Conv1d(44, 32, 2, stride=2)
        self.relu = nn.functional.relu6

        self.pool1 = torch.nn.MaxPool1d(2)
        self.pool2 = torch.nn.MaxPool1d(2)
        # n_sizes = self._get_conv_output()
        # self.fc0 = nn.Linear(n_sizes, 2048)
        self.fc1 = nn.Linear(44, 128)
        self.fc2 = nn.Linear(128, 4)
        self.accuracy = torchmetrics.Accuracy()
        self.log_softmax = nn.functional.log_softmax
        self.nll_loss = nn.functional.nll_loss


    # returns the size of the output tensor going into Linear layer from the conv block.
    # def _get_conv_output(self):
    #     batch_size = self.hparams.data.batch_size
    #     input = torch.autograd.Variable(torch.rand(batch_size, 44, 1))
    #     output = self.pool1(self.conv1(input)).clone()
    #     n_size = input.data.view(batch_size, -1).size(1)
    #     return n_size
        
    # will be used during inference
    def forward(self, data_dict):
        output_dict = {}
        x = self.external_feat(data_dict)
        x = torch.from_numpy(x).float().to(self.device)
        x = x.view(-1, x.size(0))
        # x = self.pool1(self.conv1(x)).clone()
        x = self.relu(self.fc1(x)).clone()
        x = self.log_softmax(self.fc2(x))
        output_dict["stance_scores"] = x        
        return output_dict

    def _loss(self, data_dict, output_dict):
        loss = self.nll_loss(output_dict["stance_scores"], data_dict["stance_id"])
        return loss

    def training_step(self, data_dict, idx):
        output_dict = self.forward(data_dict)
        loss = self._loss(data_dict, output_dict)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.hparams.data.batch_size)
        # training metrics
        preds = torch.argmax(output_dict["stance_scores"], dim=1)
        acc = self.accuracy(preds, data_dict["stance_id"])
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=self.hparams.data.batch_size)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, batch_size=self.hparams.data.batch_size)    
        return loss


    def validation_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self.forward(data_dict)
        loss = self._loss(data_dict, output_dict)

        # log losses
        self.log("val/loss", loss, prog_bar=True, on_step=False,
                 on_epoch=True, sync_dist=True, batch_size=1)

        # log prediction
        stance_predictions = torch.argmax(output_dict["stance_scores"])

        if self.current_epoch > self.hparams.model.prepare_epochs:
            pred_stance = LABELS[stance_predictions]
            gt_stance = data_dict["stance"]
            return pred_stance, gt_stance

    def validation_epoch_end(self, outputs):
        # evaluate instance predictions
        if self.current_epoch > self.hparams.model.prepare_epochs:
            all_pred_stances = []
            all_gt_stances = []
            for pred_obb, gt_obb in outputs:
                all_pred_stances.append(pred_obb)
                all_gt_stances.append(gt_obb)
            score, cm = score_submission(all_gt_stances, all_pred_stances)
            max_score, _ = score_submission(all_gt_stances, all_gt_stances)
            final_score = score/max_score        
            self.log("val_eval/Scores", final_score, sync_dist=True)
            print_confusion_matrix(cm)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
        return optimizer
