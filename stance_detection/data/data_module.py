from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.dataset = getattr(import_module('stance_detection.data.dataset'), data_cfg.data.dataset)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = self.dataset(self.data_cfg, "train")
            self.val_set = self.dataset(self.data_cfg, "val")
        if stage == "test" or stage is None:
            self.val_set = self.dataset(self.data_cfg, self.data_cfg.model.inference.split)
        if stage == "predict" or stage is None:
            self.test_set = self.dataset(self.data_cfg, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.data_cfg.data.batch_size, shuffle=True, pin_memory=True, drop_last=True, 
                          num_workers=self.data_cfg.data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, pin_memory=True,
                          num_workers=self.data_cfg.data.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.val_set, batch_size=1, pin_memory=True, 
    #                       num_workers=self.data_cfg.data.num_workers)

    # def predict_dataloader(self):
    #     return DataLoader(self.test_set, batch_size=1, pin_memory=True, 
    #                       num_workers=self.data_cfg.data.num_workers)


# def sparse_collate_fn(batch):
#     data = {}
#     stance_ids = []
#     external_feats = []
#     encodings = []
#     stat_feats = []


#     for i, b in enumerate(batch):
#         stance_ids.append(torch.tensor(b["stance_id"]).to(dtype=torch.int))
#         external_feats.append(torch.from_numpy(b["external_feat"]).to(dtype=torch.float32))
#         stat_feats.append(torch.from_numpy(b["stat_feat"]).to(dtype=torch.float32))
#         encodings.append(torch.from_numpy(b["encoding"]).to(dtype=torch.int))
#     data["stance_id"] = torch.tensor(stance_ids)
#     data["external_feat"] = torch.tensor(external_feats)
#     data["encoding"] = torch.tensor(encodings)
#     data["stat_feat"] = torch.tensor(stat_feats)

#     return data
