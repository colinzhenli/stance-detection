import pytorch_lightning as pl
from torch.utils.data import DataLoader
from MinkowskiEngine.utils import sparse_collate, batched_coordinates
from lib.common_ops.functions import common_ops
from importlib import import_module
import numpy as np
import torch


class DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        dataset_module = import_module('lib.data.dataset')
        self.dataset = getattr(dataset_module, data_cfg.data.dataset)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = self.dataset(self.data_cfg, "train")
            self.val_set = self.dataset(self.data_cfg, "val")
        if stage == "test" or stage is None:
            self.val_set = self.dataset(self.data_cfg, "val")
        if stage == "predict" or stage is None:
            self.test_set = self.dataset(self.data_cfg, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.data_cfg.data.batch_size, shuffle=True, pin_memory=True,
                          collate_fn=sparse_collate_fn, num_workers=self.data_cfg.data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, pin_memory=True, collate_fn=sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, pin_memory=True, collate_fn=sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, pin_memory=True, collate_fn=sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)


def scannet_collate_fn(batch):
    data = {}
    for key in batch[0].keys():
        if key in ['locs', 'locs_scaled', 'feats', 'sem_labels', 'instance_ids', 'num_instance', 'instance_info',
                   'instance_num_point', "instance_semantic_cls", 'gt_proposals_idx', 'gt_proposals_offset', 'instance_bboxes']:
            continue
        if isinstance(batch[0][key], tuple):
            coords, feats = list(zip(*[sample[key] for sample in batch]))
            coords_b = batched_coordinates(coords)
            feats_b = torch.from_numpy(np.concatenate(feats, 0)).float()
            data[key] = (coords_b, feats_b)
        elif isinstance(batch[0][key], np.ndarray):
            data[key] = torch.stack([torch.from_numpy(sample[key]) for sample in batch])
        elif isinstance(batch[0][key], torch.Tensor):
            data[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], dict):
            data[key] = sparse_collate([sample[key] for sample in batch])
        else:
            data[key] = [sample[key] for sample in batch]

    return data


def sparse_collate_fn(batch):
    data = scannet_collate_fn(batch)
    locs = []
    locs_scaled = []
    vert_batch_ids = []
    feats = []
    sem_labels = []
    instance_ids = []
    instance_info = []  # (N, 12)
    instance_num_point = []  # (total_nInst), int
    instance_offsets = [0]
    total_num_inst = 0
    total_points = 0
    instance_cls = []  # (total_nInst), long
    instance_bboxes = []
    gt_proposals_idx = []
    gt_proposals_offset = []
    scan_ids = []

    for i, b in enumerate(batch):
        scan_ids.append(b["scan_id"])
        locs.append(torch.from_numpy(b["locs"]))

        locs_scaled.append(torch.from_numpy(b["locs_scaled"]).int())
        vert_batch_ids.append(torch.full((b["locs_scaled"].shape[0],), fill_value=i, dtype=torch.int16))
        feats.append(torch.from_numpy(b["feats"]))

        if "sem_labels" in b:
            if "gt_proposals_idx" in b:
                gt_proposals_idx_i = b["gt_proposals_idx"]
                gt_proposals_idx_i[:, 0] += total_num_inst
                gt_proposals_idx_i[:, 1] += total_points
                gt_proposals_idx.append(torch.from_numpy(b["gt_proposals_idx"]))
                if gt_proposals_offset != []:
                    gt_proposals_offset_i = b["gt_proposals_offset"]
                    gt_proposals_offset_i += gt_proposals_offset[-1][-1].item()
                    gt_proposals_offset.append(torch.from_numpy(gt_proposals_offset_i[1:]))
                else:
                    gt_proposals_offset.append(torch.from_numpy(b["gt_proposals_offset"]))
            instance_ids_i = b["instance_ids"]
            instance_ids_i[np.where(instance_ids_i != -1)] += total_num_inst
            total_num_inst += b["num_instance"].item()
            total_points += len(instance_ids_i)
            instance_ids.append(torch.from_numpy(instance_ids_i))

            sem_labels.append(torch.from_numpy(b["sem_labels"]))

            instance_info.append(torch.from_numpy(b["instance_info"]))
            instance_num_point.append(torch.from_numpy(b["instance_num_point"]))
            instance_offsets.append(instance_offsets[-1] + b["num_instance"].item())

            instance_cls.extend(b["instance_semantic_cls"])
            instance_bboxes.extend(b["instance_bboxes"])

    tmp_locs_scaled = torch.cat(locs_scaled, dim=0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
    data['scan_ids'] = scan_ids
    data["locs"] = torch.cat(locs, dim=0)  # float (N, 3)
    data["vert_batch_ids"] = torch.cat(vert_batch_ids, dim=0)
    data["feats"] = torch.cat(feats, dim=0)  # .to(torch.float32)            # float (N, C)

    if len(sem_labels) != 0:
        if len(gt_proposals_idx) != 0:
            data["gt_proposals_idx"] = torch.cat(gt_proposals_idx, dim=0).to(torch.int32)
            data["gt_proposals_offset"] = torch.cat(gt_proposals_offset, dim=0).to(torch.int32)
        data["sem_labels"] = torch.cat(sem_labels, dim=0).long()  # long (N,)
        data["instance_ids"] = torch.cat(instance_ids, dim=0).long()  # long, (N,)
        data["instance_info"] = torch.cat(instance_info, dim=0)  # float (total_nInst, 12)
        data["instance_num_point"] = torch.cat(instance_num_point, dim=0)  # (total_nInst)
        data["instance_offsets"] = torch.tensor(instance_offsets, dtype=torch.int32)  # int (B+1)
        data["instance_semantic_cls"] = torch.tensor(instance_cls, dtype=torch.long)  # long (total_nInst)
        data["instance_bboxes"] = torch.tensor(instance_bboxes, dtype=torch.float32)

    # voxelize
    data["voxel_locs"], data["v2p_map"], data["p2v_map"] = common_ops.voxelization_idx(tmp_locs_scaled,
                                                                                       data["vert_batch_ids"],
                                                                                       len(batch),
                                                                                       4)
    return data