import os
import torch
import struct
import open3d as o3d
import numpy as np
from tqdm import tqdm
from minsu3d.evaluation.semantic_segmentation import *
from minsu3d.evaluation.instance_segmentation import rle_decode, rle_encode
from minsu3d.evaluation.gravity_aligned_obb import gravity_aligned_mobb
from minsu3d.evaluation.visualiztion import get_directions, o3d_render


def save_prediction(save_path, all_pred_insts, mapping_ids):
    inst_pred_path = os.path.join(save_path, "instance")
    inst_pred_masks_path = os.path.join(inst_pred_path, "predicted_masks")
    os.makedirs(inst_pred_masks_path, exist_ok=True)
    scan_instance_count = {}
    mapping_ids = list(mapping_ids)
    for preds in tqdm(all_pred_insts, desc="==> Saving predictions ..."):
        tmp_info = []
        scan_id = preds[0]["scan_id"]
        for pred in preds:
            if scan_id not in scan_instance_count:

                scan_instance_count[scan_id] = 0
            mapped_label_id = mapping_ids[pred['label_id'] - 1]
            tmp_info.append(
                f"predicted_masks/{scan_id}_{scan_instance_count[scan_id]:03d}.txt {mapped_label_id} {pred['conf']:.4f}\n")
            np.savetxt(
                os.path.join(inst_pred_masks_path, f"{scan_id}_{scan_instance_count[scan_id]:03d}.txt"),
                rle_decode(pred["pred_mask"]), fmt="%d")
            scan_instance_count[scan_id] += 1
        with open(os.path.join(inst_pred_path, f"{scan_id}.txt"), "w") as f:
            for mask_info in tmp_info:
                f.write(mask_info)


def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))
    fid.close()


def read_gt_files_from_disk(data_path):
    pth_file = torch.load(data_path)
    pth_file["xyz"] -= pth_file["xyz"].mean(axis=0)
    return pth_file["xyz"], pth_file["sem_labels"], pth_file["instance_ids"]


def read_pred_files_from_disk(data_path, gt_xyz, mapping_ids):

    sem_label_mapping = {}

    for i, item in enumerate(mapping_ids, 1):
        sem_label_mapping[item] = i
    pred_instances = []

    with open(data_path, "r") as f:
        for line in f:
            mask_relative_path, sem_label, confidence = line.strip().split()
            mask_path = os.path.join(os.path.dirname(data_path), mask_relative_path)
            pred_mask = np.loadtxt(mask_path, dtype=bool)
            pred = {"scan_id": os.path.basename(data_path), "label_id": sem_label_mapping[int(sem_label)], "conf": float(confidence),
                    "pred_mask": rle_encode(pred_mask)}
            pred_xyz = gt_xyz[pred_mask]
            pred["pred_bbox"] = np.concatenate((pred_xyz.min(0), pred_xyz.max(0)))
            pred_instances.append(pred)
    return pred_instances

def read_obb_pred_files_from_disk(data_path, gt_xyz, mapping_ids):

    sem_label_mapping = {}

    for i, item in enumerate(mapping_ids, 1):
        sem_label_mapping[item] = i
    pred_instances = []

    with open(data_path, "r") as f:
        for line in f:
            mask_relative_path, sem_label, confidence = line.strip().split()
            mask_path = os.path.join(os.path.dirname(data_path), mask_relative_path)
            pred_mask = np.loadtxt(mask_path, dtype=bool)
            pred = {"scan_id": os.path.basename(data_path), "label_id": sem_label_mapping[int(sem_label)], "conf": float(confidence),
                    "pred_mask": rle_encode(pred_mask)}
            pred_xyz = gt_xyz[pred_mask]
            center_1 =  (np.amax(pred_xyz, axis=0) + np.amin(pred_xyz, axis=0))/2.0
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(pred_xyz)
            # pred_obb = pred_pcd.get_oriented_bounding_box()
            pred_obb = {}
            obb = gravity_aligned_mobb(pred_pcd, gravity=np.array((0.0, 1.0, 0.0)), align_axis=np.array((0.0, 0.0, -1.0)))
            # R[:, 2] = trans_inv[:, 2] * -1
            pred_obb["obb"] = obb
            pred_obb["pcd"] = pred_pcd
            pred_obb["directions"] = get_directions(obb)
            pred["pred_bbox"] = pred_obb
            pcd_set = {}
            obb_set = {}
            pcd_set[0] = pred_pcd
            obb_set[0] = pred_obb
            o3d_render(pcd_set, obb_set, output="predOBB.png", win_width=640, win_height=480, with_diretions=True)
            pred_instances.append(pred)
    return pred_instances