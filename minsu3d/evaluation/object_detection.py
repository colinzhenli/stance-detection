# Modified from https://github.com/facebookresearch/votenet/blob/main/utils/eval_det.py
from re import U
from statistics import mean
import numpy as np
import open3d as o3d
# from open3d.j_visualizer  import JVisualizer
import struct
import torch
import random
from pytorch3d.ops import box3d_overlap
from .visualiztion import custom_draw_geometry_with_camera_trajectory
from .visualiztion import o3d_render, get_vertices, get_directions
from .gravity_aligned_obb import gravity_aligned_mobb

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap




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

def inside_test(points , cube3d):
    """
    cube3d  =  numpy array of the shape (8,3) with coordinates in the clockwise order. first the bottom plane is considered then the top one.
    points = array of points with shape (N, 3).

    Returns the indices of the points array which are outside the cube3d
    """
    b1,b2,b3,b4,t1,t2,t3,t4 = cube3d

    dir1 = (t1-b1)
    size1 = np.linalg.norm(dir1)
    dir1 = dir1 / size1

    dir2 = (b2-b1)
    size2 = np.linalg.norm(dir2)
    dir2 = dir2 / size2

    dir3 = (b4-b1)
    size3 = np.linalg.norm(dir3)
    dir3 = dir3 / size3

    cube3d_center = (b1 + t3)/2.0

    dir_vec = points - cube3d_center

    res1 = np.where( (np.absolute(np.dot(dir_vec, dir1)) * 2) > size1 )[0]
    res2 = np.where( (np.absolute(np.dot(dir_vec, dir2)) * 2) > size2 )[0]
    res3 = np.where( (np.absolute(np.dot(dir_vec, dir3)) * 2) > size3 )[0]

    return list( set().union(res1, res2, res3) )

def get_cosine(box_a, box_b):
    """Computes IoU of two bboxes.
    Args:
        box_a, box_b: xyzxyz
    Returns:
        cosine
    """  
    points_a = get_vertices(box_a["obb"])
    points_b = get_vertices(box_b["obb"])
    a_direction = np.cross(points_a[0]-points_a[1], points_a[0]-points_a[2])
    b_direction = np.cross(points_b[0]-points_b[1], points_b[0]-points_b[2])
    return np.dot(a_direction, b_direction) / (np.linalg.norm(a_direction)*np.linalg.norm(b_direction))

def get_iou(box_a, box_b):
    """Computes IoU of two bboxes.
    Args:
        box_a, box_b: xyzxyz
    Returns:
        iou
    """
    pcd_a = box_a["pcd"]
    pcd_b = box_b["pcd"]
    points_a = get_vertices(box_a["obb"])
    points_b = get_vertices(box_b["obb"])
    points_a = torch.from_numpy(points_a)
    points_b = torch.from_numpy(points_b)
    points_a = points_a.type(torch.FloatTensor)
    points_b = points_b.type(torch.FloatTensor)
    o3d_render([pcd_a, pcd_b], [box_a, box_b], 'iouOBBs.png', win_width=640, win_height=480)
    # intersection_vol, iou_3d = box3d_overlap(box_corner_vertices_1, box_corner_vertices_2)
    intersection_vol, iou_3d = box3d_overlap(points_a[None], points_b[None])
    # o3d.visualization.webrtc_server.enable_webrtc()
    # o3d.visualization.draw([box_a, box_b])

    # points_a = 10*np.asarray(box_a.get_box_points())
    # points_b = 10*np.asarray(box_b.get_box_points())
    # a_center = 10*box_a.get_center()
    # b_center = 10*box_b.get_center()
    # a_extend = 10*box_a.extent
    # b_extend = 10*box_b.extent
    # a_R = 10*box_a.R
    # b_R = 10*box_b.R
    # range_min_a = points_a.min(0)
    # range_max_a = points_a.max(0)
    # range_min_b = points_b.min(0)
    # range_max_b = points_b.max(0)
    # range_min = np.minimum(points_a.min(0), points_b.min(0))
    # range_max = np.maximum(points_a.max(0), points_b.max(0))
    # intersection = 0
    # union = 0
    # point_set = []
    # for i in range((range_max-range_min)[0].astype(int)):
    #     for j in range((range_max-range_min)[1].astype(int)):
    #         for k in range((range_max-range_min)[2].astype(int)):
    #             x = i + range_min[0]
    #             y = j + range_min[1]
    #             z = k + range_min[2]
    #             point_set.append(np.array((x, y, z)))
    # outside_a = inside_test(point_set , points_a)
    # outside_b = inside_test(point_set , points_b)
    # outside_ab = list(set(outside_a) & set(outside_b))
    # union = len(point_set) - len(outside_ab)
    # intersection = (len(point_set) - len(outside_a)) + (len(point_set) - len(outside_b)) - (len(point_set) - len(outside_ab)) 

    # max_a = box_a[3:]
    # max_b = box_b[3:]
    # min_max = np.array([max_a, max_b]).min(0)

    # min_a = box_a[0:3]
    # min_b = box_b[0:3]
    # max_min = np.array([min_a, min_b]).max(0)
    # if not ((min_max > max_min).all()):
    #     return 0.0

    # intersection = (min_max - max_min).prod()
    # vol_a = (box_a[3:6] - box_a[:3]).prod()
    # vol_b = (box_b[3:6] - box_b[:3]).prod()
    # if union!= 0:
    #     return 1.0 * intersection / union
    return iou_3d.item()


def get_obb_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: xyzxyz
    Returns:
        iou
    """
    points_a = box_a.get_box_points()
    points_b = box_a.get_box_points()
    max_a = box_a[3:]
    max_b = box_b[3:]
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3]
    min_b = box_b[0:3]
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = (box_a[3:6] - box_a[:3]).prod()
    vol_b = (box_b[3:6] - box_b[:3]).prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union

def get_iou_main(get_iou_func, args):
    return get_iou_func(*args)


def eval_det_cls(pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """Generic functions to compute precision/recall for object detection for a
    single class.
    Input:
        pred: map of {img_id: [(sphere, score)]} where sphere is numpy array
        gt: map of {img_id: [sphere]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if True use VOC07 11 point method
    Output:
        rec: numpy array of length nd
        prec: numpy array of length nd
        ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {}  # {img_id: {'sphere': sphere list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        sphere = gt[img_id]
        det = np.zeros(shape=len(sphere), dtype=bool)
        npos += len(sphere)
        class_recs[img_id] = {'sphere': sphere, 'det': det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'sphere': np.array([]), 'det': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    cosine = []
    sorted_BB = []
    for img_id in pred.keys():
        for sphere, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(sphere)
    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    for i in range(len(sorted_ind)):
        sorted_BB.append(BB[i])
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd, dtype=bool)
    fp = np.zeros(nd, dtype=bool)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = sorted_BB[d]
        ovmax = np.NINF
        BBGT = R['sphere']
        for j in range(len(BBGT)):
            iou = get_iou_main(get_iou_func, (bb, BBGT[j]))
            if iou > ovmax:
                cosine.append(get_cosine(bb, BBGT[j]))
                ovmax = iou
                jmax = j

        # print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = True
                R['det'][jmax] = 1
            else:
                fp[d] = True
        else:
            fp[d] = True

    # compute precision recall
    fp = np.cumsum(fp, dtype=np.uint32)
    tp = np.cumsum(tp, dtype=np.uint32)
    rec = tp.astype(np.float32) / npos
    # print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    if len(cosine)==0:
        avg_cosine = 0
    else:
        avg_cosine = mean(cosine)

    return rec, prec, ap, avg_cosine


def eval_det_cls_wrapper(arguments):
    pred, gt, ovthresh, use_07_metric, get_iou_func = arguments
    rec, prec, ap, avg_cosine = eval_det_cls(pred, gt, ovthresh, use_07_metric, get_iou_func)
    return (rec, prec, ap, avg_cosine)


def eval_det(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """Generic functions to compute precision/recall for object detection for
    multiple classes.
    Input:
        pred_all: map of {img_id: [(classname, sphere, score)]}
        gt_all: map of {img_id: [(classname, sphere)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, sphere, score in pred_all[img_id]:
            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((sphere, score))
    for img_id in gt_all.keys():
        for classname, sphere in gt_all[img_id]:
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(sphere)

    rec = {}
    prec = {}
    ap = {}
    for classname in gt.keys():
        rec[classname], prec[classname], ap[classname] = eval_det_cls(pred[classname],
                                                                      gt[classname], ovthresh,
                                                                      use_07_metric, get_iou_func)

    return rec, prec, ap


def eval_sphere(pred_all, gt_all, ovthresh, use_07_metric=False, get_iou_func=get_iou):
    """Generic functions to compute precision/recall for object detection for
    multiple classes.
    Input:
        pred_all: map of {img_id: [(classname, sphere, score)]}
        gt_all: map of {img_id: [(classname, sphere)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, sphere, score in pred_all[img_id]:

            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((sphere, score))
    for img_id in gt_all.keys():
        for classname, sphere in gt_all[img_id]:

            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(sphere)

    rec = {}
    prec = {}
    ap = {}
    avg_cosine = {}
    tmp_list = [(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func)
                        for classname in gt.keys() if classname in pred]
    ret_values = []
    for item in tmp_list:
        ret_values.append(eval_det_cls_wrapper(item))

    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname], avg_cosine[classname] = ret_values[i]
            line = ''
            line += '{}'.format('finish class: ') + '{}'.format(classname) 
            print(line)
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0
            avg_cosine[classname] = 0

    return rec, prec, ap, avg_cosine


def get_gt_bbox(xyz, instance_ids, sem_labels, ignored_label, ignore_classes):
    gt_bbox = []
    unique_inst_ids = np.unique(instance_ids)
    for instance_id in unique_inst_ids:
        if instance_id == ignored_label:
            continue
        idx = instance_ids == instance_id
        sem_label = sem_labels[idx][0]
        if sem_label in ignore_classes or sem_label == ignored_label:
            continue
        sem_label = sem_label - len(ignore_classes)

        xyz_i = xyz[idx]
        center_1 =  (np.amax(xyz_i, axis=0) + np.amin(xyz_i, axis=0))/2.0
        gt_obb = {}
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(xyz_i)
        obb = gravity_aligned_mobb(pcd=gt_pcd, gravity=np.array((0.0, 1.0, 0.0)), align_axis=np.array((0.0, 0.0, -1.0)))
        gt_obb["obb"] = obb
        gt_obb["pcd"] = gt_pcd
        gt_obb["directions"] = get_directions(obb)
        pcd_set = {}
        obb_set = {}
        pcd_set[0] = gt_pcd
        obb_set[0] = gt_obb
        o3d_render(pcd_set, obb_set, output="gtOBB.png", win_width=640, win_height=480, with_diretions=True)
        gt_bbox.append((sem_label, gt_obb))

    return gt_bbox




def evaluate_bbox_acc(all_preds, all_gts, class_names, print_result):
    iou_thresholds = [0.25, 0.5]  # adjust threshold here
    pred_all = {}
    gt_all = {}
    for i in range(len(all_preds)):
        img_id = all_preds[i][0]["scan_id"]
        pred_all[img_id] = [(pred["label_id"] - 1, pred["pred_bbox"], pred["conf"]) for pred in all_preds[i]]
        gt_all[img_id] = all_gts[i]
    bbox_aps = {}
    bbox_direction_cosine = {}
    for iou_threshold in iou_thresholds:
        eval_res = eval_sphere(pred_all, gt_all, ovthresh=iou_threshold)
        aps = list(eval_res[2].values())
        m_ap = np.mean(aps)
        eval_res[2]["avg"] = m_ap
        bbox_aps[f"all_bbox_ap_{iou_threshold}"] = eval_res[2]
        bbox_direction_cosine[f"all_bbox_ap_{iou_threshold}"] = eval_res[-1]
    if print_result:
        print_results(bbox_aps, bbox_direction_cosine, class_names)
    return bbox_aps, bbox_direction_cosine


def print_results(bbox_aps, bbox_direction_cosine, class_names):
    sep = ''
    col1 = ':'
    lineLen = 63

    print()
    print('#' * lineLen)
    line = ''
    line += '{:<15}'.format('what') + sep + col1
    line += '{:>15}'.format('BBox_AP_50%') + sep
    line += '{:>15}'.format('BBOX_AP_25%') + sep
    line += '{:>17}'.format('Direction_Cos') + sep
    print(line)
    print('#' * lineLen)

    for (li, label_name) in enumerate(class_names):
        ap_50o = bbox_aps['all_bbox_ap_0.5'][li]
        ap_25o = bbox_aps['all_bbox_ap_0.25'][li]
        avg_consine_50o = bbox_direction_cosine['all_bbox_ap_0.5'][li]
        avg_consine_25o = bbox_direction_cosine['all_bbox_ap_0.25'][li] 
        avg_cosine = (avg_consine_25o + avg_consine_50o)/ 2       
        line = '{:<15}'.format(label_name) + sep + col1
        line += sep + '{:>15.3f}'.format(ap_50o) + sep
        line += sep + '{:>15.3f}'.format(ap_25o) + sep
        line += sep + '{:>17.3f}'.format(avg_cosine) + sep
        print(line)

    all_ap_50o = bbox_aps['all_bbox_ap_0.5']["avg"]
    all_ap_25o = bbox_aps['all_bbox_ap_0.25']["avg"]

    print('-' * lineLen)
    line = '{:<15}'.format('average') + sep + col1
    line += '{:>15.3f}'.format(all_ap_50o) + sep
    line += '{:>15.3f}'.format(all_ap_25o) + sep

    print(line)
    print('#' * lineLen)
    print()
