import numpy as np
"""
Adapted from https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py
"""

from copy import deepcopy
import numpy as np
import torch

def rle_encode(mask):
    """Encode RLE (Run-length-encode) from 1D binary mask.
    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    length = mask.shape[0]
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    counts = ' '.join(str(x) for x in runs)
    rle = dict(length=length, counts=counts)
    return rle


def rle_decode(rle):
    """Decode rle to get binary mask.
    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle['length']
    counts = rle['counts']
    s = counts.split()
    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask


def get_instances(ids, class_ids, class_labels, id2label, ignored_label):
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id, ignored_label)
        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances


def get_gt_instances(semantic_labels, instance_labels, ignored_classes):
    """Get gt instances for evaluation."""
    # convert to evaluation format 0: ignore, 1->N: valid
    label_shift = len(ignored_classes)
    semantic_labels = semantic_labels - label_shift + 1
    semantic_labels[semantic_labels < 0] = 0
    instance_labels += 1
    ignore_inds = instance_labels <= 0
    # scannet encoding rule
    gt_ins = semantic_labels * 1000 + instance_labels
    gt_ins[ignore_inds] = 0
    gt_ins = gt_ins
    return gt_ins


class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id, ignored_label):
        if instance_id == ignored_label:
            return
        self.instance_id = int(instance_id)
        self.label_id = int(self.get_label_id(instance_id))
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return np.count_nonzero(mesh_vert_instances == instance_id)

    def to_dict(self):
        dict = {'instance_id': self.instance_id, 'label_id': self.label_id, 'vert_count': self.vert_count,
                'med_dist': self.med_dist, 'dist_conf': self.dist_conf}
        return dict

    def __str__(self):
        return f"({self.instance_id})"


class GeneralDatasetEvaluator(object):

    def __init__(self, class_labels, ignored_label, iou_type=None, use_label=True):
        self.valid_class_labels = class_labels
        self.ignored_label = ignored_label
        self.valid_class_ids = np.arange(len(class_labels)) + 1
        self.id2label = {}
        self.label2id = {}
        for i in range(len(self.valid_class_ids)):
            self.label2id[self.valid_class_labels[i]] = self.valid_class_ids[i]
            self.id2label[self.valid_class_ids[i]] = self.valid_class_labels[i]

        self.ious = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = np.array([100])
        self.distance_threshes = np.array([float('inf')])
        self.distance_confs = np.array([-float('inf')])

        self.iou_type = iou_type
        self.use_label = use_label
        if self.use_label:
            self.eval_class_labels = self.valid_class_labels
        else:
            self.eval_class_labels = ['class_agnostic']

    def evaluate_matches(self, matches):
        ious = self.ious
        min_region_sizes = [self.min_region_sizes[0]]
        dist_threshes = [self.distance_threshes[0]]
        dist_confs = [self.distance_confs[0]]

        # results: class x iou
        ap = np.zeros((len(dist_threshes), len(self.eval_class_labels)), np.float)
        rc = np.zeros((len(dist_threshes), len(self.eval_class_labels)), np.float)
        for di, (min_region_size, distance_thresh,
                 distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
            for li, label_name in enumerate(self.eval_class_labels):
                tp = 0
                fp = 0
                fn = 0
                for m in matches:
                    if matches[m]['sem_label']==li:
                        pred_obb_direction = matches[m]['pred']
                        gt_obb_direction = matches[m]['gt']
                        if pred_obb_direction==gt_obb_direction:
                            tp += 1
                        else:
                            fp += 1
                            fn += 1
                if (tp+fp)!=0:
                    ap_current = np.float(tp)/np.float(fp+tp)
                    rc_current = np.float(tp)/np.float(fp+fn)
                    ap[di, li] = ap_current
                    rc[di, li] = rc_current
        return ap, rc

    def compute_averages(self, aps, rcs):
        avg_dict = {}
        # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
        avg_dict['all_ap'] = np.nanmean(aps[d_inf, :])
        avg_dict['classes'] = {}
        for (li, label_name) in enumerate(self.eval_class_labels):
            avg_dict['classes'][label_name] = {}
            avg_dict['classes'][label_name]['ap'] = aps[d_inf, li]
            avg_dict['classes'][label_name]['rc'] = rcs[d_inf, li]
        return avg_dict

    def evaluate(self, pred_list, gt_list, print_result):
        """
        Args:
            pred_list:
                for each scan:
                    for each instance
                        instance = dict(scan_id, label_id, mask, conf)
            gt_list:
                for each scan:
                    for each point:
                        gt_id = class_id * 1000 + instance_id
        """
        assert len(pred_list) == len(gt_list)
        matches = {}
        for i in range(len(pred_list)):
            matches_key = f'obb_{i}'
            matches[matches_key] = {}
            matches[matches_key]['sem_label'] = pred_list[i]["sem_label"]
            matches[matches_key]['pred'] = pred_list[i]["class"]
            matches[matches_key]['gt'] = gt_list[i]["class"]
        ap_scores, rc_scores = self.evaluate_matches(matches)
        avgs = self.compute_averages(ap_scores, rc_scores)
        if print_result:
            self.print_results(avgs)
        return avgs

    def print_results(self, avgs):
        sep = ''
        col1 = ':'
        lineLen = 64

        print()
        print('#' * lineLen)
        line = ''
        line += '{:<15}'.format('what') + sep + col1
        line += '{:>8}'.format('AP') + sep
        line += '{:>8}'.format('AR') + sep

        print(line)
        print('#' * lineLen)

        for (li, label_name) in enumerate(self.eval_class_labels):
            ap_avg = avgs['classes'][label_name]['ap']
            rc_avg = avgs['classes'][label_name]['rc']
            line = '{:<15}'.format(label_name) + sep + col1
            line += sep + '{:>8.3f}'.format(ap_avg) + sep
            line += sep + '{:>8.3f}'.format(rc_avg) + sep
            print(line)

        all_ap_avg = avgs['all_ap']
        all_rc_avg = avgs['all_rc']

        print('-' * lineLen)
        line = '{:<15}'.format('average') + sep + col1
        line += '{:>8.3f}'.format(all_ap_avg) + sep
        line += '{:>8.3f}'.format(all_rc_avg) + sep
        print(line)
        print('#' * lineLen)
        print()
