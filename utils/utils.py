import torch
import numpy as np
from scipy.interpolate import interp1d
import os
import sys
import random
import config
import torch.nn as nn


def str2ind(categoryname, classlist):
    return [
        i for i in range(len(classlist))
        if categoryname == classlist[i].decode("utf-8")
    ][0]


def strlist2indlist(strlist, classlist):
    return [str2ind(s, classlist) for s in strlist]


def strlist2multihot(strlist, classlist):
    return np.sum(np.eye(len(classlist))[strlist2indlist(strlist, classlist)],
                  axis=0)


def upgrade_resolution(arr, scale):
    if scale == 1:
        return arr
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind="linear", axis=0, fill_value="extrapolate")
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


def get_proposal_oic(tList,
                     wtcam,
                     final_score,
                     c_pred,
                     scale,
                     v_len,
                     sampling_frames,
                     num_segments,
                     lambda_=0.25,
                     gamma=0.2,
                     loss_type="oic"):
    t_factor = (16 * v_len) / (scale * num_segments * sampling_frames)
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])

                len_proposal = len(grouped_temp_list[j])
                outer_s = max(
                    0, int(grouped_temp_list[j][0] - lambda_ * len_proposal))
                outer_e = min(
                    int(wtcam.shape[0] - 1),
                    int(grouped_temp_list[j][-1] + lambda_ * len_proposal),
                )

                outer_temp_list = list(
                    range(outer_s, int(grouped_temp_list[j][0]))) + list(
                        range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))

                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])

                if loss_type == "oic":
                    c_score = inner_score - outer_score + gamma * final_score[
                        c_pred[i]]
                else:
                    c_score = inner_score
                t_start = grouped_temp_list[j][0] * t_factor
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([c_pred[i], c_score, t_start, t_end])
            temp.append(c_temp)
    return temp


def result2json(result, class_dict=None):
    if class_dict is None:
        class_dict = config.class_dict
    result_file = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            line = {
                "label": class_dict[result[i][j][0]],
                "score": result[i][j][1],
                "segment": [result[i][j][2], result[i][j][3]],
            }
            result_file.append(line)
    return result_file


def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def save_best_record_thumos(test_info, file_path, tIoU_thresh=None):
    fo = open(file_path, "w")
    fo.write("Step: {}\n".format(test_info["step"][-1]))
    fo.write("Test_acc: {:.2f}\n".format(test_info["test_acc"][-1]))
    fo.write("average_mAP: {:.4f}\n".format(test_info["average_mAP"][-1]))

    if tIoU_thresh is None:
        tIoU_thresh = np.linspace(0.1, 0.9, 9)
    for i in range(len(tIoU_thresh)):
        fo.write("mAP@{:.1f}: {:.4f}\n".format(
            tIoU_thresh[i],
            test_info["mAP@{:.1f}".format(tIoU_thresh[i])][-1]))

    fo.close()


def minmax_norm(act_map):
    max_val = torch.max(act_map, dim=1)[0]
    min_val = 0.0
    ret = (act_map - min_val) / (max_val - min_val)

    return ret


def nms(proposals, thresh):
    proposals = np.array(proposals)
    x1 = proposals[:, 2]
    x2 = proposals[:, 3]
    scores = proposals[:, 1]

    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < thresh)[0]
        order = order[inds + 1]

    return keep


def soft_nms(dets, iou_thr=0.7, method='gaussian', sigma=0.3):

    dets = np.array(dets)
    x1 = dets[:, 2]
    x2 = dets[:, 3]
    scores = dets[:, 1]

    areas = x2 - x1 + 1

    # expand dets with areas, and the second dimension is
    # x1, x2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 1], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1].tolist())

        xx1 = np.maximum(dets[0, 2], dets[1:, 2])
        xx2 = np.minimum(dets[0, 3], dets[1:, 3])

        inter = np.maximum(xx2 - xx1 + 1, 0.0)
        iou = inter / (dets[0, -1] + dets[1:, -1] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 1] *= weight
        # retained_idx = np.where(dets[1:, 4] >= score_thr)[0]
        # dets = dets[retained_idx + 1, :]
        dets = dets[1:, :]

    return retained_box


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Bilinear') != -1:
        nn.init.kaiming_uniform_(mode='fan_in',
                                 nonlinearity='leaky_relu',
                                 tensor=m.weight)
        if hasattr(m, 'bias'): nn.init.zeros_(tensor=m.bias)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(mode='fan_in',
                                 nonlinearity='leaky_relu',
                                 tensor=m.weight)
        if hasattr(m, 'bias'): nn.init.zeros_(tensor=m.bias)

    elif classname.find('BatchNorm') != -1 or classname.find(
            'GroupNorm') != -1 or classname.find('LayerNorm') != -1:
        nn.init.uniform_(a=0, b=1, tensor=m.weight)
        nn.init.zeros_(tensor=m.bias)
