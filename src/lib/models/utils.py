from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2,
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

#SHLEE
def decode_tr_res(x):
    fr_ind_vec = []
    tlwhs_vec = []
    ids_vec = []
    for batch_tuple in x:
        fr_ind, tlwhs, ids = batch_tuple[-1]
        # tlwhs_ = torch.from_numpy(np.array(tlwhs)).to(fr_ind.device)
        tlwhs_ = np.array(tlwhs)
        fr_ind_vec.append(fr_ind)
        tlwhs_vec.append(tlwhs_)
        ids_vec.append(ids)
    return fr_ind_vec, tlwhs_vec, ids_vec

def make_mat(det_info, trk_info):
    # row : det, col : trk
    mat = []
    det_len = len(det_info)
    trk_len = len(trk_info)

    for i in range(det_len):
        tmp_list = []
        for j in range(trk_len):
            det = np.array(det_info[i])
            trk = np.array(trk_info[j])
            ov = compute_iou(det, trk)
            import math

            if math.isnan(ov):
                ov = 0.0
            tmp_list.append(ov)
        mat.append(tmp_list)
    mat = np.array(mat)

    return mat

def compute_iou(det_cord, trk_cord):
    """
    coord [Lx, Ly, W, H]
    trk_cord [cx, cy, lx, ly]
    :param det_cord: coordination of detection
    :param trk_cord: coordination of tracking
    :return: IOU score
    """

    det_x1 = det_cord[0]  # cx
    det_x2 = det_cord[0] + det_cord[2]
    det_y1 = det_cord[1]
    det_y2 = det_cord[1] + det_cord[3]

    trk_x1 = trk_cord[0]
    trk_x2 = trk_cord[0] + trk_cord[2]
    trk_y1 = trk_cord[1]
    trk_y2 = trk_cord[1] + trk_cord[3]

    det_area = det_cord[2] * det_cord[3]
    trk_area = trk_cord[2] * trk_cord[3]

    xx1 = trk_x1;
    xx2 = trk_x2
    yy1 = trk_y1;
    yy2 = trk_y2

    if det_x1 > trk_x1:
        xx1 = det_x1
    if det_x2 < trk_x2:
        xx2 = det_x2

    if det_y1 > trk_y1:
        yy1 = det_y1
    if det_y2 < trk_y2:
        yy2 = det_y2

    w = xx2 - xx1 + 1
    h = yy2 - yy1 + 1

    if w < 0 or h < 0:
        return 0.0

    inter = w * h
    union = det_area + trk_area - w * h
    ov = inter / union

    return ov


def compute_avgiou(det_result, trk_result, thres=-1):

    mat = make_mat(det_result, trk_result)
    if mat.size == 0:
        return torch.tensor([[0]])
    #print(mat)
    row_ind, col_ind = linear_sum_assignment(-mat)

    ln = len(row_ind)
    tmp_list = []
    for i in range(ln):
        iou = mat[row_ind[i]][col_ind[i]]
        import math
        if math.isnan(iou):
            iou = 0
        if thres > 0:
            if iou >= thres:
                tmp_list.append(iou)
        else:
            tmp_list.append(iou)

    if len(tmp_list)==0:
        return torch.tensor([[0]])

    av_iou = np.mean(np.array(tmp_list))

    return torch.tensor([[av_iou]])


def compute_card_score(gt_card, trk_card):
    if gt_card==0 and trk_card==0:
        return torch.tensor([[0.0]])
    card_ratio = min(gt_card,trk_card)/max(gt_card,trk_card)
    card_ratio = 1-card_ratio
    return torch.clamp(torch.tensor([[-np.log(card_ratio+1e-16)/np.exp(1)]]), min=0, max=1.0)

def compute_card_score_v2(gt_card, trk_card):
    if gt_card==0 and trk_card==0:
        return torch.tensor([[0.0]])
    card_ratio = min(gt_card,trk_card)/max(gt_card,trk_card)
    # card_ratio = 1-card_ratio
    return torch.clamp(torch.tensor([[card_ratio]]), min=0, max=1.0)

# implemented in torchvision modes/detection/transform.py
def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)