# utils.py
import torch
import numpy as np
import yaml
from pathlib import Path


# 早停类
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='weights/best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# 解析 data.yaml
def parse_data_yaml(data_path):
    with open(data_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


# IoU 计算
def iou(box_a, box_b):
    # box_a: (N, 4), box_b: (M, 4)
    lt = torch.max(box_a[:, None, :2], box_b[:, :2])
    rb = torch.min(box_a[:, None, 2:], box_b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = torch.prod(box_a[:, 2:] - box_a[:, :2], 1)
    area_b = torch.prod(box_b[:, 2:] - box_b[:, :2], 1)
    return inter / (area_a[:, None] + area_b - inter)


# 边界框格式转换及编码/解码
def loc_to_bbox(src_bbox, loc):
    src_width = src_bbox[:, 2] - src_bbox[:, 0]
    src_height = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height

    dx, dy, dw, dh = loc[:, 0], loc[:, 1], loc[:, 2], loc[:, 3]

    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0] = ctr_x - 0.5 * w
    dst_bbox[:, 1] = ctr_y - 0.5 * h
    dst_bbox[:, 2] = ctr_x + 0.5 * w
    dst_bbox[:, 3] = ctr_y + 0.5 * h
    return dst_bbox


def bbox_to_loc(src_bbox, dst_bbox):
    src_width = src_bbox[:, 2] - src_bbox[:, 0]
    src_height = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height

    dst_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    dst_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    dst_ctr_x = dst_bbox[:, 0] + 0.5 * dst_width
    dst_ctr_y = dst_bbox[:, 1] + 0.5 * dst_height

    eps = torch.finfo(src_height.dtype).eps
    src_width = torch.clamp(src_width, min=eps)
    src_height = torch.clamp(src_height, min=eps)

    dx = (dst_ctr_x - src_ctr_x) / src_width
    dy = (dst_ctr_y - src_ctr_y) / src_height
    dw = torch.log(dst_width / src_width)
    dh = torch.log(dst_height / src_height)

    return torch.stack((dx, dy, dw, dh), dim=1)


# 生成锚点
def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    w = base_anchor[2] - base_anchor[0] + 1
    h = base_anchor[3] - base_anchor[1] + 1
    x_ctr = base_anchor[0] + 0.5 * (w - 1)
    y_ctr = base_anchor[1] + 0.5 * (h - 1)

    h_ratios = np.round(np.sqrt(ratios))
    w_ratios = np.round(1 / h_ratios)

    ws = (w * w_ratios[:, np.newaxis] * scales).flatten()
    hs = (h * h_ratios[:, np.newaxis] * scales).flatten()

    anchors = np.zeros((len(ratios) * len(scales), 4))
    anchors[:, 0] = x_ctr - 0.5 * (ws - 1)
    anchors[:, 1] = y_ctr - 0.5 * (hs - 1)
    anchors[:, 2] = x_ctr + 0.5 * (ws - 1)
    anchors[:, 3] = y_ctr + 0.5 * (hs - 1)
    return anchors


# RPN 提议创建器
class ProposalCreator:
    def __init__(self, nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000,
                 n_test_pre_nms=6000, n_test_post_nms=300, min_size=16):
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, locs, scores, anchors, img_size, is_training=True):
        n_pre_nms = self.n_train_pre_nms if is_training else self.n_test_pre_nms
        n_post_nms = self.n_train_post_nms if is_training else self.n_test_post_nms

        rois = loc_to_bbox(anchors, locs)

        # **▼▼▼ CORRECTION START ▼▼▼**
        # Replaced the in-place clamping with an out-of-place operation.
        # Instead of modifying `rois` directly, we create a new tensor.
        x_min = torch.clamp(rois[:, 0], 0, img_size[1])
        y_min = torch.clamp(rois[:, 1], 0, img_size[0])
        x_max = torch.clamp(rois[:, 2], 0, img_size[1])
        y_max = torch.clamp(rois[:, 3], 0, img_size[0])
        rois = torch.stack([x_min, y_min, x_max, y_max], dim=1)
        # **▲▲▲ CORRECTION END ▲▲▲**

        min_size = self.min_size
        keep = torch.where((rois[:, 2] - rois[:, 0] >= min_size) & (rois[:, 3] - rois[:, 1] >= min_size))[0]
        rois = rois[keep, :]
        scores = scores[keep]

        order = torch.argsort(scores, descending=True)[:n_pre_nms]
        rois = rois[order, :]
        scores = scores[order]

        from torchvision.ops import nms
        keep = nms(rois, scores, self.nms_thresh)
        keep = keep[:n_post_nms]
        rois = rois[keep]
        return rois


# RPN 目标分配器
class AnchorTargetCreator:
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bboxes, anchors, img_size):
        device = bboxes.device
        n_anchor = len(anchors)

        inside_index = torch.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= img_size[1]) &
            (anchors[:, 3] <= img_size[0])
        )[0]

        anchors = anchors[inside_index]
        labels = torch.empty(len(inside_index), dtype=torch.long, device=device).fill_(-1)

        if len(bboxes) == 0:
            labels.fill_(0)
            return labels, torch.zeros_like(anchors)

        ious = iou(anchors, bboxes)
        max_ious, argmax_ious = ious.max(dim=1)

        labels[max_ious < self.neg_iou_thresh] = 0
        labels[max_ious >= self.pos_iou_thresh] = 1

        gt_argmax_ious = ious.argmax(dim=0)
        labels[gt_argmax_ious] = 1

        n_pos = int(self.n_sample * self.pos_ratio)
        pos_index = torch.where(labels == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = pos_index[torch.randperm(len(pos_index))[:len(pos_index) - n_pos]]
            labels[disable_index] = -1

        n_neg = self.n_sample - torch.sum(labels == 1)
        neg_index = torch.where(labels == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = neg_index[torch.randperm(len(neg_index))[:len(neg_index) - n_neg]]
            labels[disable_index] = -1

        locs = bbox_to_loc(anchors, bboxes[argmax_ious])

        # 将结果映射回原始所有锚点
        all_labels = torch.empty(n_anchor, dtype=torch.long, device=device).fill_(-1)
        all_labels[inside_index] = labels
        all_locs = torch.zeros((n_anchor, 4), dtype=torch.float32, device=device)
        all_locs[inside_index] = locs

        return all_locs, all_labels


# Fast R-CNN Head 目标分配器
class ProposalTargetCreator:
    def __init__(self, n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, rois, bboxes, labels):
        device = rois.device
        n_pos = round(self.n_sample * self.pos_ratio)

        if len(bboxes) == 0:
            # 如果没有 ground truth，所有 RoIs 都作为背景
            gt_labels = torch.zeros(len(rois), dtype=torch.long, device=device)
            gt_locs = torch.zeros(len(rois), 4, dtype=torch.float32, device=device)
            return rois, gt_locs, gt_labels

        ious = iou(rois, bboxes)
        max_ious, argmax_ious = ious.max(dim=1)
        gt_assignment = labels[argmax_ious]

        pos_index = torch.where(max_ious >= self.pos_iou_thresh)[0]
        pos_index = pos_index[torch.randperm(len(pos_index))[:n_pos]]

        neg_index = torch.where((max_ious < self.neg_iou_thresh_hi) & (max_ious >= self.neg_iou_thresh_lo))[0]
        n_neg = self.n_sample - len(pos_index)
        neg_index = neg_index[torch.randperm(len(neg_index))[:n_neg]]

        keep_index = torch.cat([pos_index, neg_index])
        sample_rois = rois[keep_index]

        gt_roi_labels = gt_assignment[keep_index]
        gt_roi_labels[len(pos_index):] = 0  # 背景类标签为0

        gt_roi_locs = bbox_to_loc(sample_rois, bboxes[argmax_ious[keep_index]])

        return sample_rois, gt_roi_locs, gt_roi_labels