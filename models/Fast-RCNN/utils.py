import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.ops import box_iou

def get_train_proposals_for_dataset(gt_boxes, img_size, roi_num_samples, roi_pos_fraction, roi_neg_iou_thresh, **kwargs):
    """为单个图像生成训练用的RoIs(Proposals)"""
    num_pos = int(roi_num_samples * roi_pos_fraction)
    if len(gt_boxes) > num_pos:
        perm = torch.randperm(len(gt_boxes))
        positive_rois = gt_boxes[perm[:num_pos]]
    else:
        positive_rois = gt_boxes

    num_neg_candidates = (roi_num_samples - len(positive_rois)) * 5

    # 随机生成两组 (x, y) 坐标
    boxes_xyxy = torch.rand(num_neg_candidates, 4, device=gt_boxes.device) * img_size
    # 使用 min/max 确保 (x1, y1) < (x2, y2)
    boxes_x1y1 = torch.min(boxes_xyxy[:, :2], boxes_xyxy[:, 2:])
    boxes_x2y2 = torch.max(boxes_xyxy[:, :2], boxes_xyxy[:, 2:])

    # 确保框有最小面积 (可选，但推荐)
    # w = (random_boxes[:, 2] - random_boxes[:, 0]).clamp(min=1)
    # h = (random_boxes[:, 3] - random_boxes[:, 1]).clamp(min=1)
    # random_boxes[:, 2] = random_boxes[:, 0] + w
    # random_boxes[:, 3] = random_boxes[:, 1] + h

    random_boxes = torch.cat([boxes_x1y1, boxes_x2y2], dim=1)

    ious = box_iou(random_boxes, gt_boxes)
    max_ious = torch.max(ious, dim=1)[0] if ious.numel() > 0 else torch.zeros(random_boxes.shape[0])

    num_neg = roi_num_samples - len(positive_rois)
    negative_rois = random_boxes[max_ious < roi_neg_iou_thresh][:num_neg]

    # 如果负样本不足，允许重复采样以凑够数量
    if len(positive_rois) + len(negative_rois) < roi_num_samples:
        if len(negative_rois) > 0:
            shortage = roi_num_samples - (len(positive_rois) + len(negative_rois))
            negative_rois = torch.cat([negative_rois, negative_rois[torch.randint(0, len(negative_rois), (shortage,))]],
                                      dim=0)

    return torch.cat([positive_rois, negative_rois]) if len(positive_rois) > 0 or len(
        negative_rois) > 0 else torch.empty(0, 4)


def compute_targets_for_dataset(proposals, gt_boxes, gt_labels, roi_pos_iou_thresh, num_classes, **kwargs):
    """为Proposals计算分类标签和回归偏移量"""
    if proposals.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, 4)

    ious = box_iou(proposals, gt_boxes)
    if ious.numel() == 0:
        target_labels = torch.full((proposals.shape[0],), num_classes - 1, dtype=torch.long)
        target_deltas = torch.zeros_like(proposals)
        return target_labels, target_deltas

    max_ious, max_indices = torch.max(ious, dim=1)
    target_labels = gt_labels[max_indices]
    target_labels[max_ious < roi_pos_iou_thresh] = num_classes - 1

    matched_gt_boxes = gt_boxes[max_indices]
    px, py = (proposals[:, 0] + proposals[:, 2]) / 2, (proposals[:, 1] + proposals[:, 3]) / 2
    pw, ph = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6), (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
    gx, gy = (matched_gt_boxes[:, 0] + matched_gt_boxes[:, 2]) / 2, (
            matched_gt_boxes[:, 1] + matched_gt_boxes[:, 3]) / 2
    gw, gh = (matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]).clamp(min=1e-6), (
            matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]).clamp(min=1e-6)

    tx:torch.Tensor = (gx - px) / pw
    ty:torch.Tensor = (gy - py) / ph
    tw:torch.Tensor = torch.log(gw / pw)
    th:torch.Tensor = torch.log(gh / ph)

    targets = torch.stack((tx, ty, tw, th), dim=1)

    BBOX_REG_MEANS = torch.tensor([0.0, 0.0, 0.0, 0.0])
    BBOX_REG_STDS = torch.tensor([0.1, 0.1, 0.2, 0.2])

    means = BBOX_REG_MEANS.to(targets.device)
    stds = BBOX_REG_STDS.to(targets.device)
    targets = (targets - means) / stds

    return target_labels, targets


class FastRCNNCollator:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)

        all_proposals = []
        all_target_labels = []
        all_target_deltas = []

        for i in range(len(images)):
            gt_boxes = targets[i]['boxes']
            gt_labels = targets[i]['labels']

            # 为每个图像动态生成 proposals 和 targets
            proposals = get_train_proposals_for_dataset(gt_boxes, **self.cfg)
            target_labels, target_deltas = compute_targets_for_dataset(proposals, gt_boxes, gt_labels, **self.cfg)

            if proposals.numel() > 0:
                # 添加 batch index 形成 rois
                batch_idx = torch.full((proposals.shape[0], 1), i, dtype=torch.float)
                rois = torch.cat([batch_idx, proposals], dim=1)
                all_proposals.append(rois)
                all_target_labels.append(target_labels)
                all_target_deltas.append(target_deltas)

        # 将整个 batch 的数据连接起来
        final_rois = torch.cat(all_proposals, dim=0) if all_proposals else torch.empty(0, 5)
        final_labels = torch.cat(all_target_labels, dim=0) if all_target_labels else torch.empty(0, dtype=torch.long)
        final_deltas = torch.cat(all_target_deltas, dim=0) if all_target_deltas else torch.empty(0, 4)

        return images, final_rois, final_labels, final_deltas


def apply_regression(boxes, deltas):
    """
    将模型预测的回归偏移量应用到提议框上。

    Args:
        boxes (torch.Tensor): [N, 4] 格式为 (x1, y1, x2, y2) 的提议框。
        deltas (torch.Tensor): [N, 4] 模型预测的偏移量 (tx, ty, tw, th)。

    Returns:
        torch.Tensor: [N, 4] 调整后的精确边界框。
    """
    # 将提议框从 (x1, y1, x2, y2) 转换为 (cx, cy, w, h)
    px = (boxes[:, 0] + boxes[:, 2]) / 2.0
    py = (boxes[:, 1] + boxes[:, 3]) / 2.0
    pw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    ph = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)

    # 提取偏移量
    BBOX_REG_MEANS = torch.tensor([0.0, 0.0, 0.0, 0.0])
    BBOX_REG_STDS = torch.tensor([0.1, 0.1, 0.2, 0.2])

    stds = BBOX_REG_STDS.to(deltas.device)
    means = BBOX_REG_MEANS.to(deltas.device)
    deltas = deltas * stds + means

    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    # 应用逆变换
    pred_cx = px + pw * dx
    pred_cy = py + ph * dy
    pred_w = pw * torch.exp(dw)
    pred_h = ph * torch.exp(dh)

    # 将计算出的新坐标转换回 (x1, y1, x2, y2) 格式
    pred_x1 = pred_cx - pred_w / 2.0
    pred_y1 = pred_cy - pred_h / 2.0
    pred_x2 = pred_cx + pred_w / 2.0
    pred_y2 = pred_cy + pred_h / 2.0

    return torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)


def compute_loss(scores, bbox_deltas, target_labels, target_deltas, num_classes):
    """计算 Fast R-CNN 的分类和回归损失"""

    cls_loss = F.cross_entropy(scores, target_labels)

    # --- 计算回归损失 ---
    # 筛选出正样本 (非背景)
    pos_mask = (target_labels < (num_classes - 1))
    num_pos = pos_mask.sum()

    if num_pos > 0:
        if bbox_deltas.shape[1] == num_classes * 4:
            # 类别相关回归 [N, num_classes * 4]
            pred_deltas_all = bbox_deltas.view(-1, num_classes, 4)
            pred_deltas_pos = pred_deltas_all[torch.arange(len(pos_mask)), target_labels]
        else:
            # 类别无关回归 [N, 4]
            # 所有正样本都使用这 [N, 4] 的预测值
            pred_deltas_pos = bbox_deltas

        reg_loss = F.smooth_l1_loss(
            pred_deltas_pos[pos_mask],  # 预测的正样本偏移量
            target_deltas[pos_mask],  # 目标的正样本偏移量
            reduction='sum'
        ) / num_pos
    else:
        # 如果没有正样本，回归损失为0
        reg_loss = torch.tensor(0.0, device=scores.device)

    return cls_loss, reg_loss

@torch.inference_mode()
def evaluate(model, val_loader, amp=True, num_classes=4, **kwargs):
    """在验证集上评估模型"""
    device = 'cuda' if amp else 'cpu'
    model.eval()
    total_val_loss = 0.0

    # @torch.inference_mode()已经替代了torch.no_grad()，所以不需要再添加
    pbar = tqdm(val_loader, desc=f"Eval")
    for images, rois, target_labels, target_deltas in pbar:

        # 在验证时也使用 autocast，以匹配训练时的数值精度
        with torch.amp.autocast(device_type=device, enabled=amp):
            scores, bbox_deltas = model(images.to(device), rois.to(device))
            cls_loss, reg_loss = compute_loss(scores, bbox_deltas, target_labels.to(device), target_deltas.to(device), num_classes)
            loss = cls_loss + reg_loss

        total_val_loss += loss.item()
        pbar.set_postfix(val_loss=loss.item())

    return total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
