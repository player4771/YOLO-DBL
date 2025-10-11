import torch
from torchvision.ops import box_iou

def get_train_proposals_for_dataset(gt_boxes, cfg):
    """为单个图像生成训练用的RoIs(Proposals)"""
    num_pos = int(cfg['roi_num_samples'] * cfg['roi_pos_fraction'])
    if len(gt_boxes) > num_pos:
        perm = torch.randperm(len(gt_boxes))
        positive_rois = gt_boxes[perm[:num_pos]]
    else:
        positive_rois = gt_boxes

    num_neg_candidates = (cfg['roi_num_samples'] - len(positive_rois)) * 5
    random_boxes = torch.zeros((num_neg_candidates, 4), device=gt_boxes.device)

    # 在图像尺寸范围内生成随机候选框
    random_boxes[:, 0] = torch.rand(num_neg_candidates) * cfg['img_size']
    random_boxes[:, 1] = torch.rand(num_neg_candidates) * cfg['img_size']
    random_boxes[:, 2] = torch.rand(num_neg_candidates) * (cfg['img_size'] - random_boxes[:, 0])
    random_boxes[:, 3] = torch.rand(num_neg_candidates) * (cfg['img_size'] - random_boxes[:, 1])
    random_boxes[:, 2:] += random_boxes[:, :2]  # x2,y2 = x1+w, y1+h

    ious = box_iou(random_boxes, gt_boxes)
    max_ious = torch.max(ious, dim=1)[0] if ious.numel() > 0 else torch.zeros(random_boxes.shape[0])

    num_neg = cfg['roi_num_samples'] - len(positive_rois)
    negative_rois = random_boxes[max_ious < cfg['roi_neg_iou_thresh']][:num_neg]

    # 如果负样本不足，允许重复采样以凑够数量
    if len(positive_rois) + len(negative_rois) < cfg['roi_num_samples']:
        if len(negative_rois) > 0:
            shortage = cfg['roi_num_samples'] - (len(positive_rois) + len(negative_rois))
            negative_rois = torch.cat([negative_rois, negative_rois[torch.randint(0, len(negative_rois), (shortage,))]],
                                      dim=0)

    return torch.cat([positive_rois, negative_rois]) if len(positive_rois) > 0 or len(
        negative_rois) > 0 else torch.empty(0, 4)


def compute_targets_for_dataset(proposals, gt_boxes, gt_labels, cfg):
    """为Proposals计算分类标签和回归偏移量"""
    if proposals.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, 4)

    ious = box_iou(proposals, gt_boxes)
    if ious.numel() == 0:
        target_labels = torch.full((proposals.shape[0],), cfg['num_classes'] - 1, dtype=torch.long)
        target_deltas = torch.zeros_like(proposals)
        return target_labels, target_deltas

    max_ious, max_indices = torch.max(ious, dim=1)
    target_labels = gt_labels[max_indices]
    target_labels[max_ious < cfg['roi_pos_iou_thresh']] = cfg['num_classes'] - 1

    matched_gt_boxes = gt_boxes[max_indices]
    px, py = (proposals[:, 0] + proposals[:, 2]) / 2, (proposals[:, 1] + proposals[:, 3]) / 2
    pw, ph = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6), (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
    gx, gy = (matched_gt_boxes[:, 0] + matched_gt_boxes[:, 2]) / 2, (
            matched_gt_boxes[:, 1] + matched_gt_boxes[:, 3]) / 2
    gw, gh = (matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]).clamp(min=1e-6), (
            matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]).clamp(min=1e-6)

    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = torch.log(gw / pw)
    th = torch.log(gh / ph)

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
            proposals = get_train_proposals_for_dataset(gt_boxes, self.cfg)
            target_labels, target_deltas = compute_targets_for_dataset(proposals, gt_boxes, gt_labels, self.cfg)

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


def convert_and_save_fp16(model_path, num_classes, output_path=None):
    """
    加载一个全精度(float32)模型,将其转换为半精度(float16),并保存.

    Args:
        model_path (str): 输入的 float32 .pth 文件路径.
        num_classes (int): 模型的类别数 (包含背景).
        output_path (str, optional): 输出的 float16 .pth 文件路径.
                                     如果为 None,则在原文件名后添加 "_fp16".
    """
    if not output_path:
        output_path = model_path.replace('.pth', '_fp16.pth')

    print(f"\n--- Starting FP16 Conversion ---")
    print(f"Loading float32 weights from: {model_path}")

    from model import FastRCNN

    # 加载模型结构并置于评估模式
    model = FastRCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 将模型转换为半精度
    print("Converting model to float16...")
    model.half()

    # 保存半精度的 state_dict
    print(f"Saving float16 model to: {output_path}")
    torch.save(model.state_dict(), output_path)
    print("Conversion complete!")

@torch.inference_mode()
def evaluate(model, val_loader, device, cfg):
    """在验证集上评估模型"""
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, rois, target_labels, target_deltas in val_loader:
            images, rois, target_labels, target_deltas = \
                images.to(device), rois.to(device), target_labels.to(device), target_deltas.to(device)

            # 在验证时也使用 autocast，以匹配训练时的数值精度
            with torch.amp.autocast(device_type='cuda', enabled=cfg['amp']):
                scores, bbox_deltas = model(images, rois)
                cls_loss = torch.nn.functional.cross_entropy(scores, target_labels)

                pos_mask = (target_labels < (cfg['num_classes'] - 1))
                num_pos = pos_mask.sum()

                if num_pos > 0:
                    # 注意：在获取回归损失的目标时，需要正确索引
                    pred_deltas = bbox_deltas.view(-1, cfg['num_classes'], 4)[
                        torch.arange(len(pos_mask)), target_labels]
                    reg_loss = torch.nn.functional.smooth_l1_loss(
                        pred_deltas[pos_mask], target_deltas[pos_mask], reduction='sum') / num_pos
                else:
                    reg_loss = torch.tensor(0.0, device=device)

            total_val_loss += (cls_loss + reg_loss).item()

    return total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
