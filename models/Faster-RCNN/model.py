# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.ops import roi_align
import numpy as np
from utils import generate_anchors, ProposalCreator, AnchorTargetCreator, ProposalTargetCreator


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet50(weights='ResNet50_Weights.DEFAULT')
        # 使用到 layer4，其输出通道数为 2048
        self.features = nn.Sequential(*list(model.children())[:-2])
        # **【修正】** 修正了不正确的属性值
        self.out_channels = 2048

    def forward(self, x):
        return self.features(x)


class RPN(nn.Module):
    # **【修正】** 将 in_channels 的默认值从 1024 改为 2048
    def __init__(self, in_channels=2048, mid_channels=512, n_anchor=9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.cls_loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)  # Regression
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)  # Classification (BG/FG)
        self.proposal_creator = ProposalCreator()

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, features, img_size, is_training=True):
        n, _, h, w = features.shape
        feature_stride = 16
        anchor_base = generate_anchors(scales=[8, 16, 32], ratios=[0.5, 1, 2])
        shift_y = torch.arange(0, h * feature_stride, feature_stride, device=features.device)
        shift_x = torch.arange(0, w * feature_stride, feature_stride, device=features.device)

        shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing='ij')

        shift = torch.stack((shift_y.ravel(), shift_x.ravel(),
                             shift_y.ravel(), shift_x.ravel()), dim=1).T.contiguous()

        shifts = shift.permute(1, 0).contiguous().view(-1, 4)
        A = anchor_base.shape[0]
        K = shifts.shape[0]

        # **▼▼▼ THIS IS THE ONLY LINE TO CHANGE ▼▼▼**
        anchors = (torch.from_numpy(anchor_base).to(features.device).reshape(1, A, 4) + shifts.reshape(K, 1, 4)).float()
        # **▲▲▲ ADD .float() AT THE END ▲▲▲**

        anchors = anchors.reshape(K * A, 4)

        x = F.relu(self.conv1(features))

        rpn_locs = self.cls_loc(x).permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(x).permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        rpn_softmax_scores = F.softmax(rpn_scores, dim=2)[:, :, 1]

        rois = []
        roi_indices = []
        for i in range(n):
            roi = self.proposal_creator(rpn_locs[i], rpn_softmax_scores[i], anchors, img_size, is_training)
            rois.append(roi)
            roi_indices.append(torch.full((len(roi),), i, dtype=torch.float32, device=features.device))

        rois = torch.cat(rois, dim=0)
        roi_indices = torch.cat(roi_indices, dim=0)

        return rpn_locs, rpn_scores, rois, roi_indices, anchors


class FastRCNNHead(nn.Module):
    def __init__(self, n_class, roi_size=7, spatial_scale=1. / 16.):
        super().__init__()
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

        # **【修正】** 同样需要修正这里的输入维度
        in_features = 2048 * roi_size * roi_size
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.cls_loc = nn.Linear(4096, n_class * 4)  # Bbox regression
        self.score = nn.Linear(4096, n_class)  # Classification

        nn.init.normal_(self.cls_loc.weight, std=0.001)
        nn.init.constant_(self.cls_loc.bias, 0)
        nn.init.normal_(self.score.weight, std=0.01)
        nn.init.constant_(self.score.bias, 0)

    def forward(self, x, rois, roi_indices):
        roi_indices = roi_indices.to(x.device)
        rois = rois.to(x.device)

        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)

        pool = roi_align(x, indices_and_rois, output_size=(self.roi_size, self.roi_size),
                         spatial_scale=self.spatial_scale)
        pool = pool.view(pool.size(0), -1)

        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        return roi_cls_locs, roi_scores


class FasterRCNN(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.backbone = Backbone()
        self.rpn = RPN()
        self.head = FastRCNNHead(n_class=n_class + 1)  # +1 for background

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

    def forward(self, imgs, bboxes=None, labels=None):
        if self.training and (bboxes is None or labels is None):
            raise ValueError("In training mode, bboxes and labels must be provided.")

        img_size = imgs.shape[2:]
        features = self.backbone(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchors = self.rpn(features, img_size, self.training)

        if self.training:
            gt_rpn_locs, gt_rpn_labels = [], []
            sample_rois, gt_roi_locs, gt_roi_labels = [], [], []

            for i in range(len(imgs)):
                bbox = bboxes[i]
                label = labels[i]

                gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchors, img_size)
                gt_rpn_locs.append(gt_rpn_loc)
                gt_rpn_labels.append(gt_rpn_label)

                current_rois = rois[roi_indices == i]
                sample_roi, gt_loc, gt_label = self.proposal_target_creator(current_rois, bbox, label)

                sample_indices = torch.full((len(sample_roi), 1), i, device=imgs.device)

                # `sample_rois` in the original code was a list of tensors.
                # Here we directly prepare for concatenation.
                sample_rois.append(sample_roi)
                # Keep track of indices for the pooled features
                gt_roi_locs.append(gt_loc)
                gt_roi_labels.append(gt_label)

            # Prepare RPN targets for loss calculation
            # Note: rpn_locs and rpn_scores are already batched from the RPN forward pass
            gt_rpn_locs = torch.cat(gt_rpn_locs, dim=0)  # Should be stack for batch
            gt_rpn_labels = torch.cat(gt_rpn_labels, dim=0)

            # This part needs careful batch handling. Let's adjust.
            batched_gt_rpn_locs = []
            batched_gt_rpn_labels = []
            for i in range(len(imgs)):
                gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bboxes[i], anchors, img_size)
                batched_gt_rpn_locs.append(gt_rpn_loc)
                batched_gt_rpn_labels.append(gt_rpn_label)

            gt_rpn_locs = torch.stack(batched_gt_rpn_locs)
            gt_rpn_labels = torch.stack(batched_gt_rpn_labels)

            rpn_loc_loss = self._calculate_rpn_loc_loss(rpn_locs, gt_rpn_locs, gt_rpn_labels, 1.0)
            rpn_cls_loss = F.cross_entropy(rpn_scores.view(-1, 2), gt_rpn_labels.view(-1), ignore_index=-1)

            # Prepare Head targets for loss calculation
            final_sample_rois = []
            final_roi_indices = []
            final_gt_roi_locs = []
            final_gt_roi_labels = []

            for i in range(len(imgs)):
                current_rois_for_img = rois[roi_indices == i]
                sample_roi, gt_loc, gt_label = self.proposal_target_creator(current_rois_for_img, bboxes[i], labels[i])

                final_sample_rois.append(sample_roi)
                final_roi_indices.append(torch.full((len(sample_roi),), i, device=imgs.device))
                final_gt_roi_locs.append(gt_loc)
                final_gt_roi_labels.append(gt_label)

            sample_rois = torch.cat(final_sample_rois, dim=0)
            sample_roi_indices = torch.cat(final_roi_indices, dim=0)
            gt_roi_locs = torch.cat(final_gt_roi_locs, dim=0)
            gt_roi_labels = torch.cat(final_gt_roi_labels, dim=0)

            roi_cls_locs, roi_scores = self.head(features, sample_rois, sample_roi_indices)

            n_sample = roi_cls_locs.size(0)
            roi_cls_locs = roi_cls_locs.view(n_sample, -1, 4)
            roi_locs = roi_cls_locs[torch.arange(0, n_sample).long(), gt_roi_labels.long()]

            roi_loc_loss = self._calculate_head_loc_loss(roi_locs, gt_roi_locs, gt_roi_labels, 1.0)
            roi_cls_loss = F.cross_entropy(roi_scores, gt_roi_labels.long())

            return rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss

        else:  # Inference mode
            return self.predict(imgs)

    def _calculate_rpn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        # pred_loc is batched: [B, N, 4], gt_loc: [B, N, 4], gt_label: [B, N]
        pos_mask = gt_label > 0
        mask = pos_mask.unsqueeze(2).expand_as(pred_loc)
        masked_pred_loc = pred_loc[mask].view(-1, 4)
        masked_gt_loc = gt_loc[mask].view(-1, 4)
        if masked_pred_loc.shape[0] == 0:
            return torch.tensor(0., device=pred_loc.device)

        loss = F.smooth_l1_loss(masked_pred_loc, masked_gt_loc, reduction='sum')
        # Normalize by number of positive samples in the entire batch
        num_pos = torch.sum(pos_mask)
        return loss / (num_pos + 1e-5)

    def _calculate_head_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        pos_mask = gt_label > 0
        masked_pred_loc = pred_loc[pos_mask]
        masked_gt_loc = gt_loc[pos_mask]
        if masked_pred_loc.shape[0] == 0:
            return torch.tensor(0., device=pred_loc.device)

        loss = F.smooth_l1_loss(masked_pred_loc, masked_gt_loc, reduction='sum')
        num_pos = torch.sum(pos_mask)
        return loss / (num_pos + 1e-5)

    def predict(self, imgs, score_thresh=0.7):
        self.eval()
        from utils import loc_to_bbox
        from torchvision.ops import nms

        img_size = imgs.shape[2:]
        features = self.backbone(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchors = self.rpn(features, img_size, is_training=False)

        roi_cls_locs, roi_scores = self.head(features, rois, roi_indices)

        roi_scores_softmax = F.softmax(roi_scores, dim=1)

        roi_cls_locs = roi_cls_locs.view(-1, self.n_class + 1, 4)
        rois_expanded = rois.unsqueeze(1).expand(-1, self.n_class + 1, -1)
        cls_bboxes = loc_to_bbox(rois_expanded.reshape(-1, 4), roi_cls_locs.reshape(-1, 4))
        cls_bboxes = cls_bboxes.view(-1, self.n_class + 1, 4)

        cls_bboxes[:, :, 0::2] = torch.clamp(cls_bboxes[:, :, 0::2], 0, img_size[1])
        cls_bboxes[:, :, 1::2] = torch.clamp(cls_bboxes[:, :, 1::2], 0, img_size[0])

        final_bboxes, final_labels, final_scores = [], [], []

        for c in range(1, self.n_class + 1):  # Skip background class (0)
            class_boxes = cls_bboxes[:, c, :]
            class_scores = roi_scores_softmax[:, c]

            keep = class_scores > score_thresh
            class_boxes = class_boxes[keep]
            class_scores = class_scores[keep]

            if class_scores.numel() == 0:
                continue

            keep = nms(class_boxes, class_scores, 0.3)
            final_bboxes.append(class_boxes[keep])
            final_labels.append(torch.full((len(keep),), c, dtype=torch.int32))
            final_scores.append(class_scores[keep])

        if not final_bboxes:
            return torch.empty(0, 4), torch.empty(0), torch.empty(0)

        return torch.cat(final_bboxes, 0), torch.cat(final_labels, 0), torch.cat(final_scores, 0)