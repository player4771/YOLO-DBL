import torch
import torch.nn as nn
import torchvision
from PIL import Image
import numpy as np
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# --- 配置和辅助函数 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 定义Fast R-CNN模型 ---
class FastRCNN_ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes

        # 加载完整的预训练ResNet50
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

        # 主干网络 (Backbone)
        # 去掉最后的平均池化层和全连接层
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 重用原始的平均池化层和全连接分类层
        self.avgpool = resnet.avgpool
        self.classifier = resnet.fc # 这个分类器是预训练好的

        # 边界框回归器(保留定义但其输出在逻辑上被忽略)
        self.bbox_regressor = nn.Linear(4096, num_classes * 4)

    def forward(self, image, rois):
        """
        :param image: 输入图像张量 (1, C, H, W)
        :param rois: 候选区域张量 (N, 5)，格式 [batch_idx, x1, y1, x2, y2]
        """
        # 提取特征
        feature_map = self.backbone(image)

        # RoI Align
        spatial_scale = feature_map.shape[2] / image.shape[2]
        roi_features = torchvision.ops.roi_align(feature_map, rois, output_size=(7, 7), spatial_scale=spatial_scale)

        # 将每个RoI特征通过原始的avgpool层和fc层进行分类
        pooled_features = self.avgpool(roi_features)
        flattened_features = torch.flatten(pooled_features, 1)
        class_scores = self.classifier(flattened_features)

        # 边界框回归器输出无效，返回零张量以保持API一致
        bbox_deltas = torch.zeros((rois.shape[0], self.num_classes * 4), device=rois.device)

        return class_scores, bbox_deltas


# --- 检测器主类 ---
class ObjectDetector:
    def __init__(self, model):
        self.model = model.to(device).eval()

        # 定义图像预处理流程
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # 加载用于可视化的类名
        self.class_names = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"]


    def get_region_proposals(self, image_np, scale=100, sigma=0.8, min_size=1000):
        """使用Selective Search生成候选区域"""
        _, regions = selectivesearch.selective_search(image_np, scale=scale, sigma=sigma, min_size=min_size)

        rects = []
        for r in regions:
            x, y, w, h = r['rect']
            if w*h != 0 : # 格式转换为 (x1, y1, x2, y2)
                rects.append([x, y, x + w, y + h])

        return np.array(rects)

    def apply_bbox_regression(self, proposals, deltas):
        """
        将回归器的输出（deltas）应用到原始候选框上。
        :param proposals: 原始候选框 (N, 4)，格式 (x1, y1, x2, y2)
        :param deltas: 模型的回归输出 (N, 4)，格式 (tx, ty, tw, th)
        :return: 调整后的边界框 (N, 4)
        """
        if proposals.shape[0] == 0:
            return proposals

        px = (proposals[:, 0] + proposals[:, 2]) * 0.5
        py = (proposals[:, 1] + proposals[:, 3]) * 0.5
        pw = proposals[:, 2] - proposals[:, 0]
        ph = proposals[:, 3] - proposals[:, 1]

        dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

        gx = px + pw * dx
        gy = py + ph * dy
        gw = pw * torch.exp(dw)
        gh = ph * torch.exp(dh)

        x1 = gx - gw * 0.5
        y1 = gy - gh * 0.5
        x2 = gx + gw * 0.5
        y2 = gy + gh * 0.5

        return torch.stack([x1, y1, x2, y2], dim=1)

    def detect(self, image_path, conf_threshold=0.5, nms_threshold=0.3):
        start_time = time.time()

        # 加载和预处理图像
        pil_image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(pil_image).unsqueeze(0).to(device)

        # 生成候选区域
        proposals_np = self.get_region_proposals(np.array(pil_image))
        print(f"生成了 {len(proposals_np)} 个候选区域。")

        # 将proposals转换为模型需要的格式: (N, 5) -> [batch_idx, x1, y1, x2, y2]
        proposals_torch = torch.from_numpy(proposals_np).float().to(device)
        batch_indices = torch.zeros((proposals_torch.shape[0], 1), device=device)
        rois = torch.cat([batch_indices, proposals_torch], dim=1)

        # 模型推理
        with torch.no_grad():
            class_scores, bbox_deltas = self.model(image_tensor, rois)

        # 后处理
        probabilities = nn.functional.softmax(class_scores, dim=1)

        final_boxes, final_scores, final_labels = [], [], []

        for class_id in range(self.model.num_classes):
            # 获取该类的分数
            class_probs_per_roi = probabilities[:, class_id] # ImageNet分类器没有背景类

            # 筛选分数高于阈值的RoI
            keep_indices = (class_probs_per_roi > conf_threshold).nonzero().squeeze(1)
            if len(keep_indices) == 0:
                continue

            # 提取对应的RoIs和分数
            selected_proposals = proposals_torch[keep_indices]
            selected_scores = class_probs_per_roi[keep_indices]

            # 应用NMS (边界框回归无效，直接对原始候选框操作)
            keep_nms = torchvision.ops.nms(selected_proposals, selected_scores, nms_threshold)

            for idx in keep_nms:
                final_boxes.append(selected_proposals[idx].cpu().numpy())
                final_scores.append(selected_scores[idx].cpu().item())
                final_labels.append(class_id)

        end_time = time.time()
        print(f"检测完成，耗时: {end_time - start_time:.2f}秒")
        print(f"检测到 {len(final_boxes)} 个物体。")

        # 可视化结果
        self.visualize(pil_image, final_boxes, final_scores, final_labels)

    def visualize(self, image, boxes, scores, labels):
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)

        for box, score, label_id in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            class_name = self.class_names[label_id]

            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

            ax.text(x1, y1 - 10, f'{class_name}: {class_name}: {score:.2f}', color='white',
                    bbox=dict(facecolor='green', alpha=0.5, pad=2))

        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    # 实例化模型
    fast_rcnn_model = FastRCNN_ResNet50(num_classes=1000)

    # 创建检测器
    detector = ObjectDetector(model=fast_rcnn_model)

    # 执行检测
    image_path = '../../data/images/bus.jpg' # 请替换为你的图片路径
    detector.detect(image_path, conf_threshold=0.6, nms_threshold=0.3)