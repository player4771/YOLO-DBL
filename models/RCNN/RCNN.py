import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import selectivesearch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class RCNN_Resnet:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #加载模型
        self.model, self.feature_extractor = self.get_pretrained_model()

    def get_pretrained_model(self):
        """加载预训练的ResNet50模型并将其改造为特征提取器"""
        # 加载在ImageNet上预训练的ResNet50模型
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

        # ResNet的特征通常取自最后一个全连接层(fc)之前
        # 我们可以通过将fc层替换为一个身份模块(Identity)来截断网络
        # 这样，模型的输出就是fc层之前的2048维特征
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

        model.to(self.device)
        feature_extractor.to(self.device)

        model.eval()
        feature_extractor.eval()

        # 我们返回原始模型用于模拟分类，返回改造后的模型用于特征提取
        return model, feature_extractor

    def get_region_proposals(self, image:np.ndarray, scale=100, sigma=0.8, min_size=50):
        """
        使用Selective Search生成候选区域
        :param image: 输入的图像 (NumPy array)
        :return: 候选区域列表，每个区域格式为 (x, y, w, h)
        """
        # Selective Search返回的rects格式是 (x, y, w, h)
        img_lbl, regions = selectivesearch.selective_search(
            image, scale=scale, sigma=sigma, min_size=min_size)

        # 过滤掉重复的区域
        rects = []
        seen = set()
        for r in regions:
            x, y, w, h = r['rect']
            # 过滤掉太小或太大/太扁或太高的区域
            if (r['rect'] in seen) or (w*h==0):
                continue

            rects.append((x, y, w, h))
            seen.add(r['rect'])

        return rects

    def extract_features(self, feature_extractor, image_tensor):
        """
        从图像张量中提取特征
        :param feature_extractor: 改造后的ResNet特征提取器
        :param image_tensor: 经过预处理的图像张量
        :return: 2048维的特征向量
        """
        # 预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(image_tensor).unsqueeze(0).to(self.device)

        # 关闭梯度计算以加速
        with torch.no_grad():
            # 将张量送入特征提取器
            features = feature_extractor(input_tensor)
            # ResNet提取的特征需要被展平
            flattened_features = torch.flatten(features, 1)

        return flattened_features.cpu().numpy().flatten()

    def visualize_detection(self, image, detections, threshold=0.5):
        """
        可视化检测结果
        :param image: 原始图像 (PIL Image)
        :param detections: 检测结果列表，每个元素为 (rect, score)
        :param threshold: 显示的置信度阈值
        """
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for rect, score in detections:
            if score > threshold:
                x, y, w, h = rect
                # 创建一个矩形框
                # Matplotlib的Rectangle需要 (x, y) 左下角坐标
                rect_patch = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                # 添加到图像上
                ax.add_patch(rect_patch)
                # 添加标签和分数
                ax.text(x, y, f'Object: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

        plt.show()

    def detect(self, image_path:str, threshold=0.5, nms_threshold=0.5):

        # 加载并准备图像
        pil_image = Image.open(image_path).convert('RGB')
        # PIL Image转为OpenCV格式 (NumPy array) 以便用于selective_search
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 生成候选区域
        candidate_rects = self.get_region_proposals(opencv_image)
        print(f"生成了 {len(candidate_rects)} 个候选区域。")

        # 对每个候选区域提取特征并进行“分类”
        detections = []
        for i, rect in enumerate(candidate_rects):
            x, y, w, h = rect

            # 从原始PIL图像中裁剪出候选区域
            region_image = pil_image.crop((x, y, x + w, y + h))

            # 使用feature_extractor提取特征
            features = self.extract_features(self.feature_extractor, region_image)
            # `features` 现在是一个2048维的向量

            # 将提取出的特征送入分类头进行预测
            features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # 只使用模型的最后分类层，避免重复计算
                outputs = self.model.fc(features_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                score, _ = torch.max(probabilities, 1)

            if score.item() > threshold:
                print(f"[{i + 1}/{len(candidate_rects)}]: 检测到物体，分数{score.item():.2f}")
                detections.append((rect, score.item()))

        # 将检测结果转换为 PyTorch Tensors
        boxes_tensor = torch.tensor([list(rect) for rect, score in detections], dtype=torch.float32)
        scores_tensor = torch.tensor([score for rect, score in detections], dtype=torch.float32)

        # torchvision.ops.nms 需要 (x1, y1, x2, y2) 格式的边界框
        # 我们的 rects 是 (x, y, w, h)，需要转换
        boxes_tensor[:, 2] = boxes_tensor[:, 0] + boxes_tensor[:, 2]  # x2 = x1 + w
        boxes_tensor[:, 3] = boxes_tensor[:, 1] + boxes_tensor[:, 3]  # y2 = y1 + h

        # 设置 NMS 的 IoU 阈值
        indices = torchvision.ops.nms(boxes_tensor, scores_tensor, nms_threshold)

        # 根据 NMS 返回的索引，筛选出最终的检测结果
        detections = [detections[i] for i in indices]
        print(f"NMS 处理后，剩余 {len(detections)} 个检测结果。")
        # --- NMS 处理结束 ---

        # 可视化结果
        print("处理完成，可视化结果...")
        self.visualize_detection(pil_image, detections, threshold=threshold)


if __name__ == '__main__':
    RCNN_Resnet().detect('../../data/images/bus.jpg',0.5, 0.3)