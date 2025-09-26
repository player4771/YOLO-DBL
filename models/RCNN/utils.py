import os
import cv2
import torch
import numpy as np
import yaml  # 导入yaml库
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)
    return config

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denominator = float(boxAArea + boxBArea - interArea)
    if denominator == 0:
        return 0.0

    iou = interArea / denominator
    return iou


def selective_search(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects


def visualize_results(**kwargs):
    image = kwargs['image']
    boxes = kwargs['boxes']
    labels = kwargs['labels']
    scores = kwargs['scores']
    class_names = kwargs['class_names']

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    cmap = plt.cm.get_cmap('hsv', len(class_names))

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1

        class_name = class_names[label]
        color = cmap(label)

        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        plt.text(x1, y1 - 5, f'{class_name}: {score:.2f}',
                 bbox=dict(facecolor=color, alpha=0.5),
                 fontsize=10, color='white')

    plt.axis('off')
    plt.show()


# --- 主要变更: __init__ 直接接收 image 和 label 目录 ---
class ObjectDetectionDataset(torch.utils.data.Dataset):
    """
    数据集类，现在直接从指定的图像和标签目录加载数据。
    """

    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        annotation_path = os.path.join(self.label_dir,
                                       self.image_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        boxes, labels = [], []
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    x1 = (x_center - width / 2) * w
                    y1 = (y_center - height / 2) * h
                    x2 = (x_center + width / 2) * w
                    y2 = (y_center + height / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(class_id))

        targets = {
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.LongTensor(labels)
        }
        return image, targets


class PreprocessedRCNNDataset(torch.utils.data.Dataset):
    """
    加载由 preprocess.py 生成的预处理数据。
    """

    def __init__(self, preprocessed_dir, transform=None):
        self.preprocessed_dir = preprocessed_dir
        self.sample_files = sorted([f for f in os.listdir(preprocessed_dir) if f.endswith('.pt')])
        self.transform = transform

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        # --- [FIX] The main fix is on this line ---
        # Explicitly set weights_only=False to allow loading NumPy arrays
        sample_data = torch.load(
            os.path.join(self.preprocessed_dir, self.sample_files[idx]),
            weights_only=False
        )

        image_path = sample_data['image_path']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pos_rois = sample_data['positive_rois']
        neg_rois = sample_data['negative_rois']
        pos_labels = sample_data['positive_labels']
        pos_reg_targets = sample_data['regression_targets']

        num_pos = min(len(pos_rois), 32)
        num_neg = min(len(neg_rois), 128 - num_pos)

        final_rois, final_labels, final_reg_targets = [], [], []
        if num_pos > 0:
            pos_indices = np.random.choice(len(pos_rois), size=num_pos, replace=False)
            final_rois.extend(pos_rois[pos_indices])
            final_labels.extend(pos_labels[pos_indices])
            final_reg_targets.extend(pos_reg_targets[pos_indices])
        if num_neg > 0:
            neg_indices = np.random.choice(len(neg_rois), size=num_neg, replace=False)
            final_rois.extend(neg_rois[neg_indices])
            final_labels.extend([0] * num_neg)

        rois_tensor_list = []
        for box in final_rois:
            x1, y1, x2, y2 = map(int, box)
            roi_img = image[y1:y2, x1:x2]
            if roi_img.shape[0] > 0 and roi_img.shape[1] > 0:
                if self.transform:
                    # Albumentations 返回的是字典
                    transformed_roi = self.transform(image=roi_img)['image']
                    rois_tensor_list.append(transformed_roi)

        if not rois_tensor_list:
            return None

        rois_tensor = torch.stack(rois_tensor_list)
        labels_tensor = torch.LongTensor(np.array(final_labels))
        targets_tensor = torch.FloatTensor(np.array(final_reg_targets))

        return rois_tensor, labels_tensor, targets_tensor

class EarlyStopping:
    """当监控的指标停止改善时，提前停止训练。"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): 在停止训练前，等待多少个epoch没有改善。
            verbose (bool): 如果为True，则为每次改善打印一条信息。
            delta (float): 监控指标的最小变化，小于此值的变化将被忽略。
            path (str): 保存最佳模型的路径。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

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
        """当验证损失减少时，保存模型。"""
        if self.verbose:
            print(f'Val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
