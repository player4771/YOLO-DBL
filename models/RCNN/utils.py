import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as infile:
        config = yaml.safe_load(infile)
    return config

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxA[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator > 0 else 0.0

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


class PreprocessedRCNNDataset(torch.utils.data.Dataset):
    """加载由 preprocess.py 生成的预处理数据。"""

    def __init__(self, preprocessed_dir, transform=None):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.sample_files = sorted(list(self.preprocessed_dir.glob('*.pt')))
        self.transform = transform

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample_data = torch.load(self.sample_files[idx], weights_only=False)
        image = cv2.imread(sample_data['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pos_rois = sample_data['positive_rois']
        neg_rois = sample_data['negative_rois']
        pos_labels = sample_data['positive_labels']
        pos_reg_targets = sample_data['regression_targets']

        # 在这里进行采样，确保每次送入模型的RoI数量可控
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
                    transformed_roi = self.transform(image=roi_img, bboxes=[], class_labels=[])['image']
                    rois_tensor_list.append(transformed_roi)

        if not rois_tensor_list:
            return None

        rois_tensor = torch.stack(rois_tensor_list)
        labels_tensor = torch.LongTensor(np.array(final_labels))
        targets_tensor = torch.FloatTensor(np.array(final_reg_targets))

        return rois_tensor, labels_tensor, targets_tensor