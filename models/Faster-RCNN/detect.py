# detect.py
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from model import FasterRCNN
from utils import parse_data_yaml
import random

# --- Config ---
DETECT_CONFIG = {
    "data_yaml": "data/data.yaml",
    "weights": "weights/best_model.pth",
    "img_path": "data/images/val/000007.jpg",  # 替换为你要检测的图片路径
    "img_size": (600, 600),
    "score_thresh": 0.7,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def preprocess_image(img_path, img_size):
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize(img_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(img).unsqueeze(0), (orig_w, orig_h)


def rescale_boxes(boxes, current_size, original_size):
    orig_w, orig_h = original_size
    cur_w, cur_h = current_size

    boxes[:, 0] = boxes[:, 0] * orig_w / cur_w
    boxes[:, 1] = boxes[:, 1] * orig_h / cur_h
    boxes[:, 2] = boxes[:, 2] * orig_w / cur_w
    boxes[:, 3] = boxes[:, 3] * orig_h / cur_h

    return boxes


def main():
    data_info = parse_data_yaml(DETECT_CONFIG["data_yaml"])
    n_class = data_info['nc']
    class_names = data_info['names']

    device = torch.device(DETECT_CONFIG["device"])

    model = FasterRCNN(n_class=n_class).to(device)
    model.load_state_dict(torch.load(DETECT_CONFIG["weights"], map_location=device))
    model.eval()

    img_tensor, (orig_w, orig_h) = preprocess_image(DETECT_CONFIG["img_path"], DETECT_CONFIG["img_size"])
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        bboxes, labels, scores = model.predict(img_tensor, score_thresh=DETECT_CONFIG["score_thresh"])

    bboxes = rescale_boxes(bboxes.cpu().numpy(), DETECT_CONFIG["img_size"], (orig_w, orig_h))
    labels = labels.cpu().numpy()
    scores = scores.cpu().numpy()

    # 可视化
    original_img = Image.open(DETECT_CONFIG["img_path"]).convert("RGB")
    draw = ImageDraw.Draw(original_img)

    # 生成随机颜色
    colors = {name: (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255)) for name in class_names}

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for box, label_idx, score in zip(bboxes, labels, scores):
        class_name = class_names[label_idx - 1]  # 标签从1开始
        color = colors[class_name]

        draw.rectangle(list(box), outline=color, width=3)
        text = f"{class_name}: {score:.2f}"

        text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        draw.rectangle([box[0], box[1] - text_h - 5, box[0] + text_w + 5, box[1]], fill=color)
        draw.text((box[0] + 2, box[1] - text_h - 2), text, fill="white", font=font)

    original_img.show()
    original_img.save("detection_result.jpg")
    print("Detection result saved to detection_result.jpg")


if __name__ == '__main__':
    main()