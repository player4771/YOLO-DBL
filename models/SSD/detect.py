import cv2
import yaml
import torch
import random
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms import v2

from utils import create_model, find_new_dir


def detect(**kwargs):
    cfg = {
        'weights': None, #权重文件
        'input': None, #要检测的图片
        'data': None, #data.yaml
        'project': './runs/',
        'name': 'detect',
        'backbone': 'vgg16',
        'conf_thres': 0.5,
        'device': 'cuda'
    }
    cfg.update(kwargs)

    output_dir = find_new_dir(Path(cfg['project'],cfg['name']))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg['data'], 'r') as infile:
        cfg.update(yaml.load(infile, Loader=yaml.FullLoader))

    if Path(cfg['input']).is_file():
        images=[Path(cfg['input'])]
    elif Path(cfg['input']).is_dir():
        images = [p for p in Path(cfg['input']).iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    else:
        raise FileNotFoundError("????????????")

    model = create_model(backbone=cfg['backbone'], num_classes=int(cfg['nc'])+1)
    model.load_state_dict(torch.load(cfg['weights'], map_location=cfg['device']))
    model.to(cfg['device'])
    model.eval()

    # 定义图像预处理流程 (与验证集保持一致)
    transform = v2.Compose([
        v2.Resize((300, 300)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    PALETTE = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
        (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199)
    ]

    print(f"Found {len(images)} images to process.")
    for img in images:
        img_pil = Image.open(img).convert("RGB")
        img_tensor = transform(img_pil).to(cfg['device']).unsqueeze(0)

        with torch.no_grad():
            preds = model(img_tensor)

        pred = preds[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        for box, label, score in zip(boxes, labels, scores):
            if score > cfg['conf_thres']:
                class_name = cfg['names'][label - 1]
                color = PALETTE[(label - 1) % len(PALETTE)]
                x_min, y_min, x_max, y_max = map(int, box)

                cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), color, 2)
                text = f"{class_name}: {score:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_cv, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1)
                cv2.putText(img_cv, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)  # AA=抗锯齿，使字体圆滑

        output = Path(output_dir,img.name)
        cv2.imwrite(str(output), img_cv)
        print(f"Saved result to {output}")

    print("\nDetection finished.")


if __name__ == '__main__':
    detect(
        weights='./runs/train4/best.pth',  # 替换为你的模型路径
        data="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",  # 替换为你的data.yaml路径
        input="E:/Projects/Datasets/example/algal+gray.jpg",  # 替换为你的测试图片或文件夹路径
        conf_thres=0.5,
    )