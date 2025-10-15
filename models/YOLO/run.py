# https://github.com/iMoonLab/yolov13.git
# https://github.com/iMoonLab/yolov13?tab=readme-ov-file#quick-start-

import torch
from ultralytics import YOLO

from global_utils import WindowsRouser


def train(model:str, data:str):
    rouser = WindowsRouser()
    rouser.start()

    model = YOLO(model)

    # Train the model
    results = model.train(
        data=data,
        project='./runs',
        epochs=300,
        batch=8, # -1:显存利用率60%;0-1:指定显存利用率
        workers=4,
        imgsz=640,
        scale=0.9,  # S:0.9; L:0.9; X:0.9
        mosaic=1.0,
        mixup=0.05,  # S:0.05; L:0.15; X:0.2
        copy_paste=0.15,  # S:0.15; L:0.5; X:0.6
        device="cuda",
        lr0=1e-3,  # n:1e-4, s:1e-3 optimizer=auto时会被忽略
        lrf=1e-2,
        patience=10,
        cos_lr=True
    )

    rouser.stop()
    return model, results

def val(model, data:str):
    # Evaluate model performance on the validation set
    metrics = model.val(data=data)
    return metrics

if __name__ == '__main__':
    data_yaml = "E:/Projects/Datasets/tea_leaf_diseases/data.yaml"

    model, _ = train('YOLOv13_edit.yaml', data_yaml)
    #val(model, data_yaml)