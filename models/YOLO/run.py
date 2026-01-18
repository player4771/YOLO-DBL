# https://github.com/iMoonLab/yolov13
# https://docs.ultralytics.com/zh/modes/train/#augmentation-settings-and-hyperparameters
import torch.nn

from ultralytics import YOLO
from global_utils import WindowsRouser

#import sys
#sys.path.append('/root/project/Paper2/')
#sys.path.append(r"E:\Projects\PyCharm\YOLO\yolov3")


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
        device="cuda",
        #lr0=1e-3,  # n:1e-4, s:1e-3. optimizer=auto时会被忽略
        lrf=1e-2,
        patience=20,
        cos_lr=True,
        amp=True,

        # 删除的是严重影响训练效果的，注释掉的是可能影响效果或者收敛速度的
        imgsz=640,
        #hsv_h=0.015,
        #hsv_s=0.7,
        #hsv_v=0.4,
        scale=0.9,  # S:0.9; L:0.9; X:0.9
        #bgr=0.5,
        mosaic=1.0,
        mixup=0.05,  # S:0.05; L:0.15; X:0.2
        copy_paste=0.15,  # S:0.15; L:0.5; X:0.6
        #erasing=0.2,

        #loss函数权重
        #box=8.5,
        #cls=0.5,
        #dfl=1.5,
    )

    rouser.stop()
    return model, results

if __name__ == '__main__':
    data_yaml = "E:/Projects/Datasets/tea_leaf_diseases/data.yaml"
    model_file = r"yolov13s_edit10.yaml"

    model, _ = train(model_file, data_yaml)
    #YOLO(model_file).val(data=data_yaml)
    #YOLO(model_file).info(detailed=False)