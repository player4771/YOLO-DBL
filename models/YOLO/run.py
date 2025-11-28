# https://github.com/iMoonLab/yolov13
# https://docs.ultralytics.com/zh/modes/train/#augmentation-settings-and-hyperparameters

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
        device="cuda",
        #lr0=1e-3,  # n:1e-4, s:1e-3. optimizer=auto时会被忽略
        lrf=1e-2,
        patience=20,
        cos_lr=True,
        amp=True,

        # 删除的是严重影响训练效果的，注释掉的是可能影响效果或者收敛速度的
        imgsz=640,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.5,  # 这个对于遮挡物体识别很重要
        scale=0.9,  # S:0.9; L:0.9; X:0.9
        bgr=0.5,
        mosaic=1.0,
        mixup=0.05,  # S:0.05; L:0.15; X:0.2
        copy_paste=0.15,  # S:0.15; L:0.5; X:0.6
        # erasing=0.9,
    )

    rouser.stop()
    return model, results


if __name__ == '__main__':
    data_yaml = "E:/Projects/Datasets/tea_leaf_diseases/data.yaml"

    model, _ = train('yolov13s_v3edit9.yaml', data_yaml)
    #val(model, data_yaml)