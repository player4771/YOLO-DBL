# https://github.com/iMoonLab/yolov13.git
# https://github.com/iMoonLab/yolov13?tab=readme-ov-file#quick-start-

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from ultralytics import YOLO

def nan_stop(trainer):
    loss = trainer.loss
    print(f'loss: {loss}, lr: {trainer.optimizer.param_groups[0]["lr"]}')
    if loss is not None and torch.isnan(loss):
        print("Stoping training...")
        trainer.stop = True

def train(model_input:str, data:str):
    assert os.path.exists(data), f'{data} does not exist'
    model = YOLO(model_input)

    model.add_callback('on_train_epoch_end', nan_stop)

    # Train the model
    results = model.train(
        data=data,
        project='./runs/detect_train',
        epochs=200,
        batch=8,#-1:显存利用率60%;0-1:指定显存利用率
        imgsz=640,
        scale=0.9,  # S:0.9; L:0.9; X:0.9
        mosaic=1.0,
        mixup=0.05,  # S:0.05; L:0.15; X:0.2
        copy_paste=0.15,  # S:0.15; L:0.5; X:0.6
        device="cuda",
        lr0=1e-3,  # n:1e-4, s:1e-3
        lrf=1e-2,
        patience=10,
        cos_lr=True,
        amp=False
    )

    return model, results

def tune(model_input:str, data:str):
    model = YOLO(model_input)

    # Define search space
    search_space = {
        "lr0": (1e-5, 1e-1),
        #"lrf": (1e-4, 1e-1),
        "degrees": (0.0, 45.0),
    }

    # Tune hyperparameters on COCO8 for 30 epochs
    results:dict = model.tune(
        data=data,
        project='./runs/detect_tune',
        epochs=5,
        iterations=20, #搜索次数
        patience=5,
        optimizer="AdamW",
        space=search_space,
        plots=False,
        save=False,
        val=False,
    )

    return results

def val(model, data:str):
    # Evaluate model performance on the validation set
    metrics = model.val(data=data)
    return metrics

def detect(model, sample:str):
    # Perform object detection on an image
    results = model(sample)
    results[0].show()
    return results

if __name__ == '__main__':
    data_yaml = "E:/Projects/Datasets/tea_leaf_diseases/data.yaml"
    #data_yaml = '/root/autodl-tmp/tea_leaf_diseases/data.yaml'
    sample = "./assets/brown_blight.png"
    #assert os.path.exists(sample), f'{sample} does not exist'

    model, _ = train('yolov13s.yaml', data_yaml)
    #val(model, data_yaml)
    #detect(model, sample)
    #detect(YOLO('./runs/detect/train10/weights/best.pt'), sample)
    #tune('yolov13s.yaml', data_yaml)