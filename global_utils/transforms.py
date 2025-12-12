import albumentations as A

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

__all__ = (
    'AlbumentationsTransform',
)

class AlbumentationsTransform:
    def __init__(self, is_train=True, size:int|tuple[int,int]=300):
        self.resize_w = size[0] if isinstance(size, tuple) else size
        self.resize_h = size[1] if isinstance(size, tuple) else size

        if is_train:
            self.transform = A.Compose([
                #缩放
                A.Resize(height=self.resize_h, width=self.resize_w),
                #随机翻转
                A.HorizontalFlip(p=0.5),
                #亮度，对比度
                A.RandomBrightnessContrast(p=0.5),
                #色调，饱和度，亮度
                A.HueSaturationValue(p=0.5),
                #缩放
                #A.Affine(), 会扰乱框的位置从而使训练崩坏
                #标准化
                A.Normalize(),
                #转为Tensor, shape: HWC -> CHW, 详见ToTensorV2的注释
                A.pytorch.ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(height=self.resize_h, width=self.resize_w),
                A.Normalize(),
                A.pytorch.ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, image, bboxes, class_labels):
        return self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
