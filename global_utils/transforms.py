import albumentations as A

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

__all__ = (
    'ATransforms',
)

class ATransforms:
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

    def __call__(self, image, boxes, class_labels):
        return self.transform(image=image, bboxes=boxes, class_labels=class_labels)

def ATransform_visualization(image):
    """
    :param image: Image Data (ndarray/Tensor/...)
    :return: Processed image
    """
    transforms=[
        A.HueSaturationValue(0.015*180, 0.7*255, 0.4*255, p=1),
        A.RandomScale(0.9, p=1),
        A.ChannelShuffle(p=1),
        A.CoarseDropout(p=1),
    ]
    results = []
    for transform in transforms:
        image = transform(image=image)['image']
        results.append(image)
    return results

if __name__ == '__main__':
    import cv2
    sample = cv2.imread(r"E:\Projects\Datasets\example\sample_v4_1.jpg", cv2.IMREAD_COLOR_RGB)
    results = ATransform_visualization(sample)
    for i, img in enumerate(results):
        cv2.imwrite(f"./cache/transform{i+1}.png", img)