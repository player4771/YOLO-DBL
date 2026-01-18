#包含函数内导入
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

    def apply_image(self, image):
        """image: np.ndarray, format:HWC"""
        return self.transform(image=image, bboxes=[], class_labels=[])['image']

    def __call__(self, image, bboxes=None, class_labels=None):
        """image: torch.Tensor, format:CHW"""
        if bboxes:
            image = image.permute(1,2,0).numpy() # CHW -> HWC
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            return transformed
        else:
            return self.apply_image(image)


def transform_visualization(image_file, sequence=False):
    """
    :param image_file: Image Path
    :param sequence: each augment for one image, or apply all
    :return: Processed image
    """
    import cv2
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    transforms=[
        A.HueSaturationValue(0.015*180, 0.7*255, 0.4*255, p=1),
        A.RandomScale(0.9, p=1),
        A.ChannelShuffle(p=1),
        A.CoarseDropout(p=1),
    ]
    results = []
    for transform in transforms:
        if sequence:
            image = transform(image=image)['image']
            results.append(image)
        else:
            results.append(transform(image=image)['image'])
    for i, img in enumerate(results):
        cv2.imwrite(f"./cache/transform{i+1}.png", img)
    return results

def image_split(image_file, x, y):
    import cv2
    img = cv2.imread(image_file, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    imgs = (img[0:y, 0:x], img[0:y, x:w], img[y:h, 0:x], img[y:h, x:w]) #左上, 右上, 左下, 右下
    for i, img in enumerate(imgs):
        cv2.imwrite(f"./cache/split{i+1}.png", img)


def transform_visualization2(image_path:str):
    import cv2
    import numpy as np
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w, c = img.shape

    results = []

    #Mosaic: 缩小并平铺为2x2
    small = cv2.resize(img, (w//2, h//2))
    results.append(np.tile(small, (2, 2, 1)))

    #HSV: 调整色调、饱和度、明度
    results.append(A.HueSaturationValue(
        hue_shift_limit=(50, 50),
        sat_shift_limit=(30, 30),
        val_shift_limit=(20, 20),
        p=1
    )(image=img)['image'])

    #Scale: 随机缩放
    scaled = cv2.resize(img, (h//2,w//2))
    h_scaled = scaled.shape[0]
    w_scaled = scaled.shape[1]
    y_off = (h - h_scaled) // 2
    x_off = (w - w_scaled) // 2
    canvas = np.full((h, w, c), 255, dtype=np.uint8)
    canvas[y_off:y_off+h_scaled, x_off:x_off+w_scaled] = scaled
    results.append(canvas)

    #Erasing: 随机挖孔 (CoarseDropout)
    results.append(A.CoarseDropout(
        num_holes_range=(3,3),
        hole_height_range=(h//5, h//5),
        hole_width_range=(w//4, w//4),
        fill=0.0,
        p=1
    )(image=img)['image'])

    #MixUp: 与水平翻转后的自身进行加权融合
    flipped = cv2.flip(img, 1)
    results.append(cv2.addWeighted(img, 0.5, flipped, 0.5, 0))

    #Copy Paste: 将中心区域裁剪并覆盖至左上角
    cp_img = img.copy()
    cw, ch = w//3, h//3
    cp_img[0:ch, 0:cw] = img[h//2:h//2+ch, w//2:w//2+cw]
    results.append(cp_img)

    #Flip: 水平翻转
    results.append(A.HorizontalFlip(p=1)(image=img)['image'])

    outfiles = []
    for i,img in enumerate(results):
        outfile = f"./cache/transform{i+1}.png"
        cv2.imwrite(outfile, img)
        outfiles.append(outfile)

    return outfiles

def display_images(img_file:list[str]|str, titles:list[str]=None, rows:int=None, cols:int=None):
    from matplotlib import pyplot as plt
    from tools import get_num_files

    files = get_num_files(img_file) if isinstance(img_file, str) else img_file

    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.axis('off')
        img = plt.imread(files[i])
        ax.imshow(img)
        ax.set_title(titles[i] if titles else f"Image{i}")

    plt.tight_layout()
    plt.savefig("./cache/augment.png", transparent=True)
    plt.show()

if __name__ == '__main__':
    img_raw = r"E:\Projects\Datasets\example\healthy.jpg"
    outfiles = transform_visualization2(img_raw)
    outfiles.insert(0, img_raw)
    display_images(outfiles, ['Original', 'Mosaic', 'HSV', 'Scale', 'Erasing', 'Mixup', 'Copy-Paste', 'Flip'], 2, 4)