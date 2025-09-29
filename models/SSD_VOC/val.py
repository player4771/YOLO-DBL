"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""

import os
import json

import torch
from tqdm import tqdm
import numpy as np

import transforms
from src import Backbone, SSD300
from dataset_voc import VOCDataSet
from train_utils import get_coco_api_from_dataset, CocoEvaluator


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def val(**kwargs):
    cfg={
        'voc_root':None, #VOC_ROOT
        'weights':None,
        'project':'./runs/',
        'device':torch.device('cuda' if torch.cuda.is_available() else "cpu"),
        'label_json_path':'./pascal_voc_classes.json',
        'batch_size':8,
        'num_workers':8,
        'num_classes':20,
    }
    cfg.update(kwargs)

    data_transforms = {
        "val": transforms.Compose([transforms.Resize(),
                                   transforms.ToTensor(),
                                   transforms.Normalization()])
    }

    assert os.path.exists(cfg['label_json_path']), "{} dose not exist.".format(cfg['label_json_path'])
    with open(cfg['label_json_path'], 'r') as f:
        class_dict = json.load(f)

    category_index = {v: k for k, v in class_dict.items()}

    # check voc root
    if os.path.exists(os.path.join(cfg['voc_root'], "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(cfg['voc_root']))

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch

    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCDataSet(cfg['voc_root'], "2012", transforms=data_transforms["val"], train_set="val.txt")
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=cfg['batch_size'],
                                                     shuffle=False,
                                                     num_workers=cfg['num_workers'],
                                                     pin_memory=True,
                                                     collate_fn=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=cfg['num_classes'] + 1)

    # 载入你自己训练好的模型权重
    assert os.path.exists(cfg['weights']), "not found {} file.".format(cfg['weights'])
    weights_dict = torch.load(cfg['weights'], map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    # print(model)

    model.to(cfg['device'])

    # evaluate on the test dataset
    coco = get_coco_api_from_dataset(val_dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    cpu_device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(val_dataset_loader, desc="validation..."):
            # 将图片传入指定设备device
            images = torch.stack(images, dim=0).to(cfg['device'])

            # inference
            results = model(images)

            outputs = []
            for index, (bboxes_out, labels_out, scores_out) in enumerate(results):
                # 将box的相对坐标信息（0-1）转为绝对值坐标(xmin, ymin, xmax, ymax)
                height_width = targets[index]["height_width"]
                # 还原回原图尺度
                bboxes_out[:, [0, 2]] = bboxes_out[:, [0, 2]] * height_width[1]
                bboxes_out[:, [1, 3]] = bboxes_out[:, [1, 3]] * height_width[0]

                info = {"boxes": bboxes_out.to(cpu_device),
                        "labels": labels_out.to(cpu_device),
                        "scores": scores_out.to(cpu_device)}
                outputs.append(info)

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_eval = coco_evaluator.coco_eval["bbox"]
    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_eval)

    # calculate voc info for every classes(IoU=0.5)
    voc_map_info_list = []
    for i in range(len(category_index)):
        stats, _ = summarize(coco_eval, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)

    # 将验证结果保存至txt文件中
    with open(cfg['project']+"record_mAP.txt", "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "mAP(IoU=0.5) for each category:",
                        print_voc]
        f.write("\n".join(record_lines))


if __name__ == "__main__":
    val(
        voc_root='E:/Projects/Datasets/',
        weights='./weights/ssd300-20.pth',
        project='./runs/train1',
        batch_size=16,
    )