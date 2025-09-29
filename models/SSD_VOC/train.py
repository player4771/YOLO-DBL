import os
import torch
import datetime
import transforms

from dataset_voc import VOCDataSet
from models.SSD.src import Backbone, SSD300
from train_utils.coco_utils import get_coco_api_from_dataset
import train_utils.train_eval_utils as utils

def create_model(num_classes=21, pre_ssd_path = "./src/nvidia_ssdpyt_fp32.pt"):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    # pre_train_path = "./src/resnet50.pth"
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)

    # https://ngc.nvidia.com/catalog/models -> search ssd -> download FP32

    if os.path.exists(pre_ssd_path) is False:
        raise FileNotFoundError("{} not find".format(pre_ssd_path))
    pre_model_dict = torch.load(pre_ssd_path, map_location='cpu')
    pre_weights_dict = pre_model_dict["model"]

    # 删除类别预测器权重，注意，回归预测器的权重可以重用，因为不涉及num_classes
    del_conf_loc_dict = {}
    for k, v in pre_weights_dict.items():
        split_key = k.split(".")
        if "conf" in split_key:
            continue
        del_conf_loc_dict.update({k: v})

    missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    return model

def train(data:str, **kwargs):
    cfg={ #default config
        'data':data,
        'project':'./runs/',
        'num_classes':20,
        'device':'cuda' if torch.cuda.is_available() else 'cpu',
        'lr':1e-3,
        'batch':8,
        'epochs':100,
        'start_epoch':0,
        'weight_decay':1e-5,
        'patience':10,
        'num_workers':8,
        'size':640,
        'resume':None, #是否恢复训练,None/文件路径
    }
    cfg.update(kwargs)

    if not os.path.exists(cfg['project']):
        os.mkdir(cfg['project'])

    results_file = cfg['project']+"results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transforms = {
        "train": transforms.Compose([transforms.SSDCropping(),
                                     transforms.Resize(),
                                     transforms.ColorJitter(),
                                     transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.Normalization(),
                                     transforms.AssignGTtoDefaultBox()]),
        "val": transforms.Compose([transforms.Resize(),
                                   transforms.ToTensor(),
                                   transforms.Normalization()])
    }

    VOC_root = cfg['data']
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_dataset = VOCDataSet(VOC_root, "2012", data_transforms['train'], train_set='train.txt')
    # 注意训练时，batch_size必须大于1
    assert cfg['batch'] > 1, "batch size must be greater than 1"
    # 防止最后一个batch_size=1，如果最后一个batch_size=1就舍去
    drop_last = True if len(train_dataset) % cfg['batch'] == 1 else False
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=cfg['batch'],
                                                    shuffle=True,
                                                    num_workers=cfg['num_workers'],
                                                    collate_fn=train_dataset.collate_fn,
                                                    drop_last=drop_last)

    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCDataSet(VOC_root, "2012", data_transforms['val'], train_set='val.txt')
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=cfg['batch'],
                                                  shuffle=False,
                                                  num_workers=cfg['num_workers'],
                                                  collate_fn=train_dataset.collate_fn)

    model = create_model(num_classes=cfg['num_classes']+1, pre_ssd_path="./nvidia_ssdpyt_amp_200703.pt")
    model.to(cfg['device'])

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=0.9, weight_decay=cfg['weight_decay'])
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if cfg['resume'] is not None:
        checkpoint = torch.load(cfg['resume'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg['start_epoch'] = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(cfg['start_epoch']))

    train_loss = []
    learning_rate = []
    val_map = []

    # 提前加载验证集数据，以免每次验证时都要重新加载一次数据，节省时间
    val_data = get_coco_api_from_dataset(val_data_loader.dataset)
    for epoch in range(cfg['start_epoch'], cfg['epochs']):
        mean_loss, lr = utils.train_one_epoch(model=model, optimizer=optimizer, data_loader=train_data_loader,
                                              device=cfg['device'], epoch=epoch, print_freq=50)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update learning rate
        lr_scheduler.step()

        coco_info = utils.evaluate(model=model, data_loader=val_data_loader,
                                   device=cfg['device'], data_set=val_data)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch
        }
        if train_loss[-1] == max(train_loss):
            torch.save(save_files, cfg['project']+"ssd300-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == '__main__':
    train(
        data='E:/Projects/Datasets/',
        project='./runs/train2',
        epochs=100,
        patience=10
    )