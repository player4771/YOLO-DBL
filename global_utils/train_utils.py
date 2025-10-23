
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from .dataset import YoloDataset
from .tools import find_new_dir, WindowsRouser
from .coco import convert_to_coco_api, coco_evaluate

class EarlyStopping:
    """
    当监控的指标停止改善时，提前停止训练。
    """
    def __init__(self, patience:int=7, delta:float=0, outfile:str|Path='./best.pth', mode:str='min', verbose:bool=True):
        """
        Args:
            patience (int): 在停止训练前，等待多少个 epoch 没有改善。
            delta (float):  被认为是改善的最小变化量。
            outfile (str):     保存最佳模型的路径。
            mode (str):     指定分数是越低越好(min)还是越高越好(max)。
            verbose (bool): 如果为 True，则为每次改善打印一条信息。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.outfile = outfile
        self.mode = mode
        assert self.mode in ['min', 'max'], f"mode 只能为 min 或 max, 但传入了{mode}"
        self.best_score = np.inf if mode == 'min' else -np.inf

    def __call__(self, score:float, model):
        """用法和效果等同于update(...)"""
        self.update(score, model)

    def update(self, score:float, model):
        """当score提高时保存模型， 否则早停计数+1"""

        improvement = (((self.mode == 'min') and (score < self.best_score - self.delta)) #越低越好
                    or ((self.mode == 'max') and (score > self.best_score + self.delta))) #越高越好

        if improvement:
            if self.verbose:
                # 在更新 best_score 之前打印
                print(f'Val score improved: {self.best_score:.6f} -> {score:.6f}.  Saving model ...')

            self.save_checkpoint(model)
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping : {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """当指标改善时，保存模型。"""
        torch.save(model.state_dict(), str(self.outfile))


class Trainer:
    def __init__(self,
                 model,
                 data,
                 project='./runs',
                 name='train',
                 amp=True,
                 epochs=20,
                 lr=0.001,
                 batch_size=8,
                 num_workers=4,
                 patience=7,
                 warmup=0,
                 weight_decay=1e-5,
                 collate_fn=None,
                 transform_train=None,
                 transform_val=None,
                 no_sleep=True,
                 **kwargs
                 ):
        self.model = model
        cfg = {
            'data':data,
            'collate_fn': collate_fn,
            'project': project,
            'name': name,
            'amp': amp,
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'patience': patience,
            'warmup': warmup,
            'weight_decay': weight_decay,
            'transform_train': transform_train,
            'transform_val': transform_val,
            'no_sleep': no_sleep,
        }
        cfg.update(kwargs)

        cfg['device'] = 'cuda' if torch.cuda.is_available() and amp else 'cpu'
        self.device = torch.device(cfg['device'])
        output_dir = find_new_dir(Path(project, name))
        output_dir.mkdir(parents=True, exist_ok=True)

        if 'dataset' not in cfg:
            with open(data, 'r') as f:
                cfg['dataset'] = yaml.load(f, Loader=yaml.FullLoader)

        data_root = Path(data).parent
        self.train_img_dir = data_root / dict(cfg['dataset'])['train']
        self.val_img_dir = data_root / dict(cfg['dataset'])['val']
        self.train_label_dir = self.train_img_dir.parent / 'labels'
        self.val_label_dir = self.val_img_dir.parent / 'labels'
        self.results_file = output_dir / 'results.csv'

        # 创建 Dataset
        dataset_train = YoloDataset(
            img_dir=self.train_img_dir, label_dir=self.train_label_dir,
            transform=transform_train
        )
        dataset_val = YoloDataset(
            img_dir=self.val_img_dir, label_dir=self.val_label_dir,
            transform=transform_val
        )

        # 创建 DataLoader
        self.train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, shuffle=True, persistent_workers=True,
            collate_fn=collate_fn
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, shuffle=False, persistent_workers=True,
            collate_fn=collate_fn
        )

        self.model = model.to(self.device)

        self.scaler = torch.amp.GradScaler(enabled=amp)
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        lf = cfg['lf'] if 'lf' in cfg else 0.01
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs,eta_min=lr * lf)
        self.early_stopper = EarlyStopping(patience=patience, outfile=output_dir/'best.pth', mode='max')
        warmup_iters = warmup * len(self.train_loader)
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-3, total_iters=warmup_iters)

        self.cfg = cfg
        with open(output_dir / 'args.yaml', 'w') as f:
            yaml.dump(cfg, f)

    def start_training(self, **kwargs):
        if len(kwargs) > 0:
            print('警告：不建议在训练时更改参数，请在初始化时提供。')
        self.cfg.update(kwargs)
        self.cfg['start_time'] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        coco_gt = convert_to_coco_api(self.val_loader.dataset)

        rouser = WindowsRouser()
        if self.cfg['no_sleep']:
            rouser.start()

        for epoch in range(self.cfg['epochs']):
            self.model.train()
            #TODO: 添加更多自定义部分，以适配更多模型
            pbar = tqdm(self.train_loader, desc=f"Train[{epoch}]")
            for (images, targets) in pbar:
                images = torch.stack(images, dim=0).to(device=self.device, non_blocking=True)
                targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]

                with torch.amp.autocast(device_type=self.cfg['device'], enabled=self.cfg['amp']):
                    loss_dict = self.model(images, targets)
                    losses = torch.Tensor(sum(loss for loss in loss_dict.values()))

                self.optimizer.zero_grad()
                self.scaler.scale(losses).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if epoch < self.cfg['warmup_epochs']:
                    self.warmup_scheduler.step()

                pbar.set_postfix(lr=self.optimizer.param_groups[0]['lr'], loss=losses.item())

            if epoch >= self.cfg['warmup_epochs']:
                self.scheduler.step()

            coco_eval = coco_evaluate(self.model, self.val_loader, self.device, coco_gt=coco_gt, outfile=self.results_file)
            mAP = coco_eval.stats[0] if coco_eval else 0.0

            self.early_stopper.update(mAP, self.model)
            if self.early_stopper.early_stop:
                print(f'Early stopping with mAP {mAP:.6f}')
                break

        print(f"Training finished, Results saved to {self.results_file}.")
        rouser.stop()

        return self.model, self.cfg