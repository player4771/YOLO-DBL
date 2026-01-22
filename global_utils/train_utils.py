
import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm
from thop import profile
from copy import deepcopy
from pprint import pprint
from pathlib import Path

from .dataset import YOLODataset
from .tools import find_new_dir, WindowsRouser, time_now_str, rand_rgb, time_sync, type_str
from .coco import convert_to_coco_api, COCOEvaluator

__all__ = (
    'EarlyStopping',
    'Trainer',
    'default_val',
    'default_detect',
)

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


def default_collate_fn(batch):
    return tuple(zip(*batch))

class Trainer:
    def __init__(
            self,
            model,
            data,
            project='./runs',
            name='train',
            amp=True,
            epochs=20,
            lr=0.001,
            lf=0.01,
            batch=8,
            workers=4,
            patience=7,
            warmup=0,
            weight_decay=1e-5,
            collate_fn=None,
            transform_train=None,
            transform_val=None,
            no_sleep=False,
            **kwargs
    ):
        self.model = model
        collate_fn = collate_fn if collate_fn else default_collate_fn
        cfg = {
            'data':data,
            'project': project,
            'name': name,
            'amp': amp,
            'epochs': epochs,
            'lr': lr,
            'lf': lf,
            'batch': batch,
            'workers': workers,
            'patience': patience,
            'warmup': warmup, #预热epoch数
            'weight_decay': weight_decay,
            'transform_train': transform_train,
            'transform_val': transform_val,
            'no_sleep': no_sleep,
            'max_norm': 10.0,
        }
        cfg.update(kwargs)

        if "device" in cfg:
            cfg['device'] = 'cuda' if torch.cuda.is_available() and amp else 'cpu'
        self.device = torch.device(cfg['device'])
        output_dir = find_new_dir(Path(project, name))
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

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
        self.dataset_train = YOLODataset(
            img_dir=self.train_img_dir, label_dir=self.train_label_dir,
            transform=transform_train
        )
        self.dataset_val = YOLODataset(
            img_dir=self.val_img_dir, label_dir=self.val_label_dir,
            transform=transform_val
        )

        # 创建 DataLoader
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset_train, batch_size=batch, num_workers=workers,
            pin_memory=True, shuffle=True, persistent_workers=True,
            collate_fn=collate_fn
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.dataset_val, batch_size=batch, num_workers=workers,
            pin_memory=True, shuffle=False, persistent_workers=True,
            collate_fn=collate_fn
        )

        self.model = model.to(self.device)

        self.scaler = torch.amp.GradScaler(enabled=amp)
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs,eta_min=lr * lf)
        self.early_stopper = EarlyStopping(patience=patience, outfile=output_dir/'best.pth', mode='max')
        warmup_iters = warmup * len(self.train_loader)
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-3, total_iters=warmup_iters)
        self.evaluator = COCOEvaluator(outdir=output_dir, coco_gt=convert_to_coco_api(self.val_loader.dataset))

        self._cfg = cfg
        self.dump_args(status='init_finished')

    def start_training(self, **kwargs):
        if len(kwargs) > 0:
            self._cfg.update(kwargs)
            print("警告：不建议在训练时更改参数，请在Trainer初始化时提供。")
        self._cfg['start_time'] = time_now_str()

        self.dump_args(verbose=True, status='training_started')

        rouser = WindowsRouser()
        if self._cfg['no_sleep']:
            rouser.start()

        flops, params = profile(deepcopy(self.model), (torch.rand(1, 3, 256, 256).to(self.device),))
        self._cfg.update({'flops': flops, 'params': params})
        print(f"Info: {params/1e6:.4f}M params, {flops/1e9:.4f} GFLOPs", flush=True)

        for epoch in range(self._cfg['epochs']):
            self.model.train()
            #TODO: 添加更多自定义部分，以适配更多模型
            pbar = tqdm(self.train_loader, desc=f"Train[{epoch}]")
            for (images, targets) in pbar:
                #idea: 自定义images/targets的转换函数？
                images = torch.stack(images, dim=0).to(device=self.device, non_blocking=True)
                targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]

                with torch.amp.autocast(device_type=self._cfg['device'], enabled=self._cfg['amp']):
                    loss_dict = self.model(images, targets)
                    losses = torch.stack(list(loss_dict.values())).sum()

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(losses).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self._cfg['max_norm'])
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if epoch < self._cfg['warmup']:
                    self.warmup_scheduler.step()

                pbar.set_postfix(lr=self.optimizer.param_groups[0]['lr'], loss=losses.item())

            if epoch >= self._cfg['warmup']:
                self.scheduler.step()

            coco_eval = self.evaluator.evaluate(self.model, self.val_loader)
            mAP = coco_eval.stats[0] if coco_eval else 0.0

            self.early_stopper.update(mAP, self.model)
            if self.early_stopper.early_stop:
                print(f'Early stopping with mAP {mAP:.6f}')
                break

        self._cfg.update({'end_time': time_now_str()})
        print(f"Training finished, Results saved to {self.results_file}.")
        self.dump_args(status='training_finished')
        if self._cfg['no_sleep']:
            rouser.stop()

    def dump_args(self, verbose=False, **kwargs):
        cfg = self._cfg | {
            'dataset_train': type_str(self.dataset_train),
            'dataset_val': type_str(self.dataset_val),
            'dataloader_train': type_str(self.train_loader),
            'dataloader_val': type_str(self.val_loader),
            "transform_train": type_str(self._cfg["transform_train"]),
            "transform_val": type_str(self._cfg["transform_val"]),
            'scaler': type_str(self.scaler),
            'optimizer': type_str(self.optimizer),
            'scheduler': type_str(self.scheduler),
            'early_stopper': type_str(self.early_stopper),
            'warmup_scheduler': type_str(self.warmup_scheduler),
            'evaluator': type_str(self.evaluator),

        }

        cfg.update(kwargs)

        if verbose:
            pprint(cfg, indent=2)

        with open(self.output_dir/"args.yaml", 'w') as f:
            yaml.dump(cfg, f, indent=4, encoding='utf-8')
        print(f"args file saved to {self.results_file}.")



@torch.inference_mode()
def default_val(model, input_dir:str|Path, transform=None, **kwargs):
    input_dir = Path(input_dir)
    cfg:dict[str, int|str] = {
        'weights': str(input_dir/'best.pth'),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    cfg.update(yaml.load((input_dir/'args.yaml').read_text(), Loader=yaml.FullLoader))
    cfg.update({'transform_val': transform})
    cfg.update(kwargs)

    with open(cfg['data'], 'r') as infile:
        data_yaml = yaml.load(infile, Loader=yaml.FullLoader)

    data_root = Path(cfg['data']).parent
    img_dir = data_root / data_yaml['val']
    label_dir = img_dir.parent / 'labels'

    dataset = YOLODataset(
        img_dir=img_dir, label_dir=label_dir,
        transform=cfg['transform_val'],
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['batch'], num_workers=cfg['workers'],
        pin_memory=True, shuffle=False, persistent_workers=True,
        collate_fn=cfg['collate_fn'] if 'collate_fn' in cfg else default_collate_fn,
    )
    evaluator = COCOEvaluator(outdir=cfg['outdir'] if 'outdir' in cfg else None)
    evaluator.evaluate(model.to(cfg['device']).eval(), loader)



@torch.inference_mode()
def default_detect(
        model,
        data:str,
        input_:str,
        project:str=None,
        name:str='detect',
        img_size:int=640,
        device=None,
        transform=None,
        conf_thres:float=0.1
):
    with open(data, 'r') as infile:
        cfg_dataset = yaml.load(infile, Loader=yaml.FullLoader)

    if Path(input_).is_file():
        images = [Path(input_)]
    else:
        images = [p for p in Path(input_).iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    output_dir = None
    if project is not None:
        output_dir = find_new_dir(Path(project, name))
        output_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    times = []
    print(f"Found {len(images)} images to process.")
    for image in images:
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        h, w, _ = img.shape

        start_time = time_sync()
        with torch.no_grad():
            preds = model([transform(img).to(device)])

        pred = preds[0]
        boxes = pred['boxes'].cpu().numpy()
        boxes[:, [0, 2]] *= w/img_size
        boxes[:, [1, 3]] *= h/img_size
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if score >= conf_thres:
                class_name = cfg_dataset['names'][label - 1]
                color = rand_rgb()
                x_min, y_min, x_max, y_max = map(int, box)

                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                text = f"{class_name}: {score:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1)
                cv2.putText(img, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)  # AA=抗锯齿，使字体圆滑

        times.append(time_sync() - start_time)
        if output_dir is not None:
            output = output_dir/Path(image).name
            cv2.imwrite(str(output), img)
            print(f"Saved result to {output}")

    if len(times) > 2:
        times.remove(max(times))
        times.remove(min(times))
    avg_time = sum(times) / len(times)
    print(f"\nDetection finished. Average time: {avg_time:.4f}")
