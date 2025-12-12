import yaml
import torch
from pathlib import Path

from train import create_model
from global_utils import coco_evaluate, YOLODataset, AlbumentationsTransform

def collate_fn(batch):
    """
    DataLoader 的 collate_fn，因为每张图的 box 数量不同。
    """
    return tuple(zip(*batch))

@torch.inference_mode()
def val(args_file:str|Path=None, **kwargs):
    cfg={}
    if args_file:
        cfg.update(yaml.load(Path(args_file).read_text(), Loader=yaml.FullLoader))
    cfg.update(kwargs)
    device = torch.device(cfg['device'])
    model = (create_model(cfg['backbone']))
    model.load_state_dict(torch.load(cfg['weights'], map_location=device))
    model.to(device)
    model.eval()

    with open(cfg['data'], 'r') as infile:
        data_yaml = yaml.load(infile, Loader=yaml.FullLoader)

    data_root = Path(cfg['data']).parent
    img_dir = data_root / data_yaml['val']
    label_dir = img_dir.parent / 'labels'

    dataset = YOLODataset(
        img_dir=img_dir, label_dir=label_dir,
        transform=AlbumentationsTransform(is_train=False, size=300)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
        pin_memory=True, shuffle=False, persistent_workers=True,
        collate_fn=collate_fn
    )

    coco_evaluate(model, loader, device, outfile=None)

if __name__ == '__main__':
    input_dir = Path("./runs/train4")
    val(
        args_file=input_dir/"args.yaml",
        backbone="vgg16",
        weights=input_dir/"best.pth",
    )