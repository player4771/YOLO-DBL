import re
import time
import torch
import ctypes
import threading
from pathlib import Path

def get_dataloader(cfg:dict, dataset, transform, is_train:bool, collate_fn): # -> dataloader
    """
    cfg requires:\n
    data(path to data.yaml from dataset),\n
    dataset(dict from data.yaml),\n
    img_size(for resize),\n
    batch_size+num_workers(for dataloader)\n
    \n
    dataset: a dataset class to use
    transform: a transform(class) to use
    """
    data_root = Path(cfg['data']).parent
    img_dir = data_root / dict(cfg['dataset'])['train']
    label_dir = img_dir.parent / 'labels'

    dataset_ = dataset(
        img_dir=img_dir, label_dir=label_dir,
        transform=transform(is_train=is_train, size=cfg['img_size'])
    )
    loader = torch.utils.data.DataLoader(
        dataset_, batch_size=cfg['batch_size'], shuffle=is_train,
        num_workers=cfg['num_workers'], pin_memory=True,
        collate_fn=collate_fn
    )
    return loader

def find_new_dir(dir_: str | Path) -> str | Path: #给定默认路径，寻找下一个未被占用的路径
    ret = str(dir_)
    while Path(ret).exists():
        num = re.search(r'\d+$', ret)
        if num: #结尾有数字：序号+1
            ret = ret[:num.start()] + str(int(num.group(0)) + 1)
        else: #结尾没数字：添加序号2
            ret = ret + '2'
    if isinstance(dir_, str): #返回类型与传入类型保持相同，用起来方便
        return str(ret)
    elif isinstance(dir_, Path):
        return Path(ret)
    else:
        raise TypeError('dir_ = str/Path')

class WindowsRouser:
    """防止电脑休眠"""
    def __init__(self, time:float=None):
        self.time:float = time #定时关闭(单位: 秒)
        self.activated:bool = False #仅用于标记是否在运行中

    def start(self):
        # prevent
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
        self.activated = True
        if self.time:
            #time后自动触发stop
            threading.Timer(self.time, self.stop)

    def stop(self):
        # set back to normal
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        self.activated = False

def this_time() -> str:
    return time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())
