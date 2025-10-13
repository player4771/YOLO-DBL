import re
import time
import torch
import threading
import pyautogui as pg
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
    """定时移动鼠标从而防止电脑休眠"""
    def __init__(self, delay:int=0, distance:int=20):
        self.delay:int = delay #每次移动鼠标的时间间隔(单位: 秒)
        self.distance:int = distance #鼠标移动的幅度(单位：像素)
        self.count:int = 0 #计数，统计当前为第几次触发
        self.running:bool = False #是否启动

    def trigger(self):
        """移动一次鼠标.\n
        偶数次时右移, 否则左移. 用时固定为0.5秒"""
        offset = self.distance if self.count % 2 == 0 else -self.distance
        pg.move(xOffset=offset, yOffset=0, duration=0.5)
        self.count += 1

    def loop(self):
        while self.running:
            self.trigger()
            time_int = self.delay // 1 #整数部分
            time_dec = self.delay - time_int #小数部分
            for i in range(time_int):
                if self.running: #保证及时响应
                    time.sleep(1)
            time.sleep(time_dec)

    def start(self, interval:float=None):
        self.running = True
        th = threading.Thread(target=self.loop, name="WSA(running)")
        th.start()
        if interval:
            threading.Timer(
                interval,
                lambda:setattr(self, 'running', False)
            ).start()

    def stop(self):
        self.running = False

def this_time() -> str:
    return time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())
