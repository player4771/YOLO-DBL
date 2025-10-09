import re
import time
import threading
import pandas as pd
import pyautogui as pg
from pathlib import Path

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

def write_coco_stat(stats:list, outfile: str | Path) -> None:
    metric_names = [
        'mAP', 'AP50', 'AP75',
        'APs', 'APm', 'APl',
        'AR1', 'AR10', 'AR100',
        'ARs', 'ARm', 'ARl'
    ]

    if not Path(outfile).exists():
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        # Dataframe默认接受一个二维列表，第一维为列，内部每个列表为行
        df = pd.DataFrame([stats], columns=metric_names)
    else:
        df = pd.read_csv(outfile)
        if df.empty:
            df = pd.DataFrame([stats], columns=metric_names)
        else:
            df = pd.concat([df, pd.DataFrame([stats], columns=metric_names)], ignore_index=True)

    df.to_csv(outfile, index=False)

class WindowsSleepAvoider:
    def __init__(self, delay:int=0, distance:int=20):
        self.delay:int = delay #每次移动鼠标的时间间隔(单位: 秒)
        self.distance:int = distance #鼠标移动的幅度(单位：像素)
        self.count:int = 0 #计数，统计当前为第几次触发
        self.running:bool = False #是否启动

    def trigger(self):
        """移动一次鼠标.\n
        偶数次时右移, 否则左移. 用时固定为0.5秒"""
        offset = self.distance if self.count % 2 == 0 else -self.distance
        pg.move(offset, 0, 0.5)
        self.count += 1

    def loop(self):
        while self.running:
            self.trigger()
            time.sleep(self.delay)

    def start(self):
        self.running = True
        th = threading.Thread(target=self.loop, name="WindowsSleepAvoider(running)")
        th.start()

    def stop(self):
        self.running = False