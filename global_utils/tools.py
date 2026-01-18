import re
import time
import torch
import ctypes
import threading
from pathlib import Path
from numpy.random import randint

__all__ = (
    'get_dataloader',
    'find_new_dir',
    'WindowsRouser',
    'time_now_str',
    'typename',
    'avg_time',
    'check_time',
    'rand_rgb',
    'get_num_files',
    'time_sync',
)

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
    """防止电脑休眠\n
    TODO:改为全局方法？"""
    def __init__(self, autostop:float=None):
        self.time:float = autostop #定时关闭(单位: 秒)，None为禁用
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

def time_now_str(sep_date='/', sep_time=':', sep_datetime=' ') -> str:
    """
    获取当前时间的格式化字符串。\n
    默认格式为"%Y/%m/%d %H:%M:%S"，如"1145/14/19 19:81:00"\n
    (好像有哪里不对)\n
    注：由于python不区分char和str，间隔符实际上可以是多个字符

    :param sep_date: 年月日分隔符
    :param sep_time: 时分秒分隔符
    :param sep_datetime: 日期与时间之间的间隔符
    :return: 格式化的日期时间字符串
    """
    format_ = f'%Y{sep_date}%m{sep_time}%d{sep_datetime}%H{sep_time}%M{sep_time}%S' #有点抽象...
    return time.strftime(format_, time.localtime())

def typename(class_):
    #如: <class 'ultralytics.nn.modules_attention.BiFormer.biformer.BiFormer'> -> BiFormer
    return re.search(r"<class '.*\.(.*)'>", str(type(class_))).group(1)

def avg_time(module, *args, repeat=10):
    result = module(*args)  # 运行一下，忽略编译时间
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for i in range(repeat):
        module(*args)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    return total_time, result

def check_time(module, *args, repeat=10, log=True, adjust=25):
    print(f"{typename(module)}:".ljust(adjust), end='')
    total_time, result = avg_time(module, *args, repeat=repeat)
    if log:
        try:
            print(f"-> {result.shape},".ljust(40), f"{total_time/repeat:.16f}s")
        except:
            print(f"-> {type(result)},".ljust(40), f"{total_time/repeat:.16f}s")
    return total_time/repeat, result

def rand_rgb():
    return randint(0, 256), randint(0, 256), randint(0, 256)

def num_sort_fn(file:str) -> list:
    """将文件的整个路径按以下规则排序：\n
    1. 对于非数字部分，逐位按字母顺序排序；\n
    2. 若某一位为数字，将从这一位开始所有连续的数字作为一个整体，按整体数字的大小排序。\n
    如:E:/test11/aaa4514.txt会被切分为：E:/test, (数字)11, /aaa, (数字)4514, .txt"""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', file)]

def get_num_files(file:str|Path) -> list[str]:
    """给定一个不带序号的文件路径，寻找同目录下所有带有序号的文件路径\n
    如给定C:/a.txt，寻找C:/a0.txt, C:/a1.txt等\n
    返回的列表保证按照后缀序号的数字大小(非字母序)升序排序"""
    file = Path(file)
    pattern = re.compile(rf".*{re.escape(file.stem)}(\d*){re.escape(file.suffix)}$")  # \d*而不是\d+是为了包含file本身
    files = [str(path) for path in file.parent.iterdir() if path.is_file()]  #Path->str
    files = [file for file in files if pattern.match(file)]
    files = sorted(files, key=num_sort_fn)
    return files

def time_sync(device=None):
    """Returns time(float) with cuda synchronized"""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return time.perf_counter()
