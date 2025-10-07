import re
import pandas as pd
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
