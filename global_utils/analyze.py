import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

__all__ = (
    'parse_results',
    'plt_coco_f1',
    'plt_coco_ap',
    'plt_coco_ar',
    'plt_coco_stats',
)

def parse_results(results:str|Path|pd.DataFrame) -> pd.DataFrame:
    """输入：results.csv的路径 或 dataframe \n
    csv/df的列格式需与cocoEval输出一致(6AP+6AR) \n
    (不进行内容检查)"""
    if isinstance(results, str):
        return pd.read_csv(results)
    elif isinstance(results, Path):
        return pd.read_csv(str(results))
    elif isinstance(results, pd.DataFrame):
        return results
    else:
        raise TypeError("Unknown input, need str or pd.DataFrame")



def plt_coco_f1(results, show=True):
    """绘制F1曲线"""
    data = parse_results(results)
    x = range(1, data.shape[0]+1)

    precision = data.iloc[:,0] #取precision = mAP = AP@IoU=0.50:0.95
    recall = data.iloc[:,8] #取recall = AR100 = AR@maxDets=100
    f1 = (2*precision*recall)/(precision+recall)

    fig, ax = plt.subplots(dpi=300)
    ax.plot(x, f1)
    ax.set_title('F1 - mAP/AR100')
    if isinstance(results, str):
        fig.savefig(Path(results).parent/'f1.png')
    if show:
        fig.show()
    return fig, ax

def plt_coco_ap(results, show=True):
    data = parse_results(results)
    x = range(1, data.shape[0]+1)

    fig, axes = plt.subplots(nrows=2, ncols=3, dpi=300)
    for i, ax in enumerate(axes.flat):
        ax.plot(x, data.iloc[:,i])
        ax.set_title(data.columns[i], fontsize=10)
    fig.tight_layout()
    if isinstance(results, str):
        fig.savefig(Path(results).parent/'results_AP.png')
    if show:
        fig.show()
    return fig, axes

def plt_coco_ar(results, show=True):
    data = parse_results(results)
    x = range(1, data.shape[0]+1)

    fig, axes = plt.subplots(nrows=2, ncols=3, dpi=300)
    for i, ax in enumerate(axes.flat):
        ax.plot(x, data.iloc[:, i+6])
        ax.set_title(data.columns[i+6], fontsize=10)
    fig.tight_layout()
    if isinstance(results, str):
        fig.savefig(Path(results).parent / 'results_AR.png')
    if show:
        fig.show()
    return fig, axes

def plt_coco_stats(results_file:str|Path, show=True):
    plt_coco_ap(results_file, show)
    plt_coco_ar(results_file, show)
    plt_coco_f1(results_file, show)

if __name__ == '__main__':
    results_file = "E:/Projects/PyCharm/Paper2/models/SSD/runs/train6/results.csv"
    plt_coco_stats(results_file, show=True)