import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from global_utils.coco import coco_stat_names

__all__ = (
    'parse_coco_stats',
    'plt_coco_f1',
    'plt_coco_ap',
    'plt_coco_ar',
    'plt_coco_stats',
)

def parse_coco_stats(results: str | Path | pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    输入列格式需与COCOeval.stats输出一致(详见coco.py) \n
    (不进行内容检查)
    """
    if isinstance(results, (str,Path)):
        return np.loadtxt(str(results), delimiter=',', skiprows=1)
    elif isinstance(results, pd.DataFrame):
        return results.values
    elif isinstance(results, np.ndarray):
        return results
    elif isinstance(results, list):
        return np.array(results)
    else:
        raise TypeError(f"Invalid input: {type(results)}, it should be like an array.")


def plt_coco_f1(coco_stats:np.ndarray):
    """绘制F1曲线"""
    precision = coco_stats[:,0] #取precision = mAP = AP@IoU=0.50:0.95
    recall = coco_stats[:,8] #取recall = AR100 = AR@maxDets=100
    f1 = (2*precision*recall)/(precision+recall)

    fig, ax = plt.subplots(dpi=300)
    ax.plot(f1)
    ax.set_title(f"F1 - mAP/AR100 (Best: {max(f1):.3f})")
    return fig, ax

def plt_coco_ap(coco_stats:np.ndarray):
    fig, axes = plt.subplots(nrows=2, ncols=3, dpi=300)
    for i, ax in enumerate(axes.flat):
        ax.plot(coco_stats[:,i])
        ax.set_title(coco_stat_names[i] +f" (Best: {max(coco_stats[:,i]):.3f})", fontsize=10)
    fig.tight_layout()
    return fig, axes

def plt_coco_ar(coco_stats:np.ndarray):
    fig, axes = plt.subplots(nrows=2, ncols=3, dpi=300)
    for i, ax in enumerate(axes.flat):
        ax.plot(coco_stats[:, i+6])
        ax.set_title(coco_stat_names[i+6] +f" (Best: {max(coco_stats[:, i+6]):.3f})", fontsize=10)
    fig.tight_layout()
    return fig, axes

def plt_coco_stats(coco_stats, show=True):
    data = parse_coco_stats(coco_stats)
    out_dir = Path(coco_stats).parent

    figs = [
        plt_coco_ap(data)[0],
        plt_coco_ar(data)[0],
        plt_coco_f1(data)[0],
    ]

    figs[0].savefig(out_dir/'results_AP.png')
    figs[1].savefig(out_dir/'results_AR.png')
    figs[2].savefig(out_dir/'f1.png')

    if show:
        [fig.show() for fig in figs]

if __name__ == '__main__':
    results_file = r"E:\Projects\PyCharm\AutoDL_Remote\SSD\runs\train\results.csv"
    plt_coco_stats(results_file, show=True)