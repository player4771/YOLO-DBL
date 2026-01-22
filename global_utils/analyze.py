import joblib
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
    'get_coco_PRF1',
)

def parse_coco_stats(results: str|Path|pd.DataFrame|np.ndarray) -> np.ndarray:
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


def get_coco_PRF1(cocoeval_file:str|Path):
    data = joblib.load(cocoeval_file)

    # 获取 precision 矩阵
    # 形状: [T*R*K*A*M]
    # T(IoU): 10个 (0=0.50, 1=0.55, ..., 9=0.95)
    # R(Recall): 101个 (0.00, 0.01, ..., 1.00)
    # K(Class): 类别数量
    # A(Area): 4个 (0=all, 1=small, 2=medium, 3=large)
    # M(MaxDets): 3个 (0=1, 1=10, 2=100)
    precision_matrix = data['eval']['precision']

    # 获取 scores 矩阵 (对应每个 P-R 点的置信度阈值)
    # 某些旧版本 pycocotools 可能不包含 scores，做个防错处理
    scores_matrix = data['eval'].get('scores', None)

    # 设置评估标准 (Strict Definition)
    iou_idx = 0  # 严格使用 IoU = 0.50
    area_idx = 0  # 面积 = all
    max_dets_idx = 2  # MaxDets = 100

    # 提取特定维度的数据 -> 形状变为 [Recall(101), Class(K)]
    # 如果某类别没数据，值为 -1
    p_curve = precision_matrix[iou_idx, :, :, area_idx, max_dets_idx]
    s_curve = scores_matrix[iou_idx, :, :, area_idx, max_dets_idx] if scores_matrix is not None else None

    # 生成标准的 Recall 轴 (0 到 1，共 101 个点)
    r_curve = np.linspace(0, 1, 101)  # Shape: (101,)
    # 扩展 Recall 维度以匹配类别数: [101, K]
    r_curve = np.tile(r_curve[:, None], (1, p_curve.shape[1]))

    # 计算 F1 曲线
    # F1 = 2 * P * R / (P + R)
    # 处理 P = -1 的情况 (无数据)，设为 0 以免报错
    valid_mask = p_curve > -1
    p_curve_clean = np.where(valid_mask, p_curve, 0.0)

    # 计算 F1 (避免除以 0)
    denominator = p_curve_clean + r_curve
    f1_curve = np.divide(
        2 * p_curve_clean * r_curve,
        denominator,
        out=np.zeros_like(p_curve_clean),
        where=denominator > 1e-6
    )

    # 寻找最佳点 (Best F1)
    num_classes = p_curve.shape[1]

    results = []
    f1_sum, p_sum, r_sum, valid_classes = 0, 0, 0, 0
    print(f"{'Class ID':<8} | {'Precision':<9} | {'Recall':<9} | {'Best F1':<9} | {'Threshold':<9}")
    for k in range(num_classes):
        # 获取该类别的 F1 曲线
        f1_k = f1_curve[:, k]

        # 找到 F1 最大的索引
        best_idx = np.argmax(f1_k)

        best_f1 = f1_k[best_idx]
        best_p = p_curve_clean[best_idx, k]
        best_r = r_curve[best_idx, k]
        best_s = s_curve[best_idx, k] if s_curve is not None else 0.0

        results.append((k, best_f1, best_p, best_r))

        f1_sum += best_f1
        p_sum += best_p
        r_sum += best_r
        valid_classes += 1

        print(f"{k:<8} | {best_p:.4f}    | {best_r:.4f}    | {best_f1:.4f}    | {best_s:.4f}")

    # 计算平均值 (Macro Average)
    avg_f1 = f1_sum / valid_classes
    avg_p = p_sum / valid_classes
    avg_r = r_sum / valid_classes

    print(f"{'MEAN':<8} | {avg_p:.4f}    | {avg_r:.4f}    | {avg_f1:.4f}    | {'-':<9}")
    print("*MEAN 为各类别 Best F1 点的算术平均值。")


def replot_conf_matrix(font_scale=1.5, figure_size=(12, 10)):
    """用于重新绘制YOLO输出的混淆矩阵，目的是调整字体大小"""
    import seaborn as sns

    labels = ['algal leaf spot', 'brown blight', 'grey-blight', 'background']
    cm_data = np.array([
        [118, 0, 0, 11],  # Predicted: algal leaf spot
        [0, 98, 0, 7],  # Predicted: brown blight
        [0, 0, 101, 4],  # Predicted: grey blight
        [9, 4, 11, 0]  # Predicted: background
    ])

    plt.figure(figsize=figure_size)

    annot_labels = np.vectorize(lambda x: str(x) if x != 0 else "")(cm_data) #将 0 替换为空字符串

    #cmap='Blues' 对应原图的蓝色系,cbar=True 显示右侧颜色条
    ax = sns.heatmap(cm_data, annot=annot_labels, fmt='', cmap='Blues',
                     xticklabels=labels, yticklabels=labels, cbar=True, square=True,
                     annot_kws={"size": 16 * font_scale, "weight": "normal"})

    ax.set_xlabel('True', fontsize=16 * font_scale, labelpad=15)
    ax.set_ylabel('Predicted', fontsize=16 * font_scale, labelpad=15)
    #ax.set_title('Confusion Matrix', fontsize=18 * font_scale, pad=20)

    plt.xticks(fontsize=12 * font_scale) #x轴刻度保持水平
    plt.yticks(fontsize=12 * font_scale, rotation=90, va='center') #y轴刻度旋转 90 度

    plt.tight_layout()
    plt.savefig("./cache/conf.png", transparent=True)
    plt.show()


if __name__ == '__main__':
    results_file = r"E:\Projects\PyCharm\AutoDL_Remote\SSD\runs\train\results.csv"
    #plt_coco_stats(results_file, show=True)
    #get_coco_PRF1(r"E:\Projects\PyCharm\Paper2\models\SSD\runs\train6\cocoeval_best.bin")
    replot_conf_matrix(font_scale=1.8)