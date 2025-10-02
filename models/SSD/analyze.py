from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def show_f1(results_file:str):
    """绘制F1, PR, ROC等曲线"""
    data = pd.read_csv(results_file)
    x = range(1, data.shape[0]+1)

    precision = data.iloc[:,0] #取precision = mAP = AP@IoU=0.50:0.95
    recall = data.iloc[:,8] #取recall = AR100 = AR@maxDets=100
    f1 = (2*precision*recall)/(precision+recall)
    plt.figure(dpi=300)
    plt.plot(x, f1)
    plt.title('F1 - mAP/AR100')
    plt.savefig(Path(results_file).parent/'f1.png')
    plt.show()

def analyze(results_file:str):
    data = pd.read_csv(results_file)
    x = range(1, data.shape[0]+1)

    plt.figure(num=1, dpi=300)
    plt.axis('off')
    for i in range(0, 6):
        plt.subplot(2, 3, i+1)
        plt.plot(x, data.iloc[:,i])
        plt.title(data.columns[i], fontsize=10)
    plt.tight_layout()
    plt.savefig(Path(results_file).parent/'results_AP.png')

    plt.figure(num=2, dpi=300)
    plt.axis('off')
    for i in range(6, 12):
        plt.subplot(2, 3, i-6+1)
        plt.plot(x, data.iloc[:, i])
        plt.title(data.columns[i], fontsize=10)
    plt.tight_layout()
    plt.savefig(Path(results_file).parent/'results_AR.png')

    plt.show()

if __name__ == '__main__':
    analyze('./runs/train5/results.csv')
    show_f1('./runs/train5/results.csv')