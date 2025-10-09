import torch
import numpy as np

class EarlyStopping:
    """
    当监控的指标停止改善时，提前停止训练。
    """
    def __init__(self, patience:int=7, delta:float=0, path:str='./checkpoint.pth', verbose:bool=True, trace_func=print):
        """
        Args:
            patience (int): 在停止训练前，等待多少个 epoch 没有改善。
            verbose (bool): 如果为 True，则为每次改善打印一条信息。
            delta (float):  被认为是改善的最小变化量。
            path (str):     保存最佳模型的路径。
            trace_func (function): 用于打印信息的函数。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, score:float, model):
        """用法和效果等同于update(...)"""
        self.update(score, model)

    def update(self, score:float, model):
        """当score提高时保存模型， 否则早停计数+1"""
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta: # 指标没有改善
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping : {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: # 指标改善
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """当指标改善时，保存模型。"""
        if self.verbose:
            self.trace_func(f'Val score improved: {self.score_min:.6f} --> {score:.6f}.  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.score_min = score

