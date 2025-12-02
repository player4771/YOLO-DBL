""" 
Copyright (c) 2025 by SLA team.

Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention}, 
  author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel import _attention
from .utils import get_block_map


class SparseLinearAttention(nn.Module):
    def __init__(self, head_dim=64, topk=0.1, feature_map='softmax', BLKQ=64, BLKK=64, use_bf16=True, tie_feature_map_qk=True):
        R'''
        Args:
            head_dim: dimension of each head.
            topk: ratio of keys selected for sparse attention, shared across all queries.
            feature_map: feature map for linear attention, one of ['hedgehog', 'elu', 'relu', 'softmax'].
            BLKQ: block size for query.
            BLKK: block size for key.
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
            tie_feature_map_qk: whether to use the same feature map for query and key.
        '''
        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=self.dtype)

        if feature_map == 'elu':
            def elu_feature_map(x):
                return F.elu(x) + 1
            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)
            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise NotImplementedError(f'Not supported feature map {feature_map}.')

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        self.init_weights_()

    def init_weights_(self):
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)

    def _forward(self, q, k, v, return_sparsity=False): #略微改了下名字，规避参数不一致的警告
        """
        Args:
            q: queries of shape (B, H, L, D).
            k: keys of shape (B, H, L, D).
            v: values of shape (B, H, L, D).
            return_sparsity: whether to return the actual sparsity.
        """
        dtype = q.dtype
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        current_topk = self.topk
        
        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=current_topk, BLKQ=self.BLKQ, BLKK=self.BLKK)

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)
        c_q = self.feature_map_q(q).contiguous().to(self.dtype)
        c_k = self.feature_map_k(k).contiguous().to(self.dtype)

        o_s, o_l = _attention.apply(q, k, v, c_q, c_k, sparse_map, lut, real_topk, self.BLKQ, self.BLKK)
        o = (o_s + self.proj_l(o_l)).to(dtype)

        if return_sparsity:
            return o, real_topk / sparse_map.shape[-1]
        else:
            return o


class SLA(SparseLinearAttention):
    """SparseLinearAttention的包装类，区别在于forward函数接受正常的NCHW向量"""
    def __init__(self, in_channels, out_channels=None, num_heads=4, head_dim=None, **kwargs):
        if head_dim is None:
            assert in_channels % num_heads == 0, f"in_channels({in_channels}) % num_heads({num_heads}) != 0"
            head_dim = in_channels // num_heads
        else:
            assert in_channels == num_heads * head_dim, f"in_channels({in_channels}) != num_heads({num_heads}) * head_dim({head_dim})"
        super().__init__(head_dim, **kwargs)

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.qkv_proj = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.zeros_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

    def forward(self, x:torch.Tensor, return_sparsity:bool=False):
        """接受 NCHW (BCHW) 张量。
        Args:
            x: input tensor of shape (B, C, H, W).
            return_sparsity: whether to return the actual sparsity.
        """
        if x.device.type == 'cpu': #SLA依赖于triton，只能使用GPU
            o = self.out_proj(x) #仅经过输出投影以保持计算图（如果需要）并确保通道数处理逻辑一致
            if return_sparsity:
                return o, 0.0
            return o

        B, C, H, W = x.shape
        L_seq = H * W  # 序列长度

        # 1. 从 x 投影 Q, K, V
        # (B, C, H, W) -> (B, 3*C, H, W)
        qkv = self.qkv_proj(x)

        # 分割成 q, k, v
        # (B, 3*C, H, W) -> 3 x (B, C, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # dim=1 是通道维度

        # 2. 重塑 (Reshape) Q, K, V 以匹配父类的 (B, H_heads, L_seq, D_head) 格式
        # (B, C, H, W) -> (B, C, L_seq) -> (B, H_heads, D_head, L_seq) -> (B, H_heads, L_seq, D_head)
        q = q.view(B, self.num_heads, self.head_dim, L_seq).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, L_seq).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, L_seq).permute(0, 1, 3, 2)

        # 3. 调用父类的 forward 方法来执行核心计算
        # out_val 是 (o, sparsity) 或 o
        out_val = super()._forward(q, k, v, return_sparsity=return_sparsity)

        o = out_val[0] if return_sparsity else out_val

        # 4. 重塑 (Reshape) 输出 o 回到 NCHW 格式
        # (B, H_heads, L_seq, D_head) -> (B, H_heads, D_head, L_seq) -> (B, C, L_seq) -> (B, C, H, W)
        o = o.permute(0, 1, 3, 2).reshape(B, C, H, W)

        # 5. 应用最终的输出投影
        # (在实际应用中，这里通常会有一个残差连接: o = x + self.out_proj(o))
        o = self.out_proj(o)

        if return_sparsity:
            return o, out_val[1]  # out_val[1] 是 sparsity
        else:
            return o