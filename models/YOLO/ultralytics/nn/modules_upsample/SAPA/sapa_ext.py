# sim_ext和atn_ext由Gemini根据C++源码编写

import torch
import torch.nn.functional as F


class sim_ext:
    """
    sim_ext 的最终优化版。
    严格遵循 unfold -> matmul -> pixel_shuffle 的最优算法，解决性能与内存问题。
    """

    @staticmethod
    def forward(query, key, kernel_size, scale_factor, output):
        B, H_k, W_k, C = key.shape
        sf = scale_factor

        # 1. Unfold key: [B, C*k*k, H_k*W_k] -> [B, H_k*W_k, C, k*k]
        unfolded_key = F.unfold(
            key.permute(0, 3, 1, 2), kernel_size=kernel_size, padding=kernel_size // 2
        ).view(B, C, kernel_size ** 2, H_k * W_k).permute(0, 3, 1, 2)

        # 2. Reshape query for matmul: [B, H_q, W_q, C] -> [B, H_k*W_k, sf*sf, C]
        query_reshaped = query.view(
            B, H_k, sf, W_k, sf, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H_k * W_k, sf ** 2, C)

        # 3. Matmul: 核心计算，[B, H_k*W_k, sf*sf, C] @ [B, H_k*W_k, C, k*k] -> [B, H_k*W_k, sf*sf, k*k]
        attn_scores = torch.matmul(query_reshaped, unfolded_key)

        # 4. Reshape and Pixel Shuffle: 高效重组为最终形状
        # [B, H_k*W_k, sf*sf, k*k] -> [B, sf*sf*k*k, H_k, W_k] -> [B, k*k, H_q, W_q]
        attn_scores = attn_scores.permute(0, 2, 3, 1).reshape(B, sf ** 2 * kernel_size ** 2, H_k, W_k)
        attn_scores = F.pixel_shuffle(attn_scores, upscale_factor=sf)

        output.copy_(attn_scores.permute(0, 2, 3, 1))

    @staticmethod
    def backward(grad_output, query, key, kernel_size, scale_factor, grad_query, grad_key):
        B, H_k, W_k, C = key.shape
        sf = scale_factor

        # 1. Pixel Unshuffle grad_output: [B, k*k, H_q, W_q] -> [B, k*k*sf*sf, H_k*W_k]
        grad_output_reshaped = F.pixel_unshuffle(grad_output.permute(0, 3, 1, 2), downscale_factor=sf)
        # -> [B, H_k*W_k, sf*sf, k*k]
        grad_output_reshaped = grad_output_reshaped.reshape(
            B, kernel_size ** 2, sf ** 2, H_k * W_k).permute(0, 3, 2, 1)

        # --- 计算 query 梯度 ---
        unfolded_key = F.unfold(
            key.permute(0, 3, 1, 2), kernel_size=kernel_size, padding=kernel_size // 2
        ).view(B, C, kernel_size ** 2, H_k * W_k).permute(0, 3, 1, 2)
        _grad_query = torch.matmul(grad_output_reshaped, unfolded_key.transpose(-1, -2))
        _grad_query = _grad_query.view(B, H_k, W_k, sf, sf, C).permute(
            0, 1, 3, 2, 4, 5).reshape(B, H_k * sf, W_k * sf, C)
        grad_query.copy_(_grad_query)

        # --- 计算 key 梯度 ---
        query_reshaped = query.view(
            B, H_k, sf, W_k, sf, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H_k * W_k, sf ** 2, C)
        grad_unfolded_key = torch.matmul(query_reshaped.transpose(-1, -2), grad_output_reshaped)
        grad_unfolded_key = grad_unfolded_key.permute(0, 2, 3, 1).reshape(B, C * kernel_size ** 2, H_k * W_k)
        _grad_key = F.fold(
            grad_unfolded_key, output_size=(H_k, W_k), kernel_size=kernel_size, padding=kernel_size // 2
        ).permute(0, 2, 3, 1)
        grad_key.copy_(_grad_key)


class atn_ext:
    """
    atn_ext 的最终优化版。
    严格遵循 unfold -> matmul -> pixel_shuffle 的最优算法，解决性能与内存问题。
    """

    @staticmethod
    def forward(attn, value, kernel_size, scale_factor, output):
        B, H_v, W_v, C = value.shape
        sf = scale_factor

        # 1. Unfold value: [B, C*k*k, H_v*W_v] -> [B, H_v*W_v, C, k*k]
        unfolded_value = F.unfold(
            value.permute(0, 3, 1, 2), kernel_size=kernel_size, padding=kernel_size // 2
        ).view(B, C, kernel_size ** 2, H_v * W_v).permute(0, 3, 1, 2)

        # 2. Reshape attn for matmul: [B, H_q, W_q, k*k] -> [B, H_v*W_v, sf*sf, k*k]
        attn_reshaped = attn.view(
            B, H_v, sf, W_v, sf, kernel_size ** 2).permute(0, 1, 3, 2, 4, 5).reshape(B, H_v * W_v, sf ** 2,
                                                                                     kernel_size ** 2)

        # 3. Matmul: [B, H_v*W_v, C, k*k] @ [B, H_v*W_v, k*k, sf*sf] -> [B, H_v*W_v, C, sf*sf]
        result = torch.matmul(unfolded_value, attn_reshaped.transpose(-1, -2))

        # 4. Pixel Shuffle: [B, H_v*W_v, C, sf*sf] -> [B, C*sf*sf, H_v, W_v] -> [B, C, H_q, W_q]
        result = result.permute(0, 2, 3, 1).reshape(B, C * sf ** 2, H_v, W_v)
        result = F.pixel_shuffle(result, upscale_factor=sf)

        output.copy_(result.permute(0, 2, 3, 1))

    @staticmethod
    def backward(grad_output, attn, value, kernel_size, scale_factor, grad_attn, grad_value):
        B, H_v, W_v, C = value.shape
        sf = scale_factor

        # 1. Pixel Unshuffle grad_output: [B, C, H_q, W_q] -> [B, H_v*W_v, C, sf*sf]
        grad_output_reshaped = F.pixel_unshuffle(grad_output.permute(0, 3, 1, 2), downscale_factor=sf)
        grad_output_reshaped = grad_output_reshaped.reshape(
            B, C, sf ** 2, H_v * W_v).permute(0, 3, 1, 2)

        # --- 计算 attn 梯度 ---
        unfolded_value = F.unfold(
            value.permute(0, 3, 1, 2), kernel_size=kernel_size, padding=kernel_size // 2
        ).view(B, C, kernel_size ** 2, H_v * W_v).permute(0, 3, 1, 2)
        _grad_attn = torch.matmul(unfolded_value.transpose(-1, -2), grad_output_reshaped)
        _grad_attn = _grad_attn.permute(0, 2, 3, 1).reshape(B, sf ** 2 * kernel_size ** 2, H_v, W_v)
        _grad_attn = F.pixel_shuffle(_grad_attn, upscale_factor=sf).permute(0, 2, 3, 1)
        grad_attn.copy_(_grad_attn)

        # --- 计算 value 梯度 ---
        attn_reshaped = attn.view(
            B, H_v, sf, W_v, sf, kernel_size ** 2).permute(0, 1, 3, 2, 4, 5).reshape(B, H_v * W_v, sf ** 2,
                                                                                     kernel_size ** 2)
        grad_unfolded_value = torch.matmul(grad_output_reshaped, attn_reshaped)
        grad_unfolded_value = grad_unfolded_value.permute(0, 2, 3, 1).reshape(B, C * kernel_size ** 2, H_v * W_v)
        _grad_value = F.fold(
            grad_unfolded_value, output_size=(H_v, W_v), kernel_size=kernel_size, padding=kernel_size // 2
        ).permute(0, 2, 3, 1)
        grad_value.copy_(_grad_value)