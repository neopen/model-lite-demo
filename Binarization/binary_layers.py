"""
@FileName: binary_layers.py
@Description: 二值化核心
@Author: HengLine
@Time: 2025/10/13 17:56
"""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as fun


class BinaryQuantize(torch.autograd.Function):
    """
    二值化函数（带 STE）
    前向: sign(x)
    反向: 梯度直通（仅 |x|<=1 时传递）
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # 二值化: >0 → +1, <=0 → -1
        out = torch.sign(input)
        # 处理 0（PyTorch sign(0)=0，我们设为 +1）
        out[out == 0] = 1
        return out

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any):
        input, = ctx.saved_tensors
        # STE: 梯度仅在 |input| <= 1 时传递
        grad_input = grad_outputs.clone()
        grad_input[torch.abs(input) > 1] = 0
        return grad_input


class TernaryQuantize(torch.autograd.Function):
    """
    三值化函数（带 STE）
    阈值 δ = 0.7 * mean(|W|)
    |W| > δ → sign(W), 否则 → 0
    """

    @staticmethod
    def forward(ctx, input, delta):
        ctx.save_for_backward(input, delta)
        out = torch.zeros_like(input)
        out[input > delta] = 1
        out[input < -delta] = -1
        return out

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any):
        input, delta = ctx.saved_tensors
        grad_input = grad_outputs.clone()
        # 仅在 [-delta, delta] 外传递梯度
        mask = (input.abs() > delta)
        grad_input[~mask] = 0
        return grad_input, None


class BinaryConv2d(nn.Module):
    """二值化卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 浮点权重（用于训练）
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        assert self.weight.shape[0] == out_channels
        assert self.bias.shape[0] == out_channels
        # print(f" BinaryConv2d: in={in_channels}, out={out_channels}, weight shape={self.weight.shape}")

        # 缩放因子 α = mean(|W|)
        self.alpha = nn.Parameter(torch.ones(out_channels, 1, 1, 1))

    def forward(self, x):
        # 计算缩放因子 alpha = mean(|W|)
        alpha = self.weight.abs().mean(dim=(1, 2, 3), keepdim=True)  # [out, 1, 1, 1]
        # 调整维度以匹配卷积输出 [B, out, H, W]
        alpha = alpha.permute(1, 0, 2, 3)  # [1, out, 1, 1]

        # 二值化权重
        weight_b = BinaryQuantize.apply(self.weight)  # [out, in, k, k]

        # 卷积计算
        out = fun.conv2d(x, weight_b, None, self.stride, self.padding)  # [B, out, H, W]

        # 添加偏置 [1, out, 1, 1]
        bias = self.bias.view(1, -1, 1, 1)

        # 最终输出
        out = out * alpha + bias
        return out


class TernaryConv2d(nn.Module):
    """三值化卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # 计算阈值 δ = 0.7 * mean(|W|)
        delta = 0.7 * self.weight.abs().mean()

        # 三值化权重
        weight_t = TernaryQuantize.apply(self.weight, delta)

        # 前向计算
        out = fun.conv2d(x, weight_t, self.bias, self.stride, self.padding)
        return out