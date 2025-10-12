"""
@FileName: model.py.py
@Description: 定义轻量 Conformer 模型
@Author: HengLine
@Time: 2025/10/12 16:29
"""
# model.py
import torch
import torch.nn as nn

class Conv2dSubsampling(nn.Module):
    """
    两层卷积下采样模块，将输入特征图尺寸缩小为原来的 1/4（时间 & 频率维度各 /2）
    输入: [B, T, F] (B: batch, T: 时间帧数, F: 频率维度，如80维log-Mel)
    输出: [B, T//4, encoder_dim]
    """
    def __init__(self, input_dim: int, encoder_dim: int):
        super().__init__()
        # 第一层卷积: stride=2 → 时间/频率维度减半
        self.conv1 = nn.Conv2d(1, encoder_dim, kernel_size=3, stride=2, padding=1)
        # 第二层卷积: 再次减半
        self.conv2 = nn.Conv2d(encoder_dim, encoder_dim, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        # 将卷积输出展平后映射到 encoder_dim 维度
        # 注意: 展平后维度 = encoder_dim * (F//4) （假设 padding 保持尺寸）
        self.out_proj = nn.Linear(encoder_dim * ((input_dim + 1) // 2 // 2), encoder_dim)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        x = x.unsqueeze(1)  # [B, T, F] -> [B, 1, T, F]
        x = self.relu(self.conv1(x))  # [B, C1, T1, F1]
        x = self.relu(self.conv2(x))  # [B, C2, T2, F2]

        x_lens = (x_lens + 1) // 2
        x_lens = (x_lens + 1) // 2

        B, C, T_prime, F_prime = x.size()
        # 动态展平：不管 C 是多少，都展平后输入 Linear
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T_prime, C * F_prime)
        x = self.out_proj(x)  # 要求: self.out_proj.in_features == C * F_prime
        return x, x_lens


class TinyConformerASR(nn.Module):
    """
    极简版 Conformer ASR 模型（仅保留前端卷积用于演示剪枝）
    实际项目中可加入 Transformer Encoder 和 CTC Head
    """
    def __init__(self, input_dim=80, encoder_dim=128, vocab_size=1000):
        super().__init__()
        self.subsampling = Conv2dSubsampling(input_dim, encoder_dim)
        # 为简化，此处省略 Encoder，直接接 CTC Head
        self.ctc_classifier = nn.Linear(encoder_dim, vocab_size)

    def forward(self, x, x_lens):
        x, x_lens = self.subsampling(x, x_lens)
        logits = self.ctc_classifier(x)  # [B, T, vocab_size]
        return logits, x_lens