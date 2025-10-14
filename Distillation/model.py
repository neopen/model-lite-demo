"""
@FileName: model.py.py
@Description: 定义 Teacher 和 Student 模型
@Author: HengLine
@Time: 2025/10/13 10:45
"""
# model.py
import torch
import torch.nn as nn

class Conv2dSubsampling(nn.Module):
    def __init__(self, input_dim: int, encoder_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, encoder_dim, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(encoder_dim, encoder_dim, 3, stride=2, padding=1)
        self.relu = nn.ReLU()
        freq_dim = ((input_dim + 1) // 2 + 1) // 2  # 80 -> 20
        self.out_proj = nn.Linear(encoder_dim * freq_dim, encoder_dim)

    def forward(self, x, x_lens):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x_lens = (x_lens + 1) // 2
        x_lens = (x_lens + 1) // 2
        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)
        x = self.out_proj(x)
        return x, x_lens

class ASRModel(nn.Module):
    """通用 ASR 模型（可配置大小）"""
    def __init__(self, input_dim=80, encoder_dim=128, vocab_size=1000):
        super().__init__()
        self.subsampling = Conv2dSubsampling(input_dim, encoder_dim)
        self.ctc_head = nn.Linear(encoder_dim, vocab_size)

    def forward(self, x, x_lens):
        x, x_lens = self.subsampling(x, x_lens)
        logits = self.ctc_head(x)
        return logits, x_lens