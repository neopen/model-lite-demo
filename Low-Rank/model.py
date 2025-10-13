"""
@FileName: model.py.py
@Description: 原始模型
@Author: HengLine
@Time: 2025/10/13 10:59
"""
import torch.nn as nn

class SimpleASR(nn.Module):
    """
    简化 ASR 模型（含 Linear 和 Conv2d，用于演示分解）
    """

    def __init__(self, input_dim=80, hidden_dim=256, vocab_size=1000):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        conv_out_dim = 64 * ((input_dim + 1) // 2)  # 80 -> 40 -> 2560
        self.fc1 = nn.Linear(conv_out_dim, hidden_dim)  # [2560, 256]
        self.fc2 = nn.Linear(hidden_dim, vocab_size)  # [256, 1000]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv(x))
        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)
        x = x.mean(dim=1)  # [B, 2560]
        x = self.relu(self.fc1(x))  # [B, 256]
        logits = self.fc2(x)  # [B, 1000]
        return logits