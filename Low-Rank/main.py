"""
@FileName: main.py.py
@Description: 主流程：分解 + 微调
@Author: HengLine
@Time: 2025/10/13 11:02
"""
# main.py
import os

import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

torch.set_num_threads(4)
device = torch.device("cpu")

from model import SimpleASR
from decompose import decompose_linear_svd
from data import get_dummy_batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def train_one_epoch(model, optimizer, criterion, num_steps=20):
    model.train()
    total_loss = 0
    for _ in range(num_steps):
        x, y = get_dummy_batch()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / num_steps


class DecomposedASR(nn.Module):
    def __init__(self, original_model, rank_ratio=0.5):
        super().__init__()
        self.conv = original_model.conv
        self.relu = nn.ReLU()

        fc1 = original_model.fc1
        # print(" 实际 fc1.in_features =", fc1.in_features)
        # print(" 实际 fc1.out_features =", fc1.out_features)
        # print(" fc1.weight.shape =", fc1.weight.shape)

        # 只分解 fc1（大输入维度层）
        print(f"分解 fc1: weight shape = {original_model.fc1.weight.shape}")
        self.fc1_1, self.fc1_2 = decompose_linear_svd(original_model.fc1, rank_ratio)

        # 不分解 fc2（小输入维度层）
        self.fc2 = original_model.fc2

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv(x))
        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)
        x = x.mean(dim=1)  # [B, 2560]
        x = self.relu(self.fc1_1(x))  # [B, 2560] -> [B, r]
        x = self.relu(self.fc1_2(x))  # [B, r] -> [B, 256]
        logits = self.fc2(x)  # [B, 256] -> [B, 1000]
        return logits


def main():
    print(f" 设备: {device}")

    # 1. 原始模型
    model_orig = SimpleASR(input_dim=80, hidden_dim=256, vocab_size=1000).to(device)
    print(f"原始模型参数量: {count_parameters(model_orig):.2f} M")

    # 2. 低秩分解（只分解 fc1）
    model_decomp = DecomposedASR(model_orig, rank_ratio=0.4).to(device)
    print(f"分解后模型参数量: {count_parameters(model_decomp):.2f} M")

    # 3. 验证前向
    x, _ = get_dummy_batch(batch_size=1)
    with torch.no_grad():
        out1 = model_orig(x.to(device))
        out2 = model_decomp(x.to(device))
        diff = torch.mean(torch.abs(out1 - out2)).item()
    print(f"分解后输出误差: {diff:.6f}")

    # 4. 微调
    print("\n 开始微调...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_decomp.parameters(), lr=1e-4)

    for epoch in range(5):
        loss = train_one_epoch(model_decomp, optimizer, criterion)
        print(f"Epoch {epoch + 1}/5, Loss: {loss:.4f}")

    print("\n 低秩分解 + 微调完成！")


if __name__ == "__main__":
    main()
