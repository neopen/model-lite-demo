"""
@FileName: main.py.py
@Description: 主流程：剪枝 + 微调
@Author: HengLine
@Time: 2025/10/12 16:33
"""
# main.py (CPU 优化版)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制禁用 GPU

import torch
import torch.nn as nn
torch.set_num_threads(4)  # 根据你的 CPU 核心数设置（如 2/4/8）
device = torch.device("cpu")

from model import TinyConformerASR
from data import generate_dummy_batch
from pruning import apply_structured_pruning

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def train_one_epoch(model, optimizer, criterion, num_steps=10, device="cpu"):
    model.train()
    total_loss = 0
    for _ in range(num_steps):
        x, x_lens, y, y_lens = generate_dummy_batch()
        x, y = x.to(device), y.to(device)
        x_lens, y_lens = x_lens.to(device), y_lens.to(device)

        # 获取模型输出和下采样后的有效长度
        logits, output_lengths = model(x, x_lens)
        log_probs = torch.log_softmax(logits, dim=2).permute(1, 0, 2)  # [T, B, V]

        # 检查维度（调试用）
        # print(f"log_probs T={log_probs.size(0)}, output_lengths max={output_lengths.max()}")

        loss = criterion(log_probs, y, output_lengths, y_lens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / num_steps

def main():
    print(f" 运行设备: {device}")
    print(f" CPU 线程数: {torch.get_num_threads()}")

    # 1. 原始模型
    original_model = TinyConformerASR(input_dim=80, encoder_dim=128, vocab_size=1000).to(device)
    print(f"原始模型参数量: {count_parameters(original_model):.2f} M")

    # 2. 剪枝
    pruned_model = apply_structured_pruning(original_model, prune_ratio=0.4).to(device)
    print(f"剪枝后模型参数量: {count_parameters(pruned_model):.2f} M")

    # 3. 验证前向
    test_x, test_x_lens, _, _ = generate_dummy_batch()
    with torch.no_grad():
        logits, _ = pruned_model(test_x.to(device), test_x_lens.to(device))
    print(f" 剪枝模型前向成功！输出形状: {logits.shape}")

    # 4. 微调
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(pruned_model.parameters(), lr=1e-4)

    for epoch in range(3):
        loss = train_one_epoch(pruned_model, optimizer, criterion, device=device)
        print(f"Epoch {epoch+1}/3, Loss: {loss:.4f}")

    print(" CPU 环境剪枝 + 微调完成！")

if __name__ == "__main__":
    main()