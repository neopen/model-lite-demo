"""
@FileName: data.py.py
@Description: 模拟数据
@Author: HengLine
@Time: 2025/10/13 10:46
"""
# data.py
import torch


def generate_batch(batch_size=4, max_time=200, input_dim=80, vocab_size=1000):
    x = torch.randn(batch_size, max_time, input_dim)
    x_lens = torch.randint(max_time // 2, max_time, (batch_size,))
    x_lens, _ = torch.sort(x_lens, descending=True)

    y = torch.randint(1, vocab_size, (batch_size, max_time // 4))
    y_lens = torch.randint(10, max_time // 4, (batch_size,))
    return x, x_lens, y, y_lens