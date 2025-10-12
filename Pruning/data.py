"""
@FileName: data.py.py
@Description: 生成模拟数据
@Author: HengLine
@Time: 2025/10/12 16:31
"""
# data.py
import torch

def generate_dummy_batch(
    batch_size=4,
    max_time=200,      # 最大音频帧数
    input_dim=80,      # log-Mel 特征维度
    vocab_size=1000,   # 词表大小
    max_label_len=50   # 最大标签长度
):
    """
    生成用于训练/微调的模拟数据（实际项目应使用真实音频特征和转录文本）
    """
    # 随机音频特征 [B, T, F]
    x = torch.randn(batch_size, max_time, input_dim)
    # 随机有效帧长（确保 > 0）
    x_lens = torch.randint(low=max_time//2, high=max_time, size=(batch_size,))
    x_lens, _ = torch.sort(x_lens, descending=True)  # 可选：降序便于 pack

    # 随机标签（CTC 标签，值 ∈ [1, vocab_size-1]，0 为 blank）
    y = torch.randint(1, vocab_size, (batch_size, max_label_len))
    y_lens = torch.randint(10, max_label_len, (batch_size,))
    return x, x_lens, y, y_lens