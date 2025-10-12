"""
@FileName: data.py.py
@Description: 模拟数据
@Author: HengLine
@Time: 2025/10/12 21:17
"""
# data.py
import torch

def get_sample_input(batch_size=1, max_time=200, input_dim=80):
    """生成用于量化的校准数据和推理输入"""
    x = torch.randn(batch_size, max_time, input_dim)
    x_lens = torch.randint(max_time//2, max_time, (batch_size,))
    return x, x_lens