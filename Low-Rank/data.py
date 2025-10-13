"""
@FileName: data.py.py
@Description: 
@Author: HengLine
@Time: 2025/10/13 11:01
"""
# data.py
import torch

def get_dummy_batch(batch_size=4, max_time=100, input_dim=80, vocab_size=1000):
    x = torch.randn(batch_size, max_time, input_dim)
    y = torch.randint(0, vocab_size, (batch_size,))
    return x, y