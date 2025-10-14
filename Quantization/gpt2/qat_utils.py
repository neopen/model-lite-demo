"""
@FileName: qat_utils.py
@Description: 
@Author: HengLine
@Time: 2025/10/13 22:30
"""
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D


def conv1d_to_linear(module):
    """
    将 Hugging Face 的 Conv1D 转换为标准 nn.Linear

    背景:
    - GPT-2 使用 Conv1D(out_features, in_features)
    - 权重形状: [in_features, out_features]
    - 标准 Linear 权重形状: [out_features, in_features]

    转换:
    - Linear.weight = Conv1D.weight.t()
    - 偏置直接复制
    """
    if isinstance(module, Conv1D):
        in_features = module.weight.shape[0]
        out_features = module.weight.shape[1]
        linear = nn.Linear(in_features, out_features, bias=module.bias is not None)

        with torch.no_grad():
            # 转置权重以匹配 Linear 格式
            linear.weight.copy_(module.weight.t())
            if module.bias is not None:
                linear.bias.copy_(module.bias)
        return linear
    return module


def replace_conv1d_with_linear(model):
    """
    递归遍历模型，将所有 Conv1D 替换为 Linear
    """
    for name, child in model.named_children():
        if isinstance(child, Conv1D):
            # 替换当前层
            setattr(model, name, conv1d_to_linear(child))
        else:
            # 递归处理子模块
            replace_conv1d_with_linear(child)
    return model