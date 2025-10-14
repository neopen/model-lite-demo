"""
@FileName: qat_gpt2.py
@Description: 量化感知训练
@Author: HengLine
@Time: 2025/10/13 21:27
"""
import torch
import torch.nn as nn
from Quantization.gpt2.qat_utils import replace_conv1d_with_linear
from torch.ao.quantization import (
    QConfig,
    prepare_qat,
    convert,
)
from torch.quantization import FakeQuantize
from torch.quantization.observer import MovingAverageMinMaxObserver
from transformers import GPT2LMHeadModel


def prepare_qat_model(model: GPT2LMHeadModel):
    """
    为 GPT-2 准备量化感知训练（QAT）

    步骤:
    1. 将 Conv1D 替换为 Linear（FX 量化要求标准层）
    2. 设置模型为训练模式（QAT 必须）
    3. 配置量化方案（x86 CPU 后端）
    4. 使用 FX 图追踪准备 QAT 模型
    """
    print(" 1: 将 Conv1D 替换为 Linear...")
    # model = replace_conv1d_with_linear(model)

    print(" 2: 设置模型为训练模式...")
    model.train()  # QAT 必须在 train() 模式下准备

    print(" 3: 配置量化方案...")
    # 创建 QAT 配置
    qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False,
        )
    )

    # 在 prepare_qat 前设置后端
    # torch.backends.quantized.engine = 'fbgemm'  # Linux/macOS
    torch.backends.quantized.engine = 'x86'    # Windows

    # 应用 qconfig 到所有 Linear 层
    def set_qconfig(module):
        if isinstance(module, nn.Linear):
            module.qconfig = qconfig

    model.apply(set_qconfig)

    # 准备 QAT
    model_qat = prepare_qat(model, inplace=False)

    print(" QAT 模型准备完成！")
    return model_qat


def convert_qat_to_int8(model_qat):
    """
    将 QAT 模型转换为 INT8 模型

    注意: 转换前必须设置为 eval() 模式
    """
    print(" 转换为 INT8 模型...")
    model_qat.eval()  # 转换必须在 eval() 模式下
    model_int8 = convert(model_qat)
    print(" INT8 模型转换完成！")
    return model_int8