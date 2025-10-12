"""
@FileName: quantize.py.py
@Description: 量化核心实现
@Author: HengLine
@Time: 2025/10/12 21:18
"""
# quantize.py
import torch
import torch.nn as nn
from Quantization.data import get_sample_input
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    prepare,
    convert,
)


def quantize_model_fp32_to_int8(model_fp32: nn.Module, calib_data) -> torch.nn.Module:
    """
    使用训练后量化（PTQ）将 FP32 模型转为 INT8（仅 CPU 支持）

    Args:
        model_fp32: 原始 FP32 模型（必须是 eval 模式）
        calib_data: 校准数据，用于确定量化范围（min/max）

    Returns:
        量化后的 INT8 模型
    """
    print("🔍 开始模型量化...")

    # Step 1: 设置量化配置（仅支持 CPU 后端）
    qconfig_mapping = get_default_qconfig_mapping("x86")  # 或 "fbgemm"（Linux）

    # Step 2: 准备模型（插入 observer）
    example_inputs = (calib_data[0], calib_data[1])
    model_prepared = prepare(model_fp32, qconfig_mapping, example_inputs)

    # Step 3: 校准（用少量数据统计激活值范围）
    print("📊 正在校准模型...")
    with torch.no_grad():
        for _ in range(10):  # 10 个 batch 足够
            x, x_lens = get_sample_input()
            model_prepared(x, x_lens)

    # Step 4: 转换为量化模型
    model_int8 = convert(model_prepared)
    print("✅ 量化完成！")
    return model_int8
