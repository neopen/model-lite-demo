"""
@FileName: quantization_benchmark.py.py
@Description: 量化前后核心指标对比
@Author: HengLine
@Time: 2025/10/12 21:59
"""
# quantization_benchmark.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.set_num_threads(4)
device = torch.device("cpu")

import time
import numpy as np
from model import TinyConformerASR
from data import get_sample_input
from quantize import quantize_model_fp32_to_int8

# def get_model_size(model):
#     """计算模型磁盘大小（MB）"""
#     torch.save(model.state_dict(), "temp_model.pth")
#     size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
#     os.remove("temp_model.pth")
#     return size_mb

# TorchScript 方式：
def get_model_size(model, example_inputs):
    # 使用 TorchScript 保存以保留量化类型
    traced = torch.jit.trace(model, example_inputs)
    torch.jit.save(traced, "temp_model.pt")
    size_mb = os.path.getsize("temp_model.pt") / (1024 * 1024)
    os.remove("temp_model.pt")
    return size_mb

def benchmark_latency(model, input_data, num_runs=50):
    """测量平均推理延迟（ms）"""
    model.eval()
    x, x_lens = input_data
    x, x_lens = x.to(device), x_lens.to(device)

    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(x, x_lens)

    # 正式计时
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(x, x_lens)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    return np.mean(times), np.std(times)

def benchmark_memory(model, input_data):
    """粗略估计推理内存占用（MB）"""
    # PyTorch 无直接 API，此处用参数量 + 激活值估算
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    x, x_lens = input_data
    x = x.to(device)
    # 前向一次，估算激活内存（简化）
    with torch.no_grad():
        logits, _ = model(x, x_lens)
    # 激活内存 ≈ 输入 + 中间特征 + 输出
    activation_size = x.element_size() * x.numel()
    activation_size += logits.element_size() * logits.numel()
    # 粗略总内存（MB）
    total_mb = (param_size + activation_size) / (1024 * 1024)
    return total_mb

def compare_models():
    print("🔧 初始化 FP32 模型...")
    model_fp32 = TinyConformerASR(input_dim=80, encoder_dim=128, vocab_size=1000).to(device)
    model_fp32.eval()

    # 准备数据
    calib_data = get_sample_input(batch_size=4)
    test_input = get_sample_input(batch_size=1)

    # 量化
    print("🔍 执行 INT8 量化...")
    model_int8 = quantize_model_fp32_to_int8(model_fp32, calib_data)

    # === 1. 模型大小 ===
    # size_fp32 = get_model_size(model_fp32)
    # size_int8 = get_model_size(model_int8)

    test_input = get_sample_input(batch_size=1)
    size_fp32 = get_model_size(model_fp32, test_input)
    size_int8 = get_model_size(model_int8, test_input)
    print(f"\n📦 模型大小:")
    print(f"  FP32: {size_fp32:.2f} MB")
    print(f"  INT8: {size_int8:.2f} MB")
    print(f"  压缩率: {(1 - size_int8/size_fp32)*100:.1f}%")

    # === 2. 推理延迟 ===
    lat_fp32, std_fp32 = benchmark_latency(model_fp32, test_input)
    lat_int8, std_int8 = benchmark_latency(model_int8, test_input)
    speedup = lat_fp32 / lat_int8
    print(f"\n⏱️  推理延迟 (batch=1, 50 runs):")
    print(f"  FP32: {lat_fp32:.2f} ± {std_fp32:.2f} ms")
    print(f"  INT8: {lat_int8:.2f} ± {std_int8:.2f} ms")
    print(f"  加速比: {speedup:.2f}x")

    # === 3. 内存占用（估算）===
    mem_fp32 = benchmark_memory(model_fp32, test_input)
    mem_int8 = benchmark_memory(model_int8, test_input)
    print(f"\n🧠 内存占用（估算）:")
    print(f"  FP32: ~{mem_fp32:.1f} MB")
    print(f"  INT8: ~{mem_int8:.1f} MB")

    # === 4. 输出精度 ===
    with torch.no_grad():
        logits_fp32, _ = model_fp32(*test_input)
        logits_int8, _ = model_int8(*test_input)
        mae = torch.mean(torch.abs(logits_fp32 - logits_int8)).item()
        max_diff = torch.max(torch.abs(logits_fp32 - logits_int8)).item()
    print(f"\n🔍 输出精度误差:")
    print(f"  平均绝对误差 (MAE): {mae:.6f}")
    print(f"  最大误差: {max_diff:.6f}")

    # === 5. WER 估算（需真实数据，此处跳过）===
    print(f"\n📝 注: 真实 WER 需在 LibriSpeech/AISHELL 上评估，通常上升 <1%。")

if __name__ == "__main__":
    compare_models()