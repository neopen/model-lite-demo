"""
@FileName: main.py.py
@Description:  主流程：量化 + 推理速度对比
@Author: HengLine
@Time: 2025/10/12 21:18
"""
# main.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.set_num_threads(4)
device = torch.device("cpu")

from model import TinyConformerASR
from data import get_sample_input
from quantize import quantize_model_fp32_to_int8
import time

def benchmark_model(model, input_data, num_runs=50):
    """Benchmark 模型推理延迟"""
    model.eval()
    x, x_lens = input_data
    x, x_lens = x.to(device), x_lens.to(device)

    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(x, x_lens)

    # 正式计时
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x, x_lens)
    end = time.time()

    avg_latency = (end - start) / num_runs * 1000  # ms
    return avg_latency

def main():
    print(f" 设备: {device}")

    # 1. 创建 FP32 模型
    model_fp32 = TinyConformerASR(input_dim=80, encoder_dim=128, vocab_size=1000).to(device)
    model_fp32.eval()

    # 2. 准备校准数据和测试输入
    calib_data = get_sample_input(batch_size=4)
    test_input = get_sample_input(batch_size=1)  # 模拟单条推理

    # 3. 量化模型
    model_int8 = quantize_model_fp32_to_int8(model_fp32, calib_data)

    # 4. Benchmark 对比
    latency_fp32 = benchmark_model(model_fp32, test_input)
    latency_int8 = benchmark_model(model_int8, test_input)

    print(f"\n 推理延迟对比 (batch=1, avg of 50 runs):")
    print(f"FP32: {latency_fp32:.2f} ms")
    print(f"INT8: {latency_int8:.2f} ms")
    print(f"加速比: {latency_fp32 / latency_int8:.2f}x")

    # 5. 验证输出一致性（可选）
    with torch.no_grad():
        logits_fp32, _ = model_fp32(*test_input)  # 解包，只取 logits
        logits_int8, _ = model_int8(*test_input)
        diff = torch.mean(torch.abs(logits_fp32 - logits_int8)).item()
        print(f"\n 输出平均绝对误差: {diff:.6f}")

    # 6. 保存量化模型（可选）
    torch.save(model_int8.state_dict(), "../data/model_int8.pth")
    print("\n 量化模型已保存为 'model_int8.pth'")

if __name__ == "__main__":
    main()