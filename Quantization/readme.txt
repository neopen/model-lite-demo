语音识别（ASR）模型量化（Quantization）Demo，专为 CPU 环境 设计，使用 PyTorch 的训练后量化（Post-Training Quantization, PTQ），适用于边缘设备部署。

仅依赖 PyTorch + torchaudio
包含：FP32 模型 → INT8 量化 → CPU 推理加速对比

1、为什么量化后模型大小没变？
    原因：state_dict() 保存的是 量化参数（scale/zero_point）+ 量化权重（int8），但 PyTorch 默认以 float32 格式序列化所有张量！
    即使权重是 int8，当你调用 torch.save(model.state_dict()) 时：
        PyTorch 会把 int8 张量自动转换为 float32 存储（为了兼容性）；
        同时还保存了 scale、zero_point 等额外参数；
        最终文件大小 ≈ 原始 FP32 模型，甚至更大！

   量化模型的体积优势只在内存和计算时体现，不在 .pth 文件中直接体现！

2、为什么推理时内存和速度有提升，但文件没变小？
    量化的核心收益在运行时，不在存储文件。部署时应使用 TorchScript / ONNX 格式。

    # 1. 量化
    model_int8 = quantize_model_fp32_to_int8(model_fp32, calib_data)

    # 2. 导出为 TorchScript（保留 int8）
    example = (torch.randn(1, 200, 80), torch.tensor([200]))
    traced = torch.jit.trace(model_int8, example)
    traced.save("asr_quantized.pt")  # ← 这个文件才真正小！

    # 3. 在目标设备加载
    deploy_model = torch.jit.load("asr_quantized.pt")
    deploy_model.eval()
