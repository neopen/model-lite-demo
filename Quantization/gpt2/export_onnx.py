"""
@FileName: export_onnx.py
@Description: 
@Author: HengLine
@Time: 2025/10/14 14:04
"""
import torch
print("支持的量化引擎:", torch.backends.quantized.supported_engines)
torch.backends.quantized.engine = 'x86'
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from qat_gpt2 import prepare_qat_model, convert_qat_to_int8


def export_onnx():
    print(" 加载模型...")
    model_name = "openai-community/gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 转换为 QAT 模型
    model = prepare_qat_model(model)
    # 模拟微调（实际应加载微调后的权重）
    # model = convert_qat_to_int8(model)
    model.eval()

    # ONNX 不支持动态控制流，必须固定序列长度
    sequence_length = 32
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, sequence_length), dtype=torch.long)

    # 关键修复 1: 禁用 FX 追踪（PyTorch 2.0+）
    with torch.no_grad():
        # 先测试前向是否返回纯张量
        test_output = model(input_ids)
        print(f"模型输出类型: {type(test_output)}")
        print(f"是否为张量: {torch.is_tensor(test_output)}")

        if not torch.is_tensor(test_output):
            raise ValueError("模型必须返回纯张量！请使用 GPT2ForQAT 包装器")

    print(" 导出 ONNX...")
    try:
        torch.onnx.export(
            model,
            input_ids,
            "gpt2.onnx",
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            # dynamic_axes=None,  # 完全静态（最稳定）
            dynamic_axes={
                'input_ids': {0: 'batch'},  # 仅 batch 动态
                'logits': {0: 'batch'}
            },
            verbose=False,
            #  关键修复 3: 禁用 FX 追踪
            dynamo=False,  # PyTorch 2.0+ 禁用 torch.compile
        )
        print(" ONNX 模型导出成功！")

        # === 关键修复 2: 验证 ONNX 模型 ===
        import onnx
        onnx_model = onnx.load("gpt2_int8.onnx")
        onnx.checker.check_model(onnx_model)
        print(" ONNX 模型验证通过！")

    except Exception as e:
        print(f" ONNX 导出失败: {e}")
        print("\n 常见解决方案:")
        print("1. 确保模型处于 eval() 模式")
        print("2. 使用固定 sequence_length（不要动态）")
        print("3. 升级 transformers >= 4.30.0")
        print("4. 如果量化模型导出失败，先导出 FP32 再量化")
        raise e

    print(" ONNX 模型已保存为 'gpt2_int8.onnx'")


if __name__ == "__main__":
    export_onnx()