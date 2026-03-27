"""
@FileName: gptq_qua.py
@Description: 
@Author: HengLine
@Time: 2025/10/14 18:42
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import Dataset
import numpy as np


def prepare_calibration_data(tokenizer, sequences=128, sequence_length=512):
    """准备校准数据集用于量化"""
    # 使用简单的文本数据作为校准集
    calibration_text = [
        "The quick brown fox jumps over the lazy dog. " * 20,
        "Machine learning is a subset of artificial intelligence. " * 15,
        "Natural language processing enables computers to understand human language. " * 12,
        "The weather today is sunny with a chance of rain in the afternoon. " * 10,
    ]

    # 编码校准数据
    encoded_data = []
    for text in calibration_text:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=sequence_length,
            truncation=True,
            padding="max_length"
        )
        encoded_data.append(encoded["input_ids"])

    # 组合所有数据
    all_input_ids = torch.cat(encoded_data, dim=0)

    # 创建校准数据集
    calibration_dataset = []
    for i in range(min(sequences, len(all_input_ids))):
        calibration_dataset.append({
            "input_ids": all_input_ids[i],
            "attention_mask": torch.ones_like(all_input_ids[i])
        })

    return calibration_dataset


def quantize_gpt2_with_gptq():
    """使用 GPTQ 量化 GPT-2 模型"""

    model_name = "openai-community/gpt2"  # 也可以使用 "gpt2-medium", "gpt2-large", "gpt2-xl"

    print("🔧 加载原始模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 添加 pad token 如果不存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("📊 准备校准数据...")
    calibration_dataset = prepare_calibration_data(tokenizer)

    # 量化配置
    quantize_config = BaseQuantizeConfig(
        bits=4,  # 4-bit 量化
        group_size=128,  # 分组大小
        desc_act=False,  # 是否使用激活排序 (对于大模型建议设为 True)
    )

    print("⚡ 开始量化过程...")

    # 使用 AutoGPTQ 进行量化
    quantized_model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        calibration_dataset=calibration_dataset,
        device_map="auto"
    )

    # 保存量化后的模型
    save_path = f"../../data/gpt2-{quantize_config.bits}bit-gptq"
    quantized_model.save_quantized(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"✅ 量化完成！模型已保存至: {save_path}")

    return quantized_model, tokenizer, save_path


def test_quantized_model(model_path):
    """测试量化后的模型"""

    print("🧪 加载量化模型进行测试...")

    # 加载量化后的模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    quantized_model = AutoGPTQForCausalLM.from_quantized(
        model_path,
        device_map="auto",
        use_triton=False  # 如果安装 triton 可以设为 True 以获得更好性能
    )

    # 测试文本生成
    test_prompts = [
        "The future of artificial intelligence",
        "In a world where machines can think",
        "The benefits of renewable energy include"
    ]

    for prompt in test_prompts:
        print(f"\n📝 提示: {prompt}")
        print("🤖 生成结果:")

        inputs = tokenizer(prompt, return_tensors="pt").to(quantized_model.device)

        with torch.no_grad():
            outputs = quantized_model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)


def compare_model_sizes(original_model_name, quantized_path):
    """比较原始模型和量化后模型的大小"""
    import os
    import glob

    def get_folder_size(folder_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # 转换为 MB

    # 获取原始模型大小（需要先下载）
    original_model = AutoModelForCausalLM.from_pretrained(original_model_name)
    original_path = f"./{original_model_name}"
    original_model.save_pretrained(original_path)
    original_size = get_folder_size(original_path)

    # 获取量化模型大小
    quantized_size = get_folder_size(quantized_path)

    print(f"\n📊 模型大小对比:")
    print(f"原始模型: {original_size:.2f} MB")
    print(f"量化模型: {quantized_size:.2f} MB")
    print(f"压缩比例: {original_size / quantized_size:.2f}x")
    print(f"大小减少: {(original_size - quantized_size) / original_size * 100:.1f}%")


if __name__ == "__main__":
    # 执行量化
    quantized_model, tokenizer, save_path = quantize_gpt2_with_gptq()

    # 测试量化模型
    test_quantized_model(save_path)

    model_name = "openai-community/gpt2"

    # 比较模型大小
    compare_model_sizes(model_name, save_path)