"""
@FileName: gptq_qua.py
@Description: 
@Author: HengLine
@Time: 2025/10/14 18:42
"""
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_name = "openai-community/gpt2"
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BitsAndBytesConfig


def quantize_with_bitsandbytes_cpu():
    """使用 bitsandbytes 在 CPU 上进行量化"""

    print("🚀 使用 bitsandbytes 进行 CPU 量化...")

    # 配置 4-bit 量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        llm_int8_enable_fp32_cpu_offload=True  # 允许 CPU 卸载
    )

    try:
        # 加载模型并应用量化
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",  # 自动分配到可用设备
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # 确保有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✅ bitsandbytes 量化成功！")
        return model, tokenizer

    except Exception as e:
        print(f"❌ bitsandbytes 量化失败: {e}")
        return None, None


def test_bitsandbytes_model(model, tokenizer):
    """测试 bitsandbytes 量化模型"""

    print("\n🧪 测试量化模型...")

    test_prompts = [
        "The future of artificial intelligence",
        "In my opinion, machine learning",
        "The weather today is"
    ]

    for prompt in test_prompts:
        print(f"\n📝 输入: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=60,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 输出: {generated_text}")


if __name__ == '__main__':
    # 使用 bitsandbytes 进行 CPU 量化
    bnb_model, bnb_tokenizer = quantize_with_bitsandbytes_cpu()

    if bnb_model and bnb_tokenizer:
        test_bitsandbytes_model(bnb_model, bnb_tokenizer)