"""
@FileName: gptq_qua.py
@Description: 
@Author: HengLine
@Time: 2025/10/14 18:42
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer


def quick_quantize():
    """快速量化 GPT-2"""

    model_name = "openai-community/gpt2"

    # 量化配置
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False
    )

    # 加载并量化模型
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        device_map="auto"
    )

    # 保存量化模型
    save_path = "../../data/gpt2-4bit-gptq"
    model.save_quantized(save_path)

    print(f"✅ 快速量化完成！模型保存至: {save_path}")
    return save_path


# 使用量化模型进行推理
def quick_inference(model_path, prompt="Hello, how are you?"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoGPTQForCausalLM.from_quantized(model_path, device_map="auto")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_length=50,
        temperature=0.7,
        do_sample=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 运行快速量化
model_path = quick_quantize()
result = quick_inference(model_path)
print("生成结果:", result)