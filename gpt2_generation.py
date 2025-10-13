"""
@FileName: gpt2_generation.py
@Description: 
@Author: HengLine
@Time: 2025/10/13 14:17
"""
import os

import torch
# 使用镜像源加速下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['TRANSFORMERS_OFFLINE'] = '0'  # 关闭离线


from transformers import pipeline, set_seed, GPT2Tokenizer, GPT2LMHeadModel


def generate_text(prompt, model_name="openai-community/gpt2", max_length=128, num_return_sequences=2, seed=42):
    generator = pipeline('text-generation', model=model_name)
    set_seed(seed)
    results = generator(prompt, max_length=max_length, max_new_tokens=512, num_return_sequences=num_return_sequences)
    return results



if __name__ == '__main__':
    # git clone https://hf-mirror.com/openai-community/gpt2
    # 对中文的支持，一塌糊涂
    results = generate_text("In a shocking finding, scientists discovered"
                            , model_name="E:\\AI\\models\\gpt2", max_length=50, num_return_sequences=3)
    for i, result in enumerate(results):
        print(f"=== Generated Text {i + 1} ===")
        print(result['generated_text'])
        print()


