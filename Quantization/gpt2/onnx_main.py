"""
@FileName: onnx_main.py
@Description: 
@Author: HengLine
@Time: 2025/10/14 14:01
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import onnxruntime as ort
from transformers import GPT2Tokenizer


def generate_onnx(model_path, tokenizer, prompt, max_length=50):
    # 创建会话
    sess = ort.InferenceSession(model_path)

    # 编码输入
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs.input_ids

    # 生成循环
    for _ in range(max_length):
        # 推理
        outputs = sess.run(None, {'input_ids': input_ids})
        logits = outputs[0]  # [batch, seq, vocab]

        # 采样下一个 token
        next_token_logits = logits[0, -1, :]
        next_token = np.random.choice(
            len(next_token_logits),
            p=np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
        )

        # 检查 EOS
        if next_token == tokenizer.eos_token_id:
            break

        # 拼接 token
        input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main():
    model_name = "openai-community/gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    prompt = "The future of AI is"

    text = generate_onnx("../../data/gpt2_int8.onnx", tokenizer, prompt)
    print(f" ONNX INT8 模型:\n{text}")


if __name__ == "__main__":
    main()
