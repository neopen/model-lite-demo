"""
@FileName: gpt2_demo.py
@Description: 分解 + 生成对比
@Author: HengLine
@Time: 2025/10/13 15:49
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

torch.set_num_threads(4)
device = torch.device("cpu")

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from decompose_gpt2 import decompose_gpt2_mlp


def generate_text(model, tokenizer, prompt, max_length=50):
    """生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# demo.py
def fine_tune_decomposed(model, tokenizer, num_steps=100):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 使用简单文本（如 WikiText）
    texts = "Artificial intelligence is a wonderful field."

    for step in range(num_steps):
        text: str = texts
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs.input_ids.to(device)

        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    model.eval()
    return model


def main():
    print(f" 设备: {device}")

    model_name = "openai-community/gpt2"

    # 1. 加载原始 GPT-2
    print(" 加载预训练 GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model_orig = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model_orig.eval()

    print(f"原始模型参数量: {count_parameters(model_orig):.2f} M")

    # 2. 复制模型用于分解
    model_decomp = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model_decomp.eval()

    # 3. 低秩分解 0.3 重复; 0.5 可用; 0.6 较好; 0.7 接近原始
    # 分解全部 + 微调
    # model_decomp = decompose_gpt2_mlp(model_decomp, rank_ratio=0.5)
    # model_decomp = fine_tune_decomposed(model_decomp, tokenizer, num_steps=60)

    # 推荐：只分解 c_proj + rank_ratio=0.5
    model_decomp = decompose_gpt2_mlp(model_decomp, rank_ratio=0.6, decompose_c_fc=False)

    print(f"分解后模型参数量: {count_parameters(model_decomp):.2f} M")

    # 4. 文本生成对比
    prompt = "The future of AI is"
    print(f"\n 提示词: '{prompt}'")

    text_orig = generate_text(model_orig, tokenizer, prompt)
    print(f"\n 原始模型:\n{text_orig}")

    text_decomp = generate_text(model_decomp, tokenizer, prompt)
    print(f"\n 分解模型:\n{text_decomp}")

    # 5. 保存分解后模型
    model_decomp.save_pretrained("../../data/gpt2_decomposed")
    tokenizer.save_pretrained("../../data/gpt2_decomposed")
    print("\n 分解模型已保存到 './data/gpt2_decomposed'")


if __name__ == "__main__":
    main()