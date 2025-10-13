"""
@FileName: gpt2_main.py
@Description: 
@Author: HengLine
@Time: 2025/10/13 19:36
"""
# demo.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

torch.set_num_threads(4)
device = torch.device("cpu")

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from prune_gpt2 import prune_gpt2_mlp


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


def fine_tune_pruned(model, tokenizer, num_steps=100):
    """微调剪枝后模型（恢复精度）"""
    print(" 开始微调剪枝后模型...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # 简单文本用于微调
    texts = [
                "Artificial intelligence is a wonderful field.",
                "Machine learning enables computers to learn from data.",
                "Natural language processing allows machines to understand human language.",
                "The future of AI is bright and full of possibilities.",
                "Deep learning has revolutionized many areas of technology."
            ] * 20  # 重复以增加数据量

    for step in range(num_steps):
        text = texts[step % len(texts)]
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        input_ids = inputs.input_ids.to(device)

        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"  Step {step}, Loss: {loss.item():.4f}")

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

    # 2. 复制模型用于剪枝
    model_pruned = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model_pruned.eval()

    # 3. 执行剪枝
    model_pruned = prune_gpt2_mlp(model_pruned, prune_ratio=0.3)

    print(f"剪枝后模型参数量: {count_parameters(model_pruned):.2f} M")

    # 4. 微调恢复精度（关键！）
    model_pruned = fine_tune_pruned(model_pruned, tokenizer, num_steps=200)

    # 5. 文本生成对比
    prompt = "The future of AI is"
    print(f"\n 提示词: '{prompt}'")

    text_orig = generate_text(model_orig, tokenizer, prompt)
    text_pruned = generate_text(model_pruned, tokenizer, prompt)

    print(f"\n 原始模型:\n{text_orig}")
    print(f"\n 剪枝模型:\n{text_pruned}")

    # 6. 保存剪枝模型
    model_pruned.save_pretrained("../../data/gpt2_pruned")
    tokenizer.save_pretrained("../../data/gpt2_pruned")
    print("\n 剪枝模型已保存到 './data/gpt2_pruned'")


if __name__ == "__main__":
    main()