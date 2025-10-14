"""
@FileName: evaluate_perplexity.py
@Description: 困惑度评估函数
@Author: HengLine
@Time: 2025/10/14 21:44
"""
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset


def evaluate_perplexity(model, tokenizer, dataset_name="wikitext", split="test", max_samples=1000):
    """
    评估语言模型的困惑度（Perplexity）

    困惑度公式: PPL = exp(average cross-entropy loss)
    - PPL 越低越好
    - GPT-2 在 WikiText-2 上的 PPL 约为 15-20

    Args:
        model: 要评估的模型（GPT2LMHeadModel）
        tokenizer: 对应的 tokenizer
        dataset_name: 数据集名称（默认 "wikitext"）
        split: 数据集分割（"test" 或 "validation"）
        max_samples: 最大评估样本数（避免耗时过长）

    Returns:
        float: 困惑度值
    """
    print(f" 开始评估困惑度（数据集: {dataset_name}, 分割: {split}）...")

    # 1. 加载数据集
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        texts = [example["text"] for example in dataset if len(example["text"].strip()) > 0]
    else:
        # 支持其他数据集（如 "ptb"）
        dataset = load_dataset(dataset_name, split=split)
        texts = [example["text"] for example in dataset if len(example["text"].strip()) > 0]

    # 限制样本数量
    texts = texts[:max_samples]
    print(f" 使用 {len(texts)} 个样本进行评估")

    # 2. 设置模型为评估模式
    model.eval()
    total_loss = 0
    total_tokens = 0

    # 3. 计算平均损失
    with torch.no_grad():
        for i, text in enumerate(texts):
            if not text.strip():
                continue

            # 编码文本
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            input_ids = encodings.input_ids.to(next(model.parameters()).device)

            # 跳过太短的文本
            if input_ids.size(1) < 2:
                continue

            # 前向计算
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            # 累加损失和 token 数
            total_loss += loss.item() * (input_ids.size(1) - 1)  # -1 因为 labels 移位
            total_tokens += (input_ids.size(1) - 1)

            # 进度显示
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1}/{len(texts)} 样本")

    # 4. 计算困惑度
    if total_tokens == 0:
        raise ValueError("没有有效的 token 用于评估！")

    average_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(average_loss)).item()

    print(f" 评估完成！")
    print(f"   平均损失: {average_loss:.4f}")
    print(f"   困惑度 (PPL): {perplexity:.2f}")

    return perplexity


# 便捷函数：快速评估常用模型
def quick_perplexity_evaluation():
    """快速评估原始 gpt2、蒸馏 gpt2 和 distilgpt2"""
    device = torch.device("cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    models_to_test = [
        ("gpt2", "gpt2"),
        ("distilgpt2", "distilgpt2"),
        ("gpt2_distilled", "./gpt2_distilled"),  # 你的蒸馏模型
    ]

    results = {}
    for name, model_path in models_to_test:
        try:
            print(f"\n{'=' * 50}")
            print(f"评估模型: {name}")
            print(f"{'=' * 50}")

            model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
            ppl = evaluate_perplexity(model, tokenizer, max_samples=500)
            results[name] = ppl

        except Exception as e:
            print(f" 评估 {name} 失败: {e}")
            results[name] = None

    # 打印对比结果
    print(f"\n{'=' * 50}")
    print("困惑度对比结果:")
    print(f"{'=' * 50}")
    for name, ppl in results.items():
        if ppl is not None:
            print(f"{name:15}: {ppl:.2f}")
        else:
            print(f"{name:15}: 评估失败")

    return results