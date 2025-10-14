"""
@FileName: gpt2_main.py
@Description: 
@Author: HengLine
@Time: 2025/10/14 20:44
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch

torch.set_num_threads(4)
device = torch.device("cpu")

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from distill_gpt2 import distillation_loss, freeze_teacher, distill_with_three_stages
from datasets import load_dataset
from evaluate_perplexity import evaluate_perplexity

def generate_text(model, tokenizer, prompt, max_length=50):
    """生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,  # ← 关键！惩罚重复 token
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def create_data_loader(tokenizer, batch_size=4, num_samples=100):
    """创建模拟数据集（实际项目应使用真实数据）"""
    # texts = [
    #             "Artificial intelligence is a wonderful field.",
    #             "Machine learning enables computers to learn from data.",
    #             "Natural language processing allows machines to understand human language.",
    #             "The future of AI is bright and full of possibilities.",
    #             "Deep learning has revolutionized many areas of technology."
    #         ] * (num_samples // 5)

    """使用真实数据集"""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [example["text"] for example in dataset if len(example["text"]) > 50]
    texts = texts[:num_samples]  # 限制样本数

    def collate_fn(batch_texts):
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        return encodings.input_ids

    # 简单数据加载器
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        yield collate_fn(batch)


def distill_gpt2(teacher_model, student_model, tokenizer, num_epochs=3):
    """执行知识蒸馏"""
    print(" 开始知识蒸馏...")

    # 冻结 Teacher
    teacher_model = freeze_teacher(teacher_model)
    teacher_model.eval()

    student_model.train()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        total_loss = 0
        total_kl = 0
        total_ce = 0

        for step, input_ids in enumerate(create_data_loader(tokenizer)):
            input_ids = input_ids.to(device)
            labels = input_ids.clone()

            # Teacher 推理（无梯度）
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids)
                teacher_logits = teacher_outputs.logits

            # Student 推理
            student_outputs = student_model(input_ids=input_ids)
            student_logits = student_outputs.logits

            # 计算蒸馏损失（alpha=0.7, temperature=2.0 时，模型陷入重复 token 循环）
            loss, kl_loss, ce_loss = distillation_loss(
                student_logits, teacher_logits, labels,
                alpha=0.5, temperature=1.5
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_kl += kl_loss.item()
            total_ce += ce_loss.item()

            if step % 20 == 0:
                print(f"  Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f} "
                      f"(KL: {kl_loss:.4f}, CE: {ce_loss:.4f})")

        avg_loss = total_loss / (step + 1)
        print(f" Epoch {epoch + 1} 完成, Avg Loss: {avg_loss:.4f}")

    student_model.eval()
    return student_model


def main():
    print(f" 设备: {device}")

    model_name = "openai-community/gpt2"
    # === 步骤 1: 加载 Tokenizer ===
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token

    # === 步骤 2: 加载 Teacher 和 Student ===
    print(" 加载 Teacher (gpt2-medium) 和 Student (gpt2)...")

    # Teacher: gpt2-medium (355M 参数)
    teacher = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-medium").to(device)

    # Student: gpt2 (124M 参数)
    student = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    # 直接微调官方蒸馏模型
    # student = GPT2LMHeadModel.from_pretrained("distilbert/distilgpt2").to(device)

    print(f"Teacher 参数量: ~355M")
    print(f"Student 参数量: ~124M")

    # === 步骤 3: 执行知识蒸馏 ===
    # distilled_student = distill_gpt2(teacher, student, tokenizer, num_epochs=2)
    # 使用三阶段蒸馏
    distilled_student = distill_with_three_stages(
        teacher, student,
        create_data_loader(tokenizer, num_samples=500),
        tokenizer, device
    )

    # === 步骤 4: 生成文本对比 ===
    print("\n 文本生成对比...")
    prompt = "The future of AI is"
    print(f"提示词: '{prompt}'")

    # Teacher 生成
    text_teacher = generate_text(teacher, tokenizer, prompt)
    # Student 生成（蒸馏后）
    text_student = generate_text(distilled_student, tokenizer, prompt)
    # 原始 Student 生成（可选）
    student_orig = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    text_orig = generate_text(student_orig, tokenizer, prompt)

    print(f"\n Teacher (gpt2-medium):\n{text_teacher}")
    print(f"\n Student (原始):\n{text_orig}")
    print(f"\n Student (蒸馏后):\n{text_student}")

    # === 步骤 5: 评估困惑度 ===
    print("\n 评估模型困惑度...")

    # 评估蒸馏后的 Student
    ppl_student = evaluate_perplexity(
        distilled_student,
        tokenizer,
        max_samples=200  # 减少样本数加速
    )
    # 评估原始 Student
    student_orig = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    ppl_orig = evaluate_perplexity(student_orig, tokenizer, max_samples=200)
    # 评估 Teacher
    ppl_teacher = evaluate_perplexity(teacher, tokenizer, max_samples=200)

    print(f"\n 困惑度对比:")
    print(f"  Teacher (gpt2-medium): {ppl_teacher:.2f}")
    print(f"  Student (原始 gpt2):   {ppl_orig:.2f}")
    print(f"  Student (蒸馏后):      {ppl_student:.2f}")

    # === 步骤 6: 保存蒸馏后模型 ===
    distilled_student.save_pretrained("../../data/gpt2_distilled")
    tokenizer.save_pretrained("../../data/gpt2_distilled")
    print("\n 蒸馏模型已保存到 './data/gpt2_distilled'")


if __name__ == "__main__":
    main()