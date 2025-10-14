"""
@FileName: distill_gpt2.py
@Description: GPT-2 知识蒸馏
@Author: HengLine
@Time: 2025/10/14 20:44
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch

import torch.nn.functional as fun

def compute_language_modeling_loss(logits, labels):
    """标准语言建模损失"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = fun.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )
    return loss


def distillation_loss(student_logits, teacher_logits, labels,
                      alpha=0.5, temperature=1.5):
    """
    知识蒸馏损失 = α * 软标签损失 + (1-α) * 硬标签损失

    Args:
        student_logits: Student 模型输出 [B, T, V]
        teacher_logits: Teacher 模型输出 [B, T, V]
        labels: 真实标签 [B, T]
        alpha: 软标签损失权重（0.5~0.9）
        temperature: 温度参数（>1 平滑分布）
    """
    # === 1. 软标签损失（KL 散度）===
    # Teacher 软化概率分布
    teacher_probs = fun.log_softmax(teacher_logits / temperature, dim=-1)
    # Student 软化概率分布
    student_probs = fun.log_softmax(student_logits / temperature, dim=-1)

    # KL 散度损失
    kl_loss = fun.kl_div(
        student_probs,
        teacher_probs,
        reduction='batchmean'
    ) * (temperature ** 2)

    # 硬标签损失（高权重）
    ce_loss = compute_language_modeling_loss(student_logits, labels)

    # 总损失：KL 权重更低
    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss

    return total_loss, kl_loss, ce_loss


def freeze_teacher(teacher_model):
    """冻结 Teacher 模型（不计算梯度）"""
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model


def pretrain_student(student_model, dataloader, tokenizer, num_epochs=1):
    """阶段 1: 预训练 Student"""
    print(" 阶段 1: 预训练 Student...")
    student_model.train()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=2e-5)

    for epoch in range(num_epochs):
        total_loss = 0
        for step, input_ids in enumerate(dataloader):
            input_ids = input_ids.to(next(student_model.parameters()).device)
            labels = input_ids.clone()

            outputs = student_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"  Pretrain Step {step}, Loss: {loss.item():.4f}")

    student_model.eval()
    return student_model


def distill_with_three_stages(teacher_model, student_model, dataloader, tokenizer, device):
    """三阶段蒸馏"""
    # 冻结 Teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    # === 阶段 1: 预训练 Student ===
    student_model = pretrain_student(student_model, dataloader, tokenizer, num_epochs=1)

    # === 阶段 2: 温和蒸馏 ===
    print(" 阶段 2: 温和蒸馏...")
    student_model.train()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)

    for epoch in range(2):  # 2 epochs 蒸馏
        total_loss = 0
        for step, input_ids in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = input_ids.clone()

            with torch.no_grad():
                teacher_logits = teacher_model(input_ids=input_ids).logits

            student_logits = student_model(input_ids=input_ids).logits
            loss, kl_loss, ce_loss = distillation_loss(
                student_logits, teacher_logits, labels,
                alpha=0.3,  # 降低 KL 权重
                temperature=1.2  # 降低温度
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"  Distill Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f} "
                      f"(KL: {kl_loss:.4f}, CE: {ce_loss:.4f})")

    # === 阶段 3: 后训练 ===
    print(" 阶段 3: 后训练...")
    student_model.train()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-6)

    for step, input_ids in enumerate(dataloader):
        if step >= 200:  # 只微调 200 步
            break
        input_ids = input_ids.to(device)
        labels = input_ids.clone()

        outputs = student_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"  Post-train Step {step}, Loss: {loss.item():.4f}")

    student_model.eval()
    return student_model