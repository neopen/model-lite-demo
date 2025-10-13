"""
@FileName: distill.py.py
@Description: 知识蒸馏核心逻辑
@Author: HengLine
@Time: 2025/10/13 10:46
"""
# distill.py
import torch
import torch.nn.functional as fun


def distillation_loss(student_logits, teacher_logits, labels,
                      student_lens, label_lens,
                      alpha=0.7, temperature=2.0):
    """
    知识蒸馏损失 = α * 软标签损失 + (1-α) * 硬标签 CTC 损失

    Args:
        student_logits: Student 模型输出 [B, T, V]
        teacher_logits: Teacher 模型输出 [B, T, V]
        labels: 真实标签 [B, L]
        student_lens: Student 输出有效长度 [B]
        label_lens: 标签有效长度 [B]
        alpha: 软标签损失权重（0.5~0.9）
        temperature: 温度参数（>1 平滑分布）
    """
    # === 1. 软标签损失（KL 散度）===
    # Teacher 软化概率
    teacher_probs = fun.log_softmax(teacher_logits / temperature, dim=2)
    # Student 软化概率
    student_probs = fun.log_softmax(student_logits / temperature, dim=2)

    # KL 散度（需 mask 无效位置）
    kl_loss = fun.kl_div(
        student_probs,
        teacher_probs,
        reduction='none'
    ).sum(dim=2)  # [B, T]

    # 创建 mask（只计算有效时间步）
    mask = torch.arange(kl_loss.size(1), device=kl_loss.device)[None, :] < student_lens[:, None]
    kl_loss = (kl_loss * mask).sum() / mask.sum()

    # 温度缩放
    kl_loss = kl_loss * (temperature ** 2)

    # === 2. 硬标签 CTC 损失 ===
    ctc_log_probs = fun.log_softmax(student_logits, dim=2).permute(1, 0, 2)
    ctc_loss = fun.ctc_loss(
        ctc_log_probs, labels, student_lens, label_lens,
        blank=0, zero_infinity=True
    )

    # === 3. 加权总损失 ===
    total_loss = alpha * kl_loss + (1 - alpha) * ctc_loss
    return total_loss, kl_loss, ctc_loss
