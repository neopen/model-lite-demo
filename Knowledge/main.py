"""
@FileName: main.py.py
@Description: 
@Author: HengLine
@Time: 2025/10/13 10:47
"""
# main.py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

torch.set_num_threads(4)
device = torch.device("cpu")

from model import ASRModel
from data import generate_batch
from distill import distillation_loss


def train_student_with_distillation(teacher, student, num_epochs=10):
    teacher.eval()  # Teacher 固定
    student.train()

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        total_loss = 0
        for step in range(20):  # 20 steps per epoch
            x, x_lens, y, y_lens = generate_batch()
            x, y = x.to(device), y.to(device)
            x_lens, y_lens = x_lens.to(device), y_lens.to(device)

            # Teacher 推理（无梯度）
            with torch.no_grad():
                teacher_logits, _ = teacher(x, x_lens)

            # Student 推理
            student_logits, student_lens = student(x, x_lens)

            # 计算蒸馏损失
            loss, kl_loss, ctc_loss = distillation_loss(
                student_logits, teacher_logits, y, student_lens, y_lens,
                alpha=0.7, temperature=2.0
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / 20
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f} "
              f"(KL: {kl_loss:.4f}, CTC: {ctc_loss:.4f})")

    return student


def main():
    print(f" 设备: {device}")

    # === 1. 创建 Teacher（大模型）和 Student（小模型）===
    teacher = ASRModel(input_dim=80, encoder_dim=128, vocab_size=1000).to(device)
    student = ASRModel(input_dim=80, encoder_dim=64, vocab_size=1000).to(device)  # 更小

    print(f"Teacher 参数量: {sum(p.numel() for p in teacher.parameters()) / 1e6:.2f} M")
    print(f"Student 参数量: {sum(p.numel() for p in student.parameters()) / 1e6:.2f} M")

    # === 2. 执行知识蒸馏 ===
    print("\n 开始知识蒸馏...")
    distilled_student = train_student_with_distillation(teacher, student, num_epochs=5)

    # === 3. 保存蒸馏后模型 ===
    torch.save(distilled_student.state_dict(), "../data/student_distilled.pth")
    print("\n 蒸馏完成！模型已保存为 'student_distilled.pth'")


if __name__ == "__main__":
    main()