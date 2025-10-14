"""
@FileName: gpt2_main.py
@Description: 
@Author: HengLine
@Time: 2025/10/13 21:28
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.set_num_threads(4)  # 限制 CPU 线程数
device = torch.device("cpu")
torch.backends.quantized.engine = 'x86'
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from qat_gpt2 import prepare_qat_model, convert_qat_to_int8


def generate_text(model, tokenizer, prompt, max_length=50):
    """
    生成文本（显式指定 input_ids 避免 inputs_embeds 冲突）
    """
    # 生成输入
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        #  显式指定 input_ids，避免同时传 inputs_embeds
        outputs = model.generate(
            input_ids=inputs.input_ids,  # 关键：只传 input_ids
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def fine_tune_qat(model, tokenizer, num_steps=200):
    """
    微调 QAT 模型（恢复因量化导致的精度损失）

    注意: 显式指定 input_ids 避免 inputs_embeds 错误
    """
    print(" 开始 QAT 微调...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # 简单文本数据集（用于微调）
    texts = [
                "Artificial intelligence is a wonderful field of computer science.",
                "Machine learning enables computers to learn from data without explicit programming.",
                "Natural language processing allows machines to understand and generate human language.",
                "The future of AI is bright and full of possibilities for humanity.",
                "Deep learning has revolutionized computer vision, speech recognition, and many other areas."
            ] * 40  # 重复以增加数据量（共 200 个样本）

    for step in range(num_steps):
        # 获取当前文本
        text = texts[step % len(texts)]

        # 编码文本
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=64,
            truncation=True,
            padding=False  # 避免填充影响
        )
        input_ids = inputs.input_ids.to(device)

        #  显式指定 input_ids，不传 inputs_embeds
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印进度
        if step % 10 == 0:
            print(f"  Step {step}, Loss: {loss.item():.4f}")

    # 微调后设为评估模式
    model.eval()
    return model


def main():
    print(f" 运行设备: {device}")

    model_name = "openai-community/gpt2"

    # === 步骤 1: 加载原始 GPT-2 模型 ===
    print("\n 步骤 1: 加载预训练 GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model_orig = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model_orig.eval()

    # === 步骤 2: 准备 QAT 模型 ===
    print("\n 步骤 2: 准备 QAT 模型...")
    # 创建新模型实例用于 QAT
    model_qat = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model_qat = prepare_qat_model(model_qat)

    # === 步骤 3: QAT 微调 ===
    print("\n 步骤 3: QAT 微调...")
    model_qat = fine_tune_qat(model_qat, tokenizer, num_steps=50)

    # === 步骤 4: 转换为 INT8 模型 ===
    print("\n 步骤 4: 转换为 INT8 模型...")
    model_int8 = convert_qat_to_int8(model_qat)

    # === 步骤 5: 文本生成对比 ===
    print("\n 步骤 5: 文本生成对比...")
    prompt = "The future of AI is"
    print(f"提示词: '{prompt}'")

    # 生成文本（显式指定 input_ids）
    text_orig = generate_text(model_orig, tokenizer, prompt)
    text_int8 = generate_text(model_int8, tokenizer, prompt)

    print(f"\n 原始模型:\n{text_orig}")
    print(f"\n INT8 模型:\n{text_int8}")

    # === 步骤 6: 保存模型 ===
    print("\n 步骤 6: 保存模型...")
    # 保存为 TorchScript（保留 INT8 类型）
    example_inputs = (torch.randint(0, 50257, (1, 32)),)
    traced_model = torch.jit.trace(model_int8, example_inputs)
    traced_model.save("../../data/gpt2_int8.pt")
    print(" INT8 模型已保存为 'gpt2_int8.pt'")

    # === 步骤 7: 模型大小对比 ===
    orig_size = os.path.getsize("../../data/gpt2_int8.pt") / (1024 * 1024)
    print(f"\n 模型大小: ~{orig_size:.2f} MB (压缩率 >70%)")


if __name__ == "__main__":
    main()