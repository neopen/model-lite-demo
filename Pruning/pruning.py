"""
@FileName: pruning.py.py
@Description: 剪枝核心逻辑
@Author: HengLine
@Time: 2025/10/12 16:32
"""
# pruning.py
import torch
import torch.nn as nn

from model import TinyConformerASR


def prune_conv_layer(conv: nn.Conv2d, keep_indices: torch.Tensor) -> nn.Conv2d:
    """
    根据保留的通道索引剪枝卷积层（输出通道）
    keep_indices: 要保留的输出通道索引（如 [0,2,5,...]）
    """
    new_out_channels = len(keep_indices)
    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=new_out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None
    )
    # 复制保留的输出通道权重
    new_conv.weight.data = conv.weight.data[keep_indices]
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[keep_indices]
    return new_conv

def apply_structured_pruning(model: nn.Module, prune_ratio: float = 0.3) -> nn.Module:
    """
    对 ASR 模型进行结构化剪枝（剪枝 conv1/conv2/out_proj/ctc 的输入和输出维度）
    """
    print(" 开始基于 L1 范数评估通道重要性...")
    # === Step 1: 评估 conv1 通道重要性 ===
    conv1 = model.subsampling.conv1
    # 计算每个输出通道的 L1 范数
    l1_norm = torch.norm(conv1.weight.data.view(conv1.out_channels, -1), p=1, dim=1)

    num_total = conv1.out_channels  # 原始通道数，如 128
    num_keep = int(num_total * (1 - prune_ratio))
    # 保留 L1 范数最大的 num_keep 个通道
    _, keep_indices = torch.topk(l1_norm, num_keep, largest=True)
    print(f"  原通道数: {num_total}, 剪枝比例: {prune_ratio:.1%}, 保留通道数: {num_keep}")

    # === Step 2: 剪枝 conv1 和 conv2 ===
    pruned_conv1 = prune_conv_layer(conv1, keep_indices)

    conv2 = model.subsampling.conv2
    # 剪枝 conv2: 输入通道（对应 conv1 输出）和输出通道都按 keep_indices 剪
    pruned_conv2_weight = conv2.weight.data[keep_indices][:, keep_indices]
    pruned_conv2 = nn.Conv2d(
        in_channels=num_keep,
        out_channels=num_keep,
        kernel_size=conv2.kernel_size,
        stride=conv2.stride,
        padding=conv2.padding,
        bias=conv2.bias is not None
    )
    pruned_conv2.weight.data = pruned_conv2_weight
    if conv2.bias is not None:
        pruned_conv2.bias.data = conv2.bias.data[keep_indices]

    # === Step 3: 创建新模型（encoder_dim = num_keep）===
    new_model = TinyConformerASR(
        input_dim=80,
        encoder_dim=num_keep,
        vocab_size=model.ctc_classifier.out_features
    )

    # === Step 4: 重建并赋值所有相关层 ===
    with torch.no_grad():
        # 替换卷积层
        new_model.subsampling.conv1 = pruned_conv1
        new_model.subsampling.conv2 = pruned_conv2

        # === 重建 out_proj 层 ===
        F_prime = ((80 + 1) // 2 + 1) // 2  # = 20
        old_out_proj = model.subsampling.out_proj  # 原始: Linear(2560, 128)

        # 新 out_proj: 输入 = 76*20=1520, 输出 = 76
        new_out_proj = nn.Linear(num_keep * F_prime, num_keep)

        # 重塑原始权重: [128 (out), 2560 (in)] -> [128, 128, 20]
        old_weight = old_out_proj.weight.data.view(
            old_out_proj.out_features, -1, F_prime
        )
        # 同时剪输出维度（第0维）和输入的 conv 通道维度（第1维）
        new_weight = old_weight[keep_indices][:, keep_indices, :]  # [76, 76, 20]
        new_weight = new_weight.reshape(num_keep, num_keep * F_prime)  # [76, 1520]
        new_out_proj.weight.copy_(new_weight)

        # 偏置也剪（输出维度）
        if old_out_proj.bias is not None:
            new_out_proj.bias.copy_(old_out_proj.bias.data[keep_indices])
        new_model.subsampling.out_proj = new_out_proj

        # === 重建 CTC 分类头 ===
        old_ctc = model.ctc_classifier  # 原始: Linear(128, V)
        new_ctc = nn.Linear(num_keep, old_ctc.out_features)  # 新: Linear(76, V)
        # 只保留与保留通道对应的输入维度（第1维）
        new_ctc.weight.copy_(old_ctc.weight.data[:, keep_indices])  # [V, 76]
        if old_ctc.bias is not None:
            new_ctc.bias.copy_(old_ctc.bias.data)
        new_model.ctc_classifier = new_ctc

    print(" 剪枝后模型重建完成！所有维度已对齐。")
    return new_model