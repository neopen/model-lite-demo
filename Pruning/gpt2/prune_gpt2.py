"""
@FileName: prune_gpt2.py
@Description: 结构化剪枝
@Author: HengLine
@Time: 2025/10/13 19:34
"""
import torch
from transformers.pytorch_utils import Conv1D
from transformers import GPT2LMHeadModel


def get_conv1d_l1_norm(conv1d_layer: Conv1D):
    """
    计算 Conv1D 层每个输出通道的 L1 范数（作为通道重要性指标）

    注意: Conv1D.weight.shape = [in_features, out_features]
    所以"输出通道"对应 weight 的第 1 维（out_features）
    """
    weight = conv1d_layer.weight.data  # [in, out]
    # 对每个输出通道（第1维）计算 L1 范数
    l1_norm = torch.norm(weight, p=1, dim=0)  # [out]
    return l1_norm


def prune_conv1d_layer(conv1d_layer: Conv1D, keep_indices: torch.Tensor):
    """
    根据保留的输出通道索引剪枝 Conv1D 层

    Args:
        conv1d_layer: 原始 Conv1D 层
        keep_indices: 要保留的输出通道索引 (shape=[num_keep])

    Returns:
        新的 Conv1D 层（输出通道数 = num_keep）
    """
    weight = conv1d_layer.weight.data  # [in, out]
    bias = conv1d_layer.bias.data if conv1d_layer.bias is not None else None

    # 剪枝权重: 保留 keep_indices 对应的输出通道
    pruned_weight = weight[:, keep_indices]  # [in, num_keep]

    # 创建新层
    in_features = weight.shape[0]
    out_features = len(keep_indices)
    new_layer = Conv1D(out_features, in_features)

    with torch.no_grad():
        new_layer.weight.copy_(pruned_weight)
        if bias is not None:
            new_layer.bias.copy_(bias[keep_indices])

    return new_layer


def prune_gpt2_mlp(model: GPT2LMHeadModel, prune_ratio=0.3):
    """
    对 GPT-2 的 MLP 层进行结构化剪枝

    剪枝策略:
    - 评估 c_fc 和 c_proj 的输出通道重要性
    - 移除 L1 范数最小的 prune_ratio 比例通道
    - 重建模型结构确保维度匹配
    """
    print(f" 开始剪枝 GPT-2 MLP 层 (prune_ratio={prune_ratio})...")

    for layer_idx, block in enumerate(model.transformer.h):
        # print(f"  处理层 {layer_idx}...")

        # === 1. 剪枝 c_fc (768 -> 3072) ===
        c_fc = block.mlp.c_fc
        l1_norm_fc = get_conv1d_l1_norm(c_fc)
        num_total_fc = len(l1_norm_fc)
        num_keep_fc = int(num_total_fc * (1 - prune_ratio))
        _, keep_indices_fc = torch.topk(l1_norm_fc, num_keep_fc, largest=True)
        pruned_c_fc = prune_conv1d_layer(c_fc, keep_indices_fc)
        block.mlp.c_fc = pruned_c_fc

        # === 2. 剪枝 c_proj 的 INPUT 通道（3072 -> num_keep_fc）===
        c_proj = block.mlp.c_proj
        # c_proj.weight.shape = [3072, 768] → 输入通道是第 0 维
        weight_proj = c_proj.weight.data  # [in=3072, out=768]
        bias_proj = c_proj.bias.data if c_proj.bias is not None else None

        # 剪枝输入通道：保留 keep_indices_fc 对应的行
        pruned_weight_proj = weight_proj[keep_indices_fc, :]  # [num_keep_fc, 768]

        # 创建新 c_proj: in=num_keep_fc, out=768
        new_c_proj = Conv1D(768, num_keep_fc)
        with torch.no_grad():
            new_c_proj.weight.copy_(pruned_weight_proj)
            if bias_proj is not None:
                new_c_proj.bias.copy_(bias_proj)  # bias 不变（输出维度仍是 768）

        block.mlp.c_proj = new_c_proj

    print(" GPT-2 MLP 剪枝完成！")
    return model
