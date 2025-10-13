"""
@FileName: decompose_gpt2.py
@Description: 低秩分解
@Author: HengLine
@Time: 2025/10/13 15:49
"""
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from transformers import GPT2LMHeadModel


def decompose_conv1d_svd(conv1d_layer: Conv1D, rank_ratio=0.5):
    """
    使用奇异值分解（SVD）对 Hugging Face 的 Conv1D 层进行低秩近似分解。

    背景说明：
    - Hugging Face 的 GPT-2/Transformer 模型使用自定义 Conv1D 层（非标准 nn.Conv1d）
    - Conv1D(out_features, in_features) 表示：输入 in_features 维，输出 out_features 维
    - 其权重矩阵 W 的形状为 [in_features, out_features]（注意：与 nn.Linear 相反！）
    - 前向计算公式：output = input @ W + bias （标准矩阵乘法，无需转置）

    分解目标：
    将原始权重矩阵 W ∈ R^{n×m}（n=in_features, m=out_features）近似分解为：
        W ≈ A @ B
    其中：
        A ∈ R^{n×r}  （第一层权重）
        B ∈ R^{r×m}  （第二层权重）
        r = rank_ratio * min(n, m)  （低秩近似秩）

    优势：
    - 原始参数量：n × m
    - 分解后参数量：n × r + r × m = r × (n + m)
    - 当 r << min(n, m) 时，显著减少参数量和计算量

    Args:
        conv1d_layer (Conv1D): 待分解的 Hugging Face Conv1D 层
        rank_ratio (float): 保留的秩比例（0.0 ~ 1.0），值越小压缩率越高，但精度损失越大

    Returns:
        nn.Sequential: 由两个 Conv1D 层组成的序列，功能等价于原始层（近似）
    """

    # === 步骤 1: 提取原始权重和偏置 ===
    # Conv1D.weight 形状: [in_features, out_features]
    W = conv1d_layer.weight.data  # 获取权重张量（不计算梯度）

    # 提取偏置（如果存在）
    # Conv1D.bias 形状: [out_features]
    bias = conv1d_layer.bias.data if conv1d_layer.bias is not None else None

    # === 步骤 2: 获取输入/输出维度 ===
    # W.shape = [in_features, out_features]
    in_features, out_features = W.shape

    # 计算最大可能秩（矩阵的秩不超过 min(行数, 列数)）
    max_rank = min(in_features, out_features)

    # 根据 rank_ratio 计算目标分解秩 r
    r = int(rank_ratio * max_rank)
    # 确保 r 至少为 1（避免秩为 0）
    r = max(1, r)
    # 确保 r 不超过 max_rank - 1（避免数值不稳定，且 SVD 需要 r < 秩）
    r = min(r, max_rank - 1)

    # === 步骤 3: 执行奇异值分解（SVD）===
    # 对权重矩阵 W 执行 SVD: W = U @ diag(S) @ Vh
    # - U: 左奇异向量矩阵，形状 [in_features, max_rank]
    # - S: 奇异值向量，形状 [max_rank]
    # - Vh: 右奇异向量矩阵的转置，形状 [max_rank, out_features]
    # 使用 full_matrices=False 以节省内存（只计算必要部分）
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    # === 步骤 4: 提取前 r 个奇异分量 ===
    # 取前 r 列的左奇异向量: [in_features, r]
    U_r = U[:, :r]
    # 取前 r 个奇异值: [r]
    S_r = S[:r]
    # 取前 r 行的右奇异向量转置: [r, out_features]
    Vh_r = Vh[:r, :]

    # === 步骤 5: 构造低秩近似矩阵 ===
    # 数学原理：W ≈ U_r @ diag(S_r) @ Vh_r
    # 为数值稳定性和对称性，将奇异值平方根分配到两边：
    #   A = U_r @ diag(sqrt(S_r))  → [in_features, r]
    #   B = diag(sqrt(S_r)) @ Vh_r  → [r, out_features]
    # 这样 A @ B = U_r @ diag(S_r) @ Vh_r ≈ W

    # 计算 sqrt(S_r) 并扩展维度以支持广播
    sqrt_S = torch.sqrt(S_r)  # [r]

    # 构造 A = U_r * sqrt(S_r)
    # 使用 unsqueeze(0) 将 sqrt_S 变为 [1, r]，与 U_r [in_features, r] 广播相乘
    A = U_r * sqrt_S.unsqueeze(0)  # 结果形状: [in_features, r]

    # 构造 B = sqrt(S_r) * Vh_r
    # 使用 unsqueeze(1) 将 sqrt_S 变为 [r, 1]，与 Vh_r [r, out_features] 广播相乘
    B = sqrt_S.unsqueeze(1) * Vh_r  # 结果形状: [r, out_features]

    # === 步骤 6: 创建两个新的 Conv1D 层 ===
    # 第一层: 输入 in_features → 输出 r
    # Conv1D(out_features=r, in_features=in_features)
    conv1 = Conv1D(r, in_features)

    # 第二层: 输入 r → 输出 out_features
    # Conv1D(out_features=out_features, in_features=r)
    # 注意：偏置只加在最后一层（与原始层一致）
    conv2 = Conv1D(out_features, r)

    # === 步骤 7: 复制分解后的权重和偏置 ===
    with torch.no_grad():  # 禁用梯度计算，避免影响优化器状态
        # 复制第一层权重 A ([in_features, r])
        # conv1.weight 形状应为 [in_features, r]
        conv1.weight.copy_(A)

        # 复制第二层权重 B ([r, out_features])
        # conv2.weight 形状应为 [r, out_features]
        conv2.weight.copy_(B)

        # 复制偏置（仅第二层需要，因为原始偏置作用于最终输出）
        if bias is not None:
            # conv2.bias 形状: [out_features]
            conv2.bias.copy_(bias)

    # === 步骤 8: 返回组合层 ===
    # 使用 nn.Sequential 将两个 Conv1D 层串联
    # 前向计算: input -> conv1 -> conv2 -> output
    # 功能等价于: output = input @ W + bias （近似）
    return nn.Sequential(conv1, conv2)


def decompose_gpt2_mlp(model: GPT2LMHeadModel, rank_ratio=0.5, decompose_c_fc=True):
    print(f" 开始分解 GPT-2 MLP 层 (rank_ratio={rank_ratio})...")

    for block in model.transformer.h:
        if decompose_c_fc:
            block.mlp.c_fc = decompose_conv1d_svd(block.mlp.c_fc, rank_ratio)
        block.mlp.c_proj = decompose_conv1d_svd(block.mlp.c_proj, rank_ratio)
    return model