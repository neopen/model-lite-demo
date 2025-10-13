"""
@FileName: decompose.py.py
@Description: 低秩分解实现
@Author: HengLine
@Time: 2025/10/13 11:01
"""
# decompose.py
import torch
import torch.nn as nn


def decompose_linear_svd(linear_layer: nn.Linear, rank_ratio=0.5):
    """
    使用 SVD 对 Linear 层进行低秩分解
    输入: Linear(in_features=n, out_features=m)
    输出: (Linear(n, r), Linear(r, m))
    """
    if not isinstance(linear_layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(linear_layer)}")

    W = linear_layer.weight.data  # [m, n] = [256, 2560]
    m, n = W.shape
    print(f"  W shape: {W.shape}")

    max_rank = min(m, n)  # = 256
    r = int(rank_ratio * max_rank)
    r = max(1, min(r, max_rank - 1))
    print(f"  Using rank r = {r}")

    #  关键：使用 linalg.svd with full_matrices=False
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    # print(f"  U shape: {U.shape}")  # [256, 256]
    # print(f"  S shape: {S.shape}")  # [256]
    # print(f"  Vh shape: {Vh.shape}")  # [256, 2560] ← 必须是这个！

    U_r = U[:, :r]  # [256, r]
    S_r = S[:r]  # [r]
    Vh_r = Vh[:r, :]  # [r, 2560] ← 关键！

    sqrt_S = torch.sqrt(S_r)
    B = sqrt_S.unsqueeze(1) * Vh_r  # [r, 2560]
    A = U_r * sqrt_S.unsqueeze(0)  # [256, r]

    # print(f"  B shape: {B.shape}")  # 应为 [r, 2560]
    # print(f"  A shape: {A.shape}")  # 应为 [256, r]

    fc1 = nn.Linear(n, r, bias=False)
    fc2 = nn.Linear(r, m, bias=linear_layer.bias is not None)

    with torch.no_grad():
        fc1.weight.copy_(B)  # [r, n] = [r, 2560]
        fc2.weight.copy_(A)  # [m, r] = [256, r]
        if linear_layer.bias is not None:
            fc2.bias.copy_(linear_layer.bias.data)

    return fc1, fc2