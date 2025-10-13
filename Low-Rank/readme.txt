低秩分解（Low-Rank Factorization）Demo，专为 语音识别（ASR）模型中的全连接层（Linear）和卷积层（Conv2d） 设计，适用于 CPU 环境。

 无需 GPU，仅依赖 PyTorch
包含：SVD 分解 Linear 层 + 张量分解 Conv2d 层 + 微调恢复精度

将一个大权重矩阵/张量近似分解为多个小矩阵/张量的乘积，从而减少参数量和计算量：

Linear 层：W∈Rm×n≈U⋅V ，其中 U∈Rm×r,V∈Rr×n ，r≪min(m,n)
Conv2d 层：将卷积核张量分解为多个低秩张量（如 CP 分解、Tucker 分解）


实际应用建议
    只分解大矩阵：参数量 > 10K 的层；
    结合微调：分解后必须微调（1~5 epoch）；
    评估 WER：在真实 ASR 数据集上验证；
    现代替代方案：低秩分解已被 剪枝 + 量化 取代，因后者更简单高效。
