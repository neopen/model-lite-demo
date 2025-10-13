"""
@FileName: model.py
@Description: 二值化 CNN
@Author: HengLine
@Time: 2025/10/13 17:57
"""
import torch.nn as nn
from binary_layers import BinaryConv2d, TernaryConv2d


class BinaryCNN(nn.Module):
    """二值化 CNN（适用于 MNIST）"""

    def __init__(self, num_classes=10, ternary=False):
        super().__init__()
        ConvLayer = TernaryConv2d if ternary else BinaryConv2d

        self.features = nn.Sequential(
            ConvLayer(1, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),  # 批归一化稳定训练

            ConvLayer(32, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
        )
        self.classifier = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x