"""
@FileName: train.py
@Description: 训练脚本
@Author: HengLine
@Time: 2025/10/13 17:57
"""
# train.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from model import BinaryCNN


def main():
    device = torch.device("cpu")
    print(f" 设备: {device}")

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 模型
    model = BinaryCNN(ternary=False).to(device)  # 设为 True 使用三值化
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练
    model.train()
    for epoch in range(10):
        total_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        acc = 100. * correct / len(train_loader.dataset)
        print(f"Epoch {epoch + 1} 完成, Accuracy: {acc:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), "../data/binary_mnist.pth")
    print(" 模型已保存为 '../data/binary_mnist.pth'")

    # 模型大小
    size_mb = os.path.getsize("../data/binary_mnist.pth") / (1024 * 1024)
    print(f"模型大小: {size_mb:.2f} MB (原始 FP32 ~3MB)")


if __name__ == "__main__":
    main()