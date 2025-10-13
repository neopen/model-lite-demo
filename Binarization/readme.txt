二值化（Binarization）和三值化（Ternarization）是极致模型压缩技术，将权重从 FP32 压缩到 1-bit（±1）或 2-bit（-1, 0, +1），适用于超低功耗边缘设备（如 MCU、IoT 传感器）。

依赖torch、torchvision
