import torch

print(torch.cuda.is_available()) # 检查 CUDA 是否可用
if torch.cuda.is_available():
    print(torch.version.cuda) # 检查 PyTorch 使用的 CUDA 版本
    print(torch.backends.cudnn.version()) # 检查 cuDNN 版本 (如果可用)