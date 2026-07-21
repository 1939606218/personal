import torch

# 打印PyTorch版本
print(f"PyTorch版本: {torch.__version__}")

# 打印PyTorch使用的CUDA版本
print(f"PyTorch对应的CUDA版本: {torch.version.cuda}")

# 检查CUDA是否可用
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 如果CUDA可用，打印当前使用的GPU信息
if torch.cuda.is_available():
    print(f"当前使用的GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU数量: {torch.cuda.device_count()}")
