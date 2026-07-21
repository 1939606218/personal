import torch
from sstvit import SSTViT
from thop import profile
import time
from tools import args

num_patches = 30

# 初始化模型
model = SSTViT(
        image_size=args.patches,
        near_band=args.band_patches,  # 接近波段或特定波段的块大小
        num_patches=num_patches,  # 图像分割成的块的数量
        num_classes=2,  # 类别数
        dim=32,  # 模型内部的维度，即特征向量的大小
        depth=2,  # Transformer模型的深度，即堆叠的Transformer层的数量
        heads=4,  # 多头注意力机制中的头数，即在注意力机制中并行计算的头的数量
        dim_head=16,  # 每个头在多头注意力机制中的维度大小
        mlp_dim=8,  # MLP（多层感知器）层的维度
        b_dim=512,  #
        b_depth=3,  #
        b_heads=8,
        b_dim_head=32,
        b_mlp_head=8,
        dropout=0.2,  # 使用的dropout比率
        emb_dropout=0.1,  # 在嵌入层使用的dropout比率
    )
# 定义输入数据
batch_size = 64
x1 = torch.randn(batch_size, num_patches, args.patches ** 2 * args.band_patches)
x2 = torch.randn(batch_size, num_patches, args.patches ** 2 * args.band_patches)

# 计算FLOPs和参数数量
macs, params = profile(model, inputs=(x1, x2))
flops = macs * 2  # 1 MAC = 2 FLOPs

print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Parameters: {params / 1e6:.2f} M")

# 计算推理时间
total_time = 0
num_trials = 100

for _ in range(num_trials):
    start_time = time.time()
    _ = model(x1, x2)
    end_time = time.time()
    total_time += end_time - start_time

average_time = total_time / num_trials
print(f"Average inference time: {average_time * 1000:.2f} ms")