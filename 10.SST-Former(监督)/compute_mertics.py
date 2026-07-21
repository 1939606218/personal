import torch
import json
from sstvit import SSTViT
from torch.utils.benchmark import Timer
from thop import profile

# 定义模型参数
image_size = 5
near_band = 30
num_patches = 30
num_classes = 2
dim = 32
depth = 2
heads = 4
mlp_dim = 8
b_dim = 512
b_depth = 3
b_heads = 8
b_dim_head = 32
b_mlp_head = 8
pool = 'cls'
channels = 1
dim_head = 16
dropout = 0.2
emb_dropout = 0.1
multi_scale_enc_depth = 1

# 初始化模型
model = SSTViT(
    image_size=image_size,
    near_band=near_band,
    num_patches=num_patches,
    num_classes=num_classes,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
    b_dim=b_dim,
    b_depth=b_depth,
    b_heads=b_heads,
    b_dim_head=b_dim_head,
    b_mlp_head=b_mlp_head,
    pool=pool,
    channels=channels,
    dim_head=dim_head,
    dropout=dropout,
    emb_dropout=emb_dropout,
    multi_scale_enc_depth=multi_scale_enc_depth
)

# 调整输入数据形状，确保与模型期望的维度匹配
# 假设模型期望的输入形状是 (batch_size, channels, height, width)
# 这里需要将 (64, 30, 5, 5) 转换为模型可以处理的形状
# 通常，transformer模型需要将图像展平为序列
batch_size = 64
x1 = torch.randn(batch_size, near_band, image_size, image_size)
x2 = torch.randn(batch_size, near_band, image_size, image_size)

# 计算总参数量
total_params = sum(p.numel() for p in model.parameters())

# 计算FLOPs
flops, _ = profile(model, inputs=(x1, x2))

# 计算平均推理时间
timer = Timer(
    stmt='model(x1, x2)',
    globals={'model': model, 'x1': x1, 'x2': x2}
)
avg_inference_time = timer.timeit(100).mean * 1000  # 转换为毫秒

# 格式化结果，使用更友好的单位表示
def format_number(value, unit_list, base=1000):
    """将大数字格式化为带单位的字符串"""
    for i, unit in enumerate(unit_list):
        if value < base ** (i + 1):
            return f"{value / (base ** i):.2f} {unit}"
    return f"{value / (base ** len(unit_list)):.2f} {unit_list[-1]}"

param_units = ['', 'K', 'M', 'B']
flop_units = ['FLOPs', 'KFLOPs', 'MFLOPs', 'GFLOPs', 'TFLOPs']

formatted_params = format_number(total_params, param_units)
formatted_flops = format_number(flops, flop_units)

result = {
    "总参数量": f"{formatted_params} 参数",
    "FLOPs": formatted_flops,
    "平均推理时间": f"{avg_inference_time:.2f} ms"
}

# 保存结果到JSON文件
with open('model_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("计算完成，结果已保存到 model_metrics.json 文件中。")
print("指标概览:")
for key, value in result.items():
    print(f"{key}: {value}")