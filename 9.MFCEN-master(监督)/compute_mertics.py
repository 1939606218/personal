import torch
import torch.nn as nn
import json
import time
from thop import profile
from vit_pytorch import ViT

# 定义模型参数
patch_size = 7
num_feats = 4  # 因为有 4 个尺度的特征
band_size = 15  # 输入数据形状 (64, 30, 7, 7)，这里假设每个时间点的波段数为 15
num_classes = 2
dim = 128
depth = 5
heads = 4
mlp_dim = 8
dropout = 0.1
emb_dropout = 0.1

# 初始化模型
model = ViT(
    patch_size=patch_size,
    num_feats=num_feats,
    band_size=band_size,
    num_classes=num_classes,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
    dropout=dropout,
    emb_dropout=emb_dropout,
)

# 生成示例输入数据
input_data = torch.randn(64, 30, 7, 7)

# 计算总参数量
total_params = sum(p.numel() for p in model.parameters())

# 计算FLOPs
flops, _ = profile(model, inputs=(input_data,))

# 计算平均推理时间
num_trials = 100
total_time = 0
with torch.no_grad():
    for _ in range(num_trials):
        start_time = time.time()
        _ = model(input_data)
        end_time = time.time()
        total_time += (end_time - start_time)
average_inference_time = total_time / num_trials * 1000  # 转换为毫秒

# 格式化结果为更直观的单位
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
formatted_time = f"{average_inference_time:.2f} ms"

# 整理结果
results = {
    "总参数量": f"{formatted_params} 参数",
    "FLOPs": formatted_flops,
    "平均推理时间": formatted_time
}

# 保存结果到JSON文件
with open('model_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("结果已保存到 model_metrics.json 文件中。")
print("指标概览:")
for key, value in results.items():
    print(f"{key}: {value}")