import torch
import json
from model import FinalModel
from thop import profile
import numpy as np

def get_human_readable_count(value):
    """将数值转换为人类可读的格式（如1K, 1M, 1G）"""
    units = ['', 'K', 'M', 'G', 'T']
    unit_index = min(int(np.floor(np.log10(abs(value)) / 3)), len(units) - 1)
    scaled_value = value / (1000 ** unit_index)
    return f"{scaled_value:.3f} {units[unit_index]}"

# 定义模型参数
seq_len, band_size, patch_size, dim, depth, heads, mlp_dim, dim_head = 8, 30, 7, 128, 4, 4, 8, 16
model = FinalModel(seq_len, band_size, patch_size, dim, depth, heads, mlp_dim, dim_head)

# 准备输入数据
input_tensor = torch.randn(64, 2 * band_size, patch_size, patch_size)

# 计算总参数量和FLOPs
flops, params = profile(model, inputs=(input_tensor,))

# 转换为人类可读的格式
params_human = get_human_readable_count(params)
flops_human = get_human_readable_count(flops)

# 计算平均推理时间
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_tensor = input_tensor.to(device)

with torch.no_grad():
    # 预热
    for _ in range(10):
        model(input_tensor)
    
    # 测量时间
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for _ in range(100):
        model(input_tensor)
    end_time.record()
    
    # 等待GPU完成
    torch.cuda.synchronize()
    avg_inference_time = start_time.elapsed_time(end_time) / 100  # 毫秒

# 保存结果到JSON文件
results = {
    "总参数量": f"{params_human} 参数",
    "FLOPs": f"{flops_human} FLOPs",
    "平均推理时间": f"{avg_inference_time:.3f} ms"
}

with open('model_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("结果已保存到 model_metrics.json 文件中。")
print("指标概览:")
for key, value in results.items():
    print(f"{key}: {value}")