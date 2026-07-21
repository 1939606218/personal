import json
import time
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
from AIWSEN import AIWSEN  # 导入你的模型

def calculate_model_metrics(model, input_shape1, input_shape2, device, num_runs=100):
    """
    计算模型的参数量、FLOPs和平均推理时间
    
    参数:
    model: PyTorch模型
    input_shape1: 第一个输入的形状
    input_shape2: 第二个输入的形状
    device: 计算设备
    num_runs: 计算平均推理时间时的运行次数
    """
    # 移至设备
    model = model.to(device)
    model.eval()
    
    # 创建输入张量
    input1 = torch.randn(input_shape1).to(device)
    input2 = torch.randn(input_shape2).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    param_size = total_params * 4 / (1024 ** 2)  # MB
    
    # 计算FLOPs
    flops, _ = profile(model, inputs=(input1, input2), verbose=False)
    flops, params = clever_format([flops, total_params], "%.3f")
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(input1, input2)
    
    # 测量推理时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input1, input2)
    end_time = time.time()
    
    # 计算平均推理时间
    avg_inference_time = (end_time - start_time) * 1000 / num_runs  # ms
    
    # 构建结果字典
    results = {
        "参数数量": params,
        "参数量大小": f"{param_size:.3f} MB",
        "FLOPs": flops,
        "平均推理时间(ms)": f"{avg_inference_time:.3f} ms"
    }
    
    return results

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    inchannel = 30
    num_head = 5  # 假设num_head为5，根据实际情况修改
    model = AIWSEN(
        device=device,
        inchannel=inchannel,
        num_head=num_head,
        layers=[1, 1, 1, 1],
        img_size=(7, 7),
        embed_dims=[96, 192, 384, 768],
        mlp_ratios=[4, 4, 4, 4],
        downsamples=[True, True, True, True],
        num_classes=2
    )
    
    # 输入形状
    input_shape1 = (64, inchannel, 7, 7)
    input_shape2 = (64, inchannel, 7, 7)
    
    # 计算指标
    results = calculate_model_metrics(model, input_shape1, input_shape2, device)
    
    # 打印结果
    for key, value in results.items():
        print(f"{key}: {value}")
    
    # 保存到JSON文件
    with open('model_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("结果已保存到 model_metrics.json")

if __name__ == "__main__":
    main()    