import torch
from HyGSTAN import hygstan
from thop import profile
import time
import json

def count_model_params():
    # 初始化模型
    image_size = 5
    num_patches = 30
    p = 12  # 修改为12，确保维度匹配
    model = hygstan(num_patches=num_patches, image_size=image_size, p=p, d=64)
    
    # 计算参数总量
    total_params = sum(p.numel() for p in model.parameters())
    
    # 创建输入数据 (batch_size, channels, height*width)
    input1 = torch.randn(64, num_patches, image_size*image_size)
    input2 = torch.randn(64, num_patches, image_size*image_size)
    
    # 计算FLOPs
    macs, params = profile(model, inputs=(input1, input2,))
    flops = macs * 2  # 将MACs转换为FLOPs
    
    # 计算推理时间
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(100):  # 运行100次取平均
            start_time = time.time()
            _ = model(input1, input2)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    avg_inference_time = sum(times[1:]) / len(times[1:])  # 去掉第一次的预热时间
    
    # 转换单位并创建结果字典
    def format_number(num):
        """将数字转换为适当的单位"""
        for unit in ['', 'K', 'M', 'B', 'T']:
            if num < 1000.0:
                return f"{num:.2f}{unit}"
            num /= 1000.0
        return f"{num:.2f}T"

    results = {
        "模型统计信息": {
            "参数数量": f"{format_number(total_params)} (总计: {total_params:,}个参数)",
            "FLOPs": f"{format_number(flops)} (总计: {flops:,}次浮点运算)",
            "平均推理时间": f"{avg_inference_time:.2f} ms"
        }
    }
    
    # 打印结果
    print("模型统计信息:")
    for key, value in results["模型统计信息"].items():
        print(f"{key}: {value}")
    
    # 保存到JSON文件
    with open('model_stats.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("\n结果已保存到 model_stats.json 文件")

if __name__ == "__main__":
    count_model_params()
