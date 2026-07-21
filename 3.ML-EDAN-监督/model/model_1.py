import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
import time
from torch.cuda import Event

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, 2, batch_first=True)

    def forward(self, x):
        r_out, (h, c) = self.rnn(x, None)
        return r_out[:, -1, :]

class ML_EDAN(nn.Module):
    def __init__(self, in_channel):
        super(ML_EDAN, self).__init__()
        self.cnn1 = AE(in_channel)
        self.cnn2 = AE(in_channel)
        self.lstm1 = RNN(6400, 512)
        self.lstm2 = RNN(2304, 256)
        self.lstm3 = RNN(512, 128)
        self.lstm4 = RNN(4608, 512)
        self.lstm5 = RNN(12800, 1024)
        self.linear1 = nn.Linear(1024, 256)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 32)
        self.linear = nn.Linear(416, 64)
        self.linear1_1 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, T1, T2):
        T1_out3, T1_out4, T1_out5, T1_out6 = self.cnn1(T1)
        T2_out3, T2_out4, T2_out5, T2_out6 = self.cnn2(T2)
        out_3 = torch.cat([T1_out3.view(T1_out3.size(0), -1).unsqueeze(1), 
                          T2_out3.view(T2_out3.size(0), -1).unsqueeze(1)], dim=1)
        out_3 = self.lstm3(out_3)
        out_4 = torch.cat([T1_out4.view(T1_out4.size(0), -1).unsqueeze(1), 
                          T2_out4.view(T2_out4.size(0), -1).unsqueeze(1)], dim=1)
        out_4 = self.lstm4(out_4)
        out_5 = torch.cat([T1_out5.view(T1_out5.size(0), -1).unsqueeze(1), 
                          T2_out5.view(T2_out5.size(0), -1).unsqueeze(1)], dim=1)
        out_5 = self.lstm5(out_5)
        out_15 = self.linear1(out_5)
        out_24 = self.linear2(out_4)
        out_33 = self.linear3(out_3)
        out = torch.cat([out_33, out_24, out_15], dim=1)
        out = self.linear1_1(self.relu(self.linear(out)))
        return out, T1_out6, T2_out6

class AE(nn.Module):
    def __init__(self, in_channel):
        super(AE, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, 3, 1, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, 2, 1)
        self.conv3 = nn.Conv2d(256, 512, 3, 2)
        self.deconv1 = nn.ConvTranspose2d(512, 256, 3, 2)
        self.deconv2 = nn.ConvTranspose2d(256, 256, 3, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(256, in_channel, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.upsample1 = nn.Upsample(size=(3, 3), mode='nearest')
        self.Conv_attention1 = nn.Conv2d(768, 256, 1, 1)
        self.trans1 = nn.Conv2d(512, 256, 1, 1)
        self.upsample2 = nn.Upsample(size=(5, 5), mode='nearest')
        self.Conv_attention2 = nn.Conv2d(512, 256, 1, 1)
        self.trans2 = nn.Conv2d(512, 256, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.deconv1(out3))
        out3_4 = self.sigmoid(self.Conv_attention1(torch.cat([out2, self.upsample1(out3)], dim=1))) * out2
        out4 = self.relu(self.trans1(torch.cat([out3_4, out4], dim=1)))
        out3_5 = self.sigmoid(self.Conv_attention2(torch.cat([out1, self.upsample2(out4)], dim=1))) * out1
        out5 = self.relu(self.deconv2(out4))
        out5 = self.relu(self.trans2(torch.cat([out3_5, out5], dim=1)))
        out6 = self.deconv3(out5)
        out_24 = torch.cat([out2, out4], dim=1)
        out_15 = torch.cat([out1, out5], dim=1)
        return out3, out_24, out_15, out6

def calculate_model_complexity(model, input_shape=(64, 30, 5, 5), device='cuda'):
    """计算模型的复杂度指标"""
    model = model.to(device)
    x1 = torch.randn(*input_shape).to(device)
    x2 = torch.randn(*input_shape).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    # 计算FLOPs
    flops, params = profile(model, inputs=(x1, x2), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    
    # 计算推理时间
    model.eval()
    
    # 预热GPU
    with torch.no_grad():
        for _ in range(10):
            _ = model(x1, x2)
            torch.cuda.synchronize()
    
    # 测量推理时间
    times = []
    with torch.no_grad():
        for _ in range(100):  # 运行100次取平均
            torch.cuda.synchronize()
            start = time.time()
            _ = model(x1, x2)
            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # 转换为毫秒
    
    avg_time = sum(times) / len(times)
    
    return {
        'total_params': f"{total_params/1e6:.2f}M",
        'flops': flops,
        'params': params,
        'inference_time': f"{avg_time:.2f}ms"
    }

if __name__ == '__main__':
    import json
    import os
    from datetime import datetime
    
    # 初始化模型
    model = ML_EDAN(in_channel=30)
    
    # 如果有GPU则使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 计算模型复杂度
    metrics = calculate_model_complexity(model, device=device)
    
    # 打印结果
    print("\n模型复杂度分析:")
    print(f"总参数量: {metrics['total_params']}")
    print(f"FLOPs: {metrics['flops']}")
    print(f"参数数量: {metrics['params']}")
    print(f"平均推理时间: {metrics['inference_time']}")
    
    # 额外信息：模型结构摘要
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024*1024)
    
    print("\n模型结构:")
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小: {model_size_mb:.2f} MB")
    
    # 创建完整的结果字典
    results = {
        "测试时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "设备信息": str(device),
        "模型复杂度": {
            "总参数量": metrics['total_params'],
            "FLOPs": metrics['flops'],
            "参数数量": metrics['params'],
            "平均推理时间(ms)": metrics['inference_time']
        },
        "模型结构": {
            "总参数量": total_params,
            "可训练参数量": trainable_params,
            "模型大小(MB)": f"{model_size_mb:.2f}"
        }
    }
    
    # 确保结果目录存在
    os.makedirs("result", exist_ok=True)
    
    # 保存结果到JSON文件
    result_file = os.path.join("result", f"model_complexity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\n结果已保存到: {result_file}")
