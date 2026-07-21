import numpy as np
import torch
from torch import optim, nn
import matplotlib.pyplot as plt
import os
import json
from load_data import generater, loadData, applyPCA, normalization
from model_2 import ASGTNet
from loss_func import DynamicCombinedLoss
from train_test import run_training, test
import random
import warnings
import torchvision.transforms as transforms
import torch.nn.functional as F

# 尝试导入FLOPs计算库
from thop import profile
FLOPS_AVAILABLE = True
print("使用 thop 库计算 FLOPs")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning, message="1Torch was not compiled with flash attention.")

def calculate_model_flops(model, input1_shape, input2_shape, device):
    """
    计算模型的FLOPs
    Args:
        model: 模型实例
        input1_shape: 第一个输入的形状 (batch_size, channels, height, width)
        input2_shape: 第二个输入的形状 (batch_size, channels, height, width)
        device: 设备
    Returns:
        flops: FLOPs数量，如果计算失败返回-1
        params: 参数数量
    """
    if not FLOPS_AVAILABLE:
        return -1, -1
    
    model.eval()
    
    try:
        # 创建输入张量
        input1 = torch.randn(input1_shape).to(device)
        input2 = torch.randn(input2_shape).to(device)
        
        # 使用thop库计算FLOPs
        flops, params = profile(model, inputs=(input1, input2), verbose=False)
        
        # 清理thop添加的额外参数，避免保存模型时出现问题
        def clean_model(module):
            if hasattr(module, 'total_ops'):
                delattr(module, 'total_ops')
            if hasattr(module, 'total_params'):
                delattr(module, 'total_params')
            for child in module.children():
                clean_model(child)
        
        clean_model(model)
        
        return flops, params
    
    except Exception as e:
        print(f"FLOPs计算失败: {e}")
        return -1, -1

def format_flops(flops):
    """格式化FLOPs显示"""
    if flops == -1:
        return "N/A"
    elif flops >= 1e12:
        return f"{flops/1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f}M"
    elif flops >= 1e3:
        return f"{flops/1e3:.2f}K"
    else:
        return f"{flops:.0f}"

def format_params(params):
    """格式化参数数量显示"""
    if params == -1:
        return "N/A"
    elif params >= 1e6:
        return f"{params/1e6:.2f}M"
    elif params >= 1e3:
        return f"{params/1e3:.2f}K"
    else:
        return f"{params:.0f}"

# 创建必要的文件夹
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('model'):
    os.makedirs('model')

# 设置要运行的数据集列表
datasets = ['farmland', 'bayArea', 'santaBarbara']

# 设置固定的超参数
num_epochs = 200
batch_size = 64
learning_rate = 0.0005
lambda_ = 0.5
patch_size = 5
train_ratio = 0.01

# 超参数实验设置
hidden_dims = [32, 64, 128, 256, 512,1024]  # 合并后的隐藏层维度列表
global_layers_list = [1, 2, 3, 4, 5]
gnn_layers_list = [1, 2, 3, 4, 5]
encoder_dims = [32,64, 128, 256, 512,1024]  # 添加encoder_dim的实验值

# 存储所有结果的字典
results = {}

for dataname in datasets:
    print(f"\n===== 开始处理数据集: {dataname} =====")
    results[dataname] = {}

    # 根据数据集设置PCA通道数
    if dataname == 'hermiston':
        pca_channel = 8
    elif dataname == 'farmland':
        pca_channel = 8
    elif dataname == 'river':
        pca_channel = 120
    elif dataname == 'bayArea':
        pca_channel = 64
    elif dataname == 'santaBarbara':
        pca_channel = 64

    # 加载数据集
    X1, X2, Y = loadData(dataname)
    X1 = normalization(X=X1)
    X2 = normalization(X=X2)
    X1_pca = applyPCA(X1, channel=pca_channel)
    X2_pca = applyPCA(X2, channel=pca_channel)

    # 生成数据迭代器（只执行一次，复用给所有超参数实验）
    (TRAIN_SIZE, TEST_SIZE, train_iter, test_iter,
     all_iter, all_position_indices, height, width,
     ce_criterion, alpha) = generater(
        X1_pca, X2_pca, Y, batch_size, train_ratio, device, windowSize=patch_size, noise_std=0
    )

    # 实验 hidden_dim（合并 global_hidden_dim 和 gnn_hidden_dim）
    results[dataname]['hidden_dim'] = {}
    for hidden_dim in hidden_dims:
        print(f"\n--- 使用 hidden_dim={hidden_dim} 训练 ---")


        # 初始化模型，同时设置 global_hidden_dim 和 gnn_hidden_dim 为相同值
        model = ASGTNet(
            num_channels=pca_channel,
            global_hidden_dim=hidden_dim,
            gnn_hidden_dim=hidden_dim,
        )
        model.to(device)

        # 计算FLOPs和参数数量
        input_shape = (1, pca_channel, patch_size, patch_size)  # (batch_size, channels, height, width)
        flops, params = calculate_model_flops(model, input_shape, input_shape, device)
        
        print(f"模型参数: {format_params(params)}, FLOPs: {format_flops(flops)}")

        # 定义优化器和学习率调度器
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # 定义损失函数
        criterion = DynamicCombinedLoss(num_classes=2, lambda_=lambda_)

        # 训练模型
        total_train_seconds, best_epoch, best_value = run_training(
            model, train_iter, test_iter, num_epochs, optimizer, criterion,
            device, scheduler, best_model_path=f"model/{dataname}_{hidden_dim}_best_model.pth"
        )

        # 加载最佳模型进行测试
        best_model = ASGTNet(
            num_channels=pca_channel,
            global_hidden_dim=hidden_dim,
            gnn_hidden_dim=hidden_dim,
        )
        best_model.load_state_dict(torch.load(f"model/{dataname}_{hidden_dim}_best_model.pth"))
        best_model.to(device)

        # 测试模型
        test_loss, test_oa, test_f1, test_precision, test_recall, test_kappa = test(
            best_model, test_iter, criterion, device
        )
        print(
            f"[Test]  Loss: {test_loss:.4f} | OA: {test_oa:.4f} | F1: {test_f1:.4f} | Pr: {test_precision:.4f} | Re: {test_recall:.4f} | kappa: {test_kappa:.4f}"
        )

        # 存储结果
        results[dataname]['hidden_dim'][hidden_dim] = {
            'test_oa': test_oa,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_kappa': test_kappa,
            'test_loss': test_loss,
            'best_epoch': best_epoch,
            'training_time': total_train_seconds,
            'flops': int(flops) if flops != -1 else -1,
            'params': int(params) if params != -1 else -1,
            'flops_formatted': format_flops(flops),
            'params_formatted': format_params(params)
        }

    # 为该数据集绘制OA值随hidden_dim变化的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(hidden_dims, [results[dataname]['hidden_dim'][hd]['test_oa'] for hd in hidden_dims], 'o-', label='OA')
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Overall Accuracy (OA)')
    plt.title(f'{dataname} Dataset - OA vs Hidden Dimension')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{dataname}_oa_vs_hidden_dim.png')
    plt.close()

    # 实验 global_layers
    results[dataname]['global_layers'] = {}
    for global_layers in global_layers_list:
        print(f"\n--- 使用 global_layers={global_layers} 训练 ---")


        # 初始化模型
        model = ASGTNet(
            num_channels=pca_channel,
            global_layers=global_layers,
        )
        model.to(device)

        # 计算FLOPs和参数数量
        input_shape = (1, pca_channel, patch_size, patch_size)
        flops, params = calculate_model_flops(model, input_shape, input_shape, device)
        
        print(f"模型参数: {format_params(params)}, FLOPs: {format_flops(flops)}")

        # 定义优化器和学习率调度器
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # 定义损失函数
        criterion = DynamicCombinedLoss(num_classes=2, lambda_=lambda_)

        # 训练模型
        total_train_seconds, best_epoch, best_value = run_training(
            model, train_iter, test_iter, num_epochs, optimizer, criterion,
            device, scheduler, best_model_path=f"model/{dataname}_{global_layers}_best_model.pth"
        )

        # 加载最佳模型进行测试
        best_model = ASGTNet(
            num_channels=pca_channel,
            global_layers=global_layers,
        )
        best_model.load_state_dict(torch.load(f"model/{dataname}_{global_layers}_best_model.pth"))
        best_model.to(device)

        # 测试模型
        test_loss, test_oa, test_f1, test_precision, test_recall, test_kappa = test(
            best_model, test_iter, criterion, device
        )
        print(
            f"[Test]  Loss: {test_loss:.4f} | OA: {test_oa:.4f} | F1: {test_f1:.4f} | Pr: {test_precision:.4f} | Re: {test_recall:.4f} | kappa: {test_kappa:.4f}"
        )

        # 存储结果
        results[dataname]['global_layers'][global_layers] = {
            'test_oa': test_oa,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_kappa': test_kappa,
            'test_loss': test_loss,
            'best_epoch': best_epoch,
            'training_time': total_train_seconds,
            'flops': int(flops) if flops != -1 else -1,
            'params': int(params) if params != -1 else -1,
            'flops_formatted': format_flops(flops),
            'params_formatted': format_params(params)
        }

    # 为该数据集绘制OA值随global_layers变化的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(global_layers_list, [results[dataname]['global_layers'][gl]['test_oa'] for gl in global_layers_list], 'o-', label='OA')
    plt.xlabel('Global Layers')
    plt.ylabel('Overall Accuracy (OA)')
    plt.title(f'{dataname} Dataset - OA vs Global Layers')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{dataname}_oa_vs_global_layers.png')
    plt.close()

    # 实验 gnn_layers
    results[dataname]['gnn_layers'] = {}
    for gnn_layers in gnn_layers_list:
        print(f"\n--- 使用 gnn_layers={gnn_layers} 训练 ---")

        # 初始化模型
        model = ASGTNet(
            num_channels=pca_channel,
            gnn_layers=gnn_layers,
        )
        model.to(device)

        # 计算FLOPs和参数数量
        input_shape = (1, pca_channel, patch_size, patch_size)
        flops, params = calculate_model_flops(model, input_shape, input_shape, device)
        
        print(f"模型参数: {format_params(params)}, FLOPs: {format_flops(flops)}")

        # 定义优化器和学习率调度器
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # 定义损失函数
        criterion = DynamicCombinedLoss(num_classes=2, lambda_=lambda_)

        # 训练模型
        total_train_seconds, best_epoch, best_value = run_training(
            model, train_iter, test_iter, num_epochs, optimizer, criterion,
            device, scheduler, best_model_path=f"model/{dataname}_{gnn_layers}_best_model.pth"
        )

        # 加载最佳模型进行测试
        best_model = ASGTNet(
            num_channels=pca_channel,
            gnn_layers=gnn_layers,
        )
        best_model.load_state_dict(torch.load(f"model/{dataname}_{gnn_layers}_best_model.pth"))
        best_model.to(device)

        # 测试模型
        test_loss, test_oa, test_f1, test_precision, test_recall, test_kappa = test(
            best_model, test_iter, criterion, device
        )
        print(
            f"[Test]  Loss: {test_loss:.4f} | OA: {test_oa:.4f} | F1: {test_f1:.4f} | Pr: {test_precision:.4f} | Re: {test_recall:.4f} | kappa: {test_kappa:.4f}"
        )

        # 存储结果
        results[dataname]['gnn_layers'][gnn_layers] = {
            'test_oa': test_oa,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_kappa': test_kappa,
            'test_loss': test_loss,
            'best_epoch': best_epoch,
            'training_time': total_train_seconds,
            'flops': int(flops) if flops != -1 else -1,
            'params': int(params) if params != -1 else -1,
            'flops_formatted': format_flops(flops),
            'params_formatted': format_params(params)
        }

    # 为该数据集绘制OA值随gnn_layers变化的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(gnn_layers_list, [results[dataname]['gnn_layers'][gl]['test_oa'] for gl in gnn_layers_list], 'o-', label='OA')
    plt.xlabel('GNN Layers')
    plt.ylabel('Overall Accuracy (OA)')
    plt.title(f'{dataname} Dataset - OA vs GNN Layers')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{dataname}_oa_vs_gnn_layers.png')
    plt.close()

    # 实验 encoder_dim
    results[dataname]['encoder_dim'] = {}
    for encoder_dim in encoder_dims:
        print(f"\n--- 使用 encoder_dim={encoder_dim} 训练 ---")


        # 初始化模型
        model = ASGTNet(
            num_channels=pca_channel,
            encoder_dim=encoder_dim
        )
        model.to(device)

        # 计算FLOPs和参数数量
        input_shape = (1, pca_channel, patch_size, patch_size)
        flops, params = calculate_model_flops(model, input_shape, input_shape, device)
        
        print(f"模型参数: {format_params(params)}, FLOPs: {format_flops(flops)}")

        # 定义优化器和学习率调度器
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # 定义损失函数
        criterion = DynamicCombinedLoss(num_classes=2, lambda_=lambda_)

        # 训练模型
        total_train_seconds, best_epoch, best_value = run_training(
            model, train_iter, test_iter, num_epochs, optimizer, criterion,
            device, scheduler, best_model_path=f"model/{dataname}_{encoder_dim}_best_model.pth"
        )

        # 加载最佳模型进行测试
        best_model = ASGTNet(
            num_channels=pca_channel,
            encoder_dim=encoder_dim
        )
        best_model.load_state_dict(torch.load(f"model/{dataname}_{encoder_dim}_best_model.pth"))
        best_model.to(device)

        # 测试模型
        test_loss, test_oa, test_f1, test_precision, test_recall, test_kappa = test(
            best_model, test_iter, criterion, device
        )
        print(
            f"[Test]  Loss: {test_loss:.4f} | OA: {test_oa:.4f} | F1: {test_f1:.4f} | Pr: {test_precision:.4f} | Re: {test_recall:.4f} | kappa: {test_kappa:.4f}"
        )

        # 存储结果
        results[dataname]['encoder_dim'][encoder_dim] = {
            'test_oa': test_oa,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_kappa': test_kappa,
            'test_loss': test_loss,
            'best_epoch': best_epoch,
            'training_time': total_train_seconds,
            'flops': int(flops) if flops != -1 else -1,
            'params': int(params) if params != -1 else -1,
            'flops_formatted': format_flops(flops),
            'params_formatted': format_params(params)
        }

    # 为该数据集绘制OA值随encoder_dim变化的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(encoder_dims, [results[dataname]['encoder_dim'][ed]['test_oa'] for ed in encoder_dims], 'o-', label='OA')
    plt.xlabel('Encoder Dimension')
    plt.ylabel('Overall Accuracy (OA)')
    plt.title(f'{dataname} Dataset - OA vs Encoder Dimension')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{dataname}_oa_vs_encoder_dim.png')
    plt.close()

# 保存所有结果到JSON文件
with open('results/all_hyperparameter_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# 创建一个FLOPs和参数数量的总结
flops_summary = {}
for dataset in datasets:
    flops_summary[dataset] = {}
    
    for param_type in ['hidden_dim', 'global_layers', 'gnn_layers', 'encoder_dim']:
        flops_summary[dataset][param_type] = {}
        
        for param_val, data in results[dataset][param_type].items():
            flops_summary[dataset][param_type][str(param_val)] = {
                'flops': data['flops'],
                'params': data['params'],
                'flops_formatted': data['flops_formatted'],
                'params_formatted': data['params_formatted'],
                'test_oa': data['test_oa']
            }

# 保存FLOPs总结
with open('results/model_complexity_summary.json', 'w') as f:
    json.dump(flops_summary, f, indent=4)

print("\n所有实验完成！结果已保存到'results'目录下。")
print("- all_hyperparameter_results.json: 完整的实验结果")
print("- model_complexity_summary.json: 模型复杂度总结（FLOPs和参数数量）")

# 打印每个数据集的复杂度范围
if FLOPS_AVAILABLE:
    print("\n模型复杂度总结:")
    for dataset in datasets:
        print(f"\n{dataset} 数据集:")
        
        all_flops = []
        all_params = []
        
        for param_type in ['hidden_dim', 'global_layers', 'gnn_layers', 'encoder_dim']:
            for data in results[dataset][param_type].values():
                if data['flops'] != -1:
                    all_flops.append(data['flops'])
                    all_params.append(data['params'])
        
        if all_flops:
            min_flops, max_flops = min(all_flops), max(all_flops)
            min_params, max_params = min(all_params), max(all_params)
            
            print(f"  FLOPs范围: {format_flops(min_flops)} - {format_flops(max_flops)}")
            print(f"  参数范围: {format_params(min_params)} - {format_params(max_params)}")
else:
    print("\n注意: 由于未安装FLOPs计算库，复杂度分析不可用")
    print("可以通过以下命令安装: pip install thop")