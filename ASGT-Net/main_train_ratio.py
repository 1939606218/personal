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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning, message="1Torch was not compiled with flash attention.")

# 设置要运行的数据集列表
datasets = [ 'farmland',  'bayArea', 'santaBarbara']

# 设置要测试的train_ratio列表
train_ratios = [0.001,0.002,0.003,0.005, 0.01, 0.02, 0.03, 0.05,0.1,0.2]

# 设置固定的超参数
num_epochs = 200
batch_size = 64
learning_rate = 0.0005
lambda_ = 0.5
patch_size = 5

# 存储所有结果的字典
results = {}

# 为每个数据集创建结果目录
if not os.path.exists('results'):
    os.makedirs('results')

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

    # 为该数据集的每个train_ratio运行实验
    for train_ratio in train_ratios:
        print(f"\n--- 使用 train_ratio={train_ratio} 训练 ---")

        # 加载数据
        (TRAIN_SIZE, TEST_SIZE, train_iter, test_iter,
         all_iter, all_position_indices, height, width,
         ce_criterion, alpha) = generater(
            X1_pca, X2_pca, Y, batch_size, train_ratio, device, windowSize=patch_size, noise_std=0
        )

        # 初始化模型
        model = ASGTNet(num_channels=pca_channel)
        model.to(device)

        # 定义优化器和学习率调度器
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # 定义损失函数
        criterion = DynamicCombinedLoss(num_classes=2, lambda_=lambda_)

        # 训练模型
        total_train_seconds, best_epoch, best_value = run_training(
            model, train_iter, test_iter, num_epochs, optimizer, criterion,
            device, scheduler, best_model_path=f"model/{dataname}_{train_ratio}_best_model.pth"
        )

        # 加载最佳模型进行测试
        best_model = ASGTNet(num_channels=pca_channel)
        best_model.load_state_dict(torch.load(f"model/{dataname}_{train_ratio}_best_model.pth"))
        best_model.to(device)

        # 测试模型
        test_loss, test_oa, test_f1, test_precision, test_recall, test_kappa = test(best_model, test_iter, criterion,
                                                                                    device)
        print(
            f"[Test]  Loss: {test_loss:.4f} | OA: {test_oa:.4f} | F1: {test_f1:.4f} | Pr: {test_precision:.4f} | Re: {test_recall:.4f} | kappa: {test_kappa:.4f}")

        # 存储结果
        results[dataname][train_ratio] = {
            'test_oa': test_oa,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_kappa': test_kappa,
            'test_loss': test_loss,
            'best_epoch': best_epoch,
            'training_time': total_train_seconds
        }

    # 为该数据集绘制OA值随train_ratio变化的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_ratios, [results[dataname][tr]['test_oa'] for tr in train_ratios], 'o-', label='OA')
    plt.xlabel('Training Ratio')
    plt.ylabel('Overall Accuracy (OA)')
    plt.title(f'{dataname} Dataset - OA vs Training Ratio')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{dataname}_oa_vs_train_ratio.png')
    plt.close()

# 保存所有结果到JSON文件
with open('results/all_ratio_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# 绘制所有数据集的综合比较图
plt.figure(figsize=(12, 8))
markers = ['o', 's', '^', 'D', 'x']
for i, dataname in enumerate(datasets):
    plt.plot(train_ratios, [results[dataname][tr]['test_oa'] for tr in train_ratios],
             marker=markers[i], linestyle='-', label=dataname)

plt.xlabel('Training Ratio')
plt.ylabel('Overall Accuracy (OA)')
plt.title('OA vs Training Ratio for Different Datasets')
plt.grid(True)
plt.legend()
plt.savefig('results/combined_oa_vs_train_ratio.png')
plt.close()

print("\n所有实验完成！结果已保存到'results'目录下。")