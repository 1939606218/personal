import numpy as np
import torch
from torch import optim, nn
from load_data1 import generater, loadData, applyPCA, normalization, visualize_full_predictions
from model_2 import ASGTNet
from loss_func import DynamicCombinedLoss
from train_test import run_training, test, predict_full_dataset
import random
import warnings
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning, message="1Torch was not compiled with flash attention.")

# 设置超参数
num_epochs = 200
batch_size = 64
learning_rate = 0.0005
# 修改：将pca_channel改为列表，包含多个需要测试的值
pca_channels = [8,12,16,20,24,28,32,64,80,96,120,128]  # 可根据需要添加更多值
lambda_ = 0.5
patch_size = 5


# 定义要运行的数据集列表
datasets = ['farmland', 'santaBarbara', 'bayArea']
# datasets = ['santaBarbara']
# 修改：遍历每个PCA通道数
for pca_channel in pca_channels:
    print(f"\n===== 开始处理 PCA 通道数: {pca_channel} =====")

    # 遍历每个数据集
    for dataname in datasets:
        print(f"\n----- 开始处理数据集: {dataname} -----")
        # 导入数据集
        X1, X2, Y = loadData(dataname)
        print(f"数据集 {dataname} 加载成功，形状: X1={X1.shape}, X2={X2.shape}, Y={Y.shape}")
        # 归一化处理
        X1 = normalization(X=X1)
        X2 = normalization(X=X2)

        # PCA降维 - 使用当前pca_channel值
        X1_pca = applyPCA(X1, channel=pca_channel)
        X2_pca = applyPCA(X2, channel=pca_channel)

        # 加载数据
        (TRAIN_SIZE, TEST_SIZE, train_iter, test_iter,
         all_iter, all_position_indices, height, width,
         ce_criterion, alpha) = generater(
            X1_pca, X2_pca, Y, batch_size, 0.01, device, windowSize=patch_size, noise_std=0
        )

        # 修改：创建输出文件夹，包含pca_channel子文件夹
        output_dir = os.path.join('output', f'pca_{pca_channel}', dataname)
        os.makedirs(output_dir, exist_ok=True)

        # 初始化最高和最低value的记录
        max_value = float('-inf')
        min_value = float('inf')
        max_value_results = {}
        min_value_results = {}

        for run_num in range(1, 11):
            print(f"\n--- 第 {run_num} 次运行 ---")
            # 初始化模型 - 使用当前pca_channel值
            model = ASGTNet(num_channels=pca_channel)
            model.to(device)

            # 定义损失函数和优化器
            criterion = DynamicCombinedLoss(num_classes=2, lambda_=lambda_)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
            from torch.optim.lr_scheduler import StepLR

            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

            # 模型保存路径 - 包含pca_channel信息
            model_path = os.path.join(output_dir, f"{dataname}_pca{pca_channel}_run_{run_num}.pth")

            # 在主循环中调用run_training函数的部分
            total_train_seconds, best_epoch, best_value = run_training(
                model, train_iter, test_iter, num_epochs, optimizer, criterion,
                device, scheduler, best_model_path=model_path
            )

            # 测试时加载最佳模型
            best_model = ASGTNet(num_channels=pca_channel)
            best_model.load_state_dict(torch.load(model_path))
            best_model.to(device)

            # 测试模型
            test_loss, test_oa, test_f1, test_precision, test_recall, test_kappa = test(
                best_model, test_iter, criterion, device
            )

            # 对整个数据集进行预测
            full_preds = predict_full_dataset(best_model, all_iter, device)

            # 可视化整个数据集的预测结果 - 包含pca_channel信息
            output_image_path = os.path.join(output_dir, f"{dataname}_pca{pca_channel}_run_{run_num}.png")
            visualize_full_predictions(
                Y,  # 原始标签图
                full_preds,  # 整个数据集的预测结果
                all_position_indices,  # 所有样本的位置索引
                height,
                width,
                dataname,
                output_image_path
            )

            value = test_oa + test_f1 + test_precision + test_recall + test_kappa
            print(f"[PCA通道: {pca_channel}, 数据集: {dataname}, 第 {run_num} 次运行] "
                  f"测试结果: Loss: {test_loss:.4f} | OA: {test_oa:.4f} | F1: {test_f1:.4f} | "
                  f"Pr: {test_precision:.4f} | Re: {test_recall:.4f} | kappa: {test_kappa:.4f}")
            print(f"Value: {value}")

            # 更新最高和最低value的记录
            if value > max_value:
                max_value = value
                max_value_results = {
                    'run_num': run_num,
                    'test_oa': test_oa,
                    'test_f1': test_f1,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_kappa': test_kappa,
                    'train_time': total_train_seconds,
                    'pca_channel': pca_channel
                }
            if value < min_value:
                min_value = value
                min_value_results = {
                    'run_num': run_num,
                    'test_oa': test_oa,
                    'test_f1': test_f1,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_kappa': test_kappa,
                    'train_time': total_train_seconds,
                    'pca_channel': pca_channel
                }

            # 释放GPU内存
            del model, best_model
            torch.cuda.empty_cache()

        # 保存结果到JSON文件 - 包含pca_channel信息
        results = {
            'pca_channel': pca_channel,
            'max_value': max_value,
            'max_value_results': max_value_results,
            'min_value': min_value,
            'min_value_results': min_value_results
        }
        json_path = os.path.join(output_dir, f"{dataname}_pca{pca_channel}_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"===== PCA通道: {pca_channel}, 数据集 {dataname} 处理完成，结果已保存至 {json_path} =====")