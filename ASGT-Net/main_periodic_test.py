import numpy as np
import torch
from torch import optim, nn
from load_data import generater, loadData, applyPCA, normalization
from model_2 import ASGTNet
from loss_func import DynamicCombinedLoss
from train_test import train, test
import random
import warnings
import torchvision.transforms as transforms
import torch.nn.functional as F
import json
import os
from datetime import datetime
import time
from datetime import timedelta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning, message="1Torch was not compiled with flash attention.")

def run_training_with_periodic_test(model, train_loader, test_loader, epochs, optimizer, criterion, device, scheduler,
                                   best_model_path="best_model.pth", test_interval=20):
    """
    修改的训练函数，每隔test_interval个epoch测试一次，并记录测试OA
    """
    total_start_time = time.time()
    epoch_durations = []
    train_losses = []
    test_oa_history = []  # 记录测试OA历史
    
    # 初始化最佳模型记录
    best_value = float('-inf')
    best_epoch = 0
    best_test_oa = 0.0

    print('Training on', device)

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # 训练阶段
        train_loss, train_oa, train_f1, train_precision, train_recall, train_kappa = train(
            model, train_loader, optimizer, criterion, device, epoch, epochs
        )
        
        print(f"[Epoch {epoch + 1}/{epochs}] "
              f"[Train] Loss: {train_loss:.4f} | OA: {train_oa:.4f} | F1: {train_f1:.4f} | "
              f"Pr: {train_precision:.4f} | Re: {train_recall:.4f} | kappa: {train_kappa:.4f}")
        
        train_losses.append(train_loss)
        
        # 更新学习率
        scheduler.step()

        # 每test_interval个epoch或最后一个epoch进行测试
        if (epoch + 1) % test_interval == 0 or (epoch + 1) == epochs:
            print(f"Testing at epoch {epoch + 1}...")
            model.eval()
            test_loss, test_oa, test_f1, test_precision, test_recall, test_kappa = test(
                model, test_loader, criterion, device
            )
            model.train()

            # 记录测试OA
            test_record = {
                'epoch': epoch + 1,
                'test_oa': float(test_oa),
                'test_f1': float(test_f1),
                'test_precision': float(test_precision),
                'test_recall': float(test_recall),
                'test_kappa': float(test_kappa),
                'test_loss': float(test_loss)
            }
            test_oa_history.append(test_record)

            # 计算value
            value = test_oa + test_f1 + test_precision + test_recall + test_kappa
            print(f"[Test] Loss: {test_loss:.4f} | OA: {test_oa:.4f} | F1: {test_f1:.4f} | "
                  f"Pr: {test_precision:.4f} | Re: {test_recall:.4f} | kappa: {test_kappa:.4f} | Value: {value:.4f}")

            # 更新最佳模型
            if value > best_value:
                best_value = value
                best_epoch = epoch + 1
                best_test_oa = test_oa
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ New best model saved at epoch {best_epoch} with value: {best_value:.4f}")

        # 时间信息
        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)
        avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = epochs - (epoch + 1)
        remaining_time = remaining_epochs * avg_epoch_time

        elapsed_time = timedelta(seconds=int(time.time() - total_start_time))
        remaining_time = timedelta(seconds=int(remaining_time))

    total_train_time = time.time() - total_start_time
    print(f"\n✅ Training completed in {timedelta(seconds=int(total_train_time))}")
    print(f"Best model found at epoch {best_epoch} with value: {best_value:.4f}")

    return test_oa_history, best_epoch, best_value, best_test_oa


def main():
    # 设置超参数
    num_epochs = 200  # 修改为200
    batch_size = 64
    learning_rate = 0.0005
    lambda_ = 0.5
    patch_size = 5
    test_interval = 20  # 每20个epoch测试一次

    # 三个数据集
    datasets = ['farmland', 'bayArea', 'santaBarbara']
    
    # 确保results文件夹存在
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 存储所有结果
    all_results = {}
    
    for dataname in datasets:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataname}")
        print(f"{'='*50}")
        
        # 设置PCA通道数
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

        train_ratio = 0.01

        # 加载和预处理数据
        X1, X2, Y = loadData(dataname)
        X1 = normalization(X=X1)
        X2 = normalization(X=X2)

        X1_pca = applyPCA(X1, channel=pca_channel)
        X2_pca = applyPCA(X2, channel=pca_channel)

        # 生成数据加载器
        (TRAIN_SIZE, TEST_SIZE, train_iter, test_iter,
         all_iter, all_position_indices, height, width,
         ce_criterion, alpha) = generater(
            X1_pca, X2_pca, Y, batch_size, train_ratio, device, windowSize=patch_size, noise_std=0
        )

        # 初始化损失函数和模型
        criterion = DynamicCombinedLoss(num_classes=2, lambda_=lambda_)
        model = ASGTNet(num_channels=pca_channel)
        model.to(device)

        # 定义优化器和调度器
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        # 设置模型保存路径
        best_model_path = os.path.join('model', f"{dataname}_best_model.pth")
        
        # 运行训练
        test_oa_history, best_epoch, best_value, best_test_oa = run_training_with_periodic_test(
            model, train_iter, test_iter, num_epochs, optimizer, criterion,
            device, scheduler, best_model_path=best_model_path, test_interval=test_interval
        )

        # 加载最佳模型进行最终测试
        best_model = ASGTNet(num_channels=pca_channel)
        best_model.load_state_dict(torch.load(best_model_path))
        best_model.to(device)

        # 最终测试
        final_test_loss, final_test_oa, final_test_f1, final_test_precision, final_test_recall, final_test_kappa = test(
            best_model, test_iter, criterion, device
        )
        
        print(f"\n[Final Test Results for {dataname}]")
        print(f"Loss: {final_test_loss:.4f} | OA: {final_test_oa:.4f} | F1: {final_test_f1:.4f} | "
              f"Pr: {final_test_precision:.4f} | Re: {final_test_recall:.4f} | kappa: {final_test_kappa:.4f}")
        
        final_test_value = final_test_oa + final_test_f1 + final_test_precision + final_test_recall + final_test_kappa
        print(f"Final Test Value: {final_test_value:.4f}")

        # 保存该数据集的结果
        dataset_results = {
            'dataset': dataname,
            'pca_channels': pca_channel,
            'num_epochs': num_epochs,
            'test_interval': test_interval,
            'best_epoch': best_epoch,
            'best_value': float(best_value),
            'best_test_oa': float(best_test_oa),
            'final_test_results': {
                'loss': float(final_test_loss),
                'oa': float(final_test_oa),
                'f1': float(final_test_f1),
                'precision': float(final_test_precision),
                'recall': float(final_test_recall),
                'kappa': float(final_test_kappa),
                'value': float(final_test_value)
            },
            'test_oa_history': test_oa_history,
            'timestamp': datetime.now().isoformat()
        }
        
        all_results[dataname] = dataset_results
        
        print(f"Results for {dataname} collected successfully")
    
    # 保存所有数据集的综合结果
    comprehensive_results = {
        'experiment_info': {
            'num_epochs': num_epochs,
            'test_interval': test_interval,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'lambda_': lambda_,
            'patch_size': patch_size,
            'train_ratio': 0.01,
            'datasets': datasets,
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    comprehensive_path = os.path.join(results_dir, 'comprehensive_periodic_test_results.json')
    with open(comprehensive_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"All experiments completed!")
    print(f"Comprehensive results saved to {comprehensive_path}")
    print(f"{'='*50}")
    
    # 打印简要总结
    print("\nSummary:")
    for dataname in datasets:
        result = all_results[dataname]
        print(f"{dataname}: Best OA = {result['best_test_oa']:.4f} at epoch {result['best_epoch']}")


if __name__ == "__main__":
    main()
