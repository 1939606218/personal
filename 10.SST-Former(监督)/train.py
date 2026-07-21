import json
import importlib
import sys
import os
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import tools
from tools import (args, chooose_train_and_test_point, mirror_hsi, train_and_test_data, 
                  train_and_test_label, train_epoch, valid_epoch, output_metric, 
                  predict_full_image, visualize_prediction)
import predata
from predata import prepare_data, create_model
from sstvit import SSTViT

# 确保必要的目录存在
if not os.path.exists('./log'):
    os.mkdir('./log')
if not os.path.exists('./results'):
    os.mkdir('./results')

# 数据集列表
datasets = ['BayArea']
# datasets = ['farmland', 'BayArea', 'Barbara']

for dataset in datasets:
    args.dataset = dataset  # 更新当前数据集名称
    print(f"\n{'='*20} 开始处理数据集: {dataset} {'='*20}")
    
    # 初始化数据集
    tools.init_dataset()
    
    # 准备数据和模型
    from tools import data_t1, data_t2, data_label
    TR, TE, num_classes, input1_normalize, input2_normalize, height, width, band = prepare_data(data_t1, data_t2, data_label)
    
    # 准备数据加载器
    total_pos_train, total_pos_test, number_train, number_test = chooose_train_and_test_point(TR, 2, train_ratio=0.01)
    
    mirror_image_t1 = mirror_hsi(height, width, band, input1_normalize, patch=args.patches)
    mirror_image_t2 = mirror_hsi(height, width, band, input2_normalize, patch=args.patches)
    
    x_train_band_t1, x_test_band_t1 = train_and_test_data(mirror_image_t1, band, total_pos_train,
                                                          total_pos_test, patch=args.patches, band_patch=args.band_patches)
    x_train_band_t2, x_test_band_t2 = train_and_test_data(mirror_image_t2, band, total_pos_train,
                                                          total_pos_test, patch=args.patches, band_patch=args.band_patches)
    y_train, y_test = train_and_test_label(number_train, number_test, num_classes)
    
    # 转换数据格式
    x_train_t1 = torch.from_numpy(x_train_band_t1.transpose(0, 2, 1)).type(torch.FloatTensor)
    x_train_t2 = torch.from_numpy(x_train_band_t2.transpose(0, 2, 1)).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    x_test_t1 = torch.from_numpy(x_test_band_t1.transpose(0, 2, 1)).type(torch.FloatTensor)
    x_test_t2 = torch.from_numpy(x_test_band_t2.transpose(0, 2, 1)).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
    
    # 创建数据加载器
    train_dataset = Data.TensorDataset(x_train_t1, x_train_t2, y_train)
    test_dataset = Data.TensorDataset(x_test_t1, x_test_t2, y_test)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 用于记录最大和最小value的结果
    max_value = float('-inf')
    min_value = float('inf')
    max_metrics = None
    min_metrics = None
    
    # 进行10次实验
    for run in range(10):
        print(f"\n开始第 {run + 1} 次实验")
        
        # 创建和初始化模型
        model, criterion, optimizer, scheduler = create_model(num_classes, args, band)
        
        tic = time.time()  # 开始计时
        
        # 训练循环
        for epoch in range(args.epoches):
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, train_loader, criterion, optimizer)
            scheduler.step()  # 调整学习率
            print("Epoch: {:03d} train_accuracy: {:.4f} train_loss: {:.4f}".format(epoch + 1, train_acc, train_obj))

            # 只在最后一个epoch进行测试
            if epoch == args.epoches - 1:
                model.eval()
                tar_v, pre_v = valid_epoch(model, test_loader, criterion)
                OA, Kappa, TN, FP, FN, TP, PR, RE, F1 = output_metric(tar_v, pre_v)
                value = F1 + OA + Kappa + PR + RE
                
                # 计算运行时间
                toc = time.time()
                run_time = toc - tic
                
                print(f"\n第 {run + 1} 次实验结果：")
                print("OA: {:.4f} | Kappa: {:.4f} | F1: {:.4f}| PR: {:.4f}| RE: {:.4f}".format(OA, Kappa, F1, PR, RE))
                print("value:", value)
                print("Running Time: {:.2f}秒".format(run_time))

                # 记录当前实验的指标
                current_metrics = {
                    'run': run + 1,
                    'epoch': epoch + 1,
                    'OA': float(OA),
                    'Kappa': float(Kappa),
                    'Precision': float(PR),
                    'Recall': float(RE),
                    'F1-Score': float(F1),
                    'TP': int(TP),
                    'TN': int(TN),
                    'FP': int(FP),
                    'FN': int(FN),
                    'value': float(value),
                    'running_time': float(run_time)
                }

                # 更新最大最小值记录
                if value > max_value:
                    max_value = value
                    max_metrics = current_metrics.copy()
                    # 保存最佳模型
                    torch.save(model.state_dict(), f"log/{dataset}_best.pth")
                
                if value < min_value:
                    min_value = value
                    min_metrics = current_metrics.copy()
                
                # 确保结果目录存在
                result_dir = os.path.join('./results', dataset)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                
                # 对整个图像进行预测
                model.eval()
                full_predictions = predict_full_image(model, input1_normalize, input2_normalize,
                                                   height, width, band, args.patches, args.band_patches)
                
                # 生成可视化结果
                vis_result = visualize_prediction(height, width, TR, full_predictions)
                
                # 保存当前实验的预测结果
                prediction_file = os.path.join(result_dir, f'prediction_run{run+1}.png')
                plt.imsave(prediction_file, vis_result)
                print(f"保存预测结果：{prediction_file}")

        print(f"\n数据集 {dataset} 的第 {run + 1} 次实验完成")

    # 一个数据集的所有实验完成后，保存最大和最小值的结果
    result_dir = os.path.join('./results', dataset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 保存最大最小值结果
    final_results = {
        'dataset': dataset,
        'max_metrics': max_metrics,
        'min_metrics': min_metrics
    }
    
    json_path = os.path.join(result_dir, 'final_results.json')
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"\n数据集 {dataset} 的所有实验完成")
    print(f"最大 value: {max_value:.4f}")
    print(f"最小 value: {min_value:.4f}")

print('\n-------------所有数据集处理完成-------------')
