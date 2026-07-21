"""
SST-Former 单次训练+预测脚本
数据集: farmland, BayArea, Barbara (各运行1次)
保存: 彩色变化检测图 DPI=600 ({dataset}_new.png) + 模型权重
"""
import sys
import os
import numpy as np
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt

# 确保 argparse 不会报错
sys.argv = ['script.py']

import tools
from tools import (chooose_train_and_test_point, mirror_hsi, train_and_test_data,
                   train_and_test_label, train_epoch, valid_epoch, output_metric,
                   predict_full_image, visualize_prediction, init_dataset)
import predata
from predata import create_model

os.makedirs('./log', exist_ok=True)
os.makedirs('./results', exist_ok=True)

datasets = ['farmland', 'BayArea', 'Barbara']

for dataset in datasets:
    tools.args.dataset = dataset
    print(f"\n{'=' * 50}")
    print(f"Processing: {dataset}")
    print(f"{'=' * 50}")

    # 1. 加载数据 + 归一化
    init_dataset()

    # 2. 准备数据（注意：prepare_data 会再做一次 MinMaxScaler 归一化，和原代码保持一致）
    TR, TE, num_classes, input1_norm, input2_norm, height, width, band = predata.prepare_data(
        tools.data_t1, tools.data_t2, tools.data_label)

    # 3. 划分训练/测试集
    total_pos_train, total_pos_test, number_train, number_test = chooose_train_and_test_point(
        TR, 2, train_ratio=0.01)

    mirror_t1 = mirror_hsi(height, width, band, input1_norm, patch=tools.args.patches)
    mirror_t2 = mirror_hsi(height, width, band, input2_norm, patch=tools.args.patches)

    x_train_t1, x_test_t1 = train_and_test_data(
        mirror_t1, band, total_pos_train, total_pos_test,
        patch=tools.args.patches, band_patch=tools.args.band_patches)
    x_train_t2, x_test_t2 = train_and_test_data(
        mirror_t2, band, total_pos_train, total_pos_test,
        patch=tools.args.patches, band_patch=tools.args.band_patches)
    y_train, y_test = train_and_test_label(number_train, number_test, num_classes)

    x_train_t1 = torch.from_numpy(x_train_t1.transpose(0, 2, 1)).type(torch.FloatTensor)
    x_train_t2 = torch.from_numpy(x_train_t2.transpose(0, 2, 1)).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    x_test_t1 = torch.from_numpy(x_test_t1.transpose(0, 2, 1)).type(torch.FloatTensor)
    x_test_t2 = torch.from_numpy(x_test_t2.transpose(0, 2, 1)).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)

    train_dataset = Data.TensorDataset(x_train_t1, x_train_t2, y_train)
    test_dataset = Data.TensorDataset(x_test_t1, x_test_t2, y_test)
    train_loader = Data.DataLoader(train_dataset, batch_size=tools.args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=tools.args.batch_size, shuffle=True)

    # 4. 创建模型并训练
    model, criterion, optimizer, scheduler = create_model(num_classes, tools.args, band)

    print(f"\n--- Training ({tools.args.epoches} epochs) ---")
    for epoch in range(tools.args.epoches):
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        print(f"Epoch {epoch + 1:03d} | train_acc: {train_acc:.4f} | train_loss: {train_obj:.4f}")

        if epoch == tools.args.epoches - 1:
            model.eval()
            tar_v, pre_v = valid_epoch(model, test_loader, criterion)
            OA, Kappa, TN, FP, FN, TP, PR, RE, F1 = output_metric(tar_v, pre_v)
            value = F1 + OA + Kappa + PR + RE
            print(f"\nResults: OA={OA:.4f} Kappa={Kappa:.4f} F1={F1:.4f} PR={PR:.4f} RE={RE:.4f}")
            print(f"Value: {value:.4f}")

    # 5. 全图预测
    print("\n--- Predicting full image ---")
    model.eval()
    full_pred = predict_full_image(model, input1_norm, input2_norm,
                                    height, width, band,
                                    tools.args.patches, tools.args.band_patches)

    # 6. 生成彩色变化检测图，DPI=600
    vis_result = visualize_prediction(height, width, TR, full_pred)
    result_dir = os.path.join('./results', dataset)
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, f'{dataset}_new.png')
    plt.imsave(save_path, vis_result, dpi=600)
    print(f"Saved: {save_path}")

    # 7. 保存模型权重
    model_path = os.path.join('./log', f'{dataset}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

print("\nAll done!")
