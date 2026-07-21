# -*- coding:utf-8 -*-
import json
import time

import numpy as np
import torch.nn

from tools.init import reports, weight_init, ContrastiveLoss
import torch.optim as optim
from Global import *
from models.SSA import SSA
import matplotlib.pyplot as plt


def train_and_test_plot(kerner_number,
                        X,
                        Y,
                        dataset,
                        BAND,
                        CLASSES_NUM,
                        train_iter,
                        test_iter,
                        TRAIN_SIZE,
                        TEST_SIZE,
                        device,
                        epoches,
                        ITER,
                        WU,
                        WC,
                        windowsize):
    # 初始化最佳和最差结果的追踪器
    best_value = float('-inf')
    worst_value = float('inf')
    best_metrics = None
    worst_metrics = None
    all_train_history = []

    for index_iter in range(ITER):
        train_loss_list = []
        train_acc_list = []
        net = SSA(kerner_number).to(device)
        net.apply(weight_init)
        optimizer = optim.RMSprop(net.parameters(), lr=0.001)
        loss = ContrastiveLoss(margin=1, WU=WU, WC=WC)
        print('\niter:', index_iter + 1)
        print('TRAIN_SIZE: ', TRAIN_SIZE)
        print('TEST_SIZE: ', TEST_SIZE)
        print('---Training on {}---\n'.format(device))
        start_time = time.time()
        total_epochs = epoches
        # 初始化优化器
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)  # 假设初始LR为0.001

        # 创建单个学习率调度器
        # 使用LambdaLR，根据epoch数调整学习率
        lr_adjust = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0 if epoch < 100 else 0.1  # 前100个epoch保持LR，之后降为0.1倍
        )

        # 记录当前迭代的训练历史
        current_train_history = {
            'iteration': index_iter + 1,
            'epochs': [],
            'total_time': 0
        }

        for epoch in range(epoches):
            train_acc_sum, train_loss_sum = 0.0, 0.0
            time_epoch = time.time()

            # 训练循环
            for step, (X1, X2, y) in enumerate(train_iter):
                x1 = X1.to(device)
                x2 = X2.to(device)
                y = y.to(device)

                y_hat = net(x1, x2)
                l = loss(y_hat, y)

                optimizer.zero_grad()
                l.backward()
                optimizer.step()  # 先更新参数

                train_loss_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(-1) == y.argmax(-1)).float().sum().cpu().item()

            # 每个epoch结束后调整学习率
            lr_adjust.step()  # 后调整学习率

            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_time = avg_epoch_time * (total_epochs - (epoch + 1))
            print('epoch %d,lr=%.6f, train loss %.6f, train acc %.6f, time %.2f sec, remaining time %.2f sec' % (
                epoch + 1, current_lr,
                train_loss_sum / len(train_iter.dataset),
                train_acc_sum / len(train_iter.dataset),
                time.time() - time_epoch,
                remaining_time))

            # 记录当前epoch的训练信息
            current_epoch_info = {
                'epoch': epoch + 1,
                'learning_rate': current_lr,
                'train_loss': train_loss_sum / len(train_iter.dataset),
                'train_accuracy': train_acc_sum / len(train_iter.dataset),
                'epoch_time': time.time() - time_epoch
            }
            current_train_history['epochs'].append(current_epoch_info)

            train_loss_list.append(train_loss_sum / len(train_iter.dataset))
            if train_loss_list[-1] <= min(train_loss_list):
                torch.save(net.state_dict(), f'./models/{dataset}_iter{index_iter + 1}.pt')
                print('***Successfully Saved Best models parametres!***\n')

        End = time.time()
        total_iter_time = End - start_time
        current_train_history['total_time'] = total_iter_time
        all_train_history.append(current_train_history)

        print('***Training End! Total Time %.1f sec***' % total_iter_time)
        net.load_state_dict(torch.load(f'./models/{dataset}_iter{index_iter + 1}.pt'))
        print('\n***Start  Testing***\n')
        metrics = Evaluate(test_iter=test_iter, dataset=dataset, model=net)

        # 计算评估指标的总和
        value = metrics['oa'] + metrics['kappa'] + metrics['precision'] + metrics['recall'] + metrics['f1']
        print(f"综合评估值 (OA+Kappa+Precision+Recall+F1): {value:.4f}")

        # 更新最佳和最差结果
        if value > best_value:
            best_value = value
            best_metrics = metrics.copy()
            best_metrics['iteration'] = index_iter + 1
            best_metrics['value'] = value
            best_metrics['model_path'] = f'./models/{dataset}_iter{index_iter + 1}.pt'
        if value < worst_value:
            worst_value = value
            worst_metrics = metrics.copy()
            worst_metrics['iteration'] = index_iter + 1
            worst_metrics['value'] = value
            worst_metrics['model_path'] = f'./models/{dataset}_iter{index_iter + 1}.pt'

    # 保存训练历史到JSON文件
    save_training_history(best_metrics, worst_metrics, dataset)

    return metrics['oa'], best_metrics, worst_metrics


def save_training_history( best_metrics, worst_metrics, dataset):
    """保存训练历史和最佳/最差结果到JSON文件"""
    results = {
        'best_result': best_metrics,
        'worst_result': worst_metrics
    }

    try:
        with open(f'./results/{dataset}_training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n=== 训练结果已保存到: ./results/{dataset}_training_results.json ===")
    except Exception as e:
        print(f"保存结果时出错: {e}")


def Evaluate(test_iter, dataset, model):
    """评估模型并打印详细结果"""
    metrics = reports(test_iter, dataset, model)

    # 打印基本指标
    print(f"\n=== 模型评估结果 ===")
    print(f"OA: {metrics['oa']:.4f}")
    print(f"Kappa: {metrics['kappa']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")

    # # 打印混淆矩阵
    # print("\n=== 混淆矩阵 ===")
    # print(metrics['confusion_matrix'])
    #
    # # 打印混淆矩阵元素（仅二分类）
    # if metrics['TN'] is not None:
    #     print(f"\n=== 混淆矩阵元素 ===")
    #     print(f"TN: {metrics['TN']}")
    #     print(f"FP: {metrics['FP']}")
    #     print(f"FN: {metrics['FN']}")
    #     print(f"TP: {metrics['TP']}")
    #
    #     # 计算并打印二分类场景下的特定指标
    #     specificity = metrics['TN'] / (metrics['TN'] + metrics['FP']) if (metrics['TN'] + metrics['FP']) > 0 else 0
    #     sensitivity = metrics['TP'] / (metrics['TP'] + metrics['FN']) if (metrics['TP'] + metrics['FN']) > 0 else 0
    #     print(f"Specificity: {specificity:.4f}")
    #     print(f"Sensitivity (Recall): {sensitivity:.4f}")
    #
    # # 打印每类别的指标
    # print(f"\n=== 每类别指标 ===")
    # num_classes = len(metrics['per_class_precision'])
    # for i in range(num_classes):
    #     print(f"Class {i} (样本数: {metrics['class_distribution'][i]}):")
    #     print(f"  Precision: {metrics['per_class_precision'][i]:.4f}")
    #     print(f"  Recall: {metrics['per_class_recall'][i]:.4f}")
    #     print(f"  F1-score: {metrics['per_class_f1'][i]:.4f}")
    #
    # # 打印样本分布
    # print(f"\n=== 样本分布 ===")
    # total_samples = np.sum(metrics['class_distribution'])
    # for i in range(num_classes):
    #     print(
    #         f"Class {i}: {metrics['class_distribution'][i]} ({metrics['class_distribution'][i] / total_samples * 100:.2f}%)")
    #

def generate_png(label, name: str, scale: float = 4.0, dpi: int = 400):
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    numlabel = numlabel.astype(np.int16)
    numlabel = np.where(numlabel > 1, 0, 1)
    plt.imshow(numlabel, cmap='gray')
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
