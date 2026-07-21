# -*- coding:utf-8 -*-
import os
import time
import numpy as np
import torch.optim.lr_scheduler
from sklearn.metrics import roc_auc_score

from init import reports, weight_init,binary
import torch.optim as optim
from network import Net
from network import ContrastiveLoss
from torch.optim import lr_scheduler

def train_test(
        dataset,
        train_iter,
        test_iter,
        TRAIN_SIZE,
        TEST_SIZE,
        TOTAL_SIZE,
        device,
        epoches,
        windowsize,
        inchannels):
    train_loss_list = []
    net = Net(in_cha=inchannels, patch=windowsize, num_class=2).to(device)
    net.apply(weight_init)  # 网络权重初始化
    optimizer = optim.Adam(net.parameters(), lr=0.0005)  # PU SGD 5e-2
    lr_adjust = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)
    loss = ContrastiveLoss()
    # 假设在训练结束后保存模型
    model_save_path = './models/' + dataset + '.pt'
    print('TORAL_SIZE: ', TOTAL_SIZE)
    print('TRAIN_SIZE: ', TRAIN_SIZE)
    print('TEST_SIZE: ', TEST_SIZE)
    print('---Training on {}---\n'.format(device))
    start = time.time()
    for epoch in range(epoches):
        train_loss_sum = 0.0
        time_epoch = time.time()
        for step, (X1, X2, X3, X4, y) in enumerate(train_iter):
            x1 = X1.to(device)
            x2 = X2.to(device)
            x3 = X3.to(device)
            x4 = X4.to(device)
            y = y.to(device)
            y_hat1, y_hat2, y_hat3, y_hat4, y_hat = net(x1, x2, x3, x4)
            l1 = loss(y_hat1, y_hat2, y.long())
            l2 = loss(y_hat3, y_hat4, y.long())
            l = 0.5 * l1 + 0.5 * l2
            optimizer.zero_grad()  # 梯度清零
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
            # 格式化输出
            # 计算剩余时间（秒）
        elapsed_time = time.time() - start
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_time = avg_epoch_time * (epoches - epoch - 1)
        print(f'epoch {epoch + 1}/{epoches}, train loss: {train_loss_sum / len(train_iter.dataset):.6f}, '
              f'remaining time: {remaining_time :.2f} s')

        train_loss_list.append(train_loss_sum / len(train_iter.dataset))
        if train_loss_list[-1] <= min(train_loss_list):
            # 确保目标目录存在，如果不存在就创建
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            # 保存模型
            torch.save(net.state_dict(), model_save_path)
            print('***Successfully Saved Best models parametres!***\n')  # 保存在训练集上损失值最好的模型效果
        if (epoch+1) % 50==0:
            print('\n***Start  Testing***\n')
            evaluate(test_iter=test_iter, model=net, device=device)
    End = time.time()
    print('***Training End! Total Time %.1f sec***' % (End - start))


def evaluate(test_iter, model, device):
    classification, confusion, oa, aa, kappa, f1, auc = reports(test_iter, model, device=device)
    # 从classification报告里提取精确率(Pr)和召回率(Re)
    lines = classification.strip().split('\n')
    # 略过表头和平均行
    pr_values = []
    re_values = []
    for line in lines[2:-3]:  # 依据实际报告格式可能需要调整
        parts = line.strip().split()
        if len(parts) >= 4:
            pr_values.append(float(parts[1]))
            re_values.append(float(parts[2]))
    # 计算宏平均精确率和召回率
    macro_pr = sum(pr_values) / len(pr_values) if pr_values else 0
    macro_re = sum(re_values) / len(re_values) if re_values else 0
    print(f'OA: {oa:.4f}, Kappa: {kappa:.4f}, F1: {f1:.4f}, Pr: {macro_pr:.4f}, Re: {macro_re:.4f}')
    # return oa, kappa, f1, macro_pr, macro_re



