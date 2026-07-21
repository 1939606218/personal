# -*- coding:utf-8 -*-
# Author:Ding
import os
import time
import argparse
import scipy.io as sio
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import set_seed, make_data, MyDataset, split_train_val, output_metric, \
    train_epoch, valid_epoch, test_epoch
from model import FinalModel
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 查找系统上所有可用的中文字体
available_fonts = [f.name for f in fm.fontManager.ttflist if 'hei' in f.name.lower() or 'song' in f.name.lower()]
print("可用的中文字体:", available_fonts)

# 然后选择其中一个字体使用
plt.rcParams["font.family"] = available_fonts[0] if available_fonts else ["sans-serif"]

def train_and_test(args):
    """执行模型训练和测试过程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: 使用 {'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 加载数据
    print(f"[Info]: 加载数据集 '{args.dataset}'...")
    all_x, all_y, labeled_index, height, width ,_= make_data(args.dataset, patch_size=args.patches)

    # ✅ 直接使用筛选后的x和y
    labeled_x = all_x.squeeze()
    labeled_y = all_y.squeeze()

    # 分割训练集和验证集
    print(f"[Info]: 分割训练集和验证集 (比例={args.ratio})...")
    train_x_set, train_y_set, val_x_set, val_y_set = split_train_val(labeled_x, labeled_y, args)

    # 创建数据加载器
    train_set = MyDataset(train_x_set, train_y_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, shuffle=True)

    val_set = MyDataset(val_x_set, val_y_set)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, shuffle=False)

    print(f"[Info]: 数据加载完成！训练样本: {len(train_x_set)}, 验证样本: {len(val_x_set)}")

    # 初始化模型
    model = FinalModel(
        seq_len=8,
        band_size=args.band_size,
        patch_size=args.patches,
        dim=args.dim,
        depth=args.depth,
        heads=args.head,
        mlp_dim=8,
        dim_head=16,
        dropout=0.1,
        emb_dropout=0.1,
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)

    # 训练记录
    train_losses = []
    train_accs = []
    val_accs = []
    val_kappas = []
    best_val_acc = -1
    best_epoch = 0

    # 训练模型
    print(f"开始训练模型 (epochs={args.epoches})...")
    total_time = 0
    epoch_times = []
    model_path = f"{args.out_dir}/{args.dataset}_model_parameter.pkl"

    for epoch in range(args.epoches):
        # 训练一个epoch
        model.train()
        start_time = time.time()
        train_acc, train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer, device)
        epoch_time = time.time() - start_time
        total_time += epoch_time
        epoch_times.append(epoch_time)

        # 将可能的GPU张量转换为CPU并转为numpy
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.cpu().item()
        if isinstance(train_acc, torch.Tensor):
            train_acc = train_acc.cpu().item()

        # 记录训练指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 计算平均每个epoch的时间和估计剩余时间
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.epoches - (epoch + 1)
        remaining_time = avg_epoch_time * remaining_epochs

        # 格式化时间
        def format_time(seconds):
                return f"{seconds:.1f}秒"

        # 每个epoch都输出训练结果和剩余时间估计
        print(f'Epoch {epoch + 1}/{args.epoches} [进度: {100 * (epoch + 1) / args.epoches:.1f}%] | '
              f'耗时: {format_time(epoch_time)} | '
              f'剩余时间估计: {format_time(remaining_time)} | '
              f'训练损失: {train_loss:.4f} | '
              f'训练OA: {train_acc:.4f} | '
              f'学习率: {optimizer.param_groups[0]["lr"]:.6f}')

        # 验证模型
        if (epoch + 1) % args.test_freq == 0:
            model.eval()
            tar_v, pre_v = valid_epoch(model, val_loader, criterion, device)
            val_acc, _, val_kappa, _ = output_metric(tar_v, pre_v)

            # 将可能的GPU张量转换为CPU并转为numpy
            if isinstance(val_acc, torch.Tensor):
                val_acc = val_acc.cpu().item()
            if isinstance(val_kappa, torch.Tensor):
                val_kappa = val_kappa.cpu().item()

            # 记录验证指标
            val_accs.append(val_acc)
            val_kappas.append(val_kappa)

            # 保存最佳模型
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_kappa = val_kappa
                torch.save(model.state_dict(), model_path)
                print(f'  验证结果: OA={val_acc:.4f}, Kappa={val_kappa:.4f} (最佳模型已保存)')
            else:
                print(f'  验证结果: OA={val_acc:.4f}, Kappa={val_kappa:.4f}')

    print(f"训练完成！总耗时: {format_time(total_time)}")
    print(f"最佳验证结果: Epoch {best_epoch}, OA={best_val_acc:.4f}, Kappa={best_kappa:.4f}")

    # 测试模型
    print("开始测试模型...")
    # 重新加载最佳模型
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 创建测试数据加载器
    all_set = MyDataset(all_x, all_y)
    all_loader = DataLoader(all_set, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, shuffle=False)

    # 执行测试
    tic = time.time()
    pre_u = test_epoch(model, all_loader, device)
    toc = time.time()
    # print(f"推理耗时: {format_time(toc - tic)}")

    # ✅ 修正：使用labeled_y而非all_y[labeled_index]
    # 注意：pre_u的索引需要与labeled_y对应，可能需要调整test_epoch的实现
    OA_test, _, Kappa_test, _ = output_metric(labeled_y, pre_u)
    F1_test = f1_score(labeled_y, pre_u, average='weighted')
    PR_test = precision_score(labeled_y, pre_u, average='weighted')
    RE_test = recall_score(labeled_y, pre_u, average='weighted')

    print("测试结果:")
    print(f"OA: {OA_test:.4f} | Kappa: {Kappa_test:.4f} | F1: {F1_test:.4f} | PR: {PR_test:.4f} | RE: {RE_test:.4f}")

    return (best_val_acc, 0#best_kappa
            ), (OA_test, Kappa_test, F1_test, PR_test, RE_test)


def get_args_parser():
    parser = argparse.ArgumentParser("HSI")
    parser.add_argument('--dataset', choices=['farmland', 'hermiston', 'river', 'Barbara', 'BayArea'],
                        default='Barbara', help='选择要使用的数据集')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--patches', type=int, default=7, help='补丁大小')
    parser.add_argument('--epoches', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.9, help='学习率衰减因子')
    parser.add_argument('--weight_decay', type=float, default=0, help='权重衰减')
    parser.add_argument('--ratio', type=float, default=0.01, help='训练集比例')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载工作线程数')
    parser.add_argument('--depth', type=int, default=4, help='Transformer深度')
    parser.add_argument('--head', type=int, default=4, help='Transformer头数')
    parser.add_argument('--dim', type=int, default=128, help='特征维度')
    parser.add_argument('--out_dir', default='./output', help='输出目录')
    parser.add_argument('--test_freq', type=int, default=100, help='验证频率(轮)')

    args = parser.parse_args()

    # 设置波段数量
    dataset_to_bands = {
        'farmland': 155,
        'hermiston': 242,
        'river': 198,
        'Barbara': 224,
        'BayArea': 224,
    }

    if args.dataset in dataset_to_bands:
        args.band_size = dataset_to_bands[args.dataset]
        print(f"已为数据集 '{args.dataset}' 设置波段数量为 {args.band_size}")
    else:
        print(f"警告: 数据集 '{args.dataset}' 的波段数量未知，使用默认值 {args.band_size}")

    return args


if __name__ == '__main__':
    args = get_args_parser()
    set_seed(seed=args.seed)

    print(f"开始实验: 数据集={args.dataset}, 训练比例={args.ratio}, 训练轮数={args.epoches}")
    (acc_train, kappa_train), (OA_test, Kappa_test, F1_test, PR_test, RE_test) = train_and_test(args)

    # 保存结果
    results = [[OA_test, Kappa_test, F1_test, PR_test, RE_test]]
    np.savetxt(f'{args.dataset}-train_test.txt', np.array(results),
               fmt="%.6f, %.6f, %.6f, %.6f, %.6f")

    print("实验完成!")