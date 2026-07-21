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
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime

from utils import set_seed, make_data, MyDataset, split_train_val, output_metric, \
    train_epoch, valid_epoch, test_epoch, calculate_metrics
from vit_pytorch import ViT

# 查找系统上所有可用的中文字体
available_fonts = [f.name for f in fm.fontManager.ttflist if 'hei' in f.name.lower() or 'song' in f.name.lower()]
print("可用的中文字体:", available_fonts)

# 然后选择其中一个字体使用
plt.rcParams["font.family"] = available_fonts[0] if available_fonts else ["sans-serif"]


def visualize_predictions_with_tpfp(labeled_y, pre_u, labeled_index, full_y, height, width, dataset, output_path):
    """
    基于已知的TN、FP、FN、TP值绘制预测图像

    参数:
        labeled_y: 标记样本的真实标签
        pre_u: 标记样本的预测标签
        labeled_index: 标记样本的索引
        full_y: 整个图像的标签（展平）
        height, width: 图像尺寸
        dataset: 数据集名称
        output_path: 可视化图像保存路径

    """
    # 创建整个图像的预测标签数组（初始值为2表示未预测）
    full_pred = np.full(full_y.shape, 2, dtype=np.uint8)

    # 将标记样本的预测结果填入对应位置
    full_pred[labeled_index] = pre_u

    # 重塑为二维数组
    pred_img = full_pred.reshape(height, width)
    true_img = full_y.reshape(height, width)

    # 创建可视化图像 (RGB)
    vis_img = np.zeros((height, width, 3), dtype=np.uint8)

    if dataset in ['BayArea', 'Barbara']:
        unlabeled = (true_img == 2)  # 未标记区域
        vis_img[unlabeled] = [100, 100, 100]  # 灰色 - 未标记区域

    # 创建标记区域的掩码
    labeled_mask = np.zeros_like(true_img, dtype=bool)
    labeled_mask.ravel()[labeled_index] = True

    # 计算TP, FP, FN, TN在整个图像中的位置
    # 注意：只在标记区域内计算
    TP_mask = np.zeros_like(true_img, dtype=bool)
    FP_mask = np.zeros_like(true_img, dtype=bool)
    FN_mask = np.zeros_like(true_img, dtype=bool)
    TN_mask = np.zeros_like(true_img, dtype=bool)

    # 二分类场景
    TP_mask[labeled_mask] = (pre_u == 1) & (labeled_y == 1)
    FP_mask[labeled_mask] = (pre_u == 1) & (labeled_y == 0)
    FN_mask[labeled_mask] = (pre_u == 0) & (labeled_y == 1)
    TN_mask[labeled_mask] = (pre_u == 0) & (labeled_y == 0)

    # 应用颜色
    vis_img[TP_mask] = [255, 255, 255]  # 白色 - TP
    vis_img[FP_mask] = [255, 0, 0]  # 红色 - FP
    vis_img[FN_mask] = [0, 255, 0]  # 绿色 - FN
    vis_img[TN_mask] = [0, 0, 0]  # 黑色 - TN

    # 保存可视化图像
    plt.imsave(output_path, vis_img)
    print(f"预测结果可视化已保存至: {output_path}")

    return vis_img


# 修改 train_and_test 函数，增加 train_loader 参数
def train_and_test(args, experiment_num, dataset_data, train_loader):
    """执行模型训练和测试过程，并返回评估指标"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: 使用 {'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    experiment_out_dir = os.path.join(args.out_dir, f"{args.dataset}_exp{experiment_num}")
    os.makedirs(experiment_out_dir, exist_ok=True)

    all_x, all_y, labeled_index, height, width, full_y = dataset_data  # 增加 full_y
    print(f"[Info]: 使用预加载的数据集 '{args.dataset}'... 实验次数: {experiment_num}")

    labeled_x = all_x.squeeze()
    labeled_y = all_y.squeeze()

    # 无需再创建 train_loader，直接使用传入的
    print(f"[Info]: 复用 DataLoader！训练样本: {len(train_loader.dataset)} 实验次数: {experiment_num}")

    # 初始化模型，将FinalModel替换为ViT
    model = ViT(
        patch_size=args.patches,
        num_feats=1 * 4,
        band_size=args.band_size,
        num_classes=2,
        dim=args.dim,
        depth=args.depth,
        heads=args.head,
        mlp_dim=8,
        dropout=0.1,
        emb_dropout=0.1,
    )
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)

    # 训练记录
    train_losses = []
    train_accs = []
    best_train_acc = -1
    best_epoch = 0

    # 训练模型
    print(f"开始训练模型 (epochs={args.epoches})... 实验次数: {experiment_num}")
    total_time = 0
    epoch_times = []
    model_path = f"{experiment_out_dir}/{args.dataset}_exp{experiment_num}_model_parameter.pkl"

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
        print(f'Epoch {epoch + 1}/{args.epoches} | '
              f'剩余时间估计: {format_time(remaining_time)} | '
              f'训练损失: {train_loss:.4f} | '
              f'训练OA: {train_acc:.4f} | '
              f'实验次数: {experiment_num}')
    
    # 训练完成后保存最终模型
    torch.save(model.state_dict(), model_path)
    print(f"训练完成！已保存最终模型至: {model_path}")
    print(f"训练总耗时: {format_time(total_time)} 实验次数: {experiment_num}")
    # 记录最后一个epoch的训练准确率作为最佳
    best_train_acc = train_accs[-1] if train_accs else 0
    best_epoch = args.epoches

    # 测试模型
    print("开始测试模型... 实验次数: {experiment_num}")
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

    # 计算评估指标
    oa, f1, precision, recall, kappa, TN, FP, FN, TP = calculate_metrics(labeled_y, pre_u)

    # 计算value值
    value = oa + f1 + precision + recall + kappa

    # 生成可视化图像
    vis_img = visualize_predictions_with_tpfp(
        labeled_y, pre_u, labeled_index, full_y,
        height, width, args.dataset,
        os.path.join(experiment_out_dir, f"{args.dataset}_exp{experiment_num}_vis.png"),
    )

    # ===================== 结束新增 =====================
    # 构建指标字符串（保留四位小数）
    print(f"OA: {oa:.4f}, Kappa: {kappa:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, TN:{TN}, FP:{FP}, FN:{FN}, TP:{TP} 实验次数: {experiment_num}")
    print(f"Value值: {value:.4f} 实验次数: {experiment_num}")

    # 保存本次实验结果
    result = {
        "experiment_num": experiment_num,
        "dataset": args.dataset,
        "oa": float(oa),
        "kappa": float(kappa),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tn": int(TN),
        "fp": int(FP),
        "fn": int(FN),
        "tp": int(TP),
        "value": float(value),
        "training_time": total_time,
        "best_epoch": best_epoch,
        "best_train_oa": float(best_train_acc),
    }

    # 保存结果到JSON文件
    result_file = os.path.join(experiment_out_dir, f"results_exp{experiment_num}.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"实验结果已保存至: {result_file}")

    return result


def get_args_parser():
    parser = argparse.ArgumentParser("HSI")
    parser.add_argument('--datasets', choices=['farmland', 'hermiston', 'river', 'Barbara', 'BayArea'],
                        nargs='+', default=['farmland','Barbara', 'BayArea'], help='选择要使用的数据集，可指定多个')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--patches', type=int, default=7, help='补丁大小')
    parser.add_argument('--epoches', type=int, default=200, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.9, help='学习率衰减因子')
    parser.add_argument('--weight_decay', type=float, default=0, help='权重衰减')
    parser.add_argument('--ratio', type=float, default=0.01, help='训练集比例')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载工作线程数')
    parser.add_argument('--depth', type=int, default=4, help='Transformer深度')
    parser.add_argument('--head', type=int, default=4, help='Transformer头数')
    parser.add_argument('--dim', type=int, default=128, help='特征维度')
    parser.add_argument('--out_dir', default='./output', help='输出目录')
    parser.add_argument('--num_experiments', type=int, default=10, help='每个数据集运行的实验次数')

    args = parser.parse_args()

    # 设置波段数量
    dataset_to_bands = {
        'farmland': 155,
        'hermiston': 242,
        'river': 198,
        'Barbara': 224,
        'BayArea': 224,
    }

    for dataset in args.datasets:
        if dataset in dataset_to_bands:
            args.band_size = dataset_to_bands[dataset]
            print(f"已为数据集 '{dataset}' 设置波段数量为 {args.band_size}")
        else:
            print(f"警告: 数据集 '{dataset}' 的波段数量未知，使用默认值 {args.band_size}")

    return args


def main():
    args = get_args_parser()
    os.makedirs(args.out_dir, exist_ok=True)
    all_datasets_summary = {}

    # 不再预加载所有数据集，改为逐个处理
    for dataset in args.datasets:
        print(f"\n===== 开始处理数据集: {dataset} =====")
        args.dataset = dataset  # 更新当前处理的数据集

        # 设置波段数量
        dataset_to_bands = {
            'farmland': 155,
            'hermiston': 242,
            'river': 198,
            'Barbara': 224,
            'BayArea': 224,
        }
        if dataset in dataset_to_bands:
            args.band_size = dataset_to_bands[dataset]

        # 加载当前数据集
        print(f"[Info]: 加载数据集 '{dataset}'...")
        all_x, all_y, labeled_index, height, width, full_y = make_data(dataset, patch_size=args.patches)
        dataset_data = (all_x, all_y, labeled_index, height, width, full_y)  # 增加 full_y

        # 分割训练集和验证集
        labeled_x = all_x.squeeze()
        labeled_y = all_y.squeeze()
        print(f"[Info]: 分割训练集和验证集 (比例={args.ratio})...")
        train_x_set, train_y_set, val_x_set, val_y_set = split_train_val(labeled_x, labeled_y, args)

        # 创建训练数据加载器
        train_set = MyDataset(train_x_set, train_y_set)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                  pin_memory=True, shuffle=True)

        print(f"[Info]: 数据加载完成！训练样本: {len(train_x_set)}")

        # 为每个数据集创建汇总目录
        dataset_summary_dir = os.path.join(args.out_dir, f"{dataset}_summary")
        os.makedirs(dataset_summary_dir, exist_ok=True)

        # 存储当前数据集的所有实验结果
        dataset_experiment_results = []

        # 每个数据集运行指定次数的实验
        for exp_num in range(1, args.num_experiments + 1):
            print(f"\n----- 开始实验 {exp_num}/{args.num_experiments} -----")
            # 设置不同的随机种子以获得不同的实验结果
            set_seed(seed=args.seed + exp_num)
            args.seed = args.seed + exp_num  # 更新种子

            # 执行训练和测试，传入当前数据集的 data 和 loader
            try:
                result = train_and_test(args, exp_num, dataset_data, train_loader)
                dataset_experiment_results.append(result)
            except Exception as e:
                print(f"实验 {exp_num} 失败: {str(e)}")
                continue

        # 如果有实验结果，分析并保存汇总信息
        if dataset_experiment_results:
            # 提取所有value值
            values = [result["value"] for result in dataset_experiment_results]

            # 找到value最大和最小的实验
            max_value_idx = np.argmax(values)
            min_value_idx = np.argmin(values)
            max_result = dataset_experiment_results[max_value_idx]
            min_result = dataset_experiment_results[min_value_idx]

            # 计算平均value和其他指标
            avg_value = np.mean(values)
            avg_oa = np.mean([result["oa"] for result in dataset_experiment_results])
            avg_kappa = np.mean([result["kappa"] for result in dataset_experiment_results])
            avg_f1 = np.mean([result["f1"] for result in dataset_experiment_results])
            avg_precision = np.mean([result["precision"] for result in dataset_experiment_results])
            avg_recall = np.mean([result["recall"] for result in dataset_experiment_results])

            # 构建汇总结果
            dataset_summary = {
                "dataset": dataset,
                "num_experiments": args.num_experiments,
                "average_value": float(avg_value),
                "average_oa": float(avg_oa),
                "average_kappa": float(avg_kappa),
                "average_f1": float(avg_f1),
                "average_precision": float(avg_precision),
                "average_recall": float(avg_recall),
                "max_value_experiment": max_result,
                "min_value_experiment": min_result,
                "all_experiments": dataset_experiment_results
            }

            # 保存数据集汇总结果
            summary_file = os.path.join(dataset_summary_dir, f"{dataset}_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(dataset_summary, f, indent=4)

            print(f"\n===== 数据集 {dataset} 实验汇总 =====")
            print(f"平均Value值: {avg_value:.4f}")
            print(f"最大Value值: {max_result['value']:.4f} (实验 {max_result['experiment_num']})")
            print(f"最小Value值: {min_result['value']:.4f} (实验 {min_result['experiment_num']})")

            # 保存到全局汇总
            all_datasets_summary[dataset] = dataset_summary

        # 处理完一个数据集后，释放相关资源
        del all_x, all_y, labeled_x, labeled_y, train_x_set, train_y_set, val_x_set, val_y_set
        del train_set, train_loader
        import gc
        gc.collect()  # 显式调用垃圾回收

        print(f"===== 数据集 {dataset} 处理完成 =====")

    # 保存所有数据集的汇总结果
    global_summary_file = os.path.join(args.out_dir, "global_summary.json")
    with open(global_summary_file, 'w') as f:
        json.dump(all_datasets_summary, f, indent=4)

    print("\n===== 所有数据集实验完成 =====")
    print(f"汇总结果已保存至: {global_summary_file}")


if __name__ == '__main__':
    print(f"开始实验: 日期={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    print("所有实验完成!")