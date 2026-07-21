"""
ASGT-Net 纯预测+可视化脚本
自动查找所有 PCA 通道中 value 最高和次高的实验，生成彩色变化检测图 DPI=600
输出: 当前目录 {dataset}_best.png 和 {dataset}_second.png
"""
import os
import json
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from load_data1 import loadData, normalization, applyPCA, generater
from model_2 import ASGTNet
from train_test import predict_full_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
PATCH_SIZE = 5


def find_top_two(dataset_name, output_dir='output'):
    """遍历所有 PCA 通道的 results.json，找到全局 value 最高和次高的实验"""
    all_experiments = []
    pattern = os.path.join(output_dir, 'pca_*', dataset_name, f'{dataset_name}_pca*_results.json')
    for json_path in glob.glob(pattern):
        with open(json_path) as f:
            data = json.load(f)
        # 添加 max 和 min 两个实验
        all_experiments.append(data['max_value_results'])
        all_experiments.append(data['min_value_results'])

    if not all_experiments:
        raise FileNotFoundError(f"No results found for {dataset_name}")

    # 按 value 降序排列，取前两名
    all_experiments.sort(key=lambda x: x['test_oa'] + x['test_f1'] + x['test_precision'] + x['test_recall'] + x['test_kappa'], reverse=True)

    return all_experiments[0], all_experiments[1]


def generate_change_map(original_labels, predictions, indices, height, width, out_path):
    """生成彩色变化检测图并保存 DPI=600"""
    full_pred = np.full((height * width), -1, dtype=np.int16)
    full_pred[indices] = predictions
    pred_img = full_pred.reshape(height, width)
    true_img = original_labels

    vis_img = np.zeros((height, width, 3), dtype=np.uint8)

    tn_mask = (true_img == 0) & (pred_img == 0)
    tp_mask = (true_img == 1) & (pred_img == 1)
    fp_mask = (true_img == 0) & (pred_img == 1)
    fn_mask = (true_img == 1) & (pred_img == 0)
    unlabeled_mask = (true_img == 2)

    vis_img[tn_mask] = [0, 0, 0]
    vis_img[tp_mask] = [255, 255, 255]
    vis_img[fp_mask] = [255, 0, 0]
    vis_img[fn_mask] = [0, 255, 0]
    vis_img[unlabeled_mask] = [100, 100, 100]

    plt.imsave(out_path, vis_img, dpi=600)
    print(f"  Saved: {out_path}")


def process_experiment(dataset_name, exp_info, label, out_dir):
    """处理单个实验：加载数据 → 加载模型 → 预测 → 保存图片"""
    pca_ch = exp_info['pca_channel']
    run_num = exp_info['run_num']
    model_path = os.path.join('output', f'pca_{pca_ch}', dataset_name,
                              f'{dataset_name}_pca{pca_ch}_run_{run_num}.pth')

    print(f"  [{label}] PCA={pca_ch}, run={run_num}")
    print(f"    OA={exp_info['test_oa']:.4f}, Kappa={exp_info['test_kappa']:.4f}")
    print(f"    Model: {model_path}")

    # 1. 加载数据
    X1, X2, Y = loadData(dataset_name)
    X1 = normalization(X1)
    X2 = normalization(X2)
    X1_pca = applyPCA(X1, channel=pca_ch)
    X2_pca = applyPCA(X2, channel=pca_ch)

    _, _, _, _, all_iter, all_position_indices, height, width, _, _ = generater(
        X1_pca, X2_pca, Y, BATCH_SIZE, 0.01, device, windowSize=PATCH_SIZE, noise_std=0
    )

    # 2. 加载模型
    model = ASGTNet(num_channels=pca_ch).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    # 3. 预测
    full_preds = predict_full_dataset(model, all_iter, device)

    # 4. 保存
    out_path = os.path.join(out_dir, f'{dataset_name}_{label}.png')
    generate_change_map(Y, full_preds, all_position_indices, height, width, out_path)

    del model
    torch.cuda.empty_cache()


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # ASGT-Net 使用的数据集名称（注意大小写）
    datasets = ['farmland', 'santaBarbara', 'bayArea']

    for dataset_name in datasets:
        print(f"\n{'=' * 50}")
        print(f"Processing: {dataset_name}")
        print(f"{'=' * 50}")

        # 找到全局最佳和次佳实验
        best_exp, second_exp = find_top_two(dataset_name)

        # 处理最佳
        process_experiment(dataset_name, best_exp, 'best', out_dir)

        # 处理次佳
        process_experiment(dataset_name, second_exp, 'second', out_dir)

        print(f"  Done: {dataset_name}!")

    print("\nAll done!")


if __name__ == '__main__':
    main()
