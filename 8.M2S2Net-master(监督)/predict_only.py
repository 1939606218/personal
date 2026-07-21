"""
M2S2Net 纯预测+可视化脚本
加载最佳预训练模型，预测变化图，保存彩色结果图 DPI=600
输出: 当前目录 {dataset}_new.png
"""
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils import make_data, MyDataset, test_epoch
from model import FinalModel


def find_best_exp(dataset):
    """从 summary JSON 中查找最佳实验编号"""
    summary_path = f'./output/{dataset}_summary/{dataset}_summary.json'
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
        best_exp = data['max_value_experiment']['experiment_num']
        best_oa = data['max_value_experiment']['oa']
        print(f"  最佳实验: exp {best_exp} (OA={best_oa:.4f})")
        return best_exp
    # 回退：检查目录
    for exp_num in range(1, 11):
        pkl = f'./output/{dataset}_exp{exp_num}/{dataset}_exp{exp_num}_model_parameter.pkl'
        if os.path.exists(pkl):
            return exp_num
    raise FileNotFoundError(f"No model found for {dataset}")


def predict_labeled_pixels(model, all_x, all_y, device, batch_size=256):
    """预测所有标记像素"""
    model.eval()
    # test_epoch 需要 (data, target)，传入 all_y 以匹配格式
    dataset = MyDataset(all_x, all_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pred = test_epoch(model, loader, device)
    return pred


def generate_change_map(y_true, y_pred, labeled_index, height, width, dataset_name, out_dir):
    """生成彩色变化检测图 (TP白/FP红/FN绿/TN黑/未标记灰)，DPI=600"""
    # 重建全图预测（未标记区域填2）
    full_pred = np.full(y_true.shape, 2, dtype=np.uint8)
    full_pred[labeled_index] = y_pred.astype(np.uint8)

    true_img = y_true.reshape(height, width)
    pred_img = full_pred.reshape(height, width)

    color_map = np.zeros((height, width, 3), dtype=np.uint8)

    TP_mask = (true_img == 1) & (pred_img == 1)
    FP_mask = (true_img == 0) & (pred_img == 1)
    FN_mask = (true_img == 1) & (pred_img == 0)
    TN_mask = (true_img == 0) & (pred_img == 0)
    unlabeled_mask = (true_img == 2)

    color_map[TP_mask] = [255, 255, 255]
    color_map[FP_mask] = [255, 0, 0]
    color_map[FN_mask] = [0, 255, 0]
    color_map[TN_mask] = [0, 0, 0]
    color_map[unlabeled_mask] = [100, 100, 100]

    save_path = os.path.join(out_dir, f'{dataset_name}_new.png')
    plt.imsave(save_path, color_map, dpi=600)
    print(f"  Saved: {save_path}")
    return color_map


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    out_dir = os.path.dirname(os.path.abspath(__file__))

    datasets = ['farmland', 'Barbara', 'BayArea']
    dataset_bands = {'farmland': 155, 'Barbara': 224, 'BayArea': 224}

    for dataset in datasets:
        print(f"\n{'=' * 50}")
        print(f"Processing: {dataset}")
        print(f"{'=' * 50}")

        # 1. 加载数据
        print("  Loading data...")
        all_x, all_y, labeled_index, height, width, full_y = make_data(dataset, patch_size=7)
        print(f"  Pixels: {height}x{width}, labeled: {len(all_y)}")

        # 2. 找最佳模型权重
        best_exp = find_best_exp(dataset)
        model_path = f'./output/{dataset}_exp{best_exp}/{dataset}_exp{best_exp}_model_parameter.pkl'
        print(f"  Loading model: {model_path}")

        # 3. 创建模型并加载权重
        band_size = dataset_bands[dataset]
        model = FinalModel(
            seq_len=8,
            band_size=band_size,
            patch_size=7,
            dim=128,
            depth=4,
            heads=4,
            mlp_dim=8,
            dim_head=16,
            dropout=0.1,
            emb_dropout=0.1,
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # 4. 预测标记像素
        print("  Predicting...")
        pre_u = predict_labeled_pixels(model, all_x, all_y, device)

        # 5. 生成并保存彩色变化图
        generate_change_map(full_y, pre_u, labeled_index, height, width, dataset, out_dir)

        # 释放资源
        del model, all_x, all_y
        torch.cuda.empty_cache()

        print(f"  Done: {dataset}!")

    print("\nAll done!")


if __name__ == '__main__':
    main()
