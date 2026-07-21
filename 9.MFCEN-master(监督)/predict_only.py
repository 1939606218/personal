"""
MFCEN 纯预测+可视化脚本
按 OA 排序，取精度 Top-3 实验，保存彩色变化检测图 DPI=600
输出: {dataset}_top1.png, {dataset}_top2.png, {dataset}_top3.png
"""
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils import make_data, MyDataset, test_epoch
from vit_pytorch import ViT


def find_top_experiments(dataset, top_k=3):
    """从 summary JSON 中按 OA 降序找前 K 个实验"""
    summary_path = f'./output/{dataset}_summary/{dataset}_summary.json'
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    with open(summary_path) as f:
        data = json.load(f)
    all_exps = sorted(data['all_experiments'], key=lambda x: x['oa'], reverse=True)
    return all_exps[:top_k]


def generate_change_map(y_true, y_pred, labeled_index, height, width, save_path):
    """生成彩色变化检测图 (TP白/FP红/FN绿/TN黑/未标记灰)，DPI=600"""
    full_pred = np.full(y_true.shape, 2, dtype=np.uint8)
    full_pred[labeled_index] = y_pred.astype(np.uint8)

    true_img = y_true.reshape(height, width)
    pred_img = full_pred.reshape(height, width)

    color_map = np.zeros((height, width, 3), dtype=np.uint8)
    color_map[(true_img == 1) & (pred_img == 1)] = [255, 255, 255]  # TP 白
    color_map[(true_img == 0) & (pred_img == 1)] = [255, 0, 0]      # FP 红
    color_map[(true_img == 1) & (pred_img == 0)] = [0, 255, 0]      # FN 绿
    color_map[(true_img == 0) & (pred_img == 0)] = [0, 0, 0]        # TN 黑
    color_map[true_img == 2] = [100, 100, 100]                       # 未标记 灰

    plt.imsave(save_path, color_map, dpi=600)
    print(f"  Saved: {save_path}")


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

        # 1. 加载数据（所有实验共享）
        print("  Loading data...")
        all_x, all_y, labeled_index, height, width, full_y = make_data(dataset, patch_size=7)
        print(f"  Pixels: {height}x{width}, labeled: {len(all_y)}")

        # 2. 找 OA Top-3 实验
        top_exps = find_top_experiments(dataset, top_k=3)
        for rank, exp in enumerate(top_exps, 1):
            exp_num = exp['experiment_num']
            oa = exp['oa']
            print(f"\n  Top-{rank}: exp {exp_num} (OA={oa:.4f})")

            model_path = f'./output/{dataset}_exp{exp_num}/{dataset}_exp{exp_num}_model_parameter.pkl'
            print(f"    Model: {model_path}")

            # 3. 创建模型并加载权重
            band_size = dataset_bands[dataset]
            model = ViT(
                patch_size=7, num_feats=4, band_size=band_size, num_classes=2,
                dim=128, depth=4, heads=4, mlp_dim=8, dropout=0.1, emb_dropout=0.1,
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # 4. 预测
            dataset_obj = MyDataset(all_x, all_y)
            loader = DataLoader(dataset_obj, batch_size=256, shuffle=False)
            pre_u = test_epoch(model, loader, device)

            # 5. 保存
            save_path = os.path.join(out_dir, f'{dataset}_top{rank}.png')
            generate_change_map(full_y, pre_u, labeled_index, height, width, save_path)

            del model
            torch.cuda.empty_cache()

        del all_x, all_y
        torch.cuda.empty_cache()
        print(f"  Done: {dataset}!")

    print("\nAll done!")


if __name__ == '__main__':
    main()
