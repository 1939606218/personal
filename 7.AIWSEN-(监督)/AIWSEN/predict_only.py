"""
AIWSEN 纯预测+可视化脚本
仅加载预训练模型进行整图预测，保存彩色变化检测图 DPI=600
输出: 当前目录下 {dataset}_new.png
"""
import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn

# 确保当前目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from get_dataset import get_dataset
from model.AIWSEN import AIWSEN


def std_norm(image):
    """按通道标准化（与训练时一致）"""
    epsilon = 1e-8
    image_np = image.permute(1, 2, 0).numpy()
    mean = torch.tensor(image_np).mean(dim=[0, 1])
    std = torch.tensor(image_np).std(dim=[0, 1])
    std = torch.where(std == 0, torch.tensor(epsilon), std)
    normalize_transform = transforms.Normalize(mean, std)
    return normalize_transform(image)


def construct_sample(img1, img2, window_size=7):
    """为所有像素构造滑动窗口坐标（镜像填充）"""
    _, height, width = img1.shape
    half_window = window_size // 2
    pad = nn.ReplicationPad2d(half_window)
    pad_img1 = pad(img1.unsqueeze(0)).squeeze(0)
    pad_img2 = pad(img2.unsqueeze(0)).squeeze(0)

    patch_coords = torch.zeros((height * width, 4), dtype=torch.long)
    t = 0
    for h in range(height):
        for w in range(width):
            patch_coords[t, :] = torch.tensor([h, h + window_size, w, w + window_size])
            t += 1
    return pad_img1, pad_img2, patch_coords


def predict_full_image(model, pad_img1, pad_img2, patch_coords, gt, device, batch_size=1000):
    """预测整幅图像所有像素"""
    height, width = gt.shape
    predictions = np.zeros((height, width), dtype=np.int64)

    model.eval()
    total = len(patch_coords)
    rows = np.arange(total) // width
    cols = np.arange(total) % width

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            n = end - start
            batch_patches1 = torch.zeros(n, pad_img1.shape[0], 7, 7)
            batch_patches2 = torch.zeros(n, pad_img2.shape[0], 7, 7)

            for k in range(n):
                idx = start + k
                h0, h1, w0, w1 = patch_coords[idx]
                batch_patches1[k] = pad_img1[:, h0:h1, w0:w1]
                batch_patches2[k] = pad_img2[:, h0:h1, w0:w1]

            batch_patches1 = batch_patches1.to(device)
            batch_patches2 = batch_patches2.to(device)

            output = model(batch_patches1, batch_patches2)
            _, pred = output.max(1)
            pred = pred.cpu().numpy()

            batch_rows = rows[start:end]
            batch_cols = cols[start:end]
            predictions[batch_rows, batch_cols] = pred

            print(f"\r  Predicting: {end}/{total}", end="")
    print()
    return predictions


def generate_change_map(y_true, y_pred, dataset_name, out_dir):
    """生成彩色变化检测图并保存，DPI=600"""
    if y_true.ndim > 2:
        y_true = y_true.squeeze()
    if y_pred.ndim > 2:
        y_pred = y_pred.squeeze()

    color_map = np.zeros((y_true.shape[0], y_true.shape[1], 3), dtype=np.uint8)

    TP_mask = (y_true == 1) & (y_pred == 1)
    FP_mask = (y_true == 0) & (y_pred == 1)
    FN_mask = (y_true == 1) & (y_pred == 0)
    TN_mask = (y_true == 0) & (y_pred == 0)
    unlabeled_mask = (y_true == 2)

    color_map[TP_mask] = [255, 255, 255]  # 白色
    color_map[FP_mask] = [255, 0, 0]      # 红色
    color_map[FN_mask] = [0, 255, 0]      # 绿色
    color_map[TN_mask] = [0, 0, 0]        # 黑色
    color_map[unlabeled_mask] = [100, 100, 100]  # 灰色

    save_path = os.path.join(out_dir, f'{dataset_name}_new.png')
    plt.imsave(save_path, color_map, dpi=600)
    print(f"Saved: {save_path}")
    return color_map


def find_best_weight(dataset_name, weights_dir='./weights'):
    """找到最佳的模型权重（文件名最长的那个，包含最多实验累积）"""
    pattern = os.path.join(weights_dir, dataset_name, f'{dataset_name}_AIWSEN_*_Final.pth')
    files = glob.glob(pattern)
    if not files:
        # 也尝试大写
        pattern = os.path.join(weights_dir, dataset_name.capitalize(), f'{dataset_name}_AIWSEN_*_Final.pth')
        files = glob.glob(pattern)
        if not files:
            # 再尝试 farmland → Farmland 的映射
            folder_map = {'farmland': 'Farmland', 'Barbara': 'Barbara', 'BayArea': 'BayArea'}
            folder = folder_map.get(dataset_name, dataset_name)
            pattern = os.path.join(weights_dir, folder, f'{dataset_name}_AIWSEN_*_Final.pth')
            files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No weight file found for {dataset_name}")
    # 选文件名最长的（包含最多实验累积）
    files.sort(key=len, reverse=True)
    return files[0]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 保存路径：当前 AIWSEN 目录
    out_dir = os.path.dirname(os.path.abspath(__file__))

    datasets_config = {
        'farmland': {'band': 155, 'num_head': 5},
        'Barbara':  {'band': 224, 'num_head': 4},
        'BayArea':  {'band': 224, 'num_head': 4},
    }

    for dataset_name, cfg in datasets_config.items():
        print(f"\n{'=' * 50}")
        print(f"Processing: {dataset_name}")
        print(f"{'=' * 50}")

        # 1. 加载原始数据
        print("  Loading data...")
        img1, img2, gt = get_dataset(dataset_name)
        # gt 是 float32 numpy，转为 int 用于后续布尔索引
        gt = gt.astype(np.int32)

        # 2. 转为 CHW tensor 并标准化
        img1_t = torch.from_numpy(img1.transpose(2, 0, 1)).float()
        img2_t = torch.from_numpy(img2.transpose(2, 0, 1)).float()
        img1_t = std_norm(img1_t)
        img2_t = std_norm(img2_t)

        # 3. 构造全图像素滑动窗口
        print("  Constructing patches...")
        pad_img1, pad_img2, patch_coords = construct_sample(img1_t, img2_t, window_size=7)
        print(f"  Patch coords: {patch_coords.shape[0]} pixels")

        # 4. 加载模型
        weight_path = find_best_weight(dataset_name)
        print(f"  Loading model: {weight_path}")
        model = AIWSEN(device=device, inchannel=cfg['band'], num_head=cfg['num_head']).to(device)
        state_dict = torch.load(weight_path, map_location=device)
        # 去掉可能的 'module.' 前缀
        new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state, strict=False)
        model.eval()

        # 5. 预测整图
        print("  Predicting full image...")
        pred_map = predict_full_image(model, pad_img1, pad_img2, patch_coords, gt, device)

        # 6. 生成并保存彩色变化图
        generate_change_map(gt, pred_map, dataset_name, out_dir)

        print(f"  Done: {dataset_name}!")

        # 释放显存
        del model, img1_t, img2_t, pad_img1, pad_img2, patch_coords
        torch.cuda.empty_cache()

    print("\nAll done!")


if __name__ == '__main__':
    main()
