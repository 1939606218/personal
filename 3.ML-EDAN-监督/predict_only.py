"""
ML-EDAN 纯预测+可视化脚本
仅加载预训练模型进行整图预测并保存变化检测图，DPI=600
图片名格式: {dataset_name}_new.png
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from data.data_process import loadData
from data.pre_process import normalization, applyPCA, create_patches
from model.model import ML_EDAN
from Global import pca_channel, batch_size, patch_sieze, device

# 推理时使用更大的 batch_size（无需反向传播，可用更多显存）
INFER_BATCH_SIZE = 512


def predict_full_map(model, X1, X2, Y, window_size=5):
    """预测整幅图像的变化图（所有像素）"""
    # 预处理：归一化 + PCA
    X1 = normalization(X1)
    X2 = normalization(X2)
    X1 = applyPCA(X1, channel=pca_channel)
    X2 = applyPCA(X2, channel=pca_channel)

    # 为整个图像所有像素创建 patches
    patches_X1, _, indices = create_patches(X1, Y, window_size=window_size)
    patches_X1 = np.transpose(patches_X1, (0, 3, 1, 2))

    patches_X2, _, _ = create_patches(X2, Y, window_size=window_size)
    patches_X2 = np.transpose(patches_X2, (0, 3, 1, 2))

    # 转为 tensor
    X1_tensor = torch.tensor(patches_X1, dtype=torch.float32)
    X2_tensor = torch.tensor(patches_X2, dtype=torch.float32)

    # 预测
    model.eval()
    pred_map = np.ones_like(Y).flatten() * 2  # 初始化为2（未标记）

    with torch.no_grad():
        for i in range(0, len(X1_tensor), INFER_BATCH_SIZE):
            end_idx = min(i + INFER_BATCH_SIZE, len(X1_tensor))
            batch_X1 = X1_tensor[i:end_idx].to(device)
            batch_X2 = X2_tensor[i:end_idx].to(device)

            outputs = model(batch_X1, batch_X2)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            _, batch_preds = torch.max(logits, 1)
            batch_preds = batch_preds.cpu().numpy()

            # 按索引映射回原始图像位置（向量化赋值）
            batch_indices = indices[i:end_idx]
            pred_map[batch_indices] = batch_preds

    pred_map = pred_map.reshape(Y.shape)
    return pred_map


def generate_change_map(y_true, y_pred, dataset_name, save_dir):
    """生成变化检测结果图，保存为 {dataset_name}_new.png，DPI=600"""
    if y_true.ndim > 2:
        y_true = y_true.squeeze()
    if y_pred.ndim > 2:
        y_pred = y_pred.squeeze()

    color_map = np.zeros((y_true.shape[0], y_true.shape[1], 3), dtype=np.uint8)

    TP_mask = (y_true == 1) & (y_pred == 1)      # 正确检测变化 → 白色
    FP_mask = (y_true == 0) & (y_pred == 1)      # 误报变化 → 红色
    FN_mask = (y_true == 1) & (y_pred == 0)      # 漏检变化 → 绿色
    TN_mask = (y_true == 0) & (y_pred == 0)      # 正确未变化 → 黑色
    unlabeled_mask = (y_true == 2)                # 未标记区域 → 灰色

    color_map[TP_mask] = [255, 255, 255]
    color_map[FP_mask] = [255, 0, 0]
    color_map[FN_mask] = [0, 255, 0]
    color_map[TN_mask] = [0, 0, 0]
    color_map[unlabeled_mask] = [100, 100, 100]

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{dataset_name}_new.png')
    plt.imsave(save_path, color_map, dpi=600)
    print(f"Saved change map to {save_path}")

    return color_map


def main():
    datasets_config = [
        {'name': 'farm', 'in_channel': 155, 'description': 'Farmland dataset'},
        {'name': 'hermiston', 'in_channel': 242, 'description': 'Hermiston dataset'},
        {'name': 'river', 'in_channel': 198, 'description': 'River dataset'},
        {'name': 'Barbara', 'in_channel': 224, 'description': 'Santa Barbara dataset'},
        {'name': 'BayArea', 'in_channel': 224, 'description': 'Bay Area dataset'},
    ]

    for config in datasets_config:
        dataname = config['name']
        model_file = os.path.join("result", dataname, "best_model.pth")

        print(f"\n{'=' * 50}")
        print(f"Processing: {config['description']} ({dataname})")

        if not os.path.exists(model_file):
            print(f"  Model not found: {model_file}, skipping...")
            continue

        # 1. 加载数据
        print("  Loading data...")
        X1, X2, Y, height, width = loadData(dataname)

        # 2. 加载预训练模型
        print(f"  Loading model from {model_file}...")
        model = ML_EDAN(in_channel=pca_channel).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()

        # 3. 预测整图
        print("  Predicting full map...")
        pred_map = predict_full_map(model, X1, X2, Y, window_size=patch_sieze)

        # 4. 保存变化检测图
        save_dir = os.path.join("result", dataname)
        generate_change_map(Y, pred_map, dataname, save_dir)

        print(f"  Done: {config['description']}!")

    print("\nAll datasets processed!")


if __name__ == '__main__':
    main()
