import torch

print(torch.__version__)
print(torch.cuda.is_available())

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from models.SSA import SSA
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import argparse


# 从 init.py 中提取需要的函数，避免导入整个模块
def pad_with_zeros(X, margin=2):
    """apply zero padding to X with margin"""
    new_X = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    new_X[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return new_X


def create_patches(X, y, window_size, remove_zero_labels=True):
    """create patch from image. suppose the image has the shape (w,h,c) then the patch shape is
    (w*h,window_size,window_size,c)"""
    margin = int((window_size - 1) / 2)
    zero_padded_X = pad_with_zeros(X, margin=margin)

    # split patches
    patches_data = np.zeros((X.shape[0] * X.shape[1], window_size, window_size, X.shape[2]))
    patchs_labels = np.zeros((X.shape[0] * X.shape[1]))
    patch_index = 0
    for r in range(margin, zero_padded_X.shape[0] - margin):
        for c in range(margin, zero_padded_X.shape[1] - margin):
            patch = zero_padded_X[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patches_data[patch_index, :, :, :] = patch
            patchs_labels[patch_index] = y[r - margin, c - margin]
            patch_index = patch_index + 1

    return patches_data, patchs_labels


def applyPCA(X, numComponents=30):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def normalization(X, type=3):
    """Normalization type 3: MinMaxScaler"""
    if type == 3:
        X_reshape = X.reshape((-1, X.shape[-1]))
        transfer = MinMaxScaler()
        X_reshape = transfer.fit_transform(X_reshape)
        X = X_reshape.reshape((X.shape[0], X.shape[1], X.shape[2]))
        return X
    else:
        return X


# 固定配置
CONFIG = {
    'model_dir': './models/',
    'default_window_size': 5,
    'default_batch_size': 64,
    # 数据集特定的 kerner_number 配置
    'kerner_numbers': {
        'hermiston': 24,
        'river': 24,
        'farmland': 24,
        'santaBarbara': 32,
        'bayArea': 32
    }
}


def generate_change_map(y_true, y_pred, dataset_name):
    """
    生成变化检测结果图
    :param y_true: 原始标签图 (2D numpy array)
    :param y_pred: 预测结果图 (2D numpy array)
    :param dataset_name: 数据集名称 (用于保存文件名)
    """
    # 确保输入是二维数组
    if y_true.ndim > 2:
        y_true = y_true.squeeze()
    if y_pred.ndim > 2:
        y_pred = y_pred.squeeze()

    # 初始化彩色图像 (H, W, 3)
    color_map = np.zeros((y_true.shape[0], y_true.shape[1], 3), dtype=np.uint8)

    # 提取不同区域
    TP_mask = (y_true == 1) & (y_pred == 1)  # 正确检测变化
    FP_mask = (y_true == 0) & (y_pred == 1)  # 误报变化
    FN_mask = (y_true == 1) & (y_pred == 0)  # 漏检变化
    TN_mask = (y_true == 0) & (y_pred == 0)  # 正确未变化
    unlabeled_mask = (y_true == 2)  # 未标记区域

    # 应用颜色映射
    color_map[TP_mask] = [255, 255, 255]  # 白色
    color_map[FP_mask] = [255, 0, 0]  # 红色
    color_map[FN_mask] = [0, 255, 0]  # 绿色
    color_map[TN_mask] = [0, 0, 0]  # 黑色
    color_map[unlabeled_mask] = [100, 100, 100]  # 灰色

    # 保存图像
    plt.imsave(f'{dataset_name}_change_map.png', color_map, dpi=600)
    print(f"Saved change map to {dataset_name}_change_map.png")

    return color_map


def load_dataset_data(dataset_name):
    """加载特定数据集的数据"""
    if dataset_name == 'hermiston':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2004.mat')['HypeRvieW']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2007.mat')['HypeRvieW']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\label.mat')['label']

    elif dataset_name == 'river':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_before.mat')['river_before']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_after.mat')['river_after']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\groundtruth.mat')['lakelabel_v1']
        labels[labels == 255] = 1

    elif dataset_name == 'farmland':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm06.mat')['imgh']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm07.mat')['imghl']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\label.mat')['label']

    elif dataset_name == 'santaBarbara':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_2013.mat')['HypeRvieW']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_2014.mat')['HypeRvieW']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_gtChanges.mat')['HypeRvieW']
        labels = np.select([labels == 0, labels == 1, labels == 2], [2, 1, 0], default=labels)

    elif dataset_name == 'bayArea':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2013.mat')['HypeRvieW']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2015.mat')['HypeRvieW']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\bayArea_gtChanges2.mat')['HypeRvieW']
        labels = np.select([labels == 0, labels == 1, labels == 2], [2, 1, 0], default=labels)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Loaded {dataset_name} dataset:")
    print(f"Data1 shape: {data1.shape}")
    print(f"Data2 shape: {data2.shape}")
    print(f"Labels shape: {labels.shape}")

    return data1, data2, labels


def predict_full_map(model, data1, data2, labels, window_size=5, batch_size=256, pca=True, pca_channel=30):
    """预测整幅图像的变化图"""
    # 预处理数据
    data1 = normalization(X=data1, type=3)
    data2 = normalization(X=data2, type=3)

    if pca:
        data1 = applyPCA(data1, numComponents=pca_channel)
        data2 = applyPCA(data2, numComponents=pca_channel)

    # 为整个图像创建patches
    patches_X1, _ = create_patches(data1, labels, window_size=window_size)
    patches_X1 = np.transpose(patches_X1, (0, 3, 1, 2))

    patches_X2, _ = create_patches(data2, labels, window_size=window_size)
    patches_X2 = np.transpose(patches_X2, (0, 3, 1, 2))

    # 将数据转换为tensor
    X1_tensor = torch.tensor(patches_X1, dtype=torch.float32)
    X2_tensor = torch.tensor(patches_X2, dtype=torch.float32)

    # 预测整幅图像
    model.eval()
    pred_map = np.zeros_like(labels)

    # 获取设备
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(0, len(X1_tensor), batch_size):
            # 获取当前批次
            end_idx = min(i + batch_size, len(X1_tensor))
            batch_X1 = X1_tensor[i:end_idx].to(device)
            batch_X2 = X2_tensor[i:end_idx].to(device)

            # 预测
            outputs = model(batch_X1, batch_X2)
            if isinstance(outputs, tuple):  # 如果返回多个输出，取最后一个
                outputs = outputs[-1]

            # 获取预测结果
            batch_preds = outputs.argmax(dim=1).cpu().numpy()

            # 将预测结果映射回原始图像
            rows, cols = np.unravel_index(range(i, i + len(batch_preds)), labels.shape)
            pred_map.flat[i:i + len(batch_preds)] = batch_preds

    return pred_map


def visualize_dataset(dataset_name, window_size=None, batch_size=None):
    """可视化指定数据集的变化检测图"""
    # 设置参数
    window_size = window_size or CONFIG['default_window_size']
    batch_size = batch_size or CONFIG['default_batch_size']

    # 设置设备 - 放在这里确保正确检测GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # 1. 加载数据集
        print(f"\nLoading {dataset_name} dataset...")
        data1, data2, labels = load_dataset_data(dataset_name)

        # 2. 构建模型路径
        model_path = os.path.join(CONFIG['model_dir'], f"{dataset_name}.pt")
        print(f"Model path: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        # 3. 加载模型
        print("Loading model...")
        # 获取数据集特定的 kerner_number
        kerner_number = CONFIG['kerner_numbers'].get(dataset_name, 24)
        print(f"Using kerner_number: {kerner_number} for {dataset_name}")

        # 创建模型时传入正确的参数
        model = SSA(kerner_number).to(device)
        model.load_state_dict(torch.load(model_path))

        # 4. 预测整幅图像
        print("Predicting full map...")
        pred_map = predict_full_map(
            model=model,
            data1=data1,
            data2=data2,
            labels=labels,
            window_size=window_size,
            batch_size=batch_size
        )

        # 5. 生成并保存变化检测图
        print("Generating change map...")
        generate_change_map(labels, pred_map, dataset_name)

        print(f"\nSuccessfully generated change map for {dataset_name}!")

    except Exception as e:
        print(f"\nError processing {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 确保正确检测GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    # 示例：直接在这里调用可视化函数
    datasets_to_visualize = [
        'hermiston',
        'river',
        'farmland',
        'santaBarbara',  # 可以取消注释需要可视化的数据集
        'bayArea'
    ]

    for dataset in datasets_to_visualize:
        print(f"\n{'=' * 50}")
        print(f"Processing dataset: {dataset}")
        print(f"{'=' * 50}")
        visualize_dataset(dataset)