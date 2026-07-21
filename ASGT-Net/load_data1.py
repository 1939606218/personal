import scipy.io as sio
import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
import torch.utils.data as Data
import torch.optim.lr_scheduler
import torch.nn.functional as F
import time

# 统计标签数量的函数
def count_labels(labels):
    """统计标签中各个值的出现次数"""
    # 方法一：使用NumPy
    unique_values, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_values, counts))

    # 方法二：使用Counter（适用于非数值标签）
    # label_counts = dict(Counter(labels.flatten()))
    return label_counts

def loadData(dataset_name):
    if dataset_name == 'hermiston':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2004.mat')['HypeRvieW']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2007.mat')['HypeRvieW']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\label.mat')['label']

    if dataset_name == 'farmland':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm06.mat')['imgh']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm07.mat')['imghl']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\label.mat')['label']

    if dataset_name == 'river':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_before.mat')['river_before']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_after.mat')['river_after']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\groundtruth.mat')['lakelabel_v1']
        labels[labels == 255] = 1

    if dataset_name == 'bayArea':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2013.mat')['HypeRvieW']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2015.mat')['HypeRvieW']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\bayArea_gtChanges2.mat')['HypeRvieW']

        # 修改标签映射：0→2，1→1，2→0
        labels = np.select(
            [labels == 0, labels == 1, labels == 2],
            [2, 1, 0],
            default=labels  # 处理其他可能的值
        )

    if dataset_name == 'santaBarbara':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_2013.mat')['HypeRvieW']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_2014.mat')['HypeRvieW']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_gtChanges.mat')['HypeRvieW']

        # 修改标签映射：0→2，1→1，2→0
        labels = np.select(
            [labels == 0, labels == 1, labels == 2],
            [2, 1, 0],
            default=labels  # 处理其他可能的值
        )

    if dataset_name == 'USA':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\USA\USA_before.mat')['USA_before']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\USA\USA_after.mat')['USA_after']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\USA\label.mat')['label']

    # 统计标签数量
    label_counts = count_labels(labels)
    print(f"数据集标签统计：{label_counts}")

    print("data1.shape", data1.shape)
    print("data2.shape", data2.shape)
    print("labels.shape", labels.shape)

    return data1, data2, labels


# 划分patch块的方法
def pad_with_zeros(X, margin=2):        #进行零填充操作
    """apply zero padding to X with margin"""

    new_X = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    new_X[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return new_X

def create_patches(X, y, window_size):
    """create patch from image. suppose the image has the shape (w,h,c) then the patch shape is
    (w*h,window_size,window_size,c)"""
    margin = int((window_size - 1) / 2)
    zero_padded_X = pad_with_zeros(X, margin=margin)
    zero_padded_y = pad_with_zeros(y[..., np.newaxis], margin=margin)[..., 0]
    # split patches
    patches_data = np.zeros((X.shape[0] * X.shape[1], window_size, window_size, X.shape[2]))
    patchs_labels = np.zeros((X.shape[0] * X.shape[1]))
    patch_index = 0
    for r in range(margin, zero_padded_X.shape[0] - margin):
        for c in range(margin, zero_padded_X.shape[1] - margin):
            patch = zero_padded_X[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patches_data[patch_index, :, :, :] = patch
            # 获取当前图像块对应的标签块
            label_patch = zero_padded_y[r - margin:r + margin + 1, c - margin:c + margin + 1]
            # 统计标签块中每个标签的出现次数
            label_counts = Counter(label_patch.flatten())
            # 选择出现次数最多的标签作为该图像块的标签
            most_common_label = label_counts.most_common(1)[0][0]
            patchs_labels[patch_index] = most_common_label
            patch_index = patch_index + 1

    return patches_data, patchs_labels

def split_train_test_set(X, y, train_ratio):
    # 分别找出标签为 0 和 1 的数据索引
    label_0_indices = np.where(y == 0)[0]
    label_1_indices = np.where(y == 1)[0]

    # 计算训练集的大小
    total_train_size = int(len(y) * train_ratio)

    # 计算标签为 0 和 1 的训练集大小
    label_1_train_size = total_train_size // 3
    label_0_train_size = 2 * label_1_train_size

    # 随机选择标签为 0 和 1 的训练集索引
    np.random.shuffle(label_0_indices)
    np.random.shuffle(label_1_indices)
    label_0_train_indices = label_0_indices[:label_0_train_size]
    label_1_train_indices = label_1_indices[:label_1_train_size]

    # 合并标签为 0 和 1 的训练集索引
    train_indices = np.concatenate((label_0_train_indices, label_1_train_indices))
    np.random.shuffle(train_indices)

    # 划分训练集和测试集
    X_train = X[train_indices]
    y_train = y[train_indices]
    test_indices = np.setdiff1d(np.arange(len(y)), train_indices)
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def applyPCA(X, channel=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=channel, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], channel))
    return newX

def normalization(X):
    """
    normalization type  1: map to [0, 1]
    normalization type  2: map to zero mean and one std
    :param X:
    :param type:
    :return: normalization X
    """           #MinMaxScaler用于对数据进行归一化处理，将数据的特征值映射到指定的区间[0,1]
    X_reshape = X.reshape((-1, X.shape[-1]))
    transfer = MinMaxScaler()
    X_reshape = transfer.fit_transform(X_reshape)
    X = X_reshape.reshape((X.shape[0], X.shape[1], X.shape[2]))
    return X

def add_gaussian_noise(tensor, mean=0., std=0.1):
    """对输入Tensor添加高斯噪声"""
    noise = torch.randn_like(tensor) * std + mean
    return tensor + noise

def generater(X_1, X_2, Y, batchsize, train_ratio, device, windowSize, noise_std=0.1, batch_num=10):
    H, W, C = X_1.shape  # 高度、宽度、通道数
    height, width = Y.shape
    assert H == height and W == width, "X和Y的空间尺寸必须一致"

    all_indices = np.arange(H * W)
    mask_all = Y.flatten() != 2
    all_position_indices = all_indices[mask_all]

    h_batch = H // batch_num  # 每批的高度

    X1_train_list = []
    X2_train_list = []
    y_train_list = []
    X1_test_list = []
    X2_test_list = []
    y_test_list = []
    X1_all_list = []
    X2_all_list = []
    Y_all_list = []

    for i in range(batch_num):
        start_h = i * h_batch
        end_h = (i + 1) * h_batch if i < batch_num - 1 else H

        X_1_batch = X_1[start_h:end_h, :, :]
        X_2_batch = X_2[start_h:end_h, :, :]
        Y_batch = Y[start_h:end_h, :]

        # 生成当前批次的patches
        alldataX1, alldataY1 = create_patches(X_1_batch, Y_batch, window_size=windowSize)
        alldataX1 = np.transpose(alldataX1, (0, 3, 1, 2)).astype(np.float32)
        alldataX2, alldataY2 = create_patches(X_2_batch, Y_batch, window_size=windowSize)
        alldataX2 = np.transpose(alldataX2, (0, 3, 1, 2)).astype(np.float32)

        # 分层抽样划分训练集和测试集（直接在批次内应用索引）
        combined_idx = np.arange(len(alldataY1))
        idx_train, idx_test, y_train, y_test = train_test_split(
            combined_idx, alldataY1,
            train_size=train_ratio,
            stratify=alldataY1,
            random_state=42
        )

        # 直接在批次数据上应用索引，避免合并后索引越界
        X1_train = alldataX1[idx_train]
        X2_train = alldataX2[idx_train]
        y_train = alldataY1[idx_train]  # 使用原始标签索引

        X1_test = alldataX1[idx_test]
        X2_test = alldataX2[idx_test]
        y_test = alldataY1[idx_test]

        # 移除标签为2的样本
        mask_train = y_train != 2
        X1_train = X1_train[mask_train]
        X2_train = X2_train[mask_train]
        y_train = y_train[mask_train]

        mask_test = y_test != 2
        X1_test = X1_test[mask_test]
        X2_test = X2_test[mask_test]
        y_test = y_test[mask_test]

        # 收集当前批次的所有数据
        mask_all_batch = Y_batch.flatten() != 2
        X1_all = alldataX1[mask_all_batch]
        X2_all = alldataX2[mask_all_batch]
        Y_all = alldataY1[mask_all_batch]

        # 追加到列表
        X1_train_list.append(X1_train)
        X2_train_list.append(X2_train)
        y_train_list.append(y_train)
        X1_test_list.append(X1_test)
        X2_test_list.append(X2_test)
        y_test_list.append(y_test)
        X1_all_list.append(X1_all)
        X2_all_list.append(X2_all)
        Y_all_list.append(Y_all)

    # 合并所有批次的数据
    X1_train = np.concatenate(X1_train_list, axis=0)
    X2_train = np.concatenate(X2_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X1_test = np.concatenate(X1_test_list, axis=0)
    X2_test = np.concatenate(X2_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    X1_all = np.concatenate(X1_all_list, axis=0)
    X2_all = np.concatenate(X2_all_list, axis=0)
    Y_all = np.concatenate(Y_all_list, axis=0)

    # 后续代码保持不变...
    X1_all_tensor = torch.from_numpy(X1_all).float()
    X2_all_tensor = torch.from_numpy(X2_all).float()
    torch_all = Data.TensorDataset(X1_all_tensor, X2_all_tensor)
    all_iter = Data.DataLoader(torch_all, batch_size=batchsize, shuffle=False, num_workers=0)

    X1_train_tensor = torch.from_numpy(X1_train).float()
    X2_train_tensor = torch.from_numpy(X2_train).float()
    Y_train_tensor = torch.from_numpy(y_train).long()
    X1_test_tensor = torch.from_numpy(X1_test).float()
    X2_test_tensor = torch.from_numpy(X2_test).float()
    Y_test_tensor = torch.from_numpy(y_test).long()

    X1_train_tensor_noised = add_gaussian_noise(X1_train_tensor, std=noise_std)
    X2_train_tensor_noised = add_gaussian_noise(X2_train_tensor, std=noise_std)

    torch_train = Data.TensorDataset(X1_train_tensor_noised, X2_train_tensor_noised, Y_train_tensor)
    torch_test = Data.TensorDataset(X1_test_tensor, X2_test_tensor, Y_test_tensor)
    train_iter = Data.DataLoader(torch_train, batch_size=batchsize, shuffle=True, num_workers=0)
    test_iter = Data.DataLoader(torch_test, batch_size=batchsize, shuffle=False, num_workers=0)

    class_counts = Counter(y_train.flatten())
    class_weights = 1.0 / torch.tensor([class_counts[0], class_counts[1]], dtype=torch.float32)
    class_weights = class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    total_samples = sum(class_counts.values())
    alpha = torch.tensor([class_counts[1] / total_samples, class_counts[0] / total_samples], device=device)

    return (len(y_train), len(y_test), train_iter, test_iter, all_iter,
            all_position_indices, height, width, criterion, alpha)



def visualize_full_predictions(original_labels, predictions, indices,
                               height, width, dataset_name, output_path):
    """
    可视化整个数据集的预测结果（包含训练集和测试集）

    参数:
        original_labels: 原始完整标签图 (2D数组)
        predictions: 整个数据集的预测结果 (1D数组)
        indices: 预测结果对应的位置索引 (1D数组)
        height, width: 图像尺寸
        dataset_name: 数据集名称
        output_path: 输出图像保存路径
    """
    # 创建全图预测结果数组（初始值-1表示未预测区域）
    full_pred = np.full((height * width), -1, dtype=np.int16)
    full_pred[indices] = predictions

    # 重塑为2D图像
    pred_img = full_pred.reshape(height, width)
    true_img = original_labels

    # 创建可视化图像 (RGB)
    vis_img = np.zeros((height, width, 3), dtype=np.uint8)

    # 获取标签区域
    tn_mask = (true_img == 0) & (pred_img == 0)
    tp_mask = (true_img == 1) & (pred_img == 1)
    fp_mask = (true_img == 0) & (pred_img == 1)
    fn_mask = (true_img == 1) & (pred_img == 0)
    unlabeled_mask = (true_img == 2)

    # 应用颜色
    vis_img[tn_mask] = [0, 0, 0]        # 黑色 - TN
    vis_img[tp_mask] = [255, 255, 255]  # 白色 - TP
    vis_img[fp_mask] = [255, 0, 0]      # 红色 - FP
    vis_img[fn_mask] = [0, 255, 0]      # 绿色 - FN
    vis_img[unlabeled_mask] = [100, 100, 100]  # 灰色 - 未标记

    # 保存可视化图像
    plt.figure(figsize=(12, 10))
    plt.imshow(vis_img)
    plt.axis('off')  # 移除坐标轴

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"完整预测结果可视化已保存至: {output_path}")
    return vis_img