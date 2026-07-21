import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
import torch.utils.data as Data
import torch.optim.lr_scheduler
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


#划分patch块的方法
def pad_with_zeros(X, margin=2):        #进行零填充操作
    """apply zero padding to X with margin"""

    new_X = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    new_X[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return new_X


def create_patches(X, y, window_size):
    """从图像创建patch，返回patch数据、标签和每个patch的索引"""
    margin = int((window_size - 1) / 2)
    zero_padded_X = pad_with_zeros(X, margin=margin)  # 假设pad_with_zeros已定义

    # 计算总patch数量
    n_patches = (zero_padded_X.shape[0] - 2 * margin) * (zero_padded_X.shape[1] - 2 * margin)
    patches_data = np.zeros((n_patches, window_size, window_size, X.shape[2]))
    patchs_labels = np.zeros(n_patches)
    indices = np.zeros(n_patches, dtype=int)  # 用于记录每个patch对应的原始像素索引

    patch_index = 0
    for r in range(margin, zero_padded_X.shape[0] - margin):
        for c in range(margin, zero_padded_X.shape[1] - margin):
            patch = zero_padded_X[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patches_data[patch_index, :, :, :] = patch
            patchs_labels[patch_index] = y[r - margin, c - margin]
            indices[patch_index] = (r - margin) * X.shape[1] + (c - margin)  # 计算原始像素索引
            patch_index += 1

    return patches_data, patchs_labels, indices


def split_train_test_set_with_indices(X, y, indices, train_ratio):
    """带索引的数据集分割函数"""
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, indices, train_size=train_ratio, random_state=345, stratify=y)
    return X_train, X_test, y_train, y_test, indices_train, indices_test


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


def generater(X_1, X_2, Y, batchsize, train_ratio, windowSize):
    # 为第一个时间点创建patch
    alldataX1, alldataY1, indices1 = create_patches(X_1, Y, window_size=windowSize)
    alldataX1 = np.transpose(alldataX1, (0, 3, 1, 2))  # (batchsize, channel, patchsize, patchsize)

    # 分割训练集和测试集，包括索引
    X1_train, X1_test, y_train, y_test, indices_train, indices_test = split_train_test_set_with_indices(
        alldataX1, alldataY1, indices1, train_ratio)

    # 为第二个时间点创建patch
    alldataX2, _, indices2 = create_patches(X_2, Y, window_size=windowSize)
    alldataX2 = np.transpose(alldataX2, (0, 3, 1, 2))

    # 对第二个时间点使用相同的索引分割
    X2_train, X2_test, _, _, _, _ = split_train_test_set_with_indices(
        alldataX2, alldataY1, indices2, train_ratio)

    # 处理训练集数据（过滤掉标签为2的样本）
    mask_train = y_train != 2
    X1_train = X1_train[mask_train]
    X2_train = X2_train[mask_train]
    y_train = y_train[mask_train]
    indices_train = indices_train[mask_train]  # 保留有效索引

    # 处理测试集数据（过滤掉标签为2的样本）
    mask_test = y_test != 2
    X1_test = X1_test[mask_test]
    X2_test = X2_test[mask_test]
    y_test = y_test[mask_test]
    indices_test = indices_test[mask_test]  # 保留有效索引

    print("训练集y_train的标签数量", Counter(y_train.flatten()))
    print("测试集y_test的标签数量", Counter(y_test.flatten()))

    ALL_SIZE = alldataX1.shape[0]
    TRAIN_SIZE = X1_train.shape[0]
    TEST_SIZE = X1_test.shape[0]

    # 转换为张量 - 关键修改：确保标签为LongTensor
    X1_train_tensor = torch.from_numpy(X1_train).type(torch.FloatTensor)
    X2_train_tensor = torch.from_numpy(X2_train).type(torch.FloatTensor)
    Y_train_tensor = torch.from_numpy(y_train).type(torch.LongTensor)  # 确保为Long类型
    indices_train_tensor = torch.from_numpy(indices_train).type(torch.LongTensor)

    X1_test_tensor = torch.from_numpy(X1_test).type(torch.FloatTensor)
    X2_test_tensor = torch.from_numpy(X2_test).type(torch.FloatTensor)
    Y_test_tensor = torch.from_numpy(y_test).type(torch.LongTensor)  # 确保为Long类型
    indices_test_tensor = torch.from_numpy(indices_test).type(torch.LongTensor)

    # 创建包含索引的TensorDataset
    torch_train = TensorDataset(X1_train_tensor, X2_train_tensor, Y_train_tensor, indices_train_tensor)
    torch_test = TensorDataset(X1_test_tensor, X2_test_tensor, Y_test_tensor, indices_test_tensor)

    print("Y_train_tensor.shape", Y_train_tensor.shape)
    print("Y_test_tensor.shape", Y_test_tensor.shape)

    train_iter = DataLoader(
        dataset=torch_train,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0
    )

    test_iter = DataLoader(
        dataset=torch_test,
        batch_size=batchsize,
        shuffle=False,
        num_workers=0
    )

    # 创建包含所有数据的DataLoader
    torch_all = TensorDataset(
        torch.cat((X1_train_tensor, X1_test_tensor), dim=0),
        torch.cat((X2_train_tensor, X2_test_tensor), dim=0),
        torch.cat((Y_train_tensor, Y_test_tensor), dim=0),
        torch.cat((indices_train_tensor, indices_test_tensor), dim=0)
    )

    total_iter = DataLoader(
        dataset=torch_all,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0
    )

    # 返回所有样本的原始标签和索引（新增）
    all_labels = alldataY1
    all_indices = indices1

    return TRAIN_SIZE, TEST_SIZE, train_iter, test_iter, total_iter, all_labels, all_indices
