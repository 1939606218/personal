import torch

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import random
from operator import truediv
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score
from collections import Counter
import torch.utils.data as Data
import torch.optim.lr_scheduler
from torch import nn
import torch.nn.functional as F
from Global import *
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score


## GLOBAL VARIABLES
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random.seed(2021)

# def convert_to_one_hot(Y, C):
#     Y = np.eye(C)[Y.reshape(-1)]
#     return Y

def loadData(names):
    if names == 'hermiston':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2004.mat')['HypeRvieW']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2007.mat')['HypeRvieW']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\label.mat')['label']

    if names == 'river':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_before.mat')['river_before']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_after.mat')['river_after']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\groundtruth.mat')['lakelabel_v1']
        labels[labels == 255] = 1

    if names == 'farmland':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm06.mat')['imgh']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm07.mat')['imghl']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\label.mat')['label']

    if names == 'santaBarbara':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_2013.mat')['HypeRvieW']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_2014.mat')['HypeRvieW']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_gtChanges.mat')['HypeRvieW']

        # 修改标签映射：0→2，1→1，2→0
        labels = np.select(
            [labels == 0, labels == 1, labels == 2],
            [2, 1, 0],
            default=labels  # 处理其他可能的值
        )

    if names == 'bayArea':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2013.mat')['HypeRvieW']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2015.mat')['HypeRvieW']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\bayArea_gtChanges2.mat')['HypeRvieW']

        # 修改标签映射：0→2，1→1，2→0
        labels = np.select(
            [labels == 0, labels == 1, labels == 2],
            [2, 1, 0],
            default=labels  # 处理其他可能的值
        )


    print("data1.shape", data1.shape)
    print("data2.shape", data2.shape)
    print("labels.shape", labels.shape)

    # 3. 直接统计 labels 中 0 和 1 的数量
    unchanged_count = np.sum(labels == 0)  # 未变化像素数量（标签=0）
    changed_count = np.sum(labels == 1)  # 变化像素数量（标签=1）
    unknown_count = np.sum(labels == 2)     #不确定的标签
    sum = unchanged_count + changed_count + unknown_count
    print(f"变化像素对数量: {changed_count}")
    print(f"未变化像素对数量: {unchanged_count}")
    print(f"不确定像素对数量: {unknown_count}")

    WC = round(0.5 / (changed_count / sum), 4)
    WU = round(0.5 / (unchanged_count / sum), 4)
    print(f"WC:{WC}")
    print(f"WU:{WU}")

    return data1, data2, labels ,WC ,WU


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1, WU=None, WC=None):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.gamma = 2.0
        self.WU = WU
        self.WC = WC

    def forward(self, output, label):
        loss_contrastive = (torch.mean((1 - label) * (torch.pow(output, 2) * 0.5) * self.WU) +
                            torch.mean(label * (torch.pow(torch.clamp(self.margin - output, min=0.0), 2) * 0.5) * self.WC))
        # loss_contrastive = (torch.mean((1 - label) * (torch.pow(output, 2) * 0.5) * 1.7235) +
        #                     torch.mean(label * (torch.pow(torch.clamp(self.margin - output, min=0.0), 2) * 0.5) * 0.7043))
        return loss_contrastive




# def set_train_sample(x, y, pos, neg):
#     np.random.seed(100)
#     rand_perm = np.random.permutation(y.shape[0])   #打乱111583
#     new_x = x[rand_perm, :, :, :]
#     new_y = y[rand_perm]    #new_x,new_y 保证一一对应
#
#     train_x0 = new_x[new_y == 0, :, :, :][:neg]     #取未变化的训练集2500个
#     train_y0 = new_y[new_y == 0][:neg]      #取未变化的训练集标签2500个
#     train_x1 = new_x[new_y == 1, :, :, :][:pos]     #取变化的训练集1250个
#     train_y1 = new_y[new_y == 1][:pos]      #取变化的训练集标签1250个
#
#
#     test_x0 = new_x[new_y == 0, :, :, :][neg:]      #取未变化的测试集从2500个开始
#     test_y0 = new_y[new_y == 0][neg:]       #取未变化的测试集标签从2500个开始
#     test_x1 = new_x[new_y == 1, :, :, :][pos:]      #取变化的测试集从1250个开始
#     test_y1 = new_y[new_y == 1][pos:]       #取变化的测试集标签从1250开始
#
#     x_train = np.concatenate((train_x0, train_x1))      #连接两个数据集
#     y_train = np.concatenate((train_y0, train_y1))
#     x_test = np.concatenate((test_x0, test_x1))
#     y_test = np.concatenate((test_y0, test_y1))
#     return  x_train, x_test, y_train, y_test

#划分patch块的方法
def pad_with_zeros(X, margin=2):        #进行零填充操作
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

    # if remove_zero_labels:
    #     patches_data = patches_data[patchs_labels > 0, :, :, :]
    #     patchs_labels = patchs_labels[patchs_labels > 0]
    #     patchs_labels -= 1
    return patches_data, patchs_labels


def split_train_test_set(X, y, train_ratio):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=train_ratio,
                                                        random_state=345,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def normalization(X, type=1):
    """
    normalization type  1: map to [0, 1]
    normalization type  2: map to zero mean and one std
    :param X:
    :param type:
    :return: normalization X
    """
    if type == 1:
        mu = np.mean(X, 0)
        X_norm = X - mu
        sigma = np.std(X_norm, 0)
        X_norm = X_norm / sigma
        return X_norm

    elif type == 2:     #映射到 [0, 1] 区间
        minX = np.min(X, 0)
        maxX = np.max(X, 0)
        X_norm = X - minX
        X_norm = X_norm / (maxX - minX)
        return X_norm

    elif type == 3:             #MinMaxScaler用于对数据进行归一化处理，将数据的特征值映射到指定的区间[0,1]
        X_reshape = X.reshape((-1, X.shape[-1]))
        transfer = MinMaxScaler()
        X_reshape = transfer.fit_transform(X_reshape)
        X = X_reshape.reshape((X.shape[0], X.shape[1], X.shape[2]))
        return X


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)  # 获取confusion_matrix的主对角线所有数值
    list_raw_sum = np.sum(confusion_matrix, axis=1)  # 将主对角线所有数求和
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))  # list_diag/list_raw_sum  对角线各个数字/对角线所有数字的总和
    average_acc = np.mean(each_acc)  #
    return np.round(each_acc, 4), average_acc

def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum = 0.0
    test_l_sum = 0
    with torch.no_grad():
        for step, (X1, X2, y) in enumerate(data_iter):
            # X = X.permute(0, 3, 1, 2)
            x1 = X1.to(device)
            x2 = X2.to(device)
            y = y.to(device)
            net.eval()
            y_hat1, y_hat2, y_hat = net(x1, x2)
            l = loss(y_hat1, y_hat2, y.long())
            test_l_sum += l.cpu().item()
            acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
    return acc_sum / len(data_iter.dataset), test_l_sum / len(data_iter.dataset)


def reports(test_iter, target_names, net, batchsize=64):
    """评估模型并返回详细指标（包括混淆矩阵元素）"""
    y_test = torch.zeros(32, 2).to(device)  # 假设二分类问题
    y_pred = torch.zeros(32, 2).to(device)

    with torch.no_grad():
        for step, (X1, X2, y) in enumerate(test_iter):
            x1 = X1.to(device)
            x2 = X2.to(device)
            y = y.to(device)
            net.eval()
            y_hat = net(x1, x2)
            y_pred = torch.cat([y_pred, y_hat], dim=0)
            y_test = torch.cat([y_test, y], dim=0)

    # 去除初始填充的零
    y_pred = y_pred[batchsize::]
    y_test = y_test[batchsize::]

    # 转换为numpy数组并获取类别标签
    y_pred_np = y_pred.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    y_pred_labels = y_pred_np.argmax(-1)
    y_test_labels = y_test_np.argmax(-1)

    # 计算基本指标
    oa = accuracy_score(y_test_labels, y_pred_labels)
    kappa = cohen_kappa_score(y_test_labels, y_pred_labels)
    precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
    f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')

    # 计算混淆矩阵并获取TN, FP, FN, TP
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

    # 检查是否为二分类问题
    if conf_matrix.shape == (2, 2):
        TN, FP, FN, TP = conf_matrix.ravel()
    else:
        # 多分类场景下，无法直接计算TN/FP/FN/TP
        TN, FP, FN, TP = None, None, None, None

    # 计算每类别的精确率、召回率和F1分数
    per_class_precision = precision_score(y_test_labels, y_pred_labels, average=None)
    per_class_recall = recall_score(y_test_labels, y_pred_labels, average=None)
    per_class_f1 = f1_score(y_test_labels, y_pred_labels, average=None)

    # 计算样本分布
    class_distribution = np.bincount(y_test_labels)

    return {
        'oa': oa,
        'kappa': kappa,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'TP': TP,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'class_distribution': class_distribution
    }

def generater(X_1, X_2, Y, batchsize, train_ratio, windowSize):
    alldataX1, alldataY1 = create_patches(X_1, Y, window_size=windowSize)   # 把像素变成patch块,那么alldataX1的形状会变成(batchsize, patch_size, patch_size, channel
    alldataX1 = np.transpose(alldataX1, (0, 3, 1, 2))       # （batchsize，channel，patchsize，patchsize）

    #X1_train, X1_test, y_train, y_test = set_train_sample(alldataX1, alldataY1, pos, neg) # 划分训练集，测试集
    X1_train, X1_test, y_train, y_test = split_train_test_set(alldataX1, alldataY1, train_ratio)

    alldataX2, alldataY2 = create_patches(X_2, Y, window_size=windowSize)
    alldataX2 = np.transpose(alldataX2, (0, 3, 1, 2))
    #X2_train, X2_test, y_train, y_test = set_train_sample(alldataX2, alldataY1, pos, neg)
    X2_train, X2_test, y_train, y_test = split_train_test_set(alldataX2, alldataY1, train_ratio)

    # 处理训练集数据
    mask_train = y_train != 2
    X1_train = X1_train[mask_train]
    X2_train = X2_train[mask_train]
    y_train = y_train[mask_train]

    # 处理测试集数据
    mask_test = y_test != 2
    X1_test = X1_test[mask_test]
    X2_test = X2_test[mask_test]
    y_test = y_test[mask_test]

    print("训练集y_train的标签数量",Counter(y_train.flatten()))
    print("测试集y_test的标签数量",Counter(y_test.flatten()))
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    ALL_SIZE = alldataX1.shape[0]
    TRAIN_SIZE = X1_train.shape[0]
    TEST_SIZE = X1_test.shape[0]        #训练集和测试集的样本数量

    X1_train_tensor = torch.from_numpy(X1_train).type(torch.FloatTensor)    #将 numpy 格式转换成 tensor 形式
    X2_train_tensor = torch.from_numpy(X2_train).type(torch.FloatTensor)
    Y_train_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)

    X1_test_tensor = torch.from_numpy(X1_test).type(torch.FloatTensor)
    X2_test_tensor = torch.from_numpy(X2_test).type(torch.FloatTensor)
    Y_test_tensor = torch.from_numpy(y_test).type(torch.FloatTensor)

    Y_train_tensor = Y_train_tensor.to(torch.int64)
    Y_train_tensor = F.one_hot(Y_train_tensor)

    Y_test_tensor = Y_test_tensor.to(torch.int64)
    Y_test_tensor = F.one_hot(Y_test_tensor)

    torch_train = Data.TensorDataset(X1_train_tensor, X2_train_tensor, Y_train_tensor)
    torch_test = Data.TensorDataset(X1_test_tensor, X2_test_tensor, Y_test_tensor)
    #Data.TensorDataset可以用来对tensor进行打包。
    print("Y_train_tensor.shape",Y_train_tensor.shape)
    print("Y_test_tensor.shape", Y_test_tensor.shape)
    train_iter = Data.DataLoader(
        dataset=torch_train,
        batch_size=batchsize,
        shuffle=True,       #决定是否在每次迭代时打乱数据顺序
        num_workers=0
    )

    test_iter = Data.DataLoader(
        dataset=torch_test,
        batch_size=batchsize,
        shuffle=False,
        num_workers=0
    )



    return TRAIN_SIZE, TEST_SIZE, train_iter, test_iter

# def pixel_select(Y):
#     random.seed(2021)
#     test_pixels = Y.copy()  # 复制Y到test_pixels
#     kinds = np.unique(Y).shape[0]
#     # print(kinds)
#     for i in range(kinds):  # i从0-15
#         num = np.sum(Y == i)
#         train_num = [2500, 1250]
#         temp1 = np.where(Y == i)
#         temp2 = random.sample(range(num), train_num[i])
#         for i in temp2:
#             test_pixels[temp1[0][temp2], temp1[1][temp2]] = 2  # 除去训练集样本
#
#     train_pixels = Y - test_pixels
#     return train_pixels, test_pixels

# def GetImageCubes(input_data, pixels_select, windowSize=11):
#     random.seed(2021)
#     Band = input_data.shape[2]
#     kind = np.unique(pixels_select).shape[0] - 1
#     paddingdata = np.pad(input_data, ((30, 30), (30, 30), (0, 0)), "edge")
#     paddinglabel = np.pad(pixels_select, ((30, 30), (30, 30)), "edge")
#     pixel = np.where(paddinglabel != 2)
#     num = np.sum(pixels_select != 2)
#     batch_out = np.zeros([num, windowSize, windowSize, Band])
#     batch_label = np.zeros([num, kind])
#     for i in range(num):
#         row_start = pixel[0][i] - windowSize // 2
#         row_end = pixel[0][i] + windowSize // 2 + 1
#         col_start = pixel[1][i] - windowSize // 2
#         col_end = pixel[1][i] + windowSize // 2 + 1
#         batch_out[i, :, :, :] = paddingdata[row_start:row_end, col_start:col_end, :]
#         temp = (paddinglabel[pixel[0][i], pixel[1][i]] - 1)
#         batch_label[i, temp] = 1
#     batch_out = batch_out.swapaxes(1, 3)
#     batch_label = np.argmax(batch_label, axis=-1)
#     # batch_out = batch_out[:, :, :, :, np.newaxis]           # np.newaxis:增加维度
#     # print('batch_out.shape:', batch_out.shape)
#     return batch_out, batch_label

#
# def aa_and_each_accuracy(confusion_matrix):
#     list_diag = np.diag(confusion_matrix)
#     list_raw_sum = np.sum(confusion_matrix, axis=1)
#     each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
#     average_acc = np.mean(each_acc)
#     return each_acc, average_acc


def weight_init(layer):
    # if isinstance(m, nn.Conv2d):  # 如果模型中是二维卷积层Conv2d那么就使用xavier_uniform 初始化
    #     init.xavier_uniform_(m.weight.data)
    #     init.constant_(m.bias.data, 0.1)  # 初始化偏置向量b为常数0.1
    # elif isinstance(m, nn.Linear):  # 如果模型中是全连接层，那么就使用如下初始化方式
    #     m.weight.data.normal_(0, 0.01)
    #     m.bias.data.zero_()
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)

    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)

# def l2_penalty(w):
#     return (w ** 2).sum / 2
