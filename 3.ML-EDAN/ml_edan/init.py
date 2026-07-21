# -*- coding:utf-8 -*-

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
from einops import reduce
import torch.utils.data as Data
import torch.optim.lr_scheduler
import torch.nn.functional as F
## GLOBAL VARIABLES
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(2021)


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        dist = F.pairwise_distance(x1, x2)
        total_loss = (1 - y) * torch.pow(dist, 2) + \
                     y * torch.pow(torch.clamp_min_(self.margin - dist, 0), 2)
        loss = torch.mean(total_loss)
        return loss


def loadData(names):
    # data_path = os.path.join(r'/home/ghm/ZYY/data')
    data_path = os.path.join(r'D:\Program Files (x86)\Anaconda\jupyter_path\dataset', 'datasets')
    if names == 'river':
        data1 = sio.loadmat(os.path.join(data_path, 'River_before.mat'))['river_before']
        data2 = sio.loadmat(os.path.join(data_path, 'River_after.mat'))['river_after']
        labels = sio.loadmat(os.path.join(data_path, 'groundtruth.mat'))['lakelabel_v1']
        labels[labels==255]=1
        #labels = sio.loadmat(os.path.join(data_path, 'ground_truth.mat')).keys()
        #print(labels)
    if names == 'farm':
        data1 = sio.loadmat(os.path.join(data_path, 'farm06.mat'))['imgh']
        data2 = sio.loadmat(os.path.join(data_path, 'farm07.mat'))['imghl']
        labels = sio.loadmat(os.path.join(data_path, 'label.mat'))['label']

    if names == 'Hermiston':
        data1 = sio.loadmat(os.path.join(data_path, 'hermiston2004.mat'))['HypeRvieW']
        data2 = sio.loadmat(os.path.join(data_path, 'hermiston2007.mat'))['HypeRvieW']
        labels = sio.loadmat(os.path.join(data_path, 'label.mat'))['label']
    # if names == 'bayArea':
    #     data_path = os.path.join(r'dataset', 'bayArea')
    #     data1 = sio.loadmat(os.path.join(data_path, 'Bay_Area_2013.mat'))['HypeRvieW']
    #     data2 = sio.loadmat(os.path.join(data_path, 'Bay_Area_2015.mat'))['HypeRvieW']
    #     labels = sio.loadmat(os.path.join(data_path, 'rdChangesHermiston_5classes.mat'))['gt5clasesHermiston']
    return data1, data2, labels

def set_train_sample(x, y, pos, neg):
    rand_perm = np.random.permutation(y.shape[0])
    new_x = x[rand_perm, :, :, :]
    new_y = y[rand_perm]

    train_x0 = new_x[new_y == 0, :, :, :][:neg]
    train_y0 = new_y[new_y == 0][:neg]
    train_x1 = new_x[new_y == 1, :, :, :][:pos]
    train_y1 = new_y[new_y == 1][:pos]

    test_x0 = new_x[new_y == 0, :, :, :][neg:]
    test_y0 = new_y[new_y == 0][neg:]
    test_x1 = new_x[new_y == 1, :, :, :][pos:]
    test_y1 = new_y[new_y == 1][pos:]

    x_train = np.concatenate((train_x0, train_x1))
    y_train = np.concatenate((train_y0, train_y1))
    x_test = np.concatenate((test_x0, test_x1))
    y_test = np.concatenate((test_y0, test_y1))
    return  x_train, x_test, y_train, y_test

#划分path块的方法
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
            patchs_labels[patch_index] = y[r - margin, c - margin] + 1
            patch_index = patch_index + 1

    if remove_zero_labels:
        patches_data = patches_data[patchs_labels > 0, :, :, :]
        patchs_labels = patchs_labels[patchs_labels > 0]
        patchs_labels -= 1
    return patches_data, patchs_labels


def split_train_test_set(X, y, train_ratio):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=345,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def normalization(X, type=1):


    if type == 1:
        x = np.zeros(shape=X.shape, dtype='float32')
        for i in range(X.shape[2]):
            temp = X[:, :, i]
            mean = np.mean(temp)
            std = np.std(temp)
            x[:, :, i] = ((temp - mean) / std)
        return x

    elif type == 2:
        x = np.zeros(shape=X.shape, dtype='float32')
        for i in range(X.shape[2]):
            min = np.min(X, axis=-1)
            max = np.max(X, axis=-1)
            x[:, :, i] = (X[:, :, i] - min) / (max-min)

        return x

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

def reports(test_iter, net, device):
    y_test = []
    y_pred = []
    with torch.no_grad():
        for step, (x1, x2, x3, x4, y) in enumerate(test_iter):
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)
            y = y.to(device)
            net.eval()
            y_hat = net(x1, x2, x3, x4)
            y_pred.extend(y_hat.cpu().argmax(dim=-1))
            y_test.extend(y.cpu())
    classification = classification_report(np.array(y_test), np.array(y_pred), digits=2)
    oa = accuracy_score(np.array(y_test), np.array(y_pred))  # 计算OA
    confusion = confusion_matrix(np.array(y_test), np.array(y_pred))  # 计算confusion
    each_acc, aa = AA_andEachClassAccuracy(confusion)  # 计算each_acc和aa
    kappa = cohen_kappa_score(np.array(y_test), np.array(y_pred))  # 计算kappa

    return classification, confusion, oa * 100, aa * 100, kappa * 100


def generater(X_1, X_2, Y, batchsize, train_ratio, windowSize):

    alldataX1, alldataY = create_patches(X_1, Y, window_size=windowSize)
    alldataX1 = np.transpose(alldataX1, (0, 3, 1, 2))
    X1_train, X1_test, y_train, y_test = set_train_sample(alldataX1, alldataY, pos=4400, neg=8800)
    # X1_train, X1_test, y_train, y_test = split_train_test_set(alldataX1, alldataY,train_ratio)
    x1_train_vit = X1_train.reshape(X1_train.shape[0], windowSize*windowSize, X1_train.shape[1])
    x1_test_vit = X1_test.reshape(X1_test.shape[0], windowSize * windowSize, X1_test.shape[1])

    alldataX2, alldataY = create_patches(X_2, Y, window_size=windowSize)
    alldataX2 = np.transpose(alldataX2, (0, 3, 1, 2))
    X2_train, X2_test, y_train, y_test = set_train_sample(alldataX2, alldataY, pos=4400, neg=8800)
    # X2_train, X2_test, y_train, y_test = split_train_test_set(alldataX2, alldataY,train_ratio)
    x2_train_vit = X2_train.reshape(X2_train.shape[0], windowSize * windowSize, X2_train.shape[1])
    x2_test_vit = X2_test.reshape(X2_test.shape[0], windowSize * windowSize, X2_test.shape[1])

    ALL_SIZE = alldataX1.shape[0]
    TRAIN_SIZE = X1_train.shape[0]
    TEST_SIZE = X1_test.shape[0]

    X1_train_tensor = torch.from_numpy(X1_train).type(torch.FloatTensor)
    X2_train_tensor = torch.from_numpy(X2_train).type(torch.FloatTensor)
    Y_train_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)
    X1_train_vit_tensor = torch.from_numpy(x1_train_vit).type(torch.FloatTensor)
    x2_train_vit_tensor = torch.from_numpy(x2_train_vit).type(torch.FloatTensor)
    # Y_train_tensor = Y_train_tensor.to(torch.int64)
    # Y_train_tensor = F.one_hot(Y_train_tensor, num_classes=2)

    X1_test_tensor = torch.from_numpy(X1_test).type(torch.FloatTensor)
    X2_test_tensor = torch.from_numpy(X2_test).type(torch.FloatTensor)
    Y_test_tensor = torch.from_numpy(y_test).type(torch.FloatTensor)
    x1_test_vit_tensor = torch.from_numpy(x1_test_vit).type(torch.FloatTensor)
    x2_test_vit_tensor = torch.from_numpy(x2_test_vit).type(torch.FloatTensor)
    # Y_test_tensor = Y_test_tensor.to(torch.int64)
    # Y_test_tensor = F.one_hot(Y_test_tensor, num_classes=2)

    X1_aLL = torch.from_numpy(alldataX1).type(torch.FloatTensor)
    X2_aLL = torch.from_numpy(alldataX2).type(torch.FloatTensor)
    Y_all  = torch.from_numpy(alldataY).type(torch.FloatTensor)

    torch_train = Data.TensorDataset(X1_train_tensor, X2_train_tensor, X1_train_vit_tensor, x2_train_vit_tensor,Y_train_tensor)
    torch_test = Data.TensorDataset(X1_test_tensor, X2_test_tensor, x1_test_vit_tensor, x2_test_vit_tensor, Y_test_tensor)
    torch_all = Data.TensorDataset(X1_aLL, X2_aLL, Y_all)

    train_iter = Data.DataLoader(
        dataset=torch_train,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0
    )

    test_iter = Data.DataLoader(
        dataset=torch_test,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0
    )

    all_iter = Data.DataLoader(
        dataset=torch_all,
        batch_size=batchsize,
        shuffle=False,
        num_workers=0
    )
    torch.save(train_iter, './dataset/train_iter.pt')
    torch.save(test_iter, './dataset/test_iter.pt')
    torch.save(all_iter, './dataset/all_iter.pt')
    return TRAIN_SIZE, TEST_SIZE,ALL_SIZE, train_iter, test_iter, all_iter


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


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

def l2_penalty(w):
    return (w ** 2).sum / 2

