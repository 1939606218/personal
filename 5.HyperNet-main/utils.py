import torch.nn as nn
import random
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def initNetParams_v2(net):
    # Init net parameters
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def diff_RX(img1, img2):
    """
    Calculate the Diff-RX change detection map between two images.
    """
    H, W, C = img1.shape
    img1_2d = np.reshape(img1, [H * W, C])
    img2_2d = np.reshape(img2, [H * W, C])
    diff = np.absolute(img1_2d - img2_2d)
    diff_mean = np.mean(diff, axis=0)

    diff_cov = np.cov(diff, rowvar=False)

    # Add a small regularization term to avoid singular matrix issues
    epsilon = 1e-6
    diff_cov += np.eye(C) * epsilon  # Add a small value to diagonal to ensure the matrix is invertible

    diff_mean0 = diff - diff_mean  # [H*W, C]
    T1 = np.matmul(diff_mean0, np.linalg.inv(diff_cov))  # [H*W, C]
    T2 = np.sum(T1 * diff_mean0, axis=1)  # [H*W]

    T2 = np.reshape(T2, [H, W])  # Reshape to the original image size
    plt.figure('Diff_RX')
    plt.imshow(T2, cmap='hot')

    return T2

def compute_kmeans_threshold(MSE_result, k=2):
    """
    对 MSE 结果进行 K 均值聚类并返回阈值。

    参数:
        MSE_result (numpy.ndarray): MSE 计算后的结果，大小为 [H, W]。
        k (int): 聚类簇的数量，默认 2（变化区域和非变化区域）。

    返回:
        threshold (float): 通过 K 均值聚类计算出来的阈值。
    """
    # 将 MSE_result 扁平化为一维数组，以适应 K 均值聚类
    MSE_result_flattened = MSE_result.flatten().reshape(-1, 1)

    # 使用 K 均值聚类来对 MSE_result 进行分类
    kmeans = KMeans(n_clusters=k, random_state=0).fit(MSE_result_flattened)

    # 获取聚类中心
    cluster_centers = kmeans.cluster_centers_.flatten()

    # 选择聚类中心的中位数作为阈值
    threshold = np.median(cluster_centers)

    print(f"Cluster Centers: {cluster_centers}")
    print(f"Selected Threshold: {threshold}")

    return threshold

# for evaluating the performance of the anomaly change detection result
def plot_roc(predict, ground_truth):
    """
    INPUTS:
     predict - anomalous change intensity map
     ground_truth - 0or1
    OUTPUTS:
     X, Y for ROC plotting
     auc
    """
    max_value = np.max(ground_truth)
    if max_value != 1:
        ground_truth = ground_truth / max_value

    # initial point（1.0, 1.0）
    x = 1.0
    y = 1.0
    hight_g, width_g = ground_truth.shape
    hight_p, width_p = predict.shape
    if hight_p != hight_g:
        predict = np.transpose(predict)

    ground_truth = ground_truth.reshape(-1)
    predict = predict.reshape(-1)
    # compuate the number of positive and negagtive pixels of the ground_truth
    pos_num = np.sum(ground_truth == 1)
    neg_num = np.sum(ground_truth == 0)
    # step in axis of  X and Y
    x_step = 1.0 / neg_num
    y_step = 1.0 / pos_num
    # ranking the result map
    index = np.argsort(list(predict))
    ground_truth = ground_truth[index]
    """ 
    for i in ground_truth:
     when ground_truth[i] = 1, TP minus 1，one y_step in the y axis, go down
     when ground_truth[i] = 0, FP minus 1，one x_step in the x axis, go left
    """
    X = np.zeros(ground_truth.shape)
    Y = np.zeros(ground_truth.shape)
    for idx in range(0, hight_g * width_g):
        if ground_truth[idx] == 1:
            y = y - y_step
        else:
            x = x - x_step
        X[idx] = x
        Y[idx] = y

    auc = -np.trapz(Y, X)
    if auc < 0.5:
        auc = -np.trapz(X, Y)
        t = X
        X = Y
        Y = t
    print('auc: ', auc)
    return X, Y, auc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def zz(seed):
    print('---------------everything will be ok-------------')
    print('current seed:', seed)


