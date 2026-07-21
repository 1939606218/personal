# -*- coding:utf-8 -*-
# Author:Ding
import os
import random

import psutil
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import platform
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader


# set experiment seed
def set_seed(seed):
    # python
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 边界拓展：镜像
def mirror_hsi(height, width, band, input_normalize, patch = 5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band)).astype(np.float16)
    # 中心区域
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    # 左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    # 右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    # 上边镜像
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    # 下边镜像
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    # print("**************************************************")
    # print("patch is : {}".format(patch))
    # print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    # print("**************************************************")
    return mirror_hsi


# -------------------------------------------------------------------------------
# 获取patch的图像数据
def get_patches(data, img_height, img_width, channel, patch_size):
    """get patches"""
    # patch_size:the size of target pixel's neighborhood
    patches = np.empty([img_height * img_width, patch_size, patch_size, channel],
                       dtype = 'float16')  # img_height * img_width
    for i in range(img_height):
        for j in range(img_width):
            patches[i * img_width + j, ...] = data[i:i + patch_size, j:j + patch_size, :]

    # patches = (img_height * img_width, patch_size, patch_size, band_size)
    return patches


def normalization(X, epsilon=1e-8):
    """
    高效的归一化函数，处理除零情况
    """
    # 计算每通道的最小值和最大值
    x_min = np.min(X, axis=(0, 1), keepdims=True)
    x_max = np.max(X, axis=(0, 1), keepdims=True)
    # 避免除零
    denominator = np.maximum(x_max - x_min, epsilon)
    # 归一化
    return (X - x_min) / denominator


def make_data(dataset: str, patch_size=5, n_class=2, flag='train'):
    """读取数据——>将数据整理为B×b×N格式——>标准化——>分批处理避免内存溢出"""
    import platform
    global path, data_t1, data_t2, y

    if platform.system().lower() == 'windows':
        print("[Info]: Use Windows!")
        path = 'E:'
    elif platform.system().lower() == 'linux':
        print("[Info]: Use Linux!")
        path = '..'

    if dataset == 'farmland':
        data_t1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm06.mat')['imgh']
        data_t2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm07.mat')['imghl']
        y = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\label.mat')['label']

    if dataset == 'hermiston':
        data_t1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2004.mat')['HypeRvieW']
        data_t2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2007.mat')['HypeRvieW']
        y = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\label.mat')['label']

    elif dataset == 'river':
        data_t1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_before.mat')['river_before']
        data_t2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_after.mat')['river_after']
        y = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\groundtruth.mat')['lakelabel_v1']
        y[y == 255] = 1
    elif dataset == 'BayArea':
        data_t1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2013.mat')['HypeRvieW']
        data_t2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2015.mat')['HypeRvieW']
        y = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\bayArea_gtChanges2.mat')['HypeRvieW']
    elif dataset == 'Barbara':
        data_t1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_2013.mat')['HypeRvieW']
        data_t2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_2014.mat')['HypeRvieW']
        y = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_gtChanges.mat')['HypeRvieW']

    img_height, img_width, channel = data_t1.shape
    print(f"图像尺寸: {img_height}×{img_width}, 通道数: {channel}")

    # ---------------------- 2. 全局归一化 ----------------------
    # 计算两个时相的全局最小/最大值（用于归一化）
    global_min = np.minimum(np.min(data_t1, axis=(0, 1)), np.min(data_t2, axis=(0, 1)))
    global_max = np.maximum(np.max(data_t1, axis=(0, 1)), np.max(data_t2, axis=(0, 1)))
    global_min = global_min.reshape(1, 1, -1)  # 扩展为[1,1,C]以支持广播
    global_max = global_max.reshape(1, 1, -1)
    epsilon = 1e-8
    denominator = np.maximum(global_max - global_min, epsilon)  # 避免除零

    # 归一化（保持浮点精度，用float32）
    data_t1_normalized = (data_t1 - global_min) / denominator
    data_t2_normalized = (data_t2 - global_min) / denominator

    # ---------------------- 3. 预生成全局镜像（关键优化！） ----------------------
    padding = patch_size // 2  # 镜像填充大小（补丁半长）
    # 对称填充，模拟镜像效果（后续直接切片，无需重复填充）
    data_t1_mirrored = np.pad(
        data_t1_normalized,
        pad_width=((padding, padding), (padding, padding), (0, 0)),
        mode='symmetric'
    )
    data_t2_mirrored = np.pad(
        data_t2_normalized,
        pad_width=((padding, padding), (padding, padding), (0, 0)),
        mode='symmetric'
    )

    # ---------------------- 4. 标签预处理 ----------------------
    y_reshaped = np.reshape(y, (-1,))  # 展平为1D数组
    if dataset in ['BayArea', 'Barbara']:
        # 0=未标记，1=变化，2=未变化 → 提取非零标签，并调整标签值
        labeled = np.argwhere(y_reshaped != 0).flatten()  # 只保留非零标签的索引
        y_reshaped = 2 - y_reshaped  # 转换为：2→0（未变化），1→1（变化）
    else:
        labeled = np.arange(len(y_reshaped))  # 所有样本都作为标记样本

    total_samples = len(labeled)
    if total_samples == 0:
        raise ValueError("没有标记样本！请检查标签文件或数据集。")
    print(f"数据集 {dataset} 标记样本数: {total_samples}")

    # ---------------------- 5. 自动计算Batch Size（适配内存） ----------------------
    available_ram = psutil.virtual_memory().available / (1024 ** 3)  # 可用内存（GB）
    # 单个补丁内存占用：patch×patch×channel×2（时相）×2（float16字节）
    single_patch_bytes = patch_size * patch_size * channel * 2 * 2
    max_allowed_bytes = available_ram * 0.7 * (1024 ** 3)  # 仅用70%内存，预留安全空间
    estimated_batch_size = int(max_allowed_bytes / single_patch_bytes)
    # 限制合理范围（避免过小或过大）
    batch_size = max(1000, min(estimated_batch_size, 100000))
    print(f"自动适配批大小: {batch_size} (可用内存: {available_ram:.2f}GB)")

    # ---------------------- 6. 分批生成补丁（预分配数组+进度条） ----------------------
    # 预分配大数组（直接填充，避免多次合并）
    patches_t1 = np.zeros((total_samples, patch_size, patch_size, channel), dtype=np.float16)
    patches_t2 = np.zeros((total_samples, patch_size, patch_size, channel), dtype=np.float16)

    # 初始化进度条
    progress_bar = tqdm(total=total_samples, desc="生成图像补丁", unit="样本")

    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_indices = labeled[batch_start:batch_end]
        # 转换为图像坐标(i, j)
        coords = [(i // img_width, i % img_width) for i in batch_indices]

        # 填充当前批次的补丁（直接从预生成的镜像中切片）
        for idx_in_batch, (i, j) in enumerate(coords):
            global_idx = batch_start + idx_in_batch  # 全局索引
            patches_t1[global_idx] = data_t1_mirrored[i:i + patch_size, j:j + patch_size, :]
            patches_t2[global_idx] = data_t2_mirrored[i:i + patch_size, j:j + patch_size, :]

        # 更新进度条
        progress_bar.update(len(coords))

    progress_bar.close()

    # ---------------------- 7. 生成模型输入 & 清理内存 ----------------------
    # 合并两个时相的补丁，调整维度为 [N, 2C, patch, patch]
    x = np.transpose(
        np.concatenate((patches_t1, patches_t2), axis=-1),
        (0, 3, 1, 2)  # 转换为 PyTorch 常用的 [N, C, H, W] 格式
    )

    # 提取标记样本的标签
    y = y_reshaped[labeled]

    # 释放大内存变量（可选，Python会自动回收，但显式释放更安全）
    del data_t1, data_t2, data_t1_mirrored, data_t2_mirrored, patches_t1, patches_t2

    # ✅ 关键验证：确保生成的样本数与labeled长度一致
    assert len(x) == len(labeled), f"样本数不匹配！x={len(x)}, labeled={len(labeled)}"
    assert len(y) == len(labeled), f"标签数不匹配！y={len(y)}, labeled={len(labeled)}"

    return x, y, labeled, img_height, img_width,  y_reshaped.copy()

class MyDataset(Dataset):
    def __init__(self, x, y = None, transform = None):
        if transform:
            self.x = x
        else:
            self.x = torch.FloatTensor(x)
        # label是需要LongTensor型
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


def split_train_val(x, y3, args):
    """ratio: the ratio of train data"""
    if not os.path.exists('./train_index'):
        os.mkdir(f'./train_index')
    ratio_str = "{:.2f}".format(args.ratio).replace('.', '')

    if not os.path.exists(f'./train_index/{args.dataset}-train-index-{ratio_str}-{args.seed}.npy'):
        train_x_set, val_x_set, train_y_set, val_y_set = train_test_split(x, y3, test_size=1 - args.ratio,
                                                                          random_state=args.seed,
                                                                          stratify=y3)
        train_index = get_train_index(x, train_x_set)
        np.save(f'./train_index/{args.dataset}-train-index-{ratio_str}-{args.seed}.npy', train_index)
    else:
        # 加载训练索引并确保其为整数类型
        train_index = np.load(f'./train_index/{args.dataset}-train-index-{ratio_str}-{args.seed}.npy').astype(np.int64)
        index = np.arange(0, len(y3))
        val_index = np.delete(index, train_index)
        train_x_set = x[train_index]
        train_y_set = y3[train_index]
        val_x_set = x[val_index]
        val_y_set = y3[val_index]

    return train_x_set, train_y_set, val_x_set, val_y_set


def get_train_index(data, data_train):
    print('Start get train index!')
    train_index = []
    pbar = tqdm(total = len(data_train), ncols = 0, desc = f"Processing", unit = "step")
    for i in range(len(data_train)):
        pbar.set_postfix(step = i + 1)
        pbar.update()
        for j in range(len(data)):
            if (data_train[i, :10, :10] == data[j, :10, :10]).all():
                train_index.append(j)
                break
    pbar.close()

    return train_index


# -------------------------------------------------------------------------------
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n = 1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# -------------------------------------------------------------------------------
def accuracy(output, target, topk = (1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


# -------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype = np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score

def calculate_metrics(y_true, y_pred):
    """
    计算分类模型的各项性能指标
    参数:
    y_true: 真实标签
    y_pred: 预测标签
    返回:
    多个评估指标和混淆矩阵元素
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 检查是否为二分类问题
    is_binary = len(set(y_true).union(set(y_pred))) == 2

    if is_binary:
        # 二分类场景
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        else:
            # 处理只有一个类别的特殊情况（实际应用中很少见）
            unique_labels = np.unique(y_true)
            if len(unique_labels) == 1 and unique_labels[0] == 0:
                TN = cm[0, 0]
                FP = 0
                FN = 0
                TP = 0
            else:
                TN = 0
                FP = 0
                FN = 0
                TP = cm[0, 0]

        oa = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

    else:
        # 多分类场景
        oa = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_true, y_pred)


        # 多分类场景下 TN、FP、FN、TP 无明确意义，设为 None
        TN, FP, FN, TP = None, None, None, None

    # 返回所有指标和混淆矩阵元素
    return oa, f1, precision, recall, kappa, TN, FP, FN, TP

# -------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, device):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        batch_pred = model(batch_data)

        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk = (1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre


def test_epoch(model, test_loader, device):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre


# -------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer, device):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk = (1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre

# Example
# data, target, labels = make_data('China', patch_size = 3, band_patch = 3)
