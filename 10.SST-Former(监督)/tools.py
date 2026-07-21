import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image
import numpy as np
import time
import os

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['farmland', 'river', 'Hermiston', 'BayArea', 'Barbara'],
                    help='dataset to use (可以在命令行指定，也可以由程序自动设置)')   
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=90, help='number of evaluation')
parser.add_argument('--patches', type=int, default=5, help='number of patches')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=200, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')   # 衰减系数
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
args = parser.parse_args()

# 选取GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True  # 使用确定性的卷积算法，保证相同的输入和网络配置，得到完全相同的输出结果
cudnn.benchmark = False     # 禁用自动调优，避免在输入数据大小变化时的性能损失

def load_dataset_data(dataset_name):
    """根据数据集名称加载相应的数据"""
    global data_t1, data_t2, data_label
    
    if dataset_name == 'farmland':
        data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm06.mat')['imgh']
        data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm07.mat')['imghl']
        data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\label.mat')['label']
        # 统计数据分布
        change_pixels = np.sum(data_label == 1)
        unchange_pixels = np.sum(data_label == 0)
        print('Farmland - 变化像素: {}, 未变化像素: {}'.format(change_pixels, unchange_pixels))

    elif args.dataset == 'river':
        data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_before.mat')['river_before']
        data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_after.mat')['river_after']
        data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\groundtruth.mat')['lakelabel_v1']
        data_label[data_label == 255] = 1
        # 统计数据分布
        change_pixels = np.sum(data_label == 1)
        unchange_pixels = np.sum(data_label == 0)
        print('River - 变化像素: {}, 未变化像素: {}'.format(change_pixels, unchange_pixels))

    elif args.dataset == 'Hermiston':
        data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2004.mat')['HypeRvieW']
        data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2007.mat')['HypeRvieW']
        data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\label.mat')['label']
        # 统计数据分布
        change_pixels = np.sum(data_label == 1)
        unchange_pixels = np.sum(data_label == 0)
        print('Hermiston - 变化像素: {}, 未变化像素: {}'.format(change_pixels, unchange_pixels))

    # BayArea和Barbara数据集
    elif args.dataset == 'BayArea':
        data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2013.mat')['HypeRvieW']
        data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2015.mat')['HypeRvieW']
        data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\bayArea_gtChanges2.mat')['HypeRvieW']
        # 修改标签映射：0→2，1→1，2→0
        data_label = np.select(
            [data_label == 0, data_label == 1, data_label == 2],
            [2, 1, 0],
            default=data_label  # 处理其他可能的值
        )
        # 统计数据分布
        change_pixels = np.sum(data_label == 1)
        unchange_pixels = np.sum(data_label == 0)
        uncertain_pixlels = np.sum(data_label == 2)
        print('BayArea - 变化像素: {}, 未变化像素: {},未确定的像素:{}'.format(change_pixels, unchange_pixels,uncertain_pixlels))


    elif args.dataset == 'Barbara':
        data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2013.mat')['HypeRvieW']
        data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2015.mat')['HypeRvieW']
        data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\bayArea_gtChanges2.mat')['HypeRvieW']
        # 修改标签映射：0→2，1→1，2→0
        data_label = np.select(
            [data_label == 0, data_label == 1, data_label == 2],
            [2, 1, 0],
            default=data_label  # 处理其他可能的值
        )
        # 统计数据分布
        change_pixels = np.sum(data_label == 1)
        unchange_pixels = np.sum(data_label == 0)
        uncertain_pixlels = np.sum(data_label == 2)
        print('BayArea - 变化像素: {}, 未变化像素: {},未确定的像素:{}'.format(change_pixels, unchange_pixels,uncertain_pixlels))


# 定位训练和测试样本(坐标)，使用随机抽样方式
def chooose_train_and_test_point(data_label, num_classes, train_ratio=0.01):
    number_train = []   # 存储训练数据中每个类别的样本数量
    number_test = []    # 存储测试数据中每个类别的样本数量
    total_pos_train = []
    total_pos_test = []
    
    np.random.seed(42)  # 设置随机种子，确保结果可重现
    
    # 只处理未变化(0)和变化(1)的样本，忽略未标记(2)的样本
    for i in range(2):  # i=0 表示未变化，i=1 表示变化
        class_indices = np.argwhere(data_label == i)  # 找出该类别的所有样本位置
        total_samples = len(class_indices)
        
        if total_samples > 0:
            # 计算训练集样本数量（1%）
            n_train = max(int(total_samples * train_ratio), 1)  # 确保至少有1个训练样本
            
            # 随机抽样
            train_indices = np.random.choice(total_samples, n_train, replace=False)
            train_samples = class_indices[train_indices]
            test_samples = np.delete(class_indices, train_indices, axis=0)
            
            # 添加到训练集和测试集
            total_pos_train.extend(train_samples)
            total_pos_test.extend(test_samples)
            
            # 记录样本数量
            number_train.append(len(train_samples))
            number_test.append(len(test_samples))
    
    # 转换为numpy数组并确保类型为int
    total_pos_train = np.array(total_pos_train).astype(int)
    total_pos_test = np.array(total_pos_test).astype(int)
    
    # 打印训练集和测试集的数据分布
    print("\n数据集分布情况:")
    print("训练集:")
    for i in range(len(number_train)):
        print(f"类别 {i}: {number_train[i]} 个样本")
    print("测试集:")
    for i in range(len(number_test)):
        print(f"类别 {i}: {number_test[i]} 个样本")
    print(f"总计 - 训练集: {sum(number_train)} 个样本, 测试集: {sum(number_test)} 个样本\n")
    
    return total_pos_train, total_pos_test, number_train, number_test
    #     训练集和测试集中所有类别样本的坐标 ；  训练集和测试集中每个类别的样本数量


# 边界拓展：镜像，通过复制图像边缘的像素到图像外围，从而扩大图像尺寸，在提取边缘像素的特征时特别有用
def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2
    mirror_hsi = np.zeros((height+2*padding, width+2*padding, band), dtype=float)   # 初始化镜像图像
    # 中心区域
    mirror_hsi[padding:(padding+height), padding:(padding+width), :] = input_normalize  # 将输入图像复制到mirror_hsi的中心区域
    # 左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding), i, :] = input_normalize[:, padding-i-1, :]
    # 右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding), width+padding+i, :] = input_normalize[:, width-1-i, :]
    # 上边镜像
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding*2-i-1, :, :]
    # 下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i, :, :] = mirror_hsi[height+padding-1-i, :, :]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


# 获取patch的图像数据，为高光谱图像中的每个像素或特定的像素点提取其邻域信息（局部区域提取方法）
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):   # 每一个patch
    x = point[i, 0]  # 样本横坐标
    y = point[i, 1]  # 样本纵坐标
    temp_image = mirror_image[x:(x+patch), y:(y+patch), :]  # 根据坐标和patch大小，提取出以(x, y)为左上角的patch x patch区域
    return temp_image   # temp_image包含了样本点周围的邻域像素，并保持了原始图像的波段信息


# 对HSI的每个样本提取其波段邻域信息，从而生成每个样本点周围的波段特征
def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    # x_train: [样本数量, patch, 波段数]；波段总数；波段邻域大小（每个样本点周围考虑的波段数量）
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)  # [样本数量, patch平方, 波段数]
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band), dtype=float)  # 存储每个样本的波段邻域信息
    # 中心区域
    x_train_band[:, nn*patch*patch:(nn+1)*patch*patch, :] = x_train_reshape
    # 左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i*patch*patch:(i+1)*patch*patch, :i+1] = x_train_reshape[:, :, band-i-1:]
            x_train_band[:, i*patch*patch:(i+1)*patch*patch, i+1:] = x_train_reshape[:, :, :band-i-1]
        else:
            x_train_band[:, i:(i+1), :(nn-i)] = x_train_reshape[:, 0:1, (band-nn+i):]
            x_train_band[:, i:(i+1), (nn-i):] = x_train_reshape[:, 0:1, :(band-nn+i)]
    # 右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn+i+1)*patch*patch:(nn+i+2)*patch*patch, :band-i-1] = x_train_reshape[:, :, i+1:]
            x_train_band[:, (nn+i+1)*patch*patch:(nn+i+2)*patch*patch, band-i-1:] = x_train_reshape[:, :, :i+1]
        else:
            x_train_band[:, (nn+1+i):(nn+2+i), (band-i-1):] = x_train_reshape[:, 0:1, :(i+1)]
            x_train_band[:, (nn+1+i):(nn+2+i), :(band-i-1)] = x_train_reshape[:, 0:1, (i+1):]

    return x_train_band


# 汇总训练数据和测试数据，利用领域像素、领域波段函数
def train_and_test_data(mirror_image, band, train_point, test_point, patch=5, band_patch=1):
    # 波段总数、包含训练集和测试集中样本点的坐标、提取的空间邻域大小（每个样本点周围考虑的patchxpatch的像素区域）、提取的波段邻域大小
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    # 针对每个样本点，利用gain_neighborhood_pixel提取其空间邻域信息，确保每个样本点都包含其周围的像素信息
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))  # 形状和数据类型
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("**************************************************")

    # 波段邻域信息处理
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape, x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape, x_test_band.dtype))
    print("**************************************************")

    return x_train_band, x_test_band


# 标签y_train, y_test
def train_and_test_label(number_train, number_test, num_classes):
    # number_train和number_test分别包含两个数：未变化(0)和变化(1)样本的数量
    # 未标记样本(2)不参与训练和测试
    y_train = []
    y_test = []
    for i in range(2):    # i=0表示未变化，i=1表示变化
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)

    y_train = np.array(y_train)     # 将列表转换为NumPy数组
    y_test = np.array(y_test)
    print("y_train: shape = {} ,type = {}".format(y_train.shape, y_train.dtype))    # 形状和数据类型信息
    print("y_test: shape = {} ,type = {}".format(y_test.shape, y_test.dtype))
    print("**************************************************")
    return y_train, y_test


class AvgrageMeter(object):    # 计算和存储平均值、总和以及计数的实用工具类
    def __init__(self):
        self.reset()    # 初始化或重置所有跟踪的指标

    def reset(self):    # 将平均值（avg）、总和（sum）和计数（cnt）重置为零
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):  # val：最新的度量值；n：val对应的样本数量，默认为 1
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):    # (1,)表示计算top-1准确率，(1, 5)表示同时计算top-1和top-5准确率
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # 输出中选取top-k的预测。maxk指取1个，1是指按行取，true指从大到小
    pred = pred.t()  # 将预测结果转置，使其与目标标签的维度对齐
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # .eq():两个Tensor进行逐元素的比较, view:reshape, expand_as:将输入tensor的维度扩展为与指定tensor相同的size

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)  # 计算每个 top-k 中的正确预测数
        res.append(correct_k.mul_(100.0/batch_size))    # 将正确预测数转换为百分比形式
    return res, target, pred.squeeze()


# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()   # 追踪损失
    top1 = AvgrageMeter()   # 追踪准确率
    tar = np.array([])  # 存储真实标签
    pre = np.array([])  # 存储模型预测
    tbar = tqdm(train_loader)   # 实时监控训练进度
    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(tbar):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        batch_pred = model(batch_data_t1, batch_data_t2)

        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        # 计算批数据的 top-1 准确率，并更新损失和准确率的度量指标，累积存储真实标签和预测结果
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data_t1.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre


# validate model（用于监测模型是否过拟合以及调整超参数）
def valid_epoch(model, valid_loader, criterion):
    objs = AvgrageMeter()   # 跟踪损失
    top1 = AvgrageMeter()   # 跟踪准确率
    tar = np.array([])  # 存储真实标签
    pre = np.array([])  # 存储模型预测
    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(valid_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data_t1, batch_data_t2)
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data_t1.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre


# test_epoch只执行数据处理和模型预测，不计算损失或准确率，只返回模型对测试数据的预测结果
def test_epoch(model, test_loader):
    pre = np.array([])
    for batch_idx, (batch_data_t1, batch_data_t2) in enumerate(test_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()

        batch_pred = model(batch_data_t1, batch_data_t2)
        _, pred = batch_pred.topk(1, 1, True, True)

        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre


# 计算和返回分类模型的性能指标，基于真实标签(tar)和预测结果(pre)
def output_metric(tar, pre):
    # 忽略未标记样本(标签为2)的评估
    mask = (tar != 2) & (pre != 2)
    tar = tar[mask]
    pre = pre[mask]
    
    matrix = confusion_matrix(tar, pre)
    
    # 确保矩阵是2x2的
    if matrix.shape == (2, 2):
        TN, FP, FN, TP = matrix.ravel()
    else:
        # 如果混淆矩阵不是2x2的，需要重新组织
        full_matrix = np.zeros((2, 2))
        for i in range(min(2, matrix.shape[0])):
            for j in range(min(2, matrix.shape[1])):
                full_matrix[i, j] = matrix[i, j]
        TN, FP, FN, TP = full_matrix.ravel()
    
    # 计算各种评估指标
    OA = (TP + TN) / (TP + TN + FP + FN)
    PRE = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / ((TP + TN + FP + FN) ** 2)
    Kappa = (OA - PRE) / (1 - PRE)
    
    # 计算精确率、召回率和F1分数
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n评估指标:")
    print(f"Overall Accuracy (OA): {OA:.4f}")
    print(f"Kappa: {Kappa:.4f}")
    print(f"Precision (PR): {precision:.4f}")
    print(f"Recall (RE): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\n混淆矩阵:")
    print("TN(预测未变化,实际未变化):", TN)
    print("FP(预测变化,实际未变化):", FP)
    print("FN(预测未变化,实际变化):", FN)
    print("TP(预测变化,实际变化):", TP)
    
    return OA, Kappa, TN, FP, FN, TP, precision, recall, f1



# 印传入的参数字典(args)的键和值，快速查看模型配置或运行时设置的参数
def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))

def visualize_prediction(height, width, true_labels, pred_labels):
    """
    生成变化检测结果的可视化图像
    Args:
        height: 图像高度
        width: 图像宽度
        true_labels: 真实标签
        pred_labels: 预测标签
    Returns:
        vis_img: 可视化结果图像
    """
    # 创建可视化图像
    vis_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 创建各种情况的掩码
    tn_mask = (true_labels == 0) & (pred_labels == 0)  # 真负例
    tp_mask = (true_labels == 1) & (pred_labels == 1)  # 真正例
    fp_mask = (true_labels == 0) & (pred_labels == 1)  # 假正例
    fn_mask = (true_labels == 1) & (pred_labels == 0)  # 假负例
    unlabeled_mask = (true_labels == 2)  # 未标记样本
    
    # 应用颜色
    vis_img[tn_mask] = [0, 0, 0]        # 黑色 - TN
    vis_img[tp_mask] = [255, 255, 255]  # 白色 - TP
    vis_img[fp_mask] = [255, 0, 0]      # 红色 - FP
    vis_img[fn_mask] = [0, 255, 0]      # 绿色 - FN
    vis_img[unlabeled_mask] = [100, 100, 100]  # 灰色 - 未标记
    
    return vis_img

def predict_full_image(model, data_t1, data_t2, height, width, band, patches, band_patches):
    """
    对整个图像进行预测，使用分批处理以节省内存
    （优化版：消除O(n²)坐标查找，使用O(1)向量化坐标计算）
    """
    model.eval()
    
    # 对整个图像进行镜像填充
    mirror_t1 = mirror_hsi(height, width, band, data_t1, patches)
    mirror_t2 = mirror_hsi(height, width, band, data_t2, patches)
    
    # 创建预测结果数组
    predictions = np.zeros((height, width))
    
    # 分批处理像素（推理时可用更大batch_size）
    batch_size = 2000
    total_pixels = height * width
    
    with torch.no_grad():
        for start_idx in range(0, total_pixels, batch_size):
            end_idx = min(start_idx + batch_size, total_pixels)
            n = end_idx - start_idx
            
            # O(1)向量化计算行列索引（替代原来的双重for循环扫描全图）
            flat_indices = np.arange(start_idx, end_idx)
            rows = (flat_indices // width).astype(np.int32)
            cols = (flat_indices % width).astype(np.int32)
            
            # 提取每个像素的patch
            batch_patches_t1 = np.zeros((n, patches, patches, band), dtype=np.float32)
            batch_patches_t2 = np.zeros((n, patches, patches, band), dtype=np.float32)
            
            for k in range(n):
                r, c = rows[k], cols[k]
                batch_patches_t1[k] = mirror_t1[r:r+patches, c:c+patches, :]
                batch_patches_t2[k] = mirror_t2[r:r+patches, c:c+patches, :]
            
            # 应用波段邻域
            batch_features_t1 = gain_neighborhood_band(batch_patches_t1, band, band_patches, patches)
            batch_features_t2 = gain_neighborhood_band(batch_patches_t2, band, band_patches, patches)
            
            # 转换为PyTorch张量并预测
            batch_t1 = torch.from_numpy(batch_features_t1.transpose(0, 2, 1)).float().cuda()
            batch_t2 = torch.from_numpy(batch_features_t2.transpose(0, 2, 1)).float().cuda()
            
            # 预测当前批次
            batch_pred = model(batch_t1, batch_t2)
            _, pred = batch_pred.max(1)
            pred = pred.cpu().numpy()
            
            # 向量化赋值（替代原来的enumerate逐像素循环）
            predictions[rows, cols] = pred
            
            # 打印进度
            print(f"\rPredicting: {end_idx}/{total_pixels}", end="")
    
    print("\nPrediction complete!")
    return predictions

# 初始化全局变量
data_t1 = None
data_t2 = None
data_label = None
input1_normalize = None
input2_normalize = None
TR = None
height = None
width = None
band = None

def init_dataset():
    """根据数据集名称加载相应的数据"""
    global data_t1, data_t2, data_label, input1_normalize, input2_normalize, TR, height, width, band
    
    if args.dataset == 'farmland':
        data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm06.mat')['imgh']
        data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm07.mat')['imghl']
        data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\label.mat')['label']
        print('Farmland - 变化像素: {}, 未变化像素: {}'.format(
            np.sum(data_label == 1), np.sum(data_label == 0)))

    elif args.dataset == 'river':
        data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_before.mat')['river_before']
        data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_after.mat')['river_after']
        data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\groundtruth.mat')['lakelabel_v1']
        data_label[data_label == 255] = 1
        print('River - 变化像素: {}, 未变化像素: {}'.format(
            np.sum(data_label == 1), np.sum(data_label == 0)))

    elif args.dataset == 'Hermiston':
        data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2004.mat')['HypeRvieW']
        data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2007.mat')['HypeRvieW']
        data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\label.mat')['label']
        print('Hermiston - 变化像素: {}, 未变化像素: {}'.format(
            np.sum(data_label == 1), np.sum(data_label == 0)))

    elif args.dataset == 'BayArea' or args.dataset == 'Barbara':
        data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2013.mat')['HypeRvieW']
        data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2015.mat')['HypeRvieW']
        data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\bayArea_gtChanges2.mat')['HypeRvieW']
        data_label = np.select(
            [data_label == 0, data_label == 1, data_label == 2],
            [2, 1, 0],
            default=data_label
        )
        print(f'{args.dataset} - 变化像素: {np.sum(data_label == 1)}, 未变化像素: {np.sum(data_label == 0)}, 未确定的像素:{np.sum(data_label == 2)}')
    
    else:
        raise ValueError(f"未知的数据集名称: {args.dataset}")
    
    # 设置数据集的基本参数
    height, width = data_t1.shape[0], data_t1.shape[1]
    band = data_t1.shape[2] if len(data_t1.shape) > 2 else 1
    
    # 数据预处理
    input1_normalize = np.zeros_like(data_t1, dtype=float)
    input2_normalize = np.zeros_like(data_t2, dtype=float)
    
    for i in range(band):
        input1_normalize[:, :, i] = (data_t1[:, :, i] - np.min(data_t1[:, :, i])) / (np.max(data_t1[:, :, i]) - np.min(data_t1[:, :, i]))
        input2_normalize[:, :, i] = (data_t2[:, :, i] - np.min(data_t2[:, :, i])) / (np.max(data_t2[:, :, i]) - np.min(data_t2[:, :, i]))
    
    TR = data_label
    
    print(f"\n数据集 {args.dataset} 标签分布:")
    print(f"- 未变化像素(0): {np.sum(data_label == 0)}")
    print(f"- 变化像素(1): {np.sum(data_label == 1)}")
    print(f"- 未标记像素(2): {np.sum(data_label == 2)}")
    total_pixels = height * width
    valid_pixels = np.sum(data_label != 2)
    print(f"总像素数: {total_pixels}")
    print(f"有效像素占比: {valid_pixels/total_pixels*100:.2f}%")
    print(f"height={height},width={width},band={band}\n")

# 定义全局变量
data_t1 = None
data_t2 = None
data_label = None
input1_normalize = None
input2_normalize = None
TR = None
height = None
width = None
band = None
