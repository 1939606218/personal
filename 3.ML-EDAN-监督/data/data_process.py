import pickle
import numpy as np
import scipy.io as sio
import torch

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

    if dataset_name == 'farm':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm06.mat')['imgh']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm07.mat')['imghl']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\label.mat')['label']

    if dataset_name == 'river':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_before.mat')['river_before']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_after.mat')['river_after']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\groundtruth.mat')['lakelabel_v1']
        labels[labels == 255] = 1

    if dataset_name == 'Barbara':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_2013.mat')['HypeRvieW']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_2014.mat')['HypeRvieW']
        labels = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_gtChanges.mat')['HypeRvieW']
        # 修改标签映射：0→2，1→1，2→0
        labels = np.select(
            [labels == 0, labels == 1, labels == 2],
            [2, 1, 0],
            default=labels  # 处理其他可能的值
        )

    if dataset_name == 'BayArea':
        data1 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2013.mat')['HypeRvieW']
        data2 = sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2015.mat')['HypeRvieW']
        labels= sio.loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\bayArea_gtChanges2.mat')['HypeRvieW']
        # 修改标签映射：0→2，1→1，2→0
        labels = np.select(
            [labels == 0, labels == 1, labels == 2],
            [2, 1, 0],
            default=labels  # 处理其他可能的值
        )

    height, width = labels.shape
    # 统计标签数量
    label_counts = count_labels(labels)
    print(f"数据集标签统计：{label_counts}")

    print("data1.shape", data1.shape)
    print("data2.shape", data2.shape)
    print("labels.shape", labels.shape)

    return data1, data2, labels , height, width





