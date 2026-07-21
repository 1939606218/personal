"""
检查MAT数据、显示图像
"""
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
from scipy.io import loadmat
import spectral as spy
import matplotlib.pyplot as plt

# '第一部分：查看数据集中的变量名称、大小和类型'
# data_path = 'datasets/HSI/farm450/'
#
# data3 = sio.whosmat(os.path.join(data_path, 'farm06_unified.mat'))    # [('HypeRvieW', (450, 140, 155), 'int16')]
# print(data3)
# data4 = sio.whosmat(os.path.join(data_path, 'farm07_unified.mat'))    # [('HypeRvieW', (450, 140, 155), 'int16')]
# print(data4)
# label2 = sio.whosmat(os.path.join(data_path, 'label_unified.mat'))    # [('HypeRvieW', (450, 140), 'uint8')]
# print(label2)


data_path3 = r'Z:\pycharm\pythonProject\SSA\dataset\river'

data1 = sio.loadmat(os.path.join(data_path3, 'river_before.mat'))
# print('数据1：', data1)
data2 = sio.loadmat(os.path.join(data_path3, 'river_after.mat'))
# print('数据2：', data2)
label = sio.loadmat(os.path.join(data_path3, 'groundtruth.mat'))    # 标签是0和1........................................
# print('标签数据：', label)

data1 = data1['river_before']    # 图像和标签的显示
band1, band2, band3 = 10, 20, 30  # 随机选择波段
rgb_image1 = np.stack((data1[:, :, band1], data1[:, :, band2], data1[:, :, band3]), axis=-1)
rgb_image1 = (rgb_image1 - np.min(rgb_image1)) / (np.max(rgb_image1) - np.min(rgb_image1))
plt.imshow(rgb_image1)
plt.axis('off')
plt.savefig(r'Z:\pycharm\pythonProject\SSA\dataset\river\before.png', dpi=1200, bbox_inches='tight', pad_inches=0)
plt.show()

data2 = data2['river_after']
band1, band2, band3 = 10, 20, 30  # 随机选择波段
rgb_image2 = np.stack((data2[:, :, band1], data2[:, :, band2], data2[:, :, band3]), axis=-1)
rgb_image2 = (rgb_image2 - np.min(rgb_image2)) / (np.max(rgb_image2) - np.min(rgb_image2))
plt.imshow(rgb_image2)
plt.axis('off')
plt.savefig(r'Z:\pycharm\pythonProject\SSA\dataset\river\after.png', dpi=1200, bbox_inches='tight', pad_inches=0)
plt.show()

label = label['lakelabel_v1']
plt.imshow(label, cmap='gray')
plt.axis('off')
plt.savefig(r'Z:\pycharm\pythonProject\SSA\dataset\river\label.png', dpi=1200, bbox_inches='tight', pad_inches=0)
plt.show()
print('原文图像显示 End')
