import scipy.io as sio
import os
import spectral
import matplotlib.pyplot as plt
from scipy.io import loadmat

data_path = './Datasets'

data1 = sio.whosmat(os.path.join(data_path, 'santaBarbara/barbara_2013.mat'))
print(data1)    # [('HypeRvieW', (984, 740, 224), 'double')]
data2 = sio.whosmat(os.path.join(data_path, 'santaBarbara/barbara_2014.mat'))
print(data2)    # [('HypeRvieW', (984, 740, 224), 'double')]
label1 = sio.whosmat(os.path.join(data_path, 'santaBarbara/barbara_gtChanges.mat'))
print(label1)   # [('HypeRvieW', (984, 740), 'uint8')]

# data_t1 = loadmat('./Datasets/farm450/farm06.mat')['imgh']
# # print(data_t1)
# data_t2 = loadmat('./Datasets/farm450/farm07.mat')['imghl']
# data_label = loadmat('./Datasets/farm450/label.mat')['label']

data_t1 = loadmat("./Datasets/bayArea/Bay_Area_2013.mat")['HypeRvieW']
# print(data_t1)
data_t2 = loadmat("./Datasets/bayArea/Bay_Area_2015.mat")['HypeRvieW']
data_label = loadmat("./Datasets/bayArea/bayArea_gtChanges2.mat")['HypeRvieW']

data_t1 = loadmat("./Datasets/santaBarbara/barbara_2013.mat")['HypeRvieW']
print(data_t1)
data_t2 = loadmat("./Datasets/santaBarbara/barbara_2014.mat")['HypeRvieW']
data_label = loadmat("./Datasets/santaBarbara/barbara_gtChanges.mat")['HypeRvieW']
#
#
# data_t1 = sio.loadmat('./Datasets/santaBarbara/barbara_2013.mat')['HypeRvieW']
# data_t2 = sio.loadmat('./Datasets/santaBarbara/barbara_2014.mat')['HypeRvieW']
# data_label = sio.loadmat('./Datasets/santaBarbara/barbara_gtChanges.mat')['HypeRvieW']
#
# view1 = spectral.imshow(data=data_t1, bands=[33, 22, 11], title="barbara_2013")  # 图像显示
# plt.pause(60)

# import numpy as np
# import matplotlib.pyplot as plt
#
# # 参数设置
# initial_lr = 5e-4
# gamma = 0.9
# epochs = 200
# step_size = 10  # 每10个epoch衰减一次
#
# # 计算每个epoch的学习率
# lrs = [initial_lr * (gamma ** (epoch // step_size)) for epoch in range(epochs)]
#
# # 绘制学习率变化图
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, epochs + 1), lrs, marker='o', linestyle='-', markersize=4, markerfacecolor='red')
# plt.title('Learning Rate Schedule')
# plt.xlabel('Epoch')
# plt.ylabel('Learning Rate')
# plt.grid(True)
# plt.show()
