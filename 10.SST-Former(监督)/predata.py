# （1）划分训练集测试集部分，将未选用点设置为0，变化点设为1，未变化点设为2。 （2）对t1时刻t2时刻图像进行归一化处理。

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sstvit import SSTViT

# 全局变量声明
TR = None
TE = None
num_classes = None
input1_normalize = None
input2_normalize = None
height = None
width = None
band = None

def prepare_data(data_t1, data_t2, data_label):
    """准备数据和模型的函数"""
    global TR, TE, num_classes, input1_normalize, input2_normalize, height, width, band
    
    # 准备训练标签矩阵TR，直接使用原始标签
    TR = data_label.copy()  # 复制标签矩阵以保留原始数据
    
    # 统计数据标签分布
    change_pixels = np.sum(TR == 1)     # 变化像素（标签为1）
    unchange_pixels = np.sum(TR == 0)   # 未变化像素（标签为0）
    uncertain_pixels = np.sum(TR == 2)   # 未标记像素（标签为2）
    print('数据集标签分布:')
    print(f'- 未变化像素(0): {unchange_pixels}')
    print(f'- 变化像素(1): {change_pixels}')
    print(f'- 未标记像素(2): {uncertain_pixels}')
    print(f'总像素数: {TR.size}')
    print(f'有效像素占比: {((change_pixels + unchange_pixels) / TR.size * 100):.2f}%')
    
    # TE和TR使用相同的标签，不需要相减操作
    TE = data_label.copy()
    
    # 设置类别数为2（0=未变化，1=变化），不包括未标记类别(2)
    num_classes = 2
    
    # 创建与原始图像数据相同形状的数组，存储归一化后的数据
    input1_normalize = np.zeros(data_t1.shape)
    input2_normalize = np.zeros(data_t2.shape)
    
    # 对每个波段分别进行归一化
    for i in range(data_t1.shape[2]):
        # 将当前波段的数据重塑为一维数组
        band_data = np.concatenate([data_t1[:, :, i].ravel(), data_t2[:, :, i].ravel()])
        
        # 创建并拟合MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        # reshape(-1, 1)是为了满足MinMaxScaler的输入要求
        scaler.fit(band_data.reshape(-1, 1))
        
        # 对两个时间点的数据分别进行转换
        input1_normalize[:, :, i] = scaler.transform(data_t1[:, :, i].reshape(-1, 1)).reshape(data_t1.shape[0], data_t1.shape[1])
        input2_normalize[:, :, i] = scaler.transform(data_t2[:, :, i].reshape(-1, 1)).reshape(data_t2.shape[0], data_t2.shape[1])
    
    height, width, band = data_t1.shape
    print("height={0},width={1},band={2}".format(height, width, band))
    
    return TR, TE, num_classes, input1_normalize, input2_normalize, height, width, band

def create_model(num_classes, args, band):
    """创建和初始化模型"""
    model = SSTViT(
        image_size=args.patches,
        near_band=args.band_patches,  # 接近波段或特定波段的块大小
        num_patches=band,  # 图像分割成的块的数量
        num_classes=num_classes,  # 类别数
        dim=32,  # 模型内部的维度，即特征向量的大小
        depth=2,  # Transformer模型的深度，即堆叠的Transformer层的数量
        heads=4,  # 多头注意力机制中的头数，即在注意力机制中并行计算的头的数量
        dim_head=16,  # 每个头在多头注意力机制中的维度大小
        mlp_dim=8,  # MLP（多层感知器）层的维度
        b_dim=512,  #
        b_depth=3,  #
        b_heads=8,
        b_dim_head=32,
        b_mlp_head=8,
        dropout=0.2,  # 使用的dropout比率
        emb_dropout=0.1,  # 在嵌入层使用的dropout比率
    )
    
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 20, gamma=args.gamma)
    
    return model, criterion, optimizer, scheduler
