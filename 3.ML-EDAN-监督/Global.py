import torch
import os




num_head = 4
train_ratio = 0.01
patch_sieze = 5      # 块的大小
EPOCHES = 200
batch_size=64
lr=0.0001

pca_channel=30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 预先定义保存路径
