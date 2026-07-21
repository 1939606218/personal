# -*- coding:utf-8 -*-

import torch
import os

# dataset = 'hermiston'
# # dataset = 'farmland'
# # dataset = 'river'
# # dataset = 'santaBarbara'
# # dataset = 'bayArea'
# if dataset =="hermiston" :
#     kerner_number=24
#     batchsize = 64
#
# if dataset =="farmland" :
#     kerner_number=24
#     batchsize = 64
#
# elif dataset =="river":
#     kerner_number = 24
#     batchsize = 64
#
# elif dataset == 'santaBarbara':
#     kerner_number = 32
#     batchsize = 64
#
# elif dataset == 'bayArea':
#     kerner_number = 32
#     batchsize = 64


train_ratio = 0.01
windowsize = 5      # 块的大小
CLASSES_NUM = 2
ITER = 1
EPOCHES = 100
pca = True

pca_channel = 30
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pos = 417
neg = 833


