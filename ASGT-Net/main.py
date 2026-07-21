import numpy as np
import torch
from torch import optim, nn
from load_data import generater, loadData, applyPCA, normalization
from model_2 import ASGTNet
from loss_func import DynamicCombinedLoss
from train_test import run_training, test
import random
import warnings
import torchvision.transforms as transforms
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning, message="1Torch was not compiled with flash attention.")

# 设置超参数
num_epochs = 200
batch_size = 64
learning_rate = 0.0005
lambda_ = 0.5
patch_size = 5

dataname = 'farmland'
# dataname = 'bayArea'
# dataname = 'santaBarbara'

train_ratio = 0.01

if dataname == 'hermiston':
    pca_channel = 8
if dataname == 'farmland':
    pca_channel = 8
if dataname == 'river':
    pca_channel = 120
if dataname == 'bayArea':
    pca_channel = 64
if dataname == 'santaBarbara':
    pca_channel = 64


X1, X2, Y = loadData(dataname)  # 导入数据集
X1 = normalization(X=X1)  # choose normalization method    正则化 变为0到1之间
X2 = normalization(X=X2)

X1_pca = applyPCA(X1, channel=pca_channel)  # pca 降维光谱波段冗余
X2_pca = applyPCA(X2, channel=pca_channel)
# print('X1.shape{},X2.shape{},Y.shape{}'.format(X1_pca.shape, X2_pca.shape, Y.shape))

# 加载数据
(TRAIN_SIZE, TEST_SIZE, train_iter, test_iter,
         all_iter, all_position_indices, height, width,
         ce_criterion, alpha) = generater(
            X1_pca, X2_pca, Y, batch_size, train_ratio, device, windowSize=patch_size, noise_std=0
        )

# 初始化
criterion = DynamicCombinedLoss(num_classes=2, lambda_=lambda_)
# 初始化模型
model = ASGTNet(num_channels=pca_channel)
model.to(device)

# 定义优化器
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

total_train_seconds, best_epoch, best_value = run_training(
                model, train_iter, test_iter, num_epochs, optimizer, criterion,
                device, scheduler, best_model_path=dataname + "_best_model.pth"
            )

# 测试时加载最佳模型
best_model = model = ASGTNet(num_channels=pca_channel)

best_model.load_state_dict(torch.load(dataname + "_best_model.pth"))
best_model.to(device)

# 测试
test_loss, test_oa, test_f1, test_precision, test_recall, test_kappa = test(best_model, test_iter,criterion, device)
print(f"[Test]  Loss: {test_loss:.4f} | OA: {test_oa:.4f} | F1: {test_f1:.4f} | Pr: {test_precision:.4f} | Re: {test_recall:.4f} | kappa: {test_kappa:.4f}")
test_ = test_oa + test_f1 + test_precision + test_recall + test_kappa
print(test_)