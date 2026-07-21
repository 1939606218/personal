from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCLoss(nn.Module):
    def __init__(self, gamma=1):
        super(PCLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.gamma = gamma

    def forward(self, z1, z2):
        cos_ = self.cos(z1, z2.detach())    #对z2的操作不会参与梯度计算
        loss = torch.mul(torch.pow((2 - cos_), self.gamma), cos_)  #得到每个样本对应的损失值，整体loss的形状依然是 [batch_size]
        L = -loss.mean() + 1   #为什么+1呢
        return L


class KLloss(nn.Module):
    def __init__(self):
        super(KLloss, self).__init__()

    def forward(self, z1, z2):
        kl = F.kl_div(z1.softmax(dim=-1).log(), z2.softmax(dim=-1), reduction='mean') #dim=-1表示在最后一个维度上进行操作
        #F.kl_div 函数是PyTorch中的一个函数，用于计算KL散度。它的工作原理是将第一个分布（z1）的对数概率与第二个分布（z2）的概率进行比较
        return kl


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # F.nll_loss 函数计算负对数似然损失，这个函数内部会计算 log_pred 和 targets 之间的损失
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss
