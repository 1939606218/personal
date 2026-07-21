# -*- coding:utf-8 -*-

from vit_pytorch import VisionTransformer
from depthwise_separable_conv import depthwise_separable_conv
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
#from einops import rearrange

class Conv_Former(nn.Module):
    def __init__(self, cha_in, windowsize, num_classes):
        super(Conv_Former, self).__init__()
        self.name = 'Conv_Former'
        self.depthwise_spe_conv = depthwise_separable_conv(ch_in=cha_in, ch_out=cha_in, kernel_size=3)
        self.vit1 = VisionTransformer(img_size=windowsize, patch_size=1, in_c=cha_in, embed_dim=513, depth=3,
                                      num_heads=3,
                                      mlp_ratio=4, representation_size=512, drop_ratio=0.1, num_classes=num_classes)
        self.vit2 = VisionTransformer(img_size=windowsize - 2, patch_size=1, in_c=cha_in, embed_dim=513, depth=3,
                                      num_heads=3,
                                      mlp_ratio=4, representation_size=512, drop_ratio=0.1, num_classes=num_classes)
        self.vit3 = VisionTransformer(img_size=windowsize - 4, patch_size=1, in_c=cha_in, embed_dim=513, depth=3,
                                      num_heads=3,
                                      mlp_ratio=4, representation_size=512, drop_ratio=0.1, num_classes=num_classes)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x1 = rearrange(x, 'b c h w -> b c (h w)')
        x1 = self.vit1(x)

        x2 = self.depthwise_spe_conv(x)
        # x3 = rearrange(x2, 'b c h w -> b c (h w)')
        x3 = self.vit2(x2)

        x4 = self.depthwise_spe_conv(x2)
        # x5 = rearrange(x4, 'b c h w -> b c (h w)')
        x5 = self.vit3(x4)

        x = x1 * x3 * x5
        #x = self.mlp_head(x)
        return x

# models = Conv_Former(cha_in=200, height=15, width=15)
# x = torch.randn(16, 200, 15, 15)
# print(models(x).shape)

def binary(x, th, bathsize):
    a = torch.zeros([bathsize])
    a = a.cuda()
    b = torch.ones([bathsize])
    b = b.cuda()
    x = torch.where(x <= th, x, b)
    x = torch.where(x > th, x, a)
    return x

class SiameseNetwork(nn.Module):
    def __init__(self, cha_in, windowsize, num_classes):
        super(SiameseNetwork, self).__init__()
        self.name = 'CTS'
        self.conv_f = Conv_Former(cha_in, windowsize, num_classes)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 2)
        )
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(1,1)
        self.conv1 = nn.Conv2d(in_channels=198, out_channels=160, kernel_size=3, padding=1)

    def forward_once(self, x):
        output = self.conv_f(x)
        #output = self.mlp_head(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output = torch.abs(output1 - output2)   #将参数传递到 torch.abs 后返回输入参数的绝对值作为输出，输入参数必须是一个 Tensor 数据类型的变量。
        output = output.reshape(output.shape[0], -1)
        #output = self.classifier(output)
        output = self.mlp_head(output)
        output = self.sigmoid(output)
        return output

#Contrastive Loss
# class ContrastiveLoss(torch.nn.Module):
#     def __init__(self, margin=1.5):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, output1, output2, label):
#         euclidean_distance = F.pairwise_distance(output1, output2)
#         loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
#                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#
#
#         return loss_contrastive
