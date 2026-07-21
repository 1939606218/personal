# -*- coding:utf-8 -*-
"""
作者：张亦严
日期:2022年05月19日
"""
from torch import nn

class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, padding=0, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)      #深度卷积中每个卷积核只处理一个通道，相比传统卷积，计算量大大减少。
        x = self.point_conv(x)
        return x