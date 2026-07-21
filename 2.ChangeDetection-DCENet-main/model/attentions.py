# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from functools import partial


class GlobalAttention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim, num_heads=4, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  #1/根号d

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        #创建了一个线性层 ,用于将输入特征映射到 q（查询）、k（键）和 v（值）三个空间，维度变为原来的 3 倍
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape   #获取输入张量 x 的形状信息，包括批量大小 B、高度 H、宽度 W 和通道数 C
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #(3，B，self.num_heads，H*W，C // self.num_heads)   3 表示 q、k、v 三个部分
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # @ 表示矩阵乘法
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #以一定概率（由attn_drop参数指定）随机将部分注意力分数置为 0。这样做可以减少模型对某些特定注意力模式的过度依赖
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #不会过度依赖于投影层输出的某些特定值，从而提高模型对不同输入特征的适应性
        return x


class GlobalTransformer(nn.Module):
    "MRA块"
    def __init__(self, dim, norm_layer=partial(nn.LayerNorm, eps=1e-6), qkv_bias=False, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.norm1 = norm_layer(dim) #对输入特征进行归一化处理
        self.attn1 = GlobalAttention(dim, num_heads=4, qkv_bias=qkv_bias,
                                     qk_scale=qk_scale, attn_drop=attn_drop)
        self.norm2 = norm_layer(dim)
        self.attn2 = GlobalAttention(dim, num_heads=4, qkv_bias=qkv_bias,
                                     qk_scale=qk_scale, attn_drop=attn_drop)  #独立的特征更新
        self.conv = nn.Conv2d(in_channels=dim * 3, out_channels=dim, kernel_size=1)  #将通道数从 dim * 3 转换为 dim

    def forward(self, x):
        # input B*C*H*W
        x_1 = x.permute(0, 2, 3, 1)   # B*H*W*C
        x_1 = x_1 + self.attn1(self.norm1(x_1))     #MSA的计算
        x_1 = x_1.permute(0, 3, 1, 2)  # B*C*H*W

        x_2 = x_1.permute(0, 2, 3, 1)
        x_2 = x_2 + self.attn2(self.norm2(x_2))
        x_2 = x_2.permute(0, 3, 1, 2)           # 多层次特征捕捉
        out = self.conv(torch.cat((x, x_1, x_2), dim=1))  #在通道维度上进行拼接

        return out
