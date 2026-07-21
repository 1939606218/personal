import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward, CrossAttention, SSTransformer
import numpy as np


class SSTTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, b_dim, b_depth, b_heads, b_dim_head, b_mlp_head, num_patches, cross_attn_depth=3, cross_attn_heads=8, dropout=0):
        super().__init__()

        self.transformer = SSTransformer(dim, depth, heads, dim_head, mlp_dim, b_dim, b_depth, b_heads, b_dim_head, b_mlp_head, num_patches, dropout)

        self.cross_attn_layers = nn.ModuleList([])  # 存储后面要添加的cross-attention layers
        for _ in range(cross_attn_depth):   # 添加指定数量的交叉注意力层到 self.cross_attn_layers
            self.cross_attn_layers.append(PreNorm(b_dim,
                                                  CrossAttention(b_dim, heads=cross_attn_heads, dim_head=dim_head, dropout=0)))

    def forward(self, x1, x2):
        x1 = self.transformer(x1)
        x2 = self.transformer(x2)

        for cross_attn in self.cross_attn_layers:
            x1_class = x1[:, 0]     # 将从 x1 中选择每个样本的第一个位置的元素，生成一个形状为（64，512）的张量 x1_class
            x1_origin = x1[:, 1:]   # 将从 x1 中选择每个样本除第一个位置外的所有元素，生成一个形状为（64，32，512）的张量 x1_origin
            x2_class = x2[:, 0]     # 将从 x2 中选择每个样本的第一个位置的元素，生成一个形状为（64，512）的张量 x2_class
            x2 = x2[:, 1:]          # 将从 x2 中选择每个样本除第一个位置外的所有元素，生成一个形状为（64，32，512）的张量 x2

            # Cross Attn
            cat1_q = x1_class.unsqueeze(1)  # 增加了一个维度，为后续的拼接操作做准备。这一步将类别信息转换成适合交叉注意力机制处理的格式
            cat1_qkv = torch.cat((cat1_q, x2), dim=1)   # 拼接张量，构造了跨注意力层所需的q k v
            cat1_out = cat1_q + cross_attn(cat1_qkv)    # 输入到跨注意力层中，并将得到的输出与原查询相加
            x1 = torch.cat((cat1_out, x1_origin), dim=1)    # 恢复了数据的原始形状
            # 类别信息 cat1_q 和 cat2_q 作为查询，而 x2 和 x1_origin 则分别作为键和值
            cat2_q = x2_class.unsqueeze(1)
            cat2_qkv = torch.cat((cat2_q, x1_origin), dim=1)
            cat2_out = cat2_q + cross_attn(cat2_qkv)    # 有助于保持信息流，并防止梯度消失问题！！！
            x2 = torch.cat((cat2_out, x2), dim=1)

        return cat1_out, cat2_out


class SSTViT(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, b_dim, b_depth, b_heads, b_dim_head, b_mlp_head, pool='cls', channels=1, dim_head = 16, dropout=0., emb_dropout=0., multi_scale_enc_depth=1):
        # 图像尺寸、频带数、补丁数、类别数等
        super().__init__()

        patch_dim = image_size ** 2 * near_band  # 每个补丁的维度，这是图像尺寸的平方乘以频带数
        self.num_patches = num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))  # 初始化位置嵌入，用于为每个补丁添加位置信息
        self.patch_to_embedding = nn.Linear(patch_dim, dim)     # 定义一个线性层，用于将补丁转换为嵌入向量
        self.cls_token_t1 = nn.Parameter(torch.randn(1, 1, dim))    # 初始化两个类别标记class token，用于聚合补丁的信息
        self.cls_token_t2 = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)  # Dropout层减少过拟合

        self.multi_scale_transformers = nn.ModuleList([])   # 存储多尺度transformer，可用于在不同尺度上处理补丁的嵌入向量
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers.append(SSTTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, b_dim,
                                                                       b_depth, b_heads, b_dim_head, b_mlp_head,
                                                                       self.num_patches, dropout=0.))

        self.pool = pool
        self.to_latent = nn.Identity()  # 恒等变换，通常用于输出层之前的数据处理

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(b_dim),
            nn.Linear(b_dim, num_classes)
        )

    def forward(self, x1, x2):
        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')
        # embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x1 = self.patch_to_embedding(x1)  # [b,n,dim]     # 改变第三维
        x2 = self.patch_to_embedding(x2)  # 将输入数据转换为嵌入向量
        b, n, _ = x1.shape

        # add position embedding
        # cls_tokens_t1 = repeat(self.cls_token_t1, '() n d -> b n d', b = b) #[b,1,dim]
        # cls_tokens_t2 = repeat(self.cls_token_t2, '() n d -> b n d', b = b)

        # x1 = torch.cat((cls_tokens_t1, x1), dim = 1) #[b,n+1,dim]

        x1 += self.pos_embedding[:, :(n + 1)]       # 嵌入位置信息
        x1 = self.dropout(x1)       # Dropout层是一种常用的正则化技术，通过随机地丢弃神经元的输出来减少过拟合，并增强模型的泛化能力。
        # x2 = torch.cat((cls_tokens_t2, x2), dim = 1) #[b,n+1,dim]
        x2 += self.pos_embedding[:, :(n + 1)]
        x2 = self.dropout(x2)
        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]

        for multi_scale_transformer in self.multi_scale_transformers:   # 多尺度变换器处理嵌入向量
            out1, out2 = multi_scale_transformer(x1, x2)
        # classification: using cls_token output
        out1 = self.to_latent(out1[:, 0])   # 提取并处理每个输入的类别标记class token输出
        out2 = self.to_latent(out2[:, 0])
        out = out1+out2
        # MLP classification layer
        return self.mlp_head(out)   # 通过多层感知器进行最终分类，并返回结果
