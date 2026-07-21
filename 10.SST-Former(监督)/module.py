import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads    # 所有头的总维度
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 将输入转换为q k v的组合，每一部分都需要inner_dim的空间
        self.to_out = nn.Sequential(    # 输出层，首先通过一个线性层将维度从inner_dim转换回原始维度dim，然后应用dropout
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads  # h为4，由 self.heads 解构获得

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # chunk函数，用于在指定维度上将张量分块，将张量沿着指定维度切分为多个块，并返回这些块组成的元组
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # transpose(k)*q / sqrt(head_dim)->[b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # dots：(b, h, i, j)，表示了q与k之间的相关性。乘以self.scale是为了对相关性进行缩放，以便在后续的注意力计算中起到适当的作用

        mask_value = -torch.finfo(dots.dtype).max  # 生成一个与 dots 张量类型相同的最小值。
        if mask is not None:    # mask value: -inf
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)  # 对dots的最后一个维度进行归一化操作，将注意力权重分配给输入序列中的不同位置。
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # 张量相乘，按照注意力权重乘以v
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')    # 重排输出张量，将多头注意力的结果合并成单个张量
        out = self.to_out(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)  # 判断是否需要输出投影。如果只有一个头且维度等于输入维度，则不需要额外的输出投影

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  # 如果需要输出投影，则使用一个线性层和dropout层；否则，使用恒等变换

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)    # 分别通过对应的线性层生成k v q
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))  # q 只针对输入的第一个元素生成（交叉注意力机制的特点之一）
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale   # 计算q k之间的点积并缩放，点积用于衡量q和k之间的相似度

        attn = dots.softmax(dim=-1)  # 对点积结果进行归一化，得到注意力权重

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # 使用注意力权重对v进行加权，得到输出
        out = rearrange(out, 'b h n d -> b n (h d)')    # 重排输出张量，将多头的结果合并成一个单一维度
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel):
        super().__init__()

        self.layers = nn.ModuleList([])  # 存储 Transformer 层
        for _ in range(depth):  # 添加指定数量的Transformer层到self.layers
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))  # 每一层包括：一个残差连接的自注意力层Attention和一个残差连接的前馈网络FeedForward

        self.skipcat = nn.ModuleList([])  # 存储跳跃连接
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))  # 添加二维卷积层到self.skipcat

    def forward(self, x, mask=None):
        for attn, ff in self.layers:  # 遍历每一层，依次应用自注意力层和前馈网络
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class SSTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, b_dim, b_depth, b_heads, b_dim_head, b_mlp_head,
                 num_patches, dropout):
        super().__init__()

        self.layers = nn.ModuleList([])  # 存储主要的 Transformer 层
        self.k_layers = nn.ModuleList([])  # 存储辅助的 Transformer 层
        self.channels_to_embedding = nn.Linear(num_patches, b_dim)  # 线性层，用于将通道转换为嵌入向量
        self.cls_token = nn.Parameter(torch.randn(1, 1, b_dim))     # 初始化一个类别标记，Transformer中常用于代表整个序列的全局信息
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))
        for _ in range(b_depth):
            self.k_layers.append(nn.ModuleList([
                Residual(PreNorm(b_dim, Attention(dim=b_dim, heads=b_heads, dim_head=b_dim_head, dropout=dropout))),
                Residual(PreNorm(b_dim, FeedForward(b_dim, b_mlp_head, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:    # 遍历主要的 Transformer 层，依次应用自注意力层和前馈网络
            x = attn(x, mask=mask)
            x = ff(x)
        x = rearrange(x, 'b n d -> b d n')  # 重新排列张量的维度
        x = self.channels_to_embedding(x)   # 通过线性层转换为嵌入向量
        b, d, n = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # 复制类别标记以匹配批次大小
        x = torch.cat((cls_tokens, x), dim=1)   # 将类别标记和嵌入向量拼接在一起

        for attn, ff in self.k_layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


# SSTransformer_pyramid在返回最终输出时，还提供了中间层的特征表示out_feature，可以用于后续任务如特征融合或多任务学习
class SSTransformer_pyramid(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, b_dim, b_depth, b_heads, b_dim_head, b_mlp_head,
                 num_patches, dropout):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.k_layers = nn.ModuleList([])
        self.channels_to_embedding = nn.Linear(num_patches, b_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, b_dim))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))
        for _ in range(b_depth):
            self.k_layers.append(nn.ModuleList([
                Residual(PreNorm(b_dim, Attention(dim=b_dim, heads=b_heads, dim_head=b_dim_head, dropout=dropout))),
                Residual(PreNorm(b_dim, FeedForward(b_dim, b_mlp_head, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        out_feature = x    # 保存当前特征表示
        x = rearrange(x, 'b n d -> b d n')
        x = self.channels_to_embedding(x)
        b, d, n = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        for attn, ff in self.k_layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x, out_feature


class ViT(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 channel_dim=1, dim_head=16, dropout=0., emb_dropout=0., mode='ViT'):
        super().__init__()

        patch_dim = image_size ** 2 * near_band  # 计算每个补丁的维度，基于图像尺寸和波段数
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 初始化位置嵌入，用于给补丁添加位置信息
        self.patch_to_embedding = nn.Linear(channel_dim, dim)   # 线性层，将图像补丁转换为嵌入向量
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))   # 初始化分类标记cls_token，用于表示整体图像的信息

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool    # 设置池化方式，默认为'cls'，即使用 cls_token 进行分类
        self.to_latent = nn.Identity()  # 恒等映射，通常用于数据的转换或连接

        self.mlp_head = nn.Sequential(  # 多层感知器头部，用于最终的分类任务
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask=None):
        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')
        # embedding every patch vector to embedding size: [batch, patch_num, embedding_size]

        x = self.patch_to_embedding(x)  # [b,n,dim]
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim]，复制class token，使其与批次大小匹配
        x = torch.cat((cls_tokens, x), dim=1)   # [b,n+1,dim]，将分类标记和嵌入向量拼接在一起
        x += self.pos_embedding[:, :(n + 1)]    # 加入position embedding
        x = self.dropout(x)

        x = self.transformer(x, mask)  # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.to_latent(x[:, 0])    # 提取经过Transformer处理后的cls_token对应的输出

        return self.mlp_head(x)     # MLP classification layer


class SSFormer_v4(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, b_dim, b_depth, b_heads, b_dim_head, b_mlp_head,
                 num_patches, dropout, mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.k_layers = nn.ModuleList([])
        self.channels_to_embedding = nn.Linear(num_patches, b_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, b_dim))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))
        for _ in range(b_depth):
            self.k_layers.append(nn.ModuleList([
                Residual(PreNorm(b_dim, Attention(dim=b_dim, heads=b_heads, dim_head=b_dim_head, dropout=dropout))),
                Residual(PreNorm(b_dim, FeedForward(b_dim, b_mlp_head, dropout=dropout)))
            ]))
        self.mode = mode

    def forward(self, x, c, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.channels_to_embedding(x)
        b, d, n = x.shape
        cls_tokens = repeat(c, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        for attn, ff in self.k_layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x
