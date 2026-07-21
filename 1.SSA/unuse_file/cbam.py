import torch
from torch import nn

print("cbam")
class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=3):    #ratio是通道注意力模块中用于降维的比例，kernel_size是空间注意力模块中卷积核的大小。
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)     #两种池化方式都是在空间维度上进行操作，将输入特征图压缩为 1x1 的大小，目的是为了获取每个通道的全局信息

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1  #padding能保持特征图尺寸，捕捉边缘信息
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)    #计算输入特征图在通道维度上的平均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)  #计算输入特征图在通道维度上的最大值
        x = torch.cat([avg_out, max_out], dim=1)    #拼接
        x = self.conv1(x)
        return self.sigmoid(x)