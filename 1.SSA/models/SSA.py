import torch.nn.functional as F
import torch
from torch import nn
from Global import *
#CBAM

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

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
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=60, out_channels=24, kernel_size=3, padding=1)
#         self.bn = nn.BatchNorm2d(24)
#     def forward(self, t):
#         t = self.conv1(t)
#         t = self.bn(t)
#         t = F.relu(t)
#         return t


class SSA(nn.Module):
    def __init__(self,kerner_number):
        super(SSA, self).__init__()
        self.name = 'SSA'   #就是把外部传来的参数 name 的值 赋值给Student类自己的属性变量 self.name
        self.conv1 = nn.Conv2d(in_channels=pca_channel, out_channels=kerner_number, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(kerner_number)  #在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.cbam = cbam_block(channel=kerner_number, ratio=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=kerner_number, out_channels=kerner_number, kernel_size=3)
        self.sigmoid = nn.Sigmoid()  #用于隐层神经元输出，取值范围为 (0,1)，它可以将一个实数映射到 (0,1)的区间，可以用来做二分类。
        self.fc = nn.Linear(in_features=kerner_number, out_features=2)

    def forward_once(self, x):
        x1 = self.conv1(x)
        x1 = self.bn(x1)
        x1 = F.relu(x1)
        x1 = self.cbam(x1)
        x2 = self.conv2(x1)
        x2 = self.bn(x2)
        x2 = F.relu(x2)
        x2 = self.cbam(x2)
        x3 = self.conv2(x2)
        x3 = self.bn(x3)
        x3 = F.relu(x3)
        # x3 = torch.flatten(x3, 1)
        return x3

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = F.pairwise_distance(output1, output2)
        #output = torch.abs(output1 - output2)# 将参数传递到 torch.abs 后返回输入参数的绝对值作为输出，输入参数必须是一个 Tensor 数据类型的变量。
        output = output.view(output.shape[0], -1)
        ##output = output.view(output.size()[0], -1)
        # euclidean_distance = F.pairwise_distance(output1, output2)
        # size = euclidean_distance.numel()
        # euclidean_distance = euclidean_distance.view(size,1)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output


if __name__ == '__main__':
    from thop import profile, clever_format

    # 参数设置
    num_channels = 30
    patch_size = 5
    batch_size = 64

    # 创建模型实例
    model = SSA()

    # 创建输入数据
    input1 = torch.randn(batch_size, num_channels, patch_size, patch_size)
    input2 = torch.randn(batch_size, num_channels, patch_size, patch_size)

    # 计算 FLOPs 和参数数量
    flops, params = profile(model, inputs=(input1, input2))
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
