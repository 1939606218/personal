# -*- coding: utf-8 -*-
from model.attentions import *


def conv3(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )   #没有设置填充（默认为 0），卷积操作会导致特征图尺寸变小


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.Up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3), #转置卷积层用于将输入张量的尺寸增大，实现上采样效果。
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Up(x)


class EDbranch(nn.Module):
    '只要fuse2的MSA,94.37 的不加ASPP的EDbranch'

    def __init__(self, in_ch):
        super(EDbranch, self).__init__()
        self.conv0 = nn.Conv2d(in_ch, 128, kernel_size=1)
        self.Down1 = conv3(128, 256)
        self.Down2 = conv3(256, 512)
        self.Down3 = conv3(512, 1024)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
#随着网络深度增加进行下采样，输入特征图的尺寸逐渐变小，但需要模型能够从更小的空间范围内提取更抽象、更高级的语义信息。增加通道数可以提供更多的维度来表示这些丰富的语义特征。
#如果在上采样过程中不减少通道数，可能会导致通道数过多，从而引入过多的冗余信息，增加模型的复杂度和过拟合风险。
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1)

    def forward(self, x):
        down0 = self.conv0(x)
        down1 = self.Down1(down0)  # 7 -> 5
        down2 = self.Down2(down1)  # 5 -> 3
        # print("down1.shape",down1.shape)

        up1 = self.up1(down2)  # 3 -> 5
        # print("up1.shape", up1.shape)
        fuse1 = self.conv1(torch.cat((up1, down1), dim=1))  #结合不同层次特征信息,调整通道数
        up2 = self.up2(fuse1)  # 5 -> 7
        fuse2 = self.conv2(torch.cat((up2, down0), dim=1))

        return down0, down1, down2, fuse1, fuse2  # 7, 5, 3, 5, 7


class DFEDSubnet(nn.Module):
    def __init__(self, in_ch):
        super(DFEDSubnet, self).__init__()
        self.edbranch1 = EDbranch(in_ch)
        self.edbranch2 = EDbranch(in_ch)

        self.Down1 = conv3(128, 256)
        self.Down2 = conv3(256, 512)
        self.transformerblock = GlobalTransformer(512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=1)

    def forward(self, x1, x2):  #DFA模块
        # two temporal branches
        d11, d12, d13, u11, u12 = self.edbranch1(x1)  # 7, 5, 3, 5, 7
        d21, d22, d23, u21, u22 = self.edbranch1(x2)
        diff1, diff2, diff3, diff4, diff5 = (d11 - d21), (d12 - d22), (d13 - d23), (u11 - u21), (u12 - u22)

        # differential branch
        diff_d1 = self.Down1(diff1)
        fuse1 = self.conv1(torch.cat((diff_d1, diff2), dim=1)) #上一层的下采样和当前层的差值拼接起来
        diff_d2 = self.Down2(fuse1)
        fuse2 = self.conv2(torch.cat((diff_d2, diff3), dim=1))
        fuse2 = self.transformerblock(fuse2)   #MRA块

        up1 = self.up1(fuse2)
        fuse3 = self.conv3(torch.cat((up1, diff4), dim=1))
        up2 = self.up2(fuse3)
        fuse4 = self.conv4(torch.cat((up2, diff5), dim=1))

        return fuse4, fuse4


class DownEncoder(nn.Module):
    def __init__(self):
        super(DownEncoder, self).__init__()
        self.Down1 = conv3(128, 256)
        self.Down2 = conv3(256, 512)
        self.Down3 = conv3(512, 1024)

    def forward(self, x):
        down1 = self.Down1(x)
        down2 = self.Down2(down1)
        down3 = self.Down3(down2)

        return down1, down2, down3
