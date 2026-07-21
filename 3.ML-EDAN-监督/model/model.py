""""
if you want to use this code
please cite this paper:
@ARTICLE{9624977,
  author={Qu, Jiahui and Hou, Shaoxiong and Dong, Wenqian and Li, Yunsong and Xie, Weiying},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={A Multilevel Encoder–Decoder Attention Network for Change Detection in Hyperspectral Images},
  year={2022},
  volume={60},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2021.3130122}}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
# ablation experimtent
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1, stride=1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1, stride=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
#
# class AttentionBasicBlock(nn.Module):
#     def __init__(self, gc):
#         super(AttentionBasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(gc, gc, 3, 1, 1)
#         self.relu = nn.PReLU()
#         self.Channelattention = ChannelAttention(gc)
#         self.Spatitalattention = SpatialAttention(7)
#         self.conv2 = nn.Conv2d(gc, gc, 3, 1, 1)
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         out = x * self.Channelattention(x)
#         out = out * self.Spatitalattention(out)
#         out = x + out
#         out = self.relu(self.conv2(out))
#         return out

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, 2, batch_first=True)      #指定输入特征数量、隐藏层大小和层数（这里为 2 层）,表示输入数据的第一个维度是批量大小

    def forward(self, x):
        r_out, (h, c) = self.rnn(x, None)               #r_out是所有时间步的输出，(h, c)是最后一个时间步的隐藏状态和细胞状态

        return r_out[:, -1, :]     #形状是(batch_size, sequence_length, hidden_size),-1表示选取最后一个时间步的输出

class ML_EDAN(nn.Module):
    def __init__(self, in_channel):
        super(ML_EDAN, self).__init__()
        ## AE model
        self.cnn1 = AE(in_channel)
        self.cnn2 = AE(in_channel)
        ## LSTM
        self.lstm1 = RNN(6400, 512)
        self.lstm2 = RNN(2304, 256)
        self.lstm3 = RNN(512, 128)   # inchannel * height * width * cat  = 512 *1 *1
        self.lstm4 = RNN(4608, 512)  # 256 * 3 * 3 * 2
        self.lstm5 = RNN(12800, 1024)   # 256 * 5 *5 * 2
        ## FC
        self.linear1 = nn.Linear(1024, 256)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 32)
        self.linear = nn.Linear(416, 64)
        self.linear1_1 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, T1, T2):
        T1_out3, T1_out4, T1_out5, T1_out6 = self.cnn1(T1)
        T2_out3, T2_out4, T2_out5, T2_out6 = self.cnn2(T2)
        out_3 = torch.cat(
            [T1_out3.reshape(T1_out3.size(0), -1).unsqueeze(1), T2_out3.reshape(T2_out3.size(0), -1).unsqueeze(1)], dim=1)
        out_3 = self.lstm3(out_3)
        out_4 = torch.cat(
            [T1_out4.reshape(T1_out4.size(0), -1).unsqueeze(1), T2_out4.reshape(T2_out4.size(0), -1).unsqueeze(1)], dim=1)
        out_4 = self.lstm4(out_4)
        out_5 = torch.cat(
            [T1_out5.reshape(T1_out5.size(0), -1).unsqueeze(1), T2_out5.reshape(T2_out5.size(0), -1).unsqueeze(1)], dim=1)
        out_5 = self.lstm5(out_5)
        out_15 = self.linear1(out_5)
        out_24 = self.linear2(out_4)
        out_33 = self.linear3(out_3)
        out = torch.cat([out_33, out_24, out_15], dim=1)
        out = self.linear1_1(self.relu(self.linear(out)))
        return out, T1_out6, T2_out6


### Autoencoder
class AE(nn.Module):
    def __init__(self, in_channel):
        super(AE, self).__init__()
        ## encoder
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)

        ## decoder
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2)
        # # 在deconv1之后添加上采样操作
        # self.upsample_after_deconv1 = nn.Upsample(size=(3, 3), mode='nearest')
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        # self.upsample_after_deconv2 = nn.Upsample(size=(5, 5), mode='nearest')
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=in_channel, kernel_size=3, stride=1, padding=1)
        ## AttentionBlock
        # self.attentionblock1 = AttentionBasicBlock(256)
        # self.attentionblock2 = AttentionBasicBlock(256)
        self.relu = nn.ReLU(inplace=True)
        self.upsample1 = nn.Upsample(size = (3, 3), mode = 'nearest')  #最近邻插值
        self.Conv_attention1 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1, stride=1)
        self.trans1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.upsample2 = nn.Upsample(size=(5, 5), mode = 'nearest')
        self.Conv_attention2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.trans2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.deconv1(out3))
        ## CIGA Attention     #在通道上拼接
        out3_4 = self.sigmoid(self.Conv_attention1(torch.cat([out2, self.upsample1(out3)], dim=1))) * out2
        out4 = self.relu(self.trans1(torch.cat([out3_4, out4], dim=1)))
        ## CIGA Attention
        out3_5 = self.sigmoid(self.Conv_attention2(torch.cat([out1, self.upsample2(out4)], dim=1))) * out1
        out5 = self.relu(self.deconv2(out4))
        out5 = self.relu(self.trans2(torch.cat([out3_5, out5], dim=1)))
        out6 = self.deconv3(out5)
        out_24 = torch.cat([out2, out4], dim=1)
        out_15 = torch.cat([out1, out5], dim=1)
        ### return three middle feature maps and the reconstructed feature map
        return out3, out_24, out_15, out6

if __name__ == '__main__':
    # 初始化模型
    model = ML_EDAN(in_channel=128)
    model.eval()  # 设置为评估模式

    # 生成虚拟输入数据
    x1 = torch.randn(64, 128, 5, 5)  # [B, C, H, W]
    x2 = torch.randn(64, 128, 5, 5)

    # 获取模型输出
    output = model(x1, x2)

    print(f"模型输出类型: {type(output)}")
    print(f"模型输出长度: {len(output)}")

    # 打印元组中每个元素的信息
    for i, item in enumerate(output):
        print(f"元素 {i}:")
        print(f"  类型: {type(item)}")
        if isinstance(item, torch.Tensor):
            print(f"  形状: {item.shape}")
            if item.dim() == 2 and item.size(1) == 2:  # 检查是否为 [B, 2] 的分类结果
                print(f"  可能是分类结果！")
        else:
            print(f"  值: {item}")

    # 提取分类结果（假设在第一个位置）
    if isinstance(output, tuple):
        logits = output[0]
        _, predicted = torch.max(logits, 1)
        print(f"\n预测结果形状: {predicted.shape}")
        print(f"预测结果前10个值: {predicted[:10]}")

    # ------------------- 计算参数量 -------------------
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())


    total_params = count_parameters(model)
    print(f"总参数量: {total_params / 1e6:.2f}M")

    # ------------------- 计算FLOPs -------------------
    # 使用thop库自动统计
    flops, params = profile(model, inputs=(x1, x2), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Params: {params}")
