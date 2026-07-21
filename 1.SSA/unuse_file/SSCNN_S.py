#coding=utf-8

from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
import sys

class SSCNN_S(nn.Module):
    def __init__(self):
        super(SSCNN_S, self).__init__()
        self.name = 'SSCNN-S'
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(24),
            nn.PReLU(24),

            nn.Conv3d(24, 12, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(12),
            nn.PReLU(12),

            nn.Conv3d(12, 12, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(12),
            nn.PReLU(12),

            nn.Conv3d(12, 12, kernel_size=(3, 1, 1),padding=(1, 0, 0) ),
            nn.BatchNorm3d(12),
            nn.PReLU(12),

            nn.Conv3d(12, 85, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(85),
            nn.PReLU(85),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(85, 12, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(12),
            nn.PReLU(12),

            nn.Conv3d(12, 12, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(12),
            nn.PReLU(12),

            nn.Conv3d(12, 12, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(12),
            nn.PReLU(12),

            nn.MaxPool3d(3, stride=2),
            nn.Flatten(),
            nn.Dropout3d(p=0.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4704, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        size = x.size()[0]
        x = x.view(size, 1, 198, 5, 5)
        output = self.conv1(x)
        output = self.conv2(output)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        dist = F.pairwise_distance(output1, output2)
        dist = self.sigmoid(dist)
        return output1, output2, dist

#Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
