import numpy as np
import torch.utils.data as Data
import torch.optim as optim
from sklearn.cluster import KMeans
import torch.nn as nn
import time
from thop import profile
from scipy import io
from einops import rearrange
from pre_data import set_train_sample
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
# from change_former import change_former
from ml_edan import ML_EDAN
from Global import *
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
from efficent import AA_andEachClassAccuracy
from tqdm import tqdm
from evaluation import two_cls_access
from sst import SSTViT


def train_test(device, train_inter, test_inter):
    net = ML_EDAN(in_channel=pca_band).to(device)

    # optimizer = optim.Adam(net.parameters(), lr)

    # net.apply(initNetParams_v2)
    optimizer = optim.Adam(net.parameters(), lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=1e-4)

    loser = 999
    loss = 0
    los = nn.CrossEntropyLoss()
    los2 = nn.L1Loss()
    print('---Training on {}---\n'.format(device))
    start = time.time()
    for epoch in range(epochs):
        train_acc_sum = 0
        loss = 0
        time_epoch = time.time()
        # for step, (x1_train, x2_train, y_train) in tqdm(enumerate(train_inter)):
        for step, (x1_train, x2_train, y_train) in enumerate(train_inter):
            # if step == 0:
            #     a = x1_train.to(device)
            #     b = x2_train.to(device)
            #     flops, params = profile(net, inputs=[a, b])
            #     print(flops)
            #     print(params)

            net.train()
            x1 = x1_train.to(device)
            x2 = x2_train.to(device)
            y_train = y_train.to(device)
            y_train = y_train.float()
            y_hat, t1, t2 = net(x1, x2)
            loss1 = los(y_hat, y_train)
            loss2 = los2(x1, t1)
            loss3 = los2(x1, t1)
            loss = 1 * loss1 + 0.5 * loss2 + 0.5 * loss3
            # loss = loss1
            optimizer.zero_grad()  # 梯度初始化为零
            loss.backward()
            optimizer.step()

            loss += loss.cpu().item()
            train_acc_sum += (y_hat.argmax(-1) == y_train.argmax(-1)).float().sum().cpu().item()
        train_ls = loss

        print('epoch %d, train loss %.6f, train acc %.6f, time %.2f sec' % (epoch + 1,
                                                                            train_ls / len(train_inter),
                                                                            train_acc_sum / (pos + neg),
                                                                            time.time() - time_epoch
                                                                            )
              )

    torch.save(net.state_dict(), 'best_' + type(net).__name__ + '_weights.pth')
        # if train_ls < loser:
        #     torch.save(net.state_dict(), 'best_' + type(net).__name__ + '_weights.pth')
        #     loser = train_ls
        #     print('model_saved')

    end = time.time()
    print('***Training End! Total Time %.1f sec***' % (end - start))

    net.load_state_dict(torch.load('best_' + type(net).__name__ + '_weights.pth'))  # 加载保存好的模型
    print('\n***Start  Testing***\n')
    net.eval()
    y_pre = torch.zeros(batch_size, 2).to(device)
    y_truth = torch.zeros(batch_size, 2).to(device)
    with torch.no_grad():
        for step, (x1_test, x2_test, y) in enumerate(test_inter):
            a1 = x1_test.to(device)
            a2 = x2_test.to(device)
            y = y.to(device)
            y_pred, b1, b2 = net(a1, a2)
            # del b1, b2
            # y_pred = y_pred.reshape(-1, 2)
            y_pre = torch.cat([y_pre, y_pred], dim=0)
            y_truth = torch.cat([y_truth, y], dim=0)

    y_pre = y_pre[batch_size::]
    y_truth = y_truth[batch_size::]

    y_pre = np.array(y_pre.cpu())
    y_truth = np.array(y_truth.cpu())

    f1 = two_cls_access(y_truth.argmax(-1), y_pre.argmax(-1))
    # 输出评价指标，accuracy表示准确率，macro avg表示宏平均，weighted avg表示带权重平均。

    oa = accuracy_score(y_truth.argmax(-1), y_pre.argmax(-1))  # 计算OA
    confusion = confusion_matrix(y_truth.argmax(-1), y_pre.argmax(-1))  # 计算confusion
    each_acc, aa = AA_andEachClassAccuracy(confusion)  # 计算each_acc和aa
    kappa = cohen_kappa_score(y_truth.argmax(-1), y_pre.argmax(-1))  # 计算kappa
    # print(f1)
    print('oa=   ' + str(oa))
    print('kappa=   ' + str(kappa))
    # print('F1=   ' + str(F1))

    # net.load_state_dict(torch.load('best_' + type(net).__name__ + '_weights.pth'))  # 加载保存好的模型

    # y_final = y_pre.argmax(-1).reshape(h, w)
    # plt.imshow(y_final, cmap='gray')
    # plt.axis('off')
    # plt.savefig('./' + str(dataset) + '/predict.png', dpi=1200)
    # plt.show()

    return oa * 100, kappa * 100, f1 * 100