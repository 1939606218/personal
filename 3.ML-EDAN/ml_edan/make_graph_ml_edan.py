import numpy as np
import torch.utils.data as Data
import torch.optim as optim
from sklearn.cluster import KMeans
import torch.nn as nn
import time
from scipy import io
from einops import rearrange
from pre_data import set_train_sample
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from ml_edan import ML_EDAN
# from change_former import change_former
from Global import *
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
from efficent import AA_andEachClassAccuracy
from tqdm import tqdm


def make_bay_graph(h, w, device, total_inter, location):
    net = ML_EDAN(in_channel=pca_band).to(device)

    net.eval()
    y_total_pred = torch.zeros(batch_size, 2).to(device)
    net.load_state_dict(torch.load('best_' + type(net).__name__ + '_weights.pth'))  # 加载保存好的模型
    final = np.zeros((h, w))
    with torch.no_grad():
        for step, (x1_total, x2_total) in enumerate(total_inter):
            a1 = x1_total.to(device)
            a2 = x2_total.to(device)

            y_pred, m, n = net(a1, a2)

            y_total_pred = torch.cat([y_total_pred, y_pred], dim=0)

    y_total_pred = y_total_pred[batch_size::]
    y_total_pred = np.array(y_total_pred.cpu())
    y_final = y_total_pred.argmax(-1) + 1
    for i in range(location.shape[0]):
        a, b = location[i]
        final[a, b] = y_final[i]

    final[final == 1] = 255
    final[final == 2] = 1
    final[final == 255] = 2

    plt.imshow(final, cmap='gray')
    plt.axis('off')
    plt.savefig('./' + str(dataset) + '/predict.png', dpi=1200)
    plt.show()


def make_graph(h, w, device, total_inter):
    net = ML_EDAN(in_channel=pca_band).to(device)

    net.eval()
    y_total_pred = torch.zeros(batch_size, 2).to(device)
    net.load_state_dict(torch.load('best_' + type(net).__name__ + '_weights.pth'))  # 加载保存好的模型
    with torch.no_grad():
        for step, (x1_total, x2_total) in enumerate(total_inter):
            a1 = x1_total.to(device)
            a2 = x2_total.to(device)

            y_pred, m, n = net(a1, a2)
            y_total_pred = torch.cat([y_total_pred, y_pred], dim=0)

    y_total_pred = y_total_pred[batch_size::]
    y_total_pred = np.array(y_total_pred.cpu())
    y_final = y_total_pred.argmax(-1).reshape(h, w)

    plt.imshow(y_final, cmap='gray')
    plt.axis('off')
    plt.savefig('./' + str(dataset) + '/predict.png', dpi=1200)
    plt.show()