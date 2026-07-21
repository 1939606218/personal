import time
import numpy as np
import sio
import torch.nn
from torch import nn
from model.model import ML_EDAN
import torch.optim as optim
from Global import *
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, recall_score, precision_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio

def calculate_metrics(y_true, y_pred):
    """计算基础评估指标（遵循原函数逻辑）"""
    cm = confusion_matrix(y_true, y_pred)
    is_binary = len(set(y_true).union(set(y_pred))) == 2

    if is_binary:
        TN, FP, FN, TP = cm.ravel()
        oa = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
    else:
        oa = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_true, y_pred)
        TN, FP, FN, TP = None, None, None, None

    return oa, f1, precision, recall, kappa, TN, FP, FN, TP, cm


def train(device, train_inter, model_path):
    net = ML_EDAN(in_channel=pca_channel).to(device)
    optimizer = optim.Adam(net.parameters(), lr)
    los = nn.CrossEntropyLoss()
    los2 = nn.L1Loss()

    print('---Training on {}---\n'.format(device))
    start = time.time()

    best_acc = 0.0  # 用于保存最佳准确率

    for epoch in range(EPOCHES):
        train_acc_sum = 0
        loss_sum = 0  # 重命名避免与loss变量冲突
        time_epoch = time.time()

        for step, (x1_train, x2_train, y_train, _) in enumerate(train_inter):
            net.train()
            x1 = x1_train.to(device)
            x2 = x2_train.to(device)
            y_train = y_train.to(device)  # 保持LongTensor类型，不要转为float

            y_hat, t1, t2 = net(x1, x2)

            # 确保y_hat形状为(batch_size, num_classes)
            # 确保y_train形状为(batch_size)，值为0到num_classes-1的整数

            loss1 = los(y_hat, y_train)
            loss2 = los2(x1, t1)
            loss3 = los2(x2, t2)
            loss = 1 * loss1 + 0.5 * loss2 + 0.5 * loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()  # 累加损失值

            # 计算准确率（直接比较预测类别和目标类别）
            train_acc_sum += (y_hat.argmax(-1) == y_train).sum().cpu().item()

        train_loss = loss_sum / len(train_inter)
        train_acc = train_acc_sum / len(train_inter.dataset)

        print('epoch %d, train loss %.6f, train acc %.6f, time %.2f sec, finish time %.2f' % (
            epoch + 1,
            train_loss,
            train_acc,
            time.time() - time_epoch,
            (time.time() - time_epoch) * (EPOCHES - epoch)
        ))

        # 保存最佳模型
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(net.state_dict(), model_path + '/best_model.pth')

    print(f"Model saved to {model_path + '/best_model.pth'}")
    end = time.time()
    print('***Training End! Total Time %.1f sec***' % (end - start))
    return end - start


def test(height, width, device, test_inter, model_path):
    # 加载模型
    model = ML_EDAN(in_channel=pca_channel).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, "best_model.pth")))
    model.eval()

    # 创建完整的预测结果矩阵（初始化为2，表示不确定）
    full_pred_labels = np.ones(height * width, dtype=np.int64) * 2

    # 存储真实标签和索引
    true_labels = []
    indices_list = []

    with torch.no_grad():
        for X1, X2, y, indices in test_inter:
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)
            output = model(X1, X2)

            # 提取分类结果（第一个元素）
            if isinstance(output, tuple):
                logits = output[0]  # 分类结果 [batch_size, 2]
            else:
                logits = output

            # 确保输出是二维张量 [batch_size, num_classes]
            _, predicted = torch.max(logits, 1)

            # 收集真实标签和索引
            true_labels.extend(y.cpu().numpy())
            indices_list.extend(indices.cpu().numpy())

            # 填充预测结果
            for i, idx in enumerate(indices.cpu().numpy()):
                full_pred_labels[idx] = predicted[i].cpu().numpy()

    # 计算评估指标
    true_labels = np.array(true_labels)
    pred_labels = full_pred_labels[indices_list]  # 只取有标签的样本

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = conf_matrix.ravel()

    # 计算评估指标
    oa = accuracy_score(true_labels, pred_labels)
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    precision_weighted = precision_score(true_labels, pred_labels, average='weighted')
    recall_weighted = recall_score(true_labels, pred_labels, average='weighted')
    kappa = cohen_kappa_score(true_labels, pred_labels)

    return oa, f1_weighted, precision_weighted, recall_weighted, kappa, tn, fp, fn, tp, conf_matrix, true_labels, full_pred_labels