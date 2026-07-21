import torch
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
import numpy as np
# 计算 Kappa, Precision, Recall, F1-score 和 OA
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score


def calculate_metrics(y_true, y_pred):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 检查是否为二分类问题
    is_binary = len(set(y_true).union(set(y_pred))) == 2

    if is_binary:
        # 二分类场景
        TN, FP, FN, TP = cm.ravel()
        oa = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
    else:
        # 多分类场景
        oa = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_true, y_pred)
        # 多分类场景下 TN、FP、FN、TP 无明确意义，可设为 None
        TN, FP, FN, TP = None, None, None, None

    # 返回所有指标和混淆矩阵元素
    return oa, f1, precision, recall, kappa, TN, FP, FN, TP


# 训练函数（支持多输出）
def train(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []

    for X1, X2, Y in train_loader:
        X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)
        optimizer.zero_grad()

        # 前向传播（获取三个输出）
        main_out= model(X1, X2)

        # 计算各分支损失
        total_loss = criterion(main_out, Y)

        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # 记录主输出结果
        running_loss += total_loss.item()
        _, preds = torch.max(main_out, 1)
        all_labels.extend(Y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    # 计算指标
    oa, f1, precision, recall, kappa, *_ = calculate_metrics(all_labels, all_preds)
    avg_loss = running_loss / len(train_loader)
    return avg_loss, oa, f1, precision, recall, kappa

# 测试函数（同理修改）
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for X1, X2, Y in test_loader:
            X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)
            main_out= model(X1, X2)

            # 计算测试损失（与训练逻辑一致）
            total_loss = criterion(main_out, Y)

            # 记录结果
            running_loss += total_loss.item()
            _, preds = torch.max(main_out, 1)
            all_labels.extend(Y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 计算指标
    oa, f1, precision, recall, kappa, *_ = calculate_metrics(all_labels, all_preds)
    avg_loss = running_loss / len(test_loader)
    return avg_loss, oa, f1, precision, recall, kappa


def run_training(model, train_loader, test_loader, epochs, optimizer, criterion, device, scheduler,
                 best_model_path="best_model.pth", patience=10, threshold=0.01):
    total_start_time = time.time()
    epoch_durations = []
    train_losses = []
    train_time = 0  # 记录纯训练时间

    # 初始化最佳模型记录
    best_value = float('-inf')
    best_epoch = 0
    
    # 收敛检测参数
    converged = False
    loss_history = []
    converge_epoch = 0

    print('Training on', device)

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # 开始训练计时器
        train_start = time.time()

        # 训练阶段
        train_loss, train_oa, train_f1, train_precision, train_recall, train_kappa = train(
            model, train_loader, optimizer, criterion, device, epoch, epochs
        )
        print(f"[Epoch {epoch + 1}/{epochs}] "
              f"[Train] Loss: {train_loss:.4f} | OA: {train_oa:.4f} | F1: {train_f1:.4f} | "
              f"Pr: {train_precision:.4f} | Re: {train_recall:.4f} | kappa: {train_kappa:.4f}")
        train_losses.append(train_loss)
        
        # 更新损失历史并检测收敛
        loss_history.append(train_loss)
        if len(loss_history) > patience:
            loss_history = loss_history[-patience:]
            # 计算最近patience个epoch的损失变化率
            loss_change = abs(loss_history[-1] - loss_history[0]) / loss_history[0]
            if loss_change < threshold and not converged:
                converged = True
                converge_epoch = epoch + 1
                print(f"🔍 Loss converged at epoch {converge_epoch}! Starting periodic testing...")

        # 结束训练计时器并累加
        train_end = time.time()
        train_time += (train_end - train_start)

        # 更新学习率
        scheduler.step()

        # 只有在收敛后才每5个epoch测试一次
        if (converged and (epoch + 1 - converge_epoch) % 99 == 0) or (epoch + 1) == epochs:
        # if (epoch + 1) == epochs:
            print(f"Testing at epoch {epoch + 1}...")
            model.eval()
            test_loss, test_oa, test_f1, test_precision, test_recall, test_kappa = test(
                model, test_loader, criterion, device
            )
            model.train()

            # 计算value
            value = test_oa + test_f1 + test_precision + test_recall + test_kappa
            print(f"[Test] Loss: {test_loss:.4f} | OA: {test_oa:.4f} | F1: {test_f1:.4f} | "
                  f"Pr: {test_precision:.4f} | Re: {test_recall:.4f} | kappa: {test_kappa:.4f} | Value: {value:.4f}")

            # 更新最佳模型
            if value > best_value:
                best_value = value
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ New best model saved at epoch {best_epoch} with value: {best_value:.4f}")

        # 时间信息
        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)
        avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = epochs - (epoch + 1)
        remaining_time = remaining_epochs * avg_epoch_time

        elapsed_time = timedelta(seconds=int(time.time() - total_start_time))
        remaining_time = timedelta(seconds=int(remaining_time))

        # print(f"⏳ Elapsed: {elapsed_time} | Est. Remain: {remaining_time}")

    # 返回训练时间（秒）而不是timedelta对象
    total_train_seconds = np.double(train_time)
    print(f"\n✅ Training completed in {timedelta(seconds=total_train_seconds)}")
    print(f"Best model found at epoch {best_epoch} with value: {best_value:.4f}")

    return total_train_seconds, best_epoch, best_value


def predict_full_dataset(model, data_loader, device):
    """预测整个数据集（包括训练集和测试集）"""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for X1, X2 in data_loader:
            X1, X2 = X1.to(device), X2.to(device)
            outputs = model(X1, X2)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds)