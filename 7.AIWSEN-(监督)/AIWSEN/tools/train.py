import torch
import time
import datetime
import math
import os

from torch import nn
from torch.utils.data import DataLoader


def adjust_lr_sub(lr_init, lr_gamma, optimizer, epoch, step_index):
    if epoch < 1:
        lr = 0.0001 * lr_init
    elif epoch <= step_index[0]:
        lr = lr_init
    elif epoch <= step_index[1]:
        lr = lr_init * lr_gamma
    elif epoch > step_index[1]:
        lr = lr_init * lr_gamma ** 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train(train_data, model, loss_fun, optimizer, device, cfg):
    """训练模型"""
    torch.autograd.set_detect_anomaly(True)

    # 配置参数
    num_workers = cfg['workers_num']
    gpu_num = cfg['gpu_num']
    save_folder = cfg['save_folder']
    save_name = cfg['save_name']
    lr_init = cfg['lr']
    lr_gamma = cfg['lr_gamma']
    lr_step = cfg['lr_step']
    lr_adjust = cfg['lr_adjust']
    epoch_size = cfg['epoch']
    batch_size = cfg['batch_size']

    # 模型设备配置
    if gpu_num > 1 and cfg['gpu_train']:
        model = nn.DataParallel(model)
    model = model.to(device)

    # 验证模型是否在GPU上
    print(f"Model device: {next(model.parameters()).device}")

    # 加载模型权重（如果需要）
    if cfg['reuse_model']:
        print('Loading model...')
        checkpoint = torch.load(cfg['reuse_file'], map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
    else:
        start_epoch = 0

    # 检查train_data类型
    if isinstance(train_data, DataLoader):
        # 如果train_data已经是DataLoader，则直接使用
        batch_data = train_data
        batch_num = len(train_data)
        dataset_size = len(train_data.dataset)
    else:
        # 如果train_data是Dataset，则创建DataLoader
        batch_data = DataLoader(train_data, batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True)
        batch_num = math.ceil(len(train_data) / batch_size)
        dataset_size = len(train_data)

    train_loss_save = []
    train_acc_save = []

    print(f'Starting training on {dataset_size} samples with {batch_num} batches...')

    for epoch in range(start_epoch + 1, epoch_size + 1):
        epoch_time0 = time.time()
        epoch_loss = 0
        predict_correct = 0
        label_num = 0

        # 调整学习率
        if lr_adjust:
            lr = adjust_lr_sub(lr_init, lr_gamma, optimizer, epoch, lr_step)
        else:
            lr = lr_init

        for batch_idx, batch_sample in enumerate(batch_data):
            iteration = (epoch - 1) * batch_num + batch_idx + 1
            batch_time0 = time.time()

            # 获取数据并移至GPU
            img1, img2, target, indices = batch_sample
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # 前向传播和反向传播
            prediction = model(img1, img2)
            loss = loss_fun(prediction, target.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算准确率
            epoch_loss += loss.item()
            predict_label = prediction.detach().argmax(dim=1, keepdim=True)
            predict_correct += predict_label.eq(target.view_as(predict_label)).sum().item()
            label_num += len(target)

        # 计算训练准确率
        train_acc = 100 * predict_correct / label_num if label_num > 0 else 0
        epoch_time1 = time.time()
        epoch_time = epoch_time1 - epoch_time0
        epoch_eta = int(epoch_time * (epoch_size - epoch))

        # 打印训练进度
        print('Epoch: {}/{} || lr: {} || loss: {} || Train acc: {:.2f}% || '
              'Epoch time: {:.4f}s || Epoch ETA: {}'
              .format(epoch, epoch_size, lr, epoch_loss / batch_num, train_acc,
                      epoch_time, str(datetime.timedelta(seconds=epoch_eta)))
              )

        # 保存训练记录
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        train_loss_save.append(epoch_loss / batch_num)
        train_acc_save.append(train_acc)

    # 保存最终模型
    save_model = dict(
        model=model.state_dict(),
        epoch=epoch_size
    )
    torch.save(save_model, os.path.join(save_folder, save_name + '_Final.pth'))

    return train_loss_save, train_acc_save