from tqdm import tqdm
import numpy as np
import scipy.io as sio
import os
from Global import *
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.decomposition import PCA

def patch_data(data, l, location):
    h, w, c = data.shape
    patch_data = []
    if l == 1:
        for idx in tqdm(list(location)):
            i, j = idx
            patch_data.append(data[i, j, :].reshape(-1))
    else:
        # for i in range(S):
        #    data[:,:,i]=(data[:,:,i] - np.mean(data[:,:,i]))/np.std(data[:,:i])
        for idx in tqdm(list(location)):
            mask = np.float32(np.zeros([l, l, c]))
            i, j = idx
            up = i - int(l / 2)
            down = i + int(l / 2)
            left = j - int(l / 2)
            right = j + int(l / 2)
            up = 0 if up < 0 else up
            left = 0 if left < 0 else left
            down = h - 1 if down > h - 1 else down
            right = w - 1 if right > w - 1 else right
            mask[int(l / 2) - (i - up):int(l / 2) + down - i + 1, int(l / 2) - (j - left):int(l / 2) + right - j + 1,
            :] = data[up:down + 1, left:right + 1, :]
            patch_data.append(mask)
    print('Data patched.')
    patch_data = np.float32(np.array(patch_data))
    patch_data = patch_data.reshape(-1, c, l, l)
    return patch_data


def get_location(label):
    location = []
    labels = []
    h, w = label.shape
    for i in range(h):
        for j in range(w):
            if label[i, j] != 0:
                location.append([i, j])
                labels.append(label[i, j]-1)
    location = np.array(location)
    labels = np.array(labels)
    return location, labels


def set_train_bay_sample(x, y, pos, neg):
    np.random.seed(100)
    rand_perm = np.random.permutation(y.shape[0])  # 打乱111583
    new_x = x[rand_perm, :, :, :]
    new_y = y[rand_perm]  # new_x,new_y 保证一一对应

    train_x0 = new_x[new_y == 1, :, :, :][:neg]  # 取未变化的训练集2500个
    train_y0 = new_y[new_y == 1][:neg]  # 取未变化的训练集标签2500个
    train_x1 = new_x[new_y == 0, :, :, :][:pos]  # 取变化的训练集1250个
    train_y1 = new_y[new_y == 0][:pos]  # 取变化的训练集标签1250个

    test_x0 = new_x[new_y == 1, :, :, :][neg:]  # 取未变化的测试集从2500个开始
    test_y0 = new_y[new_y == 1][neg:]  # 取未变化的测试集标签从2500个开始
    test_x1 = new_x[new_y == 0, :, :, :][pos:]  # 取变化的测试集从1250个开始
    test_y1 = new_y[new_y == 0][pos:]  # 取变化的测试集标签从1250开始

    x_train = np.concatenate((train_x0, train_x1))  # 连接两个数据集
    y_train = np.concatenate((train_y0, train_y1))
    x_test = np.concatenate((test_x0, test_x1))
    y_test = np.concatenate((test_y0, test_y1))
    return x_train, x_test, y_train, y_test

def pre_bay_input_dbs3tan(a, b, y):
    a_tensor = torch.from_numpy(a).type(torch.FloatTensor)
    b_tensor = torch.from_numpy(b).type(torch.FloatTensor)
    c_tensor = a_tensor.reshape(a_tensor.shape[0], windowSize * windowSize, a_tensor.shape[1])
    d_tensor = b_tensor.reshape(b_tensor.shape[0], windowSize * windowSize, b_tensor.shape[1])

    torch_total = Data.TensorDataset(a_tensor, b_tensor, c_tensor, d_tensor)


    X1_train, X1_test, y_train, y_test = set_train_bay_sample(a, y, pos=pos, neg=neg)
    x1_train_vit = X1_train.reshape(X1_train.shape[0], windowSize * windowSize, X1_train.shape[1])
    x1_test_vit = X1_test.reshape(X1_test.shape[0], windowSize * windowSize, X1_test.shape[1])

    X2_train, X2_test, y_train, y_test = set_train_bay_sample(b, y, pos=pos, neg=neg)
    x2_train_vit = X2_train.reshape(X2_train.shape[0], windowSize * windowSize, X2_train.shape[1])
    x2_test_vit = X2_test.reshape(X2_test.shape[0], windowSize * windowSize, X2_test.shape[1])

    X1_train_tensor = torch.from_numpy(X1_train).type(torch.FloatTensor)
    X2_train_tensor = torch.from_numpy(X2_train).type(torch.FloatTensor)
    Y_train_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)
    X1_train_vit_tensor = torch.from_numpy(x1_train_vit).type(torch.FloatTensor)
    x2_train_vit_tensor = torch.from_numpy(x2_train_vit).type(torch.FloatTensor)

    X1_test_tensor = torch.from_numpy(X1_test).type(torch.FloatTensor)
    X2_test_tensor = torch.from_numpy(X2_test).type(torch.FloatTensor)
    Y_test_tensor = torch.from_numpy(y_test).type(torch.FloatTensor)
    x1_test_vit_tensor = torch.from_numpy(x1_test_vit).type(torch.FloatTensor)
    x2_test_vit_tensor = torch.from_numpy(x2_test_vit).type(torch.FloatTensor)

    torch_train = Data.TensorDataset(X1_train_tensor, X2_train_tensor, X1_train_vit_tensor, x2_train_vit_tensor,
                                     Y_train_tensor)
    torch_test = Data.TensorDataset(X1_test_tensor, X2_test_tensor, x1_test_vit_tensor, x2_test_vit_tensor,
                                    Y_test_tensor)

    train_iter = Data.DataLoader(
        dataset=torch_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_iter = Data.DataLoader(
        dataset=torch_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    total_iter = Data.DataLoader(
        dataset=torch_total,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_iter, test_iter, total_iter


def pre_data_bay_loader(x1, x2, y):
    x1_train, x1_test, y_train, y_test = set_train_bay_sample(x1, y, pos, neg)  # 变化为0 未变化为1

    x2_train, x2_test, y_train, y_test = set_train_bay_sample(x2, y, pos, neg)

    x1_total_tensor = torch.from_numpy(x1).type(torch.FloatTensor)
    x2_total_tensor = torch.from_numpy(x2).type(torch.FloatTensor)

    x1_train_tensor = torch.from_numpy(x1_train).type(torch.FloatTensor)  # 将 numpy 格式转换成 tensor 形式
    x2_train_tensor = torch.from_numpy(x2_train).type(torch.FloatTensor)
    y_train_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)

    x1_test_tensor = torch.from_numpy(x1_test).type(torch.FloatTensor)
    x2_test_tensor = torch.from_numpy(x2_test).type(torch.FloatTensor)
    y_test_tensor = torch.from_numpy(y_test).type(torch.FloatTensor)

    y_train_tensor = y_train_tensor.to(torch.int64)
    y_train_tensor = F.one_hot(y_train_tensor)
    y_test_tensor = y_test_tensor.to(torch.int64)
    y_test_tensor = F.one_hot(y_test_tensor)

    torch_train = Data.TensorDataset(x1_train_tensor, x2_train_tensor, y_train_tensor)
    torch_test = Data.TensorDataset(x1_test_tensor, x2_test_tensor, y_test_tensor)
    torch_total = Data.TensorDataset(x1_total_tensor, x2_total_tensor)

    train_iter = Data.DataLoader(
        dataset=torch_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_iter = Data.DataLoader(
        dataset=torch_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    total_iter = Data.DataLoader(
        dataset=torch_total,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_iter, test_iter, total_iter

def apply_pca(X, num_components=75):
    new_x = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_x = pca.fit_transform(new_x)
    new_x = np.reshape(new_x, (X.shape[0], X.shape[1], num_components))
    return new_x