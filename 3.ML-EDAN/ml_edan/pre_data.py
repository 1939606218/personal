import scipy.io as sio
import os
from Global import *
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.decomposition import PCA

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
device = torch.device('cuda:0')


def loadData(names):
    if names == 'river':
        data1 = sio.loadmat(os.path.join('data/river.mat'))['img_1']
        data2 = sio.loadmat(os.path.join('data/river.mat'))['img_2']
        labels = sio.loadmat(os.path.join('data/river.mat'))['GT']
        labels[labels == 255] = 1

    if names == 'farmland':
        data1 = sio.loadmat(os.path.join('data/farmland.mat'))['img_1']
        data2 = sio.loadmat(os.path.join('data/farmland.mat'))['img_2']
        labels = sio.loadmat(os.path.join('data/farmland.mat'))['GT']

    if names == 'hermiston':
        data1 = sio.loadmat(os.path.join('data/hermiston.mat'))['img_1']
        data2 = sio.loadmat(os.path.join('data/hermiston.mat'))['img_2']
        labels = sio.loadmat(os.path.join('data/hermiston.mat'))['GT']

    if names == 'Bay':
        data1 = sio.loadmat(os.path.join('data/Bay.mat'))['img_1']
        data2 = sio.loadmat(os.path.join('data/Bay.mat'))['img_2']
        labels = sio.loadmat(os.path.join('data/Bay.mat'))['GT']

    if names == 'USA':
        data1 = sio.loadmat(os.path.join('data/USA.mat'))['img_1']
        data2 = sio.loadmat(os.path.join('data/USA.mat'))['img_2']
        labels = sio.loadmat(os.path.join('data/USA.mat'))['GT']

    if names == 'farm420':
        data1 = sio.loadmat(os.path.join('data/farm420.mat'))['img_1']
        data2 = sio.loadmat(os.path.join('data/farm420.mat'))['img_2']
        labels = sio.loadmat(os.path.join('data/farm420.mat'))['GT']

    if names == 'farm430':
        data1 = sio.loadmat(os.path.join('data/farm430.mat'))['img_1']
        data2 = sio.loadmat(os.path.join('data/farm430.mat'))['img_2']
        labels = sio.loadmat(os.path.join('data/farm430.mat'))['GT']

    if names == 'Barbara':
        data1 = sio.loadmat(os.path.join('data/Barbara.mat'))['img_1']
        data2 = sio.loadmat(os.path.join('data/Barbara.mat'))['img_2']
        labels = sio.loadmat(os.path.join('data/Barbara.mat'))['GT']

    return data1, data2, labels

def normalize(X, type):
    x = np.zeros(shape=X.shape, dtype='float32')
    if type == 1:
        for i in range(X.shape[2]):
            temp = X[:, :, i]
            mean = np.mean(temp)
            std = np.std(temp)
            x[:, :, i] = ((temp - mean) / std)
    if type == 2:
        for i in range(X.shape[2]):
            min = np.min(X[:, :, i])
            max = np.max(X[:, :, i])
            scale = max - min
            if scale == 0:
                scale = 1e-5
            x[:, :, i] = (X[:, :, i] - min) / scale
    return x

def pad_with_zeros(X, margin=2):
    """apply zero padding to X with margin"""

    new_X = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    new_X[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return new_X


def create_patches(X, y, window_size, remove_zero_labels=True):
    """create patch from image. suppose the image has the shape (w,h,c) then the patch shape is
    (w*h,window_size,window_size,c)"""
    margin = int((window_size - 1) / 2)
    zero_padded_X = pad_with_zeros(X, margin=margin)
    # split patches
    patches_data = np.zeros((X.shape[0] * X.shape[1], window_size, window_size, X.shape[2]))
    patchs_labels = np.zeros((X.shape[0] * X.shape[1]))
    patch_index = 0
    for r in range(margin, zero_padded_X.shape[0] - margin):
        for c in range(margin, zero_padded_X.shape[1] - margin):
            patch = zero_padded_X[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patches_data[patch_index, :, :, :] = patch
            patchs_labels[patch_index] = y[r - margin, c - margin]
            patch_index = patch_index + 1

    # if remove_zero_labels:
    #     patches_data = patches_data[patchs_labels > 0, :, :, :]
    #     patchs_labels = patchs_labels[patchs_labels > 0]
    #     patchs_labels -= 1
    return patches_data, patchs_labels


def set_train_sample(x, y, pos, neg):
    np.random.seed(100)
    rand_perm = np.random.permutation(y.shape[0])  # 打乱111583
    new_x = x[rand_perm, :, :, :]
    new_y = y[rand_perm]  # new_x,new_y 保证一一对应

    train_x0 = new_x[new_y == 0, :, :, :][:neg]  # 取未变化的训练集2500个
    train_y0 = new_y[new_y == 0][:neg]  # 取未变化的训练集标签2500个
    train_x1 = new_x[new_y == 1, :, :, :][:pos]  # 取变化的训练集1250个
    train_y1 = new_y[new_y == 1][:pos]  # 取变化的训练集标签1250个

    test_x0 = new_x[new_y == 0, :, :, :][neg:]  # 取未变化的测试集从2500个开始
    test_y0 = new_y[new_y == 0][neg:]  # 取未变化的测试集标签从2500个开始
    test_x1 = new_x[new_y == 1, :, :, :][pos:]  # 取变化的测试集从1250个开始
    test_y1 = new_y[new_y == 1][pos:]  # 取变化的测试集标签从1250开始

    x_train = np.concatenate((train_x0, train_x1))  # 连接两个数据集
    y_train = np.concatenate((train_y0, train_y1))
    x_test = np.concatenate((test_x0, test_x1))
    y_test = np.concatenate((test_y0, test_y1))
    return x_train, x_test, y_train, y_test


def make_total_tensor(x1, x2):
    x1_tensor = torch.from_numpy(x1).type(torch.FloatTensor)  # 将 numpy 格式转换成 tensor 形式
    x2_tensor = torch.from_numpy(x2).type(torch.FloatTensor)
    return x1_tensor, x2_tensor


def generater(X_1, X_2, Y, batchsize, windowSize):
    alldataX1, alldataY = create_patches(X_1, Y, window_size=windowSize)
    alldataX1 = np.transpose(alldataX1, (0, 3, 1, 2))
    X1_train, X1_test, y_train, y_test = set_train_sample(alldataX1, alldataY, pos=pos, neg=neg)
    x1_train_vit = X1_train.reshape(X1_train.shape[0], windowSize * windowSize, X1_train.shape[1])
    x1_test_vit = X1_test.reshape(X1_test.shape[0], windowSize * windowSize, X1_test.shape[1])

    alldataX2, alldataY = create_patches(X_2, Y, window_size=windowSize)
    alldataX2 = np.transpose(alldataX2, (0, 3, 1, 2))
    X2_train, X2_test, y_train, y_test = set_train_sample(alldataX2, alldataY, pos=pos, neg=neg)
    x2_train_vit = X2_train.reshape(X2_train.shape[0], windowSize * windowSize, X2_train.shape[1])
    x2_test_vit = X2_test.reshape(X2_test.shape[0], windowSize * windowSize, X2_test.shape[1])

    ALL_SIZE = alldataX1.shape[0]
    TRAIN_SIZE = X1_train.shape[0]
    TEST_SIZE = X1_test.shape[0]

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
        batch_size=batchsize,
        shuffle=True,
        num_workers=0
    )

    test_iter = Data.DataLoader(
        dataset=torch_test,
        batch_size=batchsize,
        shuffle=False,
        num_workers=0
    )


    return train_iter, test_iter


def pre_data_loader(x1, x2, y):
    x1_patch, patch_y = create_patches(x1, y, window_size=windowSize)  # (78000, 9, 9 60)
    x1_patch = np.transpose(x1_patch, (0, 3, 1, 2))  # (78000, 60, 9, 9)
    x1_train, x1_test, y_train, y_test = set_train_sample(x1_patch, patch_y, pos, neg)

    x2_patch, patch_y = create_patches(x2, y, window_size=windowSize)
    x2_patch = np.transpose(x2_patch, (0, 3, 1, 2))
    x2_train, x2_test, y_train, y_test = set_train_sample(x2_patch, patch_y, pos, neg)

    x1_total_tensor, x2_total_tensor = make_total_tensor(x1_patch, x2_patch)

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



