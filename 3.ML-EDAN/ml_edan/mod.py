# -*- coding:utf-8 -*-
"""
作者：张亦严
日期:2022年10月21日
"""
import time
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
import numpy as np
from operator import truediv
import random
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from evaluation import two_cls_access
num_class = 15


def load_data(names):
    # data_path = r'/home/ubuntu/dataset_RS/Multisource/data'
    # data_path = r'F:\science\data_server'
    data_path = os.path.join(r'/home/ubuntu/dataset_RS/change')
    if names == 'river':
        data1 = sio.loadmat(os.path.join(data_path, 'river/river_before.mat'))['river_before']
        data2 = sio.loadmat(os.path.join(data_path, 'river/river_after.mat'))['river_after']
        labels = sio.loadmat(os.path.join(data_path, 'river/groundtruth.mat'))['lakelabel_v1']
        labels[labels == 0] = 1
        labels[labels == 255] = 2

    if names == 'farm':
        data1 = sio.loadmat(os.path.join(data_path, 'farm/farm06.mat'))['imgh']
        data2 = sio.loadmat(os.path.join(data_path, 'farm/farm07.mat'))['imghl']
        labels = sio.loadmat(os.path.join(data_path, 'farm/label.mat'))['label']
        labels[labels == 1] = 2
        labels[labels == 0] = 1
    if names == 'Hermiston':
        data1 = sio.loadmat(os.path.join(data_path, 'Hermiston/hermiston2004.mat'))['HypeRvieW']
        data2 = sio.loadmat(os.path.join(data_path, 'Hermiston/hermiston2007.mat'))['HypeRvieW']
        labels = sio.loadmat(os.path.join(data_path, 'Hermiston/label.mat'))['label']
        labels[labels == 1] = 2
        labels[labels == 0] = 1
    data = abs(data1 - data2)

    return data, data1, data2, labels

def pixel_select(Y):
    test_pixels = Y.copy()
    kinds = np.unique(Y).shape[0]  # np.unique(Y)=array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16],dtype=uint8) ,kinds=分类种类数
    # print(kinds)
    for i in range(kinds):
        num = np.sum(Y == (i+1))  # 计算每个类总共有多少样本 ,从Y=0到Y=1
        train_num = [500, 500]  #river 不变：1250 变：2500  farm 不变：8800 变化:4400
        temp1 = np.where(Y == (i+1))  # 返回标签满足第i+1类的位置索引，第一次循环返回第一类的索引
        temp2 = random.sample(range(num), train_num[i])  # get random sequence,random.sample表示从某一序列中随机获取所需个数（train_num）的数并以片段的形式输出,,再这里将随机从每个种类中挑选train_num个样本
        for i in temp2:
            test_pixels[temp1[0][temp2], temp1[1][temp2]] = 0  # 除去训练集样本

    train_pixels = Y - test_pixels
    return train_pixels, test_pixels

def set_train_sample(x, y, pos, neg):
    np.random.seed(100)
    rand_perm = np.random.permutation(y.shape[0])  # 打乱111583
    new_x = x[rand_perm, :]
    new_y = y[rand_perm]  # new_x,new_y 保证一一对应

    train_x0 = new_x[new_y == 0, :][:neg]  # 取未变化的训练集2500个
    train_y0 = new_y[new_y == 0][:neg]  # 取未变化的训练集标签2500个
    train_x1 = new_x[new_y == 1, :][:pos]  # 取变化的训练集1250个
    train_y1 = new_y[new_y == 1][:pos]  # 取变化的训练集标签1250个

    test_x0 = new_x[new_y == 0, :][neg:]  # 取未变化的测试集从2500个开始
    test_y0 = new_y[new_y == 0][neg:]  # 取未变化的测试集标签从2500个开始
    test_x1 = new_x[new_y == 1, :][pos:]  # 取变化的测试集从1250个开始
    test_y1 = new_y[new_y == 1][pos:]  # 取变化的测试集标签从1250开始

    x_train = np.concatenate((train_x0, train_x1))  # 连接两个数据集
    y_train = np.concatenate((train_y0, train_y1))
    x_test = np.concatenate((test_x0, test_x1))
    y_test = np.concatenate((test_y0, test_y1))
    return  x_train, x_test, y_train, y_test

def sequence_process_traindition(x, y, pos, neg):
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1)
    x_train, x_test, y_train, y_test = set_train_sample(x, y, pos, neg)
    # train_pixels, test_pixels = pixel_select(y)
    # y_train = train_pixels.reshape(-1)
    # y_test = test_pixels.reshape(-1)
    # y = y.reshape(-1)
    #
    # X_train = x[y_train!= 0]
    # y_train = y_train[y_train!=0]
    # X_test = x[y_test!= 0]
    # y_test = y_test[y_test!=0]

    return x_train, y_train, x_test, y_test,  x, y


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)                        #获取confusion_matrix的主对角线所有数值
    list_raw_sum = np.sum(confusion_matrix, axis=1)              #将主对角线所有数求和
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))   #list_diag/list_raw_sum  对角线各个数字/对角线所有数字的总和
    average_acc = np.mean(each_acc)
    return np.round(each_acc, 4), average_acc

def reports(X_test, Y_test, clf):
    tick1 = time.time()
    Y_pred = clf.predict(X_test)
    tick2 = time.time()
    Test_time = tick2 - tick1
    print('***Testing End! Testing Time %.2fs***\n' % (Test_time))
    # Y_pred[Y_pred == 1] = 0
    # Y_pred[Y_pred == 2] = 1
    # Y_test[Y_test == 1] = 0
    # Y_test[Y_test == 2] = 1
    # target_names = ['1', '2']
    # classification = classification_report(Y_test, Y_pred, target_names=target_names, digits=4)
    # oa = accuracy_score(Y_test, Y_pred)  # 计算OA
    # confusion = confusion_matrix(Y_test, Y_pred)  # 计算confusion
    # each_acc, aa = AA_andEachClassAccuracy(confusion)  # 计算each_acc和aa
    # kappa = cohen_kappa_score(Y_test, Y_pred)  # 计算kappa

    return Y_pred, Y_test


def result(X_test, Y_test, OA, AA, EACH_ACC, KAPPA, dataset, clf, traintime):
    Y_pred, Y_test = reports(X_test, Y_test, clf)
    two_cls_access(Y_pred, Y_test)
    # OA.append(oa)
    # AA.append(aa)
    # KAPPA.append(kappa)
    # EACH_ACC.append(each_acc)
    # classification = str(classification)
    # file_name = "./results/{}/{}.txt".format(dataset, dataset)
    # with open(file_name, 'a') as x_file:
    #     x_file.write('\n**************************************************************************************\n')
    #     x_file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    #     x_file.write('\n')
    #     x_file.write('{} Overall accuracy (%)'.format(oa))
    #     x_file.write('\n')
    #     x_file.write('{} Average accuracy (%)'.format(aa))
    #     x_file.write('\n')
    #     x_file.write('{} Kappa accuracy (%)'.format(kappa))
    #     x_file.write('\n')
    #     x_file.write('\n')
    #     x_file.write('Training Time {}s'.format(traintime))
    #     x_file.write('\n')
    #     x_file.write('Testing Time {}s'.format(Test_time))
    #     x_file.write('\n')
    #     x_file.write('\n')
    #     x_file.write('{}'.format(classification))
    #     x_file.write('\n')
    #     x_file.write('\n')
    #     x_file.write('mean_OA +- std_OA is: ' + str(np.mean(OA)) + ' +- ' + str(np.std(OA)))
    #     x_file.write('\n')
    #     x_file.write('mean_AA +- std_AA is: ' + str(np.mean(AA)) + ' +- ' + str(np.std(AA)))
    #     x_file.write('\n')
    #     x_file.write('mean_KAPPA +- std_KAPPA is: ' + str(np.mean(KAPPA)) + ' +- ' + str(np.std(KAPPA)))
    #     x_file.write('\n')
    #     x_file.write('\n')
    #     x_file.write('Each class mean_OA +- std_OA is: ' + str(np.mean(EACH_ACC, axis=0))+'\n'           + ' +- '           + str(np.std(EACH_ACC, axis=0)))
    #     x_file.write('\n**************************************************************************************\n')
    #     x_file.write('\n')
    #     x_file.write('\n')


def colormap():
        # cdict = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#C0C0C0',
        #          '#808080', '#800000', '#808000', '#008000', '#800080', '#008080', '#000080', '#FFA500', '#FFD700']
        cdict = ['#000000', '#FFFFFF']
        return colors.ListedColormap(cdict)


def dis_groundtruth(dataset, gt , p=True):
    '''plt.figure(title)
    plt.title(title)'''
    plt.imshow(gt, cmap=colormap())
    # spectral.imshow(classes=gt)
    '''plt.colorbar()'''
    plt.xticks([])
    plt.yticks([])
    '''plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)'''
    if p:
        plt.savefig('./results/{}/{}.png'.format(dataset, dataset+'true'), dpi=1200, pad_inches=0.0)
    else:
        plt.savefig('./results/{}/{}.png'.format(dataset, dataset+'false'), dpi=1200, pad_inches=0.0)
    plt.show()


