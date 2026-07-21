# -*- coding:utf-8 -*-
"""
作者：张亦严
日期:2021年07月26日
"""
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def colormap():
    # cdict = ['#01018F', '#0101CF', '#000EFE', '#004EFE', '#018FFF', '#01CFFF', '#0FFFEF',
    #          '#4FFFAF', '#8FFF6F', '#CFFF2F', '#FFEF01', '#FFAF01', '#FF6F01', '#FF2F01',
    #          '#EF0101', '#AF0101', '#7F0101']   #CZH
    cdict = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#C0C0C0', '#808080', '#800000',
            '#808000', '#008000', '#800080', '#008080', '#000080', '#FFA500', '#FFD700']  # 自用

    return colors.ListedColormap(cdict)


def dis_groundtruth(gt):
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

    plt.show()


#
# def dis_groundtruth1(dataset, gt, flag):
#     '''plt.figure(title)
#     plt.title(title)'''
#     plt.imshow(gt, cmap=colormap2())
#     #spectral.imshow(classes=gt)
#     '''plt.colorbar()'''
#     plt.xticks([])
#     plt.yticks([])
#     '''plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)'''
#     plt.savefig('./results/{}/{}.png'.format(dataset, dataset + flag + 'PART'), dpi=600, pad_inches=0.0)
#
#     plt.show()
#
# def dis_groundtruth2(dataset, gt, flag):
#     '''plt.figure(title)
#     plt.title(title)'''
#     plt.imshow(gt, cmap=colormap1())
#     #spectral.imshow(classes=gt)
#     '''plt.colorbar()'''
#     plt.xticks([])
#     plt.yticks([])
#     '''plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)'''
#     plt.savefig('./results/{}/{}.png'.format(dataset, dataset + flag + 'ALL'), dpi=600, pad_inches=0.0)
#
#     plt.show()

'''
functions : 最简单的显示曲线, 可显示多条, 要求title一样即可
'''


def dis_curves_acc(title, data, xlabel='', ylabel='', label=''):
    plt.figure(title)
    plt.title(title, fontsize=14)
    plt.plot(data, label=label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig('./results/{}/acc.png'.format(title), dpi=600, pad_inches=0.0)
    plt.show()


def dis_curves_loss(title, data, xlabel='', ylabel='', label=''):
    plt.figure(title)
    plt.title(title, fontsize=14)
    plt.plot(data, label=label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig('./results/{}/loss.png'.format(title), dpi=600, pad_inches=0.0)
    plt.show()


def dis_curves(data_1, data_2, xlabel='', ylabel='', label=''):
    plt.figure(dpi=600)
    plt.plot(data_1, label=label)
    plt.plot(data_2, label=label)
    plt.legend(["Acc", "Loss"])
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True)
    plt.savefig('acc-loss.png', dpi=600, pad_inches=0.0)
    plt.show()


'''
functions： 显示地物类别
input：含有类别标记的矩阵, 画图的标题
output：地物分类的图片
'''


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y
