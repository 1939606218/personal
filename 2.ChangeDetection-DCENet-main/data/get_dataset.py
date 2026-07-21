'''
 Dataset Source:
    Farmland: http://crabwq.github.io/
    Hermiston: https://citius.usc.es/investigacion/datasets/hyperspectral-change-detection-dataset
'''
import numpy as np
from scipy.io import loadmat

def get_Hermiston_dataset():
        data_set_before = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\Hermiston\hermiston2004.mat')['HypeRvieW']
        data_set_after = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\Hermiston\hermiston2007.mat')['HypeRvieW']
        ground_truth = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\Hermiston\label.mat')['label']
        img1 = data_set_before.astype('float32')  # (420, 140, 154)
        img2 = data_set_after.astype('float32')  # (420, 140, 154)
        gt = ground_truth.astype('float32')  # (420, 140)
        return img1, img2, gt

def get_farmland_dataset():
    data_set_before = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\farm\farm06.mat')['imgh']
    data_set_after = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\farm\farm07.mat')['imghl']
    ground_truth = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\farm\label.mat')['label']
    img1 = data_set_before.astype('float32')  # (420, 140, 154)
    img2 = data_set_after.astype('float32')  # (420, 140, 154)
    gt = ground_truth.astype('float32')  # (420, 140)
    return img1, img2, gt

def get_bayArea_dataset():
    data_set_before = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\bayArea\mat\Bay_Area_2013.mat')['HypeRvieW']
    data_set_after = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\bayArea\mat\Bay_Area_2015.mat')['HypeRvieW']
    ground_truth = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\bayArea\mat\bayArea_gtChanges2.mat.mat')['HypeRvieW']
    img1 = data_set_before.astype('float32')  # (420, 140, 154)
    img2 = data_set_after.astype('float32')  # (420, 140, 154)
    gt = ground_truth.astype('float32')  # (420, 140)
    return img1, img2, gt

def get_santaBarbara_dataset():
    data_set_before = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\santaBarbara\mat\barbara_2013.mat')['HypeRvieW']
    data_set_after = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\santaBarbara\mat\barbara_2014.mat')['HypeRvieW']
    ground_truth = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\santaBarbara\mat\barbara_gtChanges.mat')['HypeRvieW']
    img1 = data_set_before.astype('float32')  # (420, 140, 154)
    img2 = data_set_after.astype('float32')  # (420, 140, 154)
    gt = ground_truth.astype('float32')  # (420, 140)
    return img1, img2, gt

def get_river_dataset():
    data_set_before = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\river\river_before.mat')['river_before']
    data_set_after = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\river\river_after.mat')['river_after']
    ground_truth = loadmat(r'Z:\pycharm\pythonProject\2.ChangeDetection-DCENet-main\data\river\groundtruth.mat')['lakelabel_v1']
    img1 = data_set_before.astype('float32')  # (420, 140, 154)
    img2 = data_set_after.astype('float32')  # (420, 140, 154)
    gt = ground_truth.astype('float32')  # (420, 140)
    gt[gt == 255] = 1
    return img1, img2, gt

def get_dataset(current_dataset):
    if current_dataset == 'Hermiston':
        return get_Hermiston_dataset()  # Hermiston(307, 241, 154), gt[0. 1.]
    if current_dataset == 'farmland':
        return get_farmland_dataset()
    if current_dataset == 'bayArea':
        return get_bayArea_dataset()
    if current_dataset == 'santaBarbara':
        return get_santaBarbara_dataset()
    if current_dataset == 'river':
        return get_river_dataset()