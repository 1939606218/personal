import numpy as np
from scipy.io import loadmat

# 统计标签数量的函数
def count_labels(labels):
    """统计标签中各个值的出现次数"""
    # 方法一：使用NumPy
    unique_values, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_values, counts))

    # 方法二：使用Counter（适用于非数值标签）
    # label_counts = dict(Counter(labels.flatten()))

    return label_counts
def get_dataset(current_dataset):
    if current_dataset == 'farmland':
        data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm06.mat')['imgh']
        data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm07.mat')['imghl']
        data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\label.mat')['label']

    if current_dataset == 'hermiston':
        data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2004.mat')['HypeRvieW']
        data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2007.mat')['HypeRvieW']
        data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\label.mat')['label']

    if current_dataset == 'river':
        data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_before.mat')['river_before']
        data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_after.mat')['river_after']
        data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\groundtruth.mat')['lakelabel_v1']
        data_label[data_label == 255] = 1

    if current_dataset == 'Barbara':
        data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_2013.mat')['HypeRvieW']
        data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_2014.mat')['HypeRvieW']
        data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_gtChanges.mat')['HypeRvieW']
        # 修改标签映射：0→2，1→1，2→0
        data_label = np.select(
            [data_label == 0,data_label == 1, data_label == 2],
            [2, 1, 0],
            default=data_label  # 处理其他可能的值
        )
    if current_dataset == 'BayArea':
        data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2013.mat')['HypeRvieW']
        data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2015.mat')['HypeRvieW']
        data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\bayArea_gtChanges2.mat')['HypeRvieW']
        # 修改标签映射：0→2，1→1，2→0
        data_label = np.select(
            [data_label == 0, data_label == 1, data_label == 2],
            [2, 1, 0],
            default=data_label  # 处理其他可能的值
        )

    img1 =  data_t1.astype('float32')
    img2 = data_t2.astype('float32')
    gt = data_label.astype('float32')
    print(img1.shape,img2.shape,gt.shape)

    return img1, img2, gt
