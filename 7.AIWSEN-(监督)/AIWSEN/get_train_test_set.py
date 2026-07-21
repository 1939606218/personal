import data_preprocess as data_preprocess
from get_dataset import get_dataset as getdata
import torch
import numpy as np
from sklearn.decomposition import PCA

def pca_transform(img, n_components=30):
    """
    对高光谱图像进行PCA降维
    参数:
    img (torch.Tensor): 输入图像，形状为 [C, H, W]
    n_components (int): 降维后的维度

    返回:
    torch.Tensor: 降维后的图像，形状为 [n_components, H, W]
    """
    # 将图像转换为numpy数组并重塑为 [C, H*W]
    c, h, w = img.shape
    img_2d = img.reshape(c, -1).numpy().T  # [H*W, C]

    # 应用PCA降维
    pca = PCA(n_components=n_components)
    img_pca = pca.fit_transform(img_2d)  # [H*W, n_components]

    # 将结果转换回 [n_components, H, W] 的形状
    img_pca = img_pca.T.reshape(n_components, h, w)

    # 转回torch.Tensor
    return torch.from_numpy(img_pca).float()

def get_train_test_set(cfg):
    current_dataset = cfg['current_dataset']
    train_set_num = cfg['train_set_num']
    patch_size = cfg['patch_size']
    use_pca = cfg.get('use_pca', False)  # 新增配置项，控制是否使用PCA
    pca_components = cfg.get('pca_components', 155)  # 新增配置项，PCA降维后的维度

    img1, img2, gt = getdata(current_dataset)

    img1 = torch.from_numpy(img1)
    img2 = torch.from_numpy(img2)
    gt = torch.from_numpy(gt)

    img1 = img1.permute(2, 0, 1)
    img2 = img2.permute(2, 0, 1)

    # 应用PCA降维
    if use_pca:
        print(f"Applying PCA to reduce dimensions to {pca_components}")
        img1 = pca_transform(img1, n_components=pca_components)
        img2 = pca_transform(img2, n_components=pca_components)

    img1 = data_preprocess.std_norm(img1)
    img2 = data_preprocess.std_norm(img2)
    img_gt = gt
    img1_pad, img2_pad, patch_coordinates = data_preprocess.construct_sample(img1, img2, patch_size)

    data_sample = data_preprocess.select_sample(img_gt, train_set_num)

    data_sample['img1_pad'] = img1_pad
    data_sample['img2_pad'] = img2_pad

    data_sample['patch_coordinates'] = patch_coordinates
    data_sample['img_gt'] = img_gt  #
    data_sample['ori_gt'] = gt

    # 保存原始和降维后的维度信息
    data_sample['original_dim'] = img1.shape[0] if not use_pca else None
    data_sample['reduced_dim'] = img1.shape[0] if use_pca else None

    return data_sample