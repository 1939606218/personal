import torch.utils.data as data

class HSICD_data(data.Dataset):
    def __init__(self, data_sample, cfg):
        self.phase = cfg['phase']
        self.img1 = data_sample['img1_pad']
        self.img2 = data_sample['img2_pad']
        self.patch_coordinates = data_sample['patch_coordinates']
        self.gt = data_sample['img_gt']
        if self.phase == 'trl':
            self.data_indices = data_sample['trainl_sample_center']   #sample_center中第一维是样本索引，第二维是行索引，第三维是列索引
        elif self.phase == 'tru':
            self.data_indices = data_sample['trainu_sample_center']
        elif self.phase == 'test':
            self.data_indices = data_sample['test_sample_center']

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):  #自动为每次迭代生成合适的idx值，从0开始,到len(self.data_indices) - 1
        index = self.data_indices[idx]
        img_index = self.patch_coordinates[index[0]]  #获取具体的图像块坐标信息 index[0]表示第几行
        #patch_coordinates 每一行存储的是一个图像块的坐标信息，形式为 [h, h + window_size, w, w + window_size]
        # From pad_img intercept samples based on coordinates
        img1 = self.img1[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
        img2 = self.img2[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]  #img_index[0] 表示图像块在高度方向上的起始坐标。
                                                                                    #img_index[1] 表示图像块在高度方向上的结束坐标。
                                                                                    # img_index[2] 表示图像块在宽度方向上的起始坐标。
                                                                                    # img_index[3] 表示图像块在宽度方向上的结束坐标。
        label_gt = self.gt[index[1], index[2]]

        return img1, img2, label_gt, index
