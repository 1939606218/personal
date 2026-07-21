import torch
import torch.nn as nn
from torchvision import transforms
from configs.configs import *

def std_norm(image):  # input tensor image size with CxHxW
    image = image.permute(1, 2, 0).numpy()  #变成 H×W×C
    if current_dataset == 'Hermiston':
        trans = transforms.Compose([
                transforms.ToTensor(), #transforms.ToTensor()会将numpy数组格式的图像转换回torch张量格式。如果原始image的numpy数组形状是HxWxC，转换后张量的形状会变回CxHxW
                transforms.Normalize(torch.tensor(image).mean(dim=[0, 1,2]), torch.tensor(image).std(dim=[0, 1,2]))
            ])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(torch.tensor(image).mean(dim=[0, 1]), torch.tensor(image).std(dim=[0, 1]))
        ])
    tensor_image = trans(image)
    return tensor_image


def construct_sample(img1, img2, window_size=5):   #对输入的两张图像进行填充处理，并确定图像上各个位置对应的特定大小窗口（图像块）的坐标信息
    _, height, width = img1.shape  # input float tensor image size with CxHxW
    half_window = int(window_size // 2)

    # padding
    pad = nn.ReplicationPad2d(half_window)  #复制图像边界的像素值来扩展图像的尺寸，填充的大小由 half_window 决定
    pad_img1 = pad(img1.unsqueeze(0)).squeeze(0)
    pad_img2 = pad(img2.unsqueeze(0)).squeeze(0)

    #  get coordinates
    patch_coordinates = torch.zeros((height * width, 4), dtype=torch.long)
    t = 0
    for h in range(height):
        for w in range(width):
            patch_coordinates[t, :] = torch.tensor([h, h + window_size, w, w + window_size])
            t += 1
    #通过循环实现了为所有图像块生成并存储坐标信息的功能
    return pad_img1, pad_img2, patch_coordinates  #pad_img1和pad_img2最终是三维张量，维度格式为Cx(H + pad_size)x(W + pad_size)


def select_sample(gt, ntr):  # input tensor data with NxCxHxW, tensor gt with HxW
    gt_vector = gt.reshape(-1, 1).squeeze(1)  #-1 表示自动根据原张量元素个数计算出合适的维度大小，使其变为 (H * W, 1) 的形状，使用 squeeze(1) 操作去掉维度大小为 1 的那一维
    label = torch.unique(gt)  #获取 gt 张量中所有不重复的标注值（也就是图像中出现的不同类别标签）
    first_time = True
    for each in range(2):
        indices_vector = torch.where(gt_vector == label[each])
        indices = torch.where(gt == label[each])

        indices_vector = indices_vector[0]
        indices_row = indices[0]
        indices_column = indices[1]

        class_num = torch.tensor(len(indices_vector))
        ntr_trl = ntr[0]
        ntr_tru = ntr[1]

        # get select_num
        if ntr_trl < 1:
            select_num_trl = int(ntr_trl * class_num)
        else:
            select_num_trl = int(ntr_trl)

        if ntr_tru < 1:
            select_num_tru = int(ntr_tru * class_num)
        else:
            select_num_tru = int(ntr_tru)

        select_num_trl = torch.tensor(select_num_trl)  # River 不变0->20377, 变化1->1939
        select_num_tru = torch.tensor(select_num_tru)

        # disorganize
        torch.manual_seed(4)
        rand_indices0 = torch.randperm(class_num)  #生成一个从 0 到 class_num - 1 的随机排列的一维张量
        rand_indices = indices_vector[rand_indices0]

        # Divide train and test
        trl_ind0 = rand_indices0[0:select_num_trl]  # trl train label 训练集，有标记样本
        tru_ind0 = rand_indices0[select_num_trl:select_num_trl + select_num_tru]  # tru train unlabel 训练集，无标记样本
        te_ind0 = rand_indices0[select_num_trl + select_num_tru:]  # test 测试集
        trl_ind = rand_indices[0:select_num_trl]
        tru_ind = rand_indices[select_num_trl:select_num_trl + select_num_tru]
        te_ind = rand_indices[select_num_trl + select_num_tru:]

        # index+Sample center coordinate
        # train label
        select_trl_ind = torch.cat([trl_ind.unsqueeze(1),    #有标记训练样本索引 trl_ind
                                    indices_row[trl_ind0].unsqueeze(1),  #行索引 indices_row[trl_ind0]
                                    indices_column[trl_ind0].unsqueeze(1)],  #列索引 indices_column[trl_ind0],都添加一维拼接
                                   dim=1
                                   )  # torch.Size([x, 3])
        # train unlabel
        select_tru_ind = torch.cat([tru_ind.unsqueeze(1),
                                    indices_row[tru_ind0].unsqueeze(1),
                                    indices_column[tru_ind0].unsqueeze(1)],
                                   dim=1
                                   )
        # test
        select_te_ind = torch.cat([te_ind.unsqueeze(1),
                                   indices_row[te_ind0].unsqueeze(1),
                                   indices_column[te_ind0].unsqueeze(1)],
                                  dim=1
                                  )

        if first_time:
            first_time = False

            trainl_sample_center = select_trl_ind
            trainl_sample_num = select_num_trl.unsqueeze(0)

            trainu_sample_center = select_tru_ind
            trainu_sample_num = select_num_tru.unsqueeze(0)

            test_sample_center = select_te_ind
            test_sample_num = (class_num - select_num_trl - select_num_tru).unsqueeze(0)

        else:
            trainl_sample_center = torch.cat([trainl_sample_center, select_trl_ind], dim=0)
            trainl_sample_num = torch.cat([trainl_sample_num, select_num_trl.unsqueeze(0)])

            trainu_sample_center = torch.cat([trainu_sample_center, select_tru_ind], dim=0)
            trainu_sample_num = torch.cat([trainu_sample_num, select_num_tru.unsqueeze(0)])

            test_sample_center = torch.cat([test_sample_center, select_te_ind], dim=0)
            test_sample_num = torch.cat(
                [test_sample_num, (class_num - select_num_trl - select_num_tru).unsqueeze(0)])

        rand_trl_ind = torch.randperm(trainl_sample_num.sum()) #.sun用于计算张量在指定维度上元素的总和。如果不指定维度，它会计算整个张量所有元素的总和。
        trainl_sample_center = trainl_sample_center[rand_trl_ind,]
        rand_tru_ind = torch.randperm(trainu_sample_num.sum())
        trainu_sample_center = trainu_sample_center[rand_tru_ind,]
        rand_te_ind = torch.randperm(test_sample_num.sum())
        test_sample_center = test_sample_center[rand_te_ind,]

    data_sample = {'trainl_sample_center': trainl_sample_center, 'trainl_sample_num': trainl_sample_num,
                   'trainu_sample_center': trainu_sample_center, 'trainu_sample_num': trainu_sample_num,
                   'test_sample_center': test_sample_center, 'test_sample_num': test_sample_num,
                   }

    return data_sample
