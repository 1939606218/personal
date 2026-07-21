import os
import torch.nn as nn
import scipy.io as io
import configs.configs as cfg
import torch.optim as optim
from HSICD_data import HSICD_data
from get_train_test_set import get_train_test_set as get_set
from tools.train import train as fun_train
from tools.test import test as fun_test
import imageio
from tools.show import *
from tools.assessment import *
from model.AIWSEN import AIWSEN as fun_model


def main():
    current_dataset = cfg.current_dataset
    current_model = cfg.current_model
    model_name = current_dataset + current_model
    print('model {}'.format(model_name))
    cfg_data = cfg.data
    cfg_model = cfg.model
    cfg_train = cfg.train['train_model']
    cfg_optim = cfg.train['optimizer']
    cfg_test = cfg.test

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    data_sets = get_set(cfg_data)
    img_gt = data_sets['img_gt']
    train_data = HSICD_data(data_sets, cfg_data['train_data'])
    test_data = HSICD_data(data_sets, cfg_data['test_data'])
    # Load model
    model = fun_model(device = device,in_chans = cfg_model['in_fea_num'],).to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg_optim['lr'], momentum=cfg_optim['momentum'], weight_decay=cfg_optim['weight_decay'])
    fun_train(train_data, model, loss_fun, optimizer, device, cfg_train)
    # test
    pred_train_label, pred_train_acc = fun_test(train_data, data_sets['img_gt'], model, device, cfg_test)
    pred_test_label, pred_test_acc = fun_test(test_data, data_sets['img_gt'], model, device, cfg_test)

    predict_label = torch.cat([pred_train_label, pred_test_label], dim=0)
    print('pred_train_acc {:.2f}%, pred_test_acc {:.2f}%'.format(pred_train_acc, pred_test_acc))
    predict_img = Predict_Label2Img(predict_label, img_gt)

    # 可视化修改：增加未标记区域处理
    change = np.zeros(predict_img.shape + (3,), dtype=np.uint8)

    # 未标记区域（标签2）
    unlabeled = (img_gt == 2)
    change[unlabeled] = [100, 100, 100]  # 灰色表示未标记

    # 变化检测结果（只处理标签0和1的区域）
    valid_area = (img_gt != 2)

    # 在有效区域内
    FP = (predict_img == 1) & (img_gt == 0) & valid_area
    FN = (predict_img == 0) & (img_gt == 1) & valid_area
    TP = (predict_img == 1) & (img_gt == 1) & valid_area
    TN = (predict_img == 0) & (img_gt == 0) & valid_area

    change[TP] = [0, 255, 0]  # 绿色：正确检测变化
    change[FP] = [255, 0, 0]  # 红色：误报变化
    change[FN] = [0, 0, 255]  # 蓝色：漏检变化
    change[TN] = [200, 200, 200]  # 浅灰色：正确未变化

    # 创建掩码：忽略标签2（未标记区域）
    valid_mask = (img_gt != 2)  # 只处理标签0和1
    valid_gt = img_gt[valid_mask]
    valid_pred = predict_img[valid_mask]

    # 仅使用有效像素计算指标
    conf_mat, oa, kappa_co, P, R, F1, acc = accuracy_assessment(valid_gt, valid_pred)
    # 构建带指标名称的评估结果列表（不含换行）
    assessment_result = [
        f"数据集:{current_dataset}",
        f"OA:{round(oa, 4):.4f}",
        f"Kappa:{round(kappa_co, 4):.4f}",
        f"F1:{round(F1, 4):.4f}",
        f"Precision:{round(P, 4):.4f}",
        f"Recall:{round(R, 4):.4f}",
        f"模型:{model_name}"
    ]

    # 用空格或分隔符连接所有指标并打印
    print("\n=== 评估结果 ===")
    print(" | ".join(assessment_result))

    # Store
    save_folder = cfg_test['save_folder']
    save_name = cfg_test['save_name']

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    io.savemat(save_folder + '/' + save_name + ".mat",
               {"predict_img": np.array(predict_img.cpu()), "oa": assessment_result})
    imageio.imwrite(save_folder + '/' + save_name + '+predict_img.png', change)
    print('save predict_img successful!')


if __name__ == '__main__':
    main()

