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
import numpy as np
import json
import time  # 新增：用于计时
from torch.utils.data import DataLoader


def run_experiment(cfg_data, cfg_model, cfg_train, cfg_test, current_dataset, current_model, current_band_now,
                   num_head_now, experiment_num):
    model_name = current_dataset + current_model
    print('model {}'.format(model_name))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: 使用 {'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    data_sets = get_set(cfg_data)
    img_gt = data_sets['img_gt']

    train_data = HSICD_data(data_sets, cfg_data['train_data'])
    train_loader = DataLoader(train_data, batch_size=cfg_train['train_model']['batch_size'], shuffle=True)

    test_data = HSICD_data(data_sets, cfg_data['test_data'])
    test_loader = DataLoader(test_data, batch_size=cfg_test['batch_size'], shuffle=False)

    # Load model
    model = fun_model(device=device, inchannel=current_band_now, num_head=num_head_now).to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg_train['optimizer']['lr'],
                          momentum=cfg_train['optimizer']['momentum'],
                          weight_decay=cfg_train['optimizer']['weight_decay'])

    # 开始计时
    start_time = time.time()

    # 构建包含实验次数的保存名称
    experiment_suffix = f"_exp{experiment_num}"
    save_folder = cfg_test['save_folder']
    save_name = cfg_test['save_name'] + experiment_suffix

    # 更新训练配置中的保存路径，包含实验次数
    train_save_folder = cfg_train['train_model']['save_folder']
    train_save_name = cfg_train['train_model']['save_name'] + experiment_suffix
    train_final_model_path = os.path.join(train_save_folder, train_save_name + '_Final.pth')

    # 修改训练和测试调用部分
    # 更新训练配置中的保存路径
    cfg_train['train_model']['save_folder'] = train_save_folder
    cfg_train['train_model']['save_name'] = train_save_name

    fun_train(train_loader, model, loss_fun, optimizer, device, cfg_train['train_model'])

    # 结束计时并计算训练时间（秒）
    training_time = time.time() - start_time
    print(f"训练耗时: {training_time:.2f} 秒")

    # 测试时加载带实验次数的模型
    cfg_test['model_weights'] = train_final_model_path

    # test
    pred_train_label, pred_train_acc = fun_test(train_loader, data_sets['img_gt'], model, device, cfg_test)
    pred_test_label, pred_test_acc = fun_test(test_loader, data_sets['img_gt'], model, device, cfg_test)

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

    # 修改颜色映射：TP-白色，TN-黑色，FP-红色，FN-绿色
    change[TP] = [255, 255, 255]  # 白色：正确检测变化
    change[FP] = [255, 0, 0]  # 红色：误报变化
    change[FN] = [0, 255, 0]  # 绿色：漏检变化
    change[TN] = [0, 0, 0]  # 黑色：正确未变化

    # 创建掩码：忽略标签2（未标记区域）
    valid_mask = (img_gt != 2)  # 只处理标签0和1
    valid_gt = img_gt[valid_mask]
    valid_pred = predict_img[valid_mask]

    # 仅使用有效像素计算指标
    conf_mat, oa, kappa_co, P, R, F1, acc = accuracy_assessment(valid_gt, valid_pred)
    value = oa + F1 + P + R + kappa_co

    result = {
        "oa": oa,
        "kappa": kappa_co,
        "f1": F1,
        "precision": P,
        "recall": R,
        "value": value,
        "training_time": training_time,
        "model_path": train_final_model_path,  # 保存模型路径到结果中
        "mat_path": save_folder + '/' + save_name + ".mat",  # 保存.mat路径
        "png_path": save_folder + '/' + save_name + '+predict_img.png'  # 保存.png路径
    }

    # Store
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if not os.path.exists(train_save_folder):
        os.makedirs(train_save_folder)

    io.savemat(save_folder + '/' + save_name + ".mat",
               {"predict_img": np.array(predict_img.cpu()), "result": result})
    imageio.imwrite(save_folder + '/' + save_name + '+predict_img.png', change)
    print(f'save predict_img successful! 路径: {save_folder + "/" + save_name}')

    # 清理资源
    del model, loss_fun, optimizer, train_data, test_data, data_sets, predict_label, predict_img, change
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return result

def main():
    num_experiments = 10  # 每个数据集运行的实验次数
    all_datasets_summary = {}
    dataset_summary_dir = './dataset_summary'
    if not os.path.exists(dataset_summary_dir):
        os.mkdir(dataset_summary_dir)

    for dataset_info in cfg.datasets:
        cfg_data, cfg_model, cfg_train, cfg_test, current_band_now, num_head_now = cfg.get_config(dataset_info)
        current_dataset = dataset_info['current_dataset']

        # 更新配置以确保使用正确的数据集路径
        cfg_train['train_model']['save_folder'] = f'./weights/{current_dataset}/'
        cfg_test['model_weights'] = f'./weights/{current_dataset}/{current_dataset}{cfg.current_model}_Final.pth'
        cfg_test['save_folder'] = f'./result/{current_dataset}'

        print(f"\n===== 开始处理数据集: {current_dataset} =====")
        dataset_experiment_results = []

        for i in range(num_experiments):
            print(f"\n===== 数据集 {current_dataset} 实验 {i + 1}/{num_experiments} =====")
            # 传递实验次数给run_experiment函数
            result = run_experiment(cfg_data, cfg_model, cfg_train, cfg_test, current_dataset, cfg.current_model,current_band_now, num_head_now, i+1)
            result["experiment_num"] = i + 1
            dataset_experiment_results.append(result)

            # 打印当前实验结果（包含训练时间）
            print(f"实验 {i + 1} 结果: OA={result['oa']:.4f}, Kappa={result['kappa']:.4f}, "
                  f"F1={result['f1']:.4f}, Precision={result['precision']:.4f}, "
                  f"Recall={result['recall']:.4f}, Value={result['value']:.4f}, "
                  f"训练时间={result['training_time']:.2f}秒"
                  f"模型路径: {result['model_path']}")

        # 分析实验结果
        if dataset_experiment_results:
            # 提取所有value值
            values = [result["value"] for result in dataset_experiment_results]
            training_times = [result["training_time"] for result in dataset_experiment_results]

            # 找到value最大和最小的实验
            max_value_idx = np.argmax(values)
            min_value_idx = np.argmin(values)
            max_result = dataset_experiment_results[max_value_idx]
            min_result = dataset_experiment_results[min_value_idx]

            # 计算平均value和其他指标（包括训练时间）
            avg_value = np.mean(values)
            avg_oa = np.mean([result["oa"] for result in dataset_experiment_results])
            avg_kappa = np.mean([result["kappa"] for result in dataset_experiment_results])
            avg_f1 = np.mean([result["f1"] for result in dataset_experiment_results])
            avg_precision = np.mean([result["precision"] for result in dataset_experiment_results])
            avg_recall = np.mean([result["recall"] for result in dataset_experiment_results])
            avg_training_time = np.mean(training_times)

            # 构建汇总结果
            dataset_summary = {
                "dataset": current_dataset,
                "num_experiments": num_experiments,
                "average_value": float(avg_value),
                "average_oa": float(avg_oa),
                "average_kappa": float(avg_kappa),
                "average_f1": float(avg_f1),
                "average_precision": float(avg_precision),
                "average_recall": float(avg_recall),
                "average_training_time": float(avg_training_time),
                "max_value_experiment": max_result,
                "min_value_experiment": min_result,
                "all_experiments": dataset_experiment_results
            }

            # 保存数据集汇总结果
            summary_file = os.path.join(dataset_summary_dir, f"{current_dataset}_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(dataset_summary, f, indent=4)

            print(f"\n===== 数据集 {current_dataset} 实验汇总 =====")
            print(f"平均Value值: {avg_value:.4f}")
            print(f"最大Value值: {max_result['value']:.4f} (实验 {max_result['experiment_num']})")
            print(f"最小Value值: {min_result['value']:.4f} (实验 {min_result['experiment_num']})")
            print(f"平均训练时间: {avg_training_time:.2f}秒")

            # 保存到全局汇总
            all_datasets_summary[current_dataset] = dataset_summary

        # 释放资源
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        print(f"===== 数据集 {current_dataset} 处理完成 =====")

    # 保存所有数据集的汇总结果
    global_summary_file = os.path.join('./', "global_summary.json")
    with open(global_summary_file, 'w') as f:
        json.dump(all_datasets_summary, f, indent=4)
    print(f"\n===== 所有数据集处理完成，全局汇总已保存到 {global_summary_file} =====")

if __name__ == '__main__':
    main()