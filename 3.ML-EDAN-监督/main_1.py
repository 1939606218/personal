from data.data_process import loadData
from data.pre_process import normalization, generater, applyPCA
from Global import *
from tools.train import train, test
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import json  # 添加JSON模块导入


def calculate_value(oa, f1, precision, recall, kappa):
    return oa + f1 + precision + recall + kappa

def plot_change_detection_result(true_labels, pred_labels, height, width, model_path, run_num):
    # 定义颜色映射
    colors = {
        'TP': [255, 255, 255],  # 白色：正确检测变化
        'FP': [255, 0, 0],  # 红色：误报变化
        'FN': [0, 255, 0],  # 绿色：漏检变化
        'TN': [0, 0, 0],  # 黑色：正确未变化
        'NO_CHANGE': [100, 100, 100]  # 灰色：标签为2的不确定区域
    }

    # 初始化结果图像
    result_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 逐像素比较预测和真实标签，确定每个像素的类别
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            true_label = true_labels[idx]
            pred_label = pred_labels[idx]

            # 处理标签为2的不确定区域
            if true_label == 2:
                result_image[i, j] = colors['NO_CHANGE']
            # 根据真实标签和预测标签确定像素类别
            elif true_label == 1 and pred_label == 1:
                # 真实变化，预测变化 -> TP
                result_image[i, j] = colors['TP']
            elif true_label == 0 and pred_label == 1:
                # 真实未变化，预测变化 -> FP
                result_image[i, j] = colors['FP']
            elif true_label == 1 and pred_label == 0:
                # 真实变化，预测未变化 -> FN
                result_image[i, j] = colors['FN']
            elif true_label == 0 and pred_label == 0:
                # 真实未变化，预测未变化 -> TN
                result_image[i, j] = colors['TN']

    # 保存图像
    plt.figure(figsize=(10, 10))
    plt.imshow(result_image)
    plt.axis('off')
    plt.title(f'Change Detection Result - Run {run_num}')
    plt.savefig(os.path.join(model_path, f'change_detection_result_run_{run_num}.png'),
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def main():
    # 定义所有数据集及其特定参数
    datasets_config = [
        {
            'name': 'farm',
            'in_channel': 155,
            'description': 'Farmland dataset'
        },
        # {
        #     'name': 'hermiston',
        #     'in_channel': 242,
        #     'description': 'Hermiston dataset'
        # },
        # {
        #     'name': 'river',
        #     'in_channel': 198,
        #     'description': 'River dataset'
        # },
        {
            'name': 'Barbara',
            'in_channel': 224,
            'description': 'Santa Barbara dataset'
        },
        {
            'name': 'BayArea',
            'in_channel': 224,
            'description': 'Bay Area dataset'
        }
    ]

    # 记录总体开始时间
    total_start_time = time.time()

    # 循环处理每个数据集
    for config in datasets_config:
        dataname = config['name']
        model_path = os.path.join("result", dataname)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        in_channel = config['in_channel']
        description = config['description']

        print(f"\n\n=== 开始处理数据集: {dataname} ({description}) ===")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        max_value = -np.inf
        min_value = np.inf
        max_result = None
        min_result = None

        for run_num in range(1, 11):
            print(f"\n--- 第 {run_num} 次运行 ---")
            run_start_time = time.time()

            try:
                # 加载和预处理数据
                X1, X2, Y, height, width = loadData(dataname)
                X1 = normalization(X=X1)
                X2 = normalization(X=X2)

                X1 = applyPCA(X1, channel=pca_channel)
                X2 = applyPCA(X2, channel=pca_channel)
                print(f'X1.shape{X1.shape}, X2.shape{X2.shape}, Y.shape{Y.shape}')

                # 修改generater调用，获取所有标签和索引
                TRAIN_SIZE, TEST_SIZE, train_iter, test_iter, total_iter, all_labels, all_indices = generater(
                    X_1=X1,
                    X_2=X2,
                    Y=Y,
                    batchsize=batch_size,
                    train_ratio=train_ratio,
                    windowSize=patch_sieze
                )

                # 训练模型
                print("开始训练模型...")
                train_time = train(device=device, train_inter=train_iter, model_path=model_path)

                # 测试模型
                print("开始测试模型...")
                oa, f1_weighted, precision_weighted, recall_weighted, kappa, TN, FP, FN, TP, conf_matrix, true_labels, full_pred_labels = test(
                    height, width, device=device, test_inter=test_iter, model_path=model_path)
                # 绘制变化检测结果图像（使用完整标签和预测）
                plot_change_detection_result(Y.flatten(), full_pred_labels, height, width, model_path, run_num)
                value = calculate_value(oa, f1_weighted, precision_weighted, recall_weighted, kappa)

                # 在记录max_result和min_result时添加run_num
                if value > max_value:
                    max_value = value
                    max_result = {
                        'run_num': run_num,  # 添加运行次数
                        'oa': oa,
                        'f1': f1_weighted,
                        'precision': precision_weighted,
                        'recall': recall_weighted,
                        'kappa': kappa,
                        'TN': TN,
                        'FP': FP,
                        'FN': FN,
                        'TP': TP,
                        'conf_matrix': conf_matrix,
                        'value': value,
                        'train_time': train_time
                    }

                if value < min_value:
                    min_value = value
                    min_result = {
                        'run_num': run_num,  # 添加运行次数
                        'oa': oa,
                        'f1': f1_weighted,
                        'precision': precision_weighted,
                        'recall': recall_weighted,
                        'kappa': kappa,
                        'TN': TN,
                        'FP': FP,
                        'FN': FN,
                        'TP': TP,
                        'conf_matrix': conf_matrix,
                        'value': value,
                        'train_time': train_time
                    }

                run_elapsed_time = time.time() - run_start_time
                print(f"--- 第 {run_num} 次运行完成，耗时: {run_elapsed_time:.2f} 秒 ---")

            except Exception as e:
                print(f"!!! 第 {run_num} 次运行数据集 {dataname} 时出错: {str(e)}")
                # 继续下一次运行
                continue

        print(f"\n=== 数据集 {dataname} 10 次运行完成 ===")
        print(f"最大 value ({max_value:.6f}) 结果 - 第 {max_result['run_num']} 次运行:")
        print(
            f"OA: {max_result['oa']:.6f}, F1: {max_result['f1']:.6f}, Pr: {max_result['precision']:.6f}, Re: {max_result['recall']:.6f}, Kappa: {max_result['kappa']:.6f}")
        print(f"最小 value ({min_value:.6f}) 结果 - 第 {min_result['run_num']} 次运行:")
        print(
            f"OA: {min_result['oa']:.6f}, F1: {min_result['f1']:.6f}, Pr: {min_result['precision']:.6f}, Re: {min_result['recall']:.6f}, Kappa: {min_result['kappa']:.6f}")
        # 保存最高和最低value的结果到JSON文件
        save_results_to_json(dataname, model_path, max_result, min_result)

    # 计算总体处理时间
    total_elapsed_time = time.time() - total_start_time
    print(f"\n\n=== 所有数据集处理完成，总耗时: {total_elapsed_time:.2f} 秒 ===")


def save_results_to_json(dataname, model_path, max_result, min_result):
    """将最高和最低value的结果保存到JSON文件"""
    # 提取需要保存的指标
    results_to_save = {
        'dataset': dataname,
        'max_value_result': {
            'run_num': max_result['run_num'],  # 添加运行次数
            'oa': float(max_result['oa']),
            'f1': float(max_result['f1']),
            'precision': float(max_result['precision']),
            'recall': float(max_result['recall']),
            'kappa': float(max_result['kappa']),
            'value': float(max_result['value'])
        },
        'min_value_result': {
            'run_num': min_result['run_num'],  # 添加运行次数
            'oa': float(min_result['oa']),
            'f1': float(min_result['f1']),
            'precision': float(min_result['precision']),
            'recall': float(min_result['recall']),
            'kappa': float(min_result['kappa']),
            'value': float(min_result['value'])
        }
    }

    # 保存到JSON文件
    json_path = os.path.join(model_path, 'best_and_worst_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_to_save, f, indent=4)

    print(f"结果已保存到 {json_path}")


if __name__ == '__main__':
    main()