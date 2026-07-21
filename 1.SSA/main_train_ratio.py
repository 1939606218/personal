from tools.train_test import train_and_test_plot
from tools.init import loadData, normalization, generater, applyPCA
import torch
import os
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt

def main():
    # 全局参数配置
    windowsize = 5  # 块的大小
    CLASSES_NUM = 2
    ITER = 10
    EPOCHES = 100
    pca = True
    pca_channel = 30
    train_ratios = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义所有数据集及其特定参数
    datasets_config = [
        {
            'name': 'hermiston',
            'kerner_number': 24,
            'batchsize': 64
        },
        {
            'name': 'farmland',
            'kerner_number': 24,
            'batchsize': 64
        },
        {
            'name': 'river',
            'kerner_number': 24,
            'batchsize': 64
        },
        {
            'name': 'santaBarbara',
            'kerner_number': 32,
            'batchsize': 64
        },
        {
            'name': 'bayArea',
            'kerner_number': 32,
            'batchsize': 64
        }
    ]

    # 记录总体开始时间
    total_start_time = time.time()

    # 存储每个数据集在不同train_ratio下的OA结果
    all_results = {}

    # 循环处理每个数据集
    for config in datasets_config:
        dataset = config['name']
        kerner_number = config['kerner_number']
        batchsize = config['batchsize']

        print(f"\n\n=== 开始处理数据集: {dataset} ===")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 存储当前数据集在不同train_ratio下的OA结果
        dataset_results = {}

        for train_ratio in train_ratios:
            print(f"\n=== 开始处理 train_ratio: {train_ratio} ===")

            # 记录单个train_ratio处理时间
            ratio_start_time = time.time()

            try:
                # 加载和预处理数据
                X1, X2, Y, WC, WU = loadData(dataset)
                X1 = normalization(X=X1, type=3)
                X2 = normalization(X=X2, type=3)

                BAND = X1.shape[-1]
                if pca:
                    X1 = applyPCA(X1, numComponents=pca_channel)
                    X2 = applyPCA(X2, numComponents=pca_channel)
                    BAND = pca_channel

                print(f'X1.shape{X1.shape}, X2.shape{X2.shape}, Y.shape{Y.shape}')

                TRAIN_SIZE, TEST_SIZE, train_iter, test_iter = generater(
                    X_1=X1,
                    X_2=X2,
                    Y=Y,
                    batchsize=batchsize,
                    train_ratio=train_ratio,
                    windowSize=windowsize
                )

                # 训练和测试模型
                oa, best_metrics, worst_metrics = train_and_test_plot(
                    kerner_number,
                    X=X1,
                    Y=Y,
                    dataset=dataset,
                    BAND=BAND,
                    CLASSES_NUM=CLASSES_NUM,
                    train_iter=train_iter,
                    test_iter=test_iter,
                    TRAIN_SIZE=TRAIN_SIZE,
                    TEST_SIZE=TEST_SIZE,
                    device=device,
                    epoches=EPOCHES,
                    ITER=ITER,
                    windowsize=windowsize,
                    WU=WU,
                    WC=WC
                )

                # 存储当前train_ratio的OA结果
                dataset_results[str(train_ratio)] = oa

                # 计算train_ratio处理时间
                ratio_elapsed_time = time.time() - ratio_start_time
                print(f"=== train_ratio {train_ratio} 处理完成，耗时: {ratio_elapsed_time:.2f} 秒 ===")

            except Exception as e:
                print(f"!!! 处理 train_ratio {train_ratio} 时出错: {str(e)}")
                # 继续处理下一个train_ratio
                continue

        # 存储当前数据集的所有结果
        all_results[dataset] = dataset_results

        # 绘制当前数据集的曲线
        plt.figure()
        plt.plot(train_ratios, [dataset_results[str(ratio)] for ratio in train_ratios])
        plt.xlabel('Train Ratio')
        plt.ylabel('Test OA')
        plt.title(f'Test OA vs Train Ratio for {dataset}')
        plt.savefig(f'./results/{dataset}_train_ratio_oa.png')
        plt.close()

        print(f"=== 数据集 {dataset} 所有 train_ratio 处理完成 ===")

    # 保存所有结果到JSON文件
    try:
        with open('./results/all_train_ratio_results.json', 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\n=== 所有训练结果已保存到: ./results/all_train_ratio_results.json ===")
    except Exception as e:
        print(f"保存结果时出错: {e}")

    # 计算总体处理时间
    total_elapsed_time = time.time() - total_start_time
    print(f"\n\n=== 所有数据集和train_ratio处理完成，总耗时: {total_elapsed_time:.2f} 秒 ===")

if __name__ == '__main__':
    main()