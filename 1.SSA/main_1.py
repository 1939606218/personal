from tools.train_test import train_and_test_plot
from tools.init import loadData, normalization, generater, applyPCA
import torch
import os
import time
from datetime import datetime

def main():
    # 全局参数配置
    train_ratio = 0.01
    windowsize = 5  # 块的大小
    CLASSES_NUM = 2
    ITER = 10
    EPOCHES = 200
    pca = True
    pca_channel = 30

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义所有数据集及其特定参数
    datasets_config = [
        # {
        #     'name': 'hermiston',
        #     'kerner_number': 24,
        #     'batchsize': 64
        # },
        {
            'name': 'farmland',
            'kerner_number': 24,
            'batchsize': 64
        },
        # {
        #     'name': 'river',
        #     'kerner_number': 24,
        #     'batchsize': 64
        # },
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

    # 循环处理每个数据集
    for config in datasets_config:
        dataset = config['name']
        kerner_number = config['kerner_number']
        batchsize = config['batchsize']

        print(f"\n\n=== 开始处理数据集: {dataset} ===")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 记录单个数据集处理时间
        dataset_start_time = time.time()

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
            train_and_test_plot(
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

            # 计算数据集处理时间
            dataset_elapsed_time = time.time() - dataset_start_time
            print(f"=== 数据集 {dataset} 处理完成，耗时: {dataset_elapsed_time:.2f} 秒 ===")

        except Exception as e:
            print(f"!!! 处理数据集 {dataset} 时出错: {str(e)}")
            # 继续处理下一个数据集
            continue

    # 计算总体处理时间
    total_elapsed_time = time.time() - total_start_time
    print(f"\n\n=== 所有数据集处理完成，总耗时: {total_elapsed_time:.2f} 秒 ===")


if __name__ == '__main__':
    main()