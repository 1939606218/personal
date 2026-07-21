from tools.train_test import train_and_test_plot
from Global import *
from tools.init import loadData, normalization, generater, applyPCA


def main():
    X1, X2, Y, WC, WU = loadData(dataset)   # 导入数据集
    X1 = normalization(X=X1, type=3)  # choose normalization method    正则化 变为0到1之间
    X2 = normalization(X=X2, type=3)

    BAND = X1.shape[-1]     #指最后一维数
    if pca:
        X1 = applyPCA(X1, numComponents=pca_channel)   # pca 降维光谱波段冗余
        X2 = applyPCA(X2, numComponents=pca_channel)
        BAND = pca_channel

    print('X1.shape{},X2.shape{},Y.shape{}'.format(X1.shape, X2.shape, Y.shape))
    # model = SiameseNetwork(cha_in=BAND, windowsize=windowsize, num_classes=CLASSES_NUM).to(device)

    print('---Selecting Small Pieces from the Original Cube Data---')

    # 数据的预处理，patch块的处理，维度，划分训练集测试集，转换为张量，打包到一块了。
    TRAIN_SIZE, TEST_SIZE, train_iter, test_iter= generater(X_1=X1,
                                                            X_2=X2,
                                                            Y=Y,
                                                            batchsize=64,
                                                            train_ratio=train_ratio,
                                                            windowSize=windowsize)

    train_and_test_plot(X = X1,
                        Y = Y,
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


if __name__ == '__main__':
    main()
