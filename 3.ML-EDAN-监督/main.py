from data.data_process import loadData
from data.pre_process import normalization, generater, applyPCA
from Global import *
from tools.train import train , test

def main():
    dataname = 'farm'
    in_channel = 155
    # dataname = 'hermiston'
    # in_channel = 242
    # dataname = 'river'
    # in_channel= 198
    # dataname = 'Barbara'
    # in_channel = 224
    # dataname = 'BayArea'
    # in_channel = 224
    model_path = os.path.join("result", dataname)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    X1, X2, Y , height , width = loadData(dataname)   # 导入数据集
    X1 = normalization(X=X1)  # choose normalization method    正则化 变为0到1之间
    X2 = normalization(X=X2)

    X1 = applyPCA(X1, channel=pca_channel)  # pca 降维光谱波段冗余
    X2 = applyPCA(X2, channel=pca_channel)
    print('X1.shape{},X2.shape{},Y.shape{}'.format(X1.shape, X2.shape, Y.shape))

    TRAIN_SIZE, TEST_SIZE, train_iter, test_iter , total_iter = generater(
        X_1=X1,
        X_2=X2,
        Y=Y,
        batchsize=batch_size,
        train_ratio=train_ratio,
        windowSize=patch_sieze)

    train(device=device, train_inter=train_iter, model_path=model_path)

    test(height, width, device=device, test_inter=test_iter, model_path=model_path)



if __name__ == '__main__':
    main()
