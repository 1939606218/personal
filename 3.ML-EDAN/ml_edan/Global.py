import torch
pos = 500    # 变化的选取索引个数
neg = 500    # 未变化的选取索引个数

dataset = 'Barbara'

epochs = 200

windowSize = 7

# depth = 5
# b_depth = 5

batch_size = 64
lr = 0.0001


if dataset == 'hermiston':
    band = 242
    pca_band = 30
    hid = 32
elif dataset == 'river':
    layernum = [198, 128, 64, 10, 2]
elif dataset == 'farmland':
    band = 155
    pca_band = 30
    hid = 24
elif dataset == 'USA':
    layernum = [154, 128, 64, 10, 2]
elif dataset == 'Bay':
    band = 224
    pca_band = 30
    hid = 32
elif dataset == 'farm420':
    band = 154
elif dataset == 'farm430':
    layernum = [132, 128, 64, 10, 2]
elif dataset == 'Barbara':
    band = 224
    pca_band = 30
    hid = 32


