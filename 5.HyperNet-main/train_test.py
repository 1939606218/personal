import torch.backends.cudnn as cudnn
import scipy.io as sio
import matplotlib.pyplot as plt
from HyperNet_model import HyperNet, BasicBlock
from evaluation import two_cls_access,two_cls_access_for_Bay_Barbara
from skimage.filters import threshold_otsu
import torch
import numpy as np
import os
from utils import initNetParams_v2,adjust_learning_rate,compute_kmeans_threshold

def train_HyperNet_HBCD(args):
    print('---------------------------func: train_HyperNet_HBCD---------------------------')
    print('\n')
    model, idx = [], []
    if args.EX_num == 'Hermiston':
        model = HyperNet(BasicBlock, layernum=[242, 121, 242, 121], gamma=args.gamma)
        idx = sio.loadmat(args.idx_file)['label']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        # 准备参考标注数据（真实标签数据），这里需要根据实际情况替换为真实的reference_data
        reference_data = sio.loadmat(args.idx_file)['label']
        data_before = sio.loadmat(args.data_before)
        data_after = sio.loadmat(args.data_after)
        img_1 = data_before['HypeRvieW']
        img_2 = data_after['HypeRvieW']  # H,W,C

        print('unchanged idx path:', args.idx_file)

    if args.EX_num == 'farm':
        model = HyperNet(BasicBlock, layernum=[155, 78, 156, 78], gamma=args.gamma)
        idx = sio.loadmat(args.idx_file)['label']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        # 准备参考标注数据（真实标签数据），这里需要根据实际情况替换为真实的reference_data
        reference_data = sio.loadmat(args.idx_file)['label']
        data_before = sio.loadmat(args.data_before)
        data_after = sio.loadmat(args.data_after)
        img_1 = data_before['imgh']
        img_2 = data_after['imghl']  # H,W,C

        print('unchanged idx path:', args.idx_file)

    if args.EX_num == 'river':
        model = HyperNet(BasicBlock, layernum=[198, 99, 198, 99], gamma=args.gamma)
        idx = sio.loadmat(args.idx_file)['lakelabel_v1']
        idx[idx == 255] = 1
        idx = torch.from_numpy(idx.squeeze()).cuda()
        # 准备参考标注数据（真实标签数据），这里需要根据实际情况替换为真实的reference_data
        reference_data = sio.loadmat(args.idx_file)['lakelabel_v1']
        reference_data[reference_data == 255] = 1
        data_before = sio.loadmat(args.data_before)
        data_after = sio.loadmat(args.data_after)
        img_1 = data_before['river_before']
        img_2 = data_after['river_after']  # H,W,C

        print('unchanged idx path:', args.idx_file)

    elif args.EX_num == 'Bay':
        model =HyperNet(BasicBlock,  layernum=[224, 112, 224, 112], gamma=1)  # for Bay area dataset
        idx = sio.loadmat(args.idx_file)['HypeRvieW']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        reference_data = sio.loadmat(args.idx_file)['HypeRvieW']
        data_before = sio.loadmat(args.data_before)
        data_after = sio.loadmat(args.data_after)
        img_1 = data_before['HypeRvieW']
        img_2 = data_after['HypeRvieW']  # H,W,C
        print('unchanged idx path:', args.idx_file)

    elif args.EX_num == 'Barbara':
        print('------------training for Barbara dataset ------------------')
        model = HyperNet(BasicBlock, layernum=[224, 112, 224, 112], gamma=1)  # for Santa Barbara dataset
        idx = sio.loadmat(args.idx_file)['HypeRvieW']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        reference_data = sio.loadmat(args.idx_file)['HypeRvieW']
        data_before = sio.loadmat(args.data_before)
        data_after = sio.loadmat(args.data_after)
        img_1 = data_before['HypeRvieW']
        img_2 = data_after['HypeRvieW']  # H,W,C

        print('unchanged idx path:', args.idx_file)

    model.apply(initNetParams_v2)
    print('----------------model.apply(initNetParams_v2)-----------------')
    print('img_1 and img_2 is input for test')
    H, W, C = img_1.shape
    X1 = torch.tensor(np.transpose(img_1, [2, 0, 1]), dtype=torch.float32).unsqueeze(0).cuda()
    X2 = torch.tensor(np.transpose(img_2, [2, 0, 1]), dtype=torch.float32).unsqueeze(0).cuda()
    print('input.shape:', X1.shape)
    del img_1, img_2, data_after, data_before

    total_elements = idx.numel()
    idx = idx.reshape(total_elements)
    print("After reshape, idx shape:", idx.shape)
    idx = idx.long()
    print("After type conversion, idx dtype:", idx.dtype)

    init_lr = args.lr
    model.cuda()
    optim_params = model.parameters()
    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    Tra_ls = []
    print('trainging begins----------------------------')
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        loss = model(X1, X2, idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Tra_ls.append(loss.item())
        if epoch % 10 == 0:
            print('epoch [{}/{}],train:{:.4f}'.format(epoch, args.epochs, loss.item()))
    print('--------------SSL: training is sucessfully down--------------')
    print('--------------model save_path:', args.save_model_path, sep='\n')
    torch.save(model.state_dict(), args.save_model_path)
    root_dir = str(args.save_result_path)
    # 如果目录不存在则创建
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    plt.figure(args.EX_num)
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(args.epochs), np.asarray(Tra_ls), 'r-o', label="SSL")
    plt.legend()
    # 保存图像
    save_path = os.path.join(root_dir, 'SSL.jpg')
    plt.savefig(save_path)

    model.load_state_dict(torch.load(args.save_model_path))
    model.eval()
    f1, f2 = model(X1, X2, 0)
    f1, f2 = f1.squeeze(), f2.squeeze()
    f1 = f1.permute(1, 2, 0)
    f2 = f2.permute(1, 2, 0)
    print('f1.shape',f1.shape)
    f1 = f1.reshape([-1, C])
    f2 = f2.reshape([-1, C])

    print('input shape:', f1.shape)
    mse_criterion = torch.nn.MSELoss(reduction='none')
    MSE_result = mse_criterion(f1, f2)
    MSE_result = np.mean(MSE_result.numpy(), axis=1).reshape([H, W])
    threshold = compute_kmeans_threshold(MSE_result,k=2)
    print('threshold',threshold)
    result_data = (MSE_result > threshold).astype(int)
    plt.imshow(result_data, cmap='hot')
    # 调用two_cls_access函数进行评价
    oa_kappa = two_cls_access(reference_data, result_data)
    print("OA and Kappa values and other metrics:", oa_kappa)

    plt.figure()
    plt.imshow(MSE_result)
    plt.title('MSE')

    # 保存图像
    save_path = os.path.join(root_dir, 'mse_result.jpg')
    plt.savefig(save_path)

    # sio.savemat(args.save_result_path, { 'MSE_result': MSE_result})
    print('--------------save_result_path:', args.save_result_path, sep='\n')
    return MSE_result