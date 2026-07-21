import torch
from torch import nn
from torch.utils.data import DataLoader


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'

    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix前缀 'module.' '''
    print('remove prefix \'{}\''.format(prefix))

    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))

    if load_to_cpu == torch.device('cpu'):
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)['model']
    else:
        device = torch.cuda.current_device()    # gpu
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))['model']

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

    return model


def test(test_data, origin_gt, model, device, cfg):
    """测试模型"""
    num_workers = cfg['workers_num']
    gpu_num = cfg['gpu_num']
    batch_size = cfg['batch_size']

    # 加载模型权重
    model = load_model(model, cfg['model_weights'], device)
    model.eval()

    # 模型设备配置
    if gpu_num > 1 and cfg['gpu_train']:
        model = nn.DataParallel(model)
    model = model.to(device)

    # 检查test_data类型
    if isinstance(test_data, DataLoader):
        # 如果test_data已经是DataLoader，则直接使用
        batch_data = test_data
        batch_num = len(test_data)
        dataset_size = len(test_data.dataset)
    else:
        # 如果test_data是Dataset，则创建DataLoader
        batch_data = DataLoader(test_data, batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        batch_num = math.ceil(len(test_data) / batch_size)
        dataset_size = len(test_data)

    print(f'Starting testing on {dataset_size} samples with {batch_num} batches...')

    predict_correct = 0
    label_num = 0
    predict_label = []

    for batch_idx, batch_sample in enumerate(batch_data):
        # 获取数据并移至GPU
        img1, img2, target, indices = batch_sample
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)

        # 前向传播
        with torch.no_grad():
            prediction = model(img1, img2)

        # 计算预测结果
        label = prediction.cpu().argmax(dim=1, keepdim=True)

        # 计算准确率
        if target.sum() > 0:
            predict_correct += label.eq(target.view_as(label)).sum().item()
            label_num += len(target)

        predict_label.append(torch.cat([indices, label], dim=1))

    # 合并所有预测结果
    predict_label = torch.cat(predict_label, dim=0)

    # 计算总体准确率
    test_acc = 100 * predict_correct / label_num if label_num > 0 else 0
    if label_num > 0:
        print('OA {:.2f}%'.format(test_acc))

    return predict_label, test_acc
