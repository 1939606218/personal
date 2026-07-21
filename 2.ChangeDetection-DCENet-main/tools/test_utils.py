import torch
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

    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x     #
    #使用split方法以prefix为分隔符对x进行分割，并且只分割一次（参数1表示分割次数），然后取分割后的结果列表中的最后一个元素（[-1]表示取最后一个元素），这样就相当于把前缀从参数名称中移除了。
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # 引用时：model = load_model(model, cfg['model_weights'], device)
    print('Loading pretrained model from {}'.format(pretrained_path))

    if load_to_cpu == torch.device('cpu'):
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)['model']   #在这里lambda storage, loc: storage表示将参数直接保留在原存储位置（CPU）
    else:
        device = torch.cuda.current_device()  # gpu
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))['model']  #将模型参数存储位置转换到指定的 GPU 设备

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

    return model


def test(test_data, img_gt, model, device, cfg):
    num_workers = cfg['workers_num']
    gpu_num = cfg['gpu_num']
    batch_size = cfg['batch_size']

    # Model load
    model = load_model(model, cfg['model_weights'], device)
    model.eval()
    # Data load
    batch_data = DataLoader(test_data, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    predict_correct = 0
    label_num = 0
    predict_label = []

    for batch_idx, batch_sample in enumerate(batch_data):
        img1, img2, gt, indices = batch_sample
        img1 = img1.to(device)
        img2 = img2.to(device)
        gt = gt.to(device)

        with torch.no_grad():
            prediction = model(img1, img2, img1, img2)

        label = prediction.cpu().argmax(dim=1, keepdim=True)
        if gt.sum() > 0:
            predict_correct += label.eq(gt.cpu().view_as(label)).sum().item() #对于布尔值类型的张量，sum() 函数会将 True 视为 1，False 视为 0，然后沿着张量的所有维度进行求和
            label_num += len(gt)  #gt的形状是(batch_size, num_classes)，len求第0维的长度
        predict_label.append(torch.cat([indices, label], dim=1))   #在列拼接

    predict_label = torch.cat(predict_label, dim=0)    #在行拼接

    test_acc = 100 * predict_correct / label_num
    if label_num > 0:
        print('OA {:.2f}%'.format(test_acc))

    return predict_label, test_acc
