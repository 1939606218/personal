import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from HyGSTAN import hygstan
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch.nn.functional as F
import time
import os
from PIL import Image

# from image_mask import color_matrix

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['BayArea', 'Barbara', 'farmland', 'river', 'hermiston'], default='BayArea', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='test', help='testing mark')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=100, help='number of evaluation')
parser.add_argument('--patches', type=int, default=5, help='number of patches')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=100, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0.1, help='weight_decay')
parser.add_argument('--train_number', type=int, default=0.01, help='train_number')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
#-------------------------------------------------------------------------------
# 定位训练和测试样本
def chooose_train_and_test_point(train_data, test_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    # number_true = []
    # pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    return total_pos_train, total_pos_test, number_train, number_test
#-------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    # x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("**************************************************")

    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band
#-------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, num_classes):
    y_train = []
    y_test = []
    # y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("**************************************************")
    return y_train, y_test
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------

class Loss_fn(nn.Module):
    def __init__(self, delta=1.0, lambda_param=0.5, num_classes=2):
        super(Loss_fn, self).__init__()
        self.delta = delta
        self.lambda_param = lambda_param
        self.num_classes = num_classes

    def forward(self, logits, labels):
        probabilities = F.softmax(logits, dim=1)
        Pt = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        loss_primary = -(1 - Pt) ** self.delta * torch.log(Pt)
        loss_secondary = self.lambda_param * (1 - Pt) ** (self.delta + 1)
        total_loss = loss_primary + loss_secondary
        return total_loss.mean()
#-------------------------------------------------------------------------------

# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(train_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()
        optimizer.zero_grad()
        batch_pred = model(batch_data_t1,batch_data_t2)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data_t1.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
#-------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(valid_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data_t1,batch_data_t2)

        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data_t1.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre

def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(test_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data_t1,batch_data_t2)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre
#-------------------------------------------------------------------------------
# def output_metric(tar, pre):
#     matrix = confusion_matrix(tar, pre)
#     OA, AA_mean, Kappa, AA = cal_results(matrix)
#     return OA, AA_mean, Kappa, AA

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
def output_metric(tar, pre):
    # 计算混淆矩阵
    matrix = confusion_matrix(tar, pre)

    # 计算总体准确率OA
    total_samples = np.sum(matrix)
    OA = np.sum(np.diag(matrix)) / total_samples if total_samples > 0 else 0.0

    # 计算各类的精确率、召回率和F1分数，设置zero_division=0处理无样本情况
    class_precision = precision_score(tar, pre, average=None, zero_division=0)
    class_recall = recall_score(tar, pre, average=None, zero_division=0)
    class_f1 = f1_score(tar, pre, average=None, zero_division=0)

    # 计算宏平均指标
    Pr = np.mean(class_precision)
    Re = np.mean(class_recall)
    F1 = np.mean(class_f1)

    # 计算Kappa系数
    po = OA  # 观察到的准确率
    pe = np.sum(np.sum(matrix, axis=0) * np.sum(matrix, axis=1)) / (total_samples ** 2) if total_samples > 0 else 0.0
    Kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    # 计算平均准确率AA，处理除零情况
    row_sums = np.sum(matrix, axis=1)
    valid_classes = row_sums > 0

    # 创建AA数组并初始化为0
    AA = np.zeros(len(row_sums))

    # 只对有样本的类别计算准确率
    if np.any(valid_classes):
        AA[valid_classes] = np.divide(
            np.diag(matrix)[valid_classes],
            row_sums[valid_classes],
            out=np.zeros(np.sum(valid_classes)),
            where=row_sums[valid_classes] > 0
        )

    # 计算平均准确率
    AA_mean = np.mean(AA) if len(AA) > 0 else 0.0

    return OA, F1, Pr, Re, Kappa, AA_mean, AA  # 返回所有需要的指标
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------

LABEL_CONFIG = {
    'hermiston': {'change': 1, 'no_change': 2},
    'farmland': {'change': 1, 'no_change': 2},
    'river': {'change': 1, 'no_change': 2},  # 修正：预处理后变为 [1,2]
    'Barbara': {'change': 1, 'no_change': 2},  # 假设 Barbara 也执行了 data_label[data_label == 0] = 2
    'BayArea': {'change': 1, 'no_change': 2},   # 同理
}

# 绘制预测图片
def plot_prediction(prediction_matrix, labels, dataset, save_path):
    # 从配置字典获取标签映射
    config = LABEL_CONFIG.get(dataset)
    if not config:
        raise ValueError(f"Dataset {dataset} not found in LABEL_CONFIG")

    change_label = config['change']
    no_change_label = config['no_change']

    # 计算分类结果
    tp = (prediction_matrix == change_label) & (labels == change_label)
    fp = (prediction_matrix == change_label) & (labels == no_change_label)
    fn = (prediction_matrix == no_change_label) & (labels == change_label)
    tn = (prediction_matrix == no_change_label) & (labels == no_change_label)
    other = ~(tp | fp | fn | tn)

    # 颜色映射（保持原逻辑）
    result = np.zeros((*labels.shape, 3), dtype=np.uint8)
    result[tp] = [255, 255, 255]  # 白色：正确变化
    result[fp] = [255, 0, 0]  # 红色：误报
    result[fn] = [0, 255, 0]  # 绿色：漏报
    result[tn] = [0, 0, 0]  # 黑色：正确未变化
    result[other] = [100, 100, 100]

    img = Image.fromarray(result)
    img.save(save_path)


# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# prepare data
if args.dataset == 'BayArea':
    data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2013.mat')['HypeRvieW']
    data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\Bay_Area_2015.mat')['HypeRvieW']
    data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\bayArea\bayArea_gtChanges2.mat')['HypeRvieW']
    uc_position = np.array(np.where(data_label==2)).transpose(1,0)
    c_position = np.array(np.where(data_label==1)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))
    print(args.dataset)

elif args.dataset == 'Barbara':
    data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_2013.mat')['HypeRvieW']
    data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_2014.mat')['HypeRvieW']
    data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\santaBarbara\barbara_gtChanges.mat')['HypeRvieW']
    uc_position = np.array(np.where(data_label==2)).transpose(1,0)
    c_position = np.array(np.where(data_label==1)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))
    print(args.dataset)

elif args.dataset == 'river':
    data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_before.mat')['river_before']
    data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\river_after.mat')['river_after']
    data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\river\groundtruth.mat')['lakelabel_v1']
    uc_position = np.array(np.where(data_label==0)).transpose(1,0)
    c_position = np.array(np.where(data_label==255)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))
    print(args.dataset)
    data_label = (data_label-data_label.min())/(data_label.max()-data_label.min())
    data_label[data_label==0]=2

elif args.dataset == 'farmland':
    data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm06.mat')['imgh']
    data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\farm07.mat')['imghl']
    data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\farm\label.mat')['label']
    uc_position = np.array(np.where(data_label==0)).transpose(1,0)
    c_position = np.array(np.where(data_label==1)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))
    print(args.dataset)
    data_label[data_label==0]=2

elif args.dataset == 'hermiston':
    data_t1 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2004.mat')['HypeRvieW']
    data_t2 = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\hermiston2007.mat')['HypeRvieW']
    data_label = loadmat(r'Z:\pycharm\pythonProject\1.SSA\dataset\Hermiston\label.mat')['label']
    uc_position = np.array(np.where(data_label==0)).transpose(1,0)
    c_position = np.array(np.where(data_label==1)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))
    print(args.dataset)
    data_label[data_label==0]=2
else:
    raise ValueError("Unkknow dataset")

selected_uc = np.random.choice(uc_position.shape[0], int(args.train_number * uc_position.shape[0]), replace = False)
selected_c = np.random.choice(c_position.shape[0], int(args.train_number * c_position.shape[0]), replace = False)
print(int(args.train_number * uc_position.shape[0]))
print(int(args.train_number * c_position.shape[0]))
selected_uc_position=uc_position[selected_uc]
selected_c_position=c_position[selected_c]
TR = np.zeros(data_label.shape)
for i in range (int(args.train_number*c_position.shape[0])):
    TR[selected_c_position[i][0],selected_c_position[i][1]]=1
for i in range (int(args.train_number*uc_position.shape[0])):
    TR[selected_uc_position[i][0],selected_uc_position[i][1]]=2
#--------------测试样本-----------------
TE=data_label-TR

# color_mat = loadmat('./data/AVIRIS_colormap.mat')


num_classes = np.max(TR)
num_classes=int(num_classes)
# color_mat_list = list(color_mat)
# color_matrix = color_mat[color_mat_list[3]] #(17,3)
# normalize data by band norm


input1_normalize = np.zeros(data_t1.shape)
input2_normalize = np.zeros(data_t1.shape)
# for i in range(data_t1.shape[2]):
#     input_max = max(np.max(data_t1[:,:,i]),np.max(data_t2[:,:,i]))
#     input_min = min(np.min(data_t1[:,:,i]),np.min(data_t2[:,:,i]))
#     input1_normalize[:,:,i] = (data_t1[:,:,i]-input_min)/(input_max-input_min)
#     input2_normalize[:,:,i] = (data_t2[:,:,i]-input_min)/(input_max-input_min)
# 分波段进行归一化
for i in range(data_t1.shape[2]):
    # 计算当前波段在两个数据集中的最大值和最小值
    input_max = max(np.max(data_t1[:, :, i]), np.max(data_t2[:, :, i]))
    input_min = min(np.min(data_t1[:, :, i]), np.min(data_t2[:, :, i]))

    # 计算分母
    denominator = input_max - input_min

    # 处理分母为零的情况
    if denominator == 0:
        input1_normalize[:, :, i] = 1  # 可以根据需要修改为其他常数值
        input2_normalize[:, :, i] = 1
    else:
        # 正常的归一化操作
        input1_normalize[:, :, i] = (data_t1[:, :, i] - input_min) / denominator
        input2_normalize[:, :, i] = (data_t2[:, :, i] - input_min) / denominator

height, width, band = data_t1.shape

#-------------------------------------------------------------------------------

if args.flag_test=='train':
    total_pos_train, total_pos_test, number_train, number_test = chooose_train_and_test_point(TR, TE, num_classes)
    mirror_image_t1 = mirror_hsi(height, width, band, input1_normalize, patch=args.patches)
    mirror_image_t2 = mirror_hsi(height, width, band, input2_normalize, patch=args.patches)
    x_train_band_t1, x_test_band_t1 = train_and_test_data(mirror_image_t1, band, total_pos_train, total_pos_test, patch=args.patches, band_patch=args.band_patches)
    x_train_band_t2, x_test_band_t2 = train_and_test_data(mirror_image_t2, band, total_pos_train, total_pos_test, patch=args.patches, band_patch=args.band_patches)
    y_train, y_test = train_and_test_label(number_train, number_test, num_classes)
    #-------------------------------------------------------------------------------
    # load data
    x_train_t1=torch.from_numpy(x_train_band_t1.transpose(0,2,1)).type(torch.FloatTensor)
    x_train_t2=torch.from_numpy(x_train_band_t2.transpose(0,2,1)).type(torch.FloatTensor)
    y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
    Label_train=Data.TensorDataset(x_train_t1,x_train_t2,y_train)
    x_test_t1=torch.from_numpy(x_test_band_t1.transpose(0,2,1)).type(torch.FloatTensor)
    x_test_t2=torch.from_numpy(x_test_band_t2.transpose(0,2,1)).type(torch.FloatTensor)
    y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
    Label_test=Data.TensorDataset(x_test_t1,x_test_t2,y_test)


    label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
    label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)

elif args.flag_test=='test':
    mirror_image_t1 = mirror_hsi(height, width, band, input1_normalize, patch=args.patches)
    mirror_image_t2 = mirror_hsi(height, width, band, input2_normalize, patch=args.patches)
    x1_true = np.zeros((height*width, args.patches, args.patches, band), dtype=float)
    x2_true = np.zeros((height*width, args.patches, args.patches, band), dtype=float)
    y_true=[]
    for i in range(height):
        for j in range(width):
            x1_true[i*width+j,:,:,:]=mirror_image_t1[i:(i+args.patches),j:(j+args.patches),:]
            x2_true[i*width+j,:,:,:]=mirror_image_t2[i:(i+args.patches),j:(j+args.patches),:]
            y_true.append(i)
    y_true = np.array(y_true)
    x1_true_band = gain_neighborhood_band(x1_true, band, args.band_patches, args.patches)
    x2_true_band = gain_neighborhood_band(x2_true, band, args.band_patches, args.patches)
    x1_true_band=torch.from_numpy(x1_true_band.transpose(0,2,1)).type(torch.FloatTensor)
    x2_true_band=torch.from_numpy(x2_true_band.transpose(0,2,1)).type(torch.FloatTensor)
    y_true=torch.from_numpy(y_true).type(torch.LongTensor)
    Label_true=Data.TensorDataset(x1_true_band,x2_true_band,y_true)
    label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)
    print('------测试数据加载完毕------')

#-------------------------------------------------------------------------------
# create model
model = hygstan(
    image_size = args.patches,
    num_patches = band,
    p=12,
    d=64,
)

model = model.cuda()
# criterion
criterion = Loss_fn(delta=4.0, lambda_param=0.3).cuda()

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//20, gamma=args.gamma)
#-------------------------------------------------------------------------------
if args.flag_test == 'test':
    print("start testing")
    model_path = f'./log/hygstan_{args.dataset}.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # output classification maps
    pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(height):
        for j in range(width):
            prediction_matrix[i,j] = pre_u[i*width+j] + 1

    # 绘制并保存预测图片
    plot_save_path = f"output/{args.dataset}_prediction.png"
    plot_prediction(prediction_matrix, data_label, args.dataset, plot_save_path)
    print("图片绘制成功")
    # plt.subplot(1,1,1)
    #plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    # from PIL import Image
    # im=Image.fromarray((1-(prediction_matrix-1))*255)
    # import ipdb; ipdb.set_trace()
    # savemat('matrix.mat',{'P':prediction_matrix, 'label':label})
elif args.flag_test == 'train':
    print("start training")
    tic = time.time()
    for epoch in range(args.epoches):
        # train model
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
        OA1, F11, Pr1, Re1, Kappa1, AA_mean1, AA1 = output_metric(tar_t, pre_t)
        print(
            "Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f} | F1: {:.4f} | Pr: {:.4f} | Re: {:.4f} | Kappa: {:.4f}"
            .format(epoch + 1, train_obj, train_acc, F11, Pr1, Re1, Kappa1))

        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
            model.eval()
            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
            OA2, F12, Pr2, Re2, Kappa2, AA_mean2, AA2 = output_metric(tar_v, pre_v)
            print("OA: {:.4f} | F1: {:.4f} | Pr: {:.4f} | Re: {:.4f} | Kappa: {:.4f} | AA: {:.4f}"
                  .format(OA2, F12, Pr2, Re2, Kappa2, AA_mean2))
            scheduler.step()

    toc = time.time()
    print("Running Time: {:.2f}".format(toc - tic))
    print("**************************************************")

    print("Final result:")
    print("OA: {:.4f} | F1: {:.4f} | Pr: {:.4f} | Re: {:.4f} | Kappa: {:.4f} | AA: {:.4f}"
              .format(OA2, F12, Pr2, Re2, Kappa2, AA_mean2))
    print("Class-wise Accuracy:", AA2)
    print("**************************************************")
    print("Parameter:")
    torch.save(model.state_dict(), "log/hygstan_BayArea.pth")

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

print_args(vars(args))
