import argparse
import os



def get_args_Hermiston(seed):
    print('---------------------------func: get_args_Hermiston---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_before', default='./data/Hermiston/hermiston2004.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--data_after', default='./data/Hermiston/hermiston2007.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='Hermiston',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--idx_file',
                        default='./data/Hermiston/label.mat',
                        type=str, help='path filename of the trained model')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')
    parser.add_argument('--save_model_path',default=path + '/Hermiston_HyperNet_' + str(seed) + '.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path', default=path + '/Hermiston_HyperNet_result_' + str(seed) + '.mat',
                        type=str, help='path filename of the trained model')


    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_before,args.data_after)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)

    return args

def get_args_farm(seed):
    print('---------------------------func: get_args_farm---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_before', default='./data/farm/farm06.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--data_after', default='./data/farm/farm07.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='farm',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--idx_file',
                        default='./data/farm/label.mat',
                        type=str, help='path filename of the trained model')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')
    parser.add_argument('--save_model_path',default=path + '/farm_HyperNet_' + str(seed) + '.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path', default=path + '/farm_HyperNet_result_' + str(seed) + '.mat',
                        type=str, help='path filename of the trained model')


    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_before,args.data_after)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)

    return args
def get_args_Bay(seed):
    print('---------------------------func: get_args_Bay---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_before', default='./data/bayArea/mat/Bay_Area_2013.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--data_after', default='./data/bayArea/mat/Bay_Area_2015.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='Bay',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--idx_file', default='./data/bayArea/mat/bayArea_gtChanges2.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)

    parser.add_argument('--save_model_path',
                        default= path + '/Bay_HyperNet_'+str(seed)+'.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path',
                        default= path + '/Bay_HyperNet_result'+str(seed)+'.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_before,args.data_after)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args


def get_args_Barbara(seed):
    print('---------------------------func: get_args_Barbara---------------------------')
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--EX_num', default='Barbara',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)

    parser.add_argument('--data_before',default='./data/santaBarbara/mat/barbara_2013.mat',
                            type=str, help='path filename of training data')
    parser.add_argument('--data_after', default='./data/santaBarbara/mat/barbara_2014.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--idx_file',default='./data/santaBarbara/mat/barbara_gtChanges.mat',
                            type=str, help='path filename of the trained model')
    parser.add_argument('--save_model_path', default=path +'/Barbara_half1_HyperNet_' + str(seed) + '.pkl',
                            type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path', default=path +'/Barbara_half1_HyperNet_result' + str(seed) + '.mat',
                            type=str, help='path filename of the trained model')

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_before,args.data_after)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args

def get_args_farm(seed):
    print('---------------------------func: get_args_farm---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_before', default='./data/farm/farm06.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--data_after', default='./data/farm/farm07.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='farm',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--idx_file',
                        default='./data/farm/label.mat',
                        type=str, help='path filename of the trained model')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')
    parser.add_argument('--save_model_path',default=path + '/farm_HyperNet_' + str(seed) + '.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path', default=path + '/farm_HyperNet_result_' + str(seed) + '.mat',
                        type=str, help='path filename of the trained model')


    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_before,args.data_after)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args

def get_args_river(seed):
    print('---------------------------func: get_args_river---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_before', default='./data/river/river_before.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--data_after', default='./data/river/river_after.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='river',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--idx_file',
                        default='./data/river/groundtruth.mat',
                        type=str, help='path filename of the trained model')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')
    parser.add_argument('--save_model_path',default=path + '/river_HyperNet_' + str(seed) + '.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path', default=path + '/river_HyperNet_result_' + str(seed) + '.mat',
                        type=str, help='path filename of the trained model')


    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_before,args.data_after)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)

    return args