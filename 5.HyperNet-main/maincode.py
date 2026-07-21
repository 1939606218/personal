# pytorch code of TGRS paper
# "HyperNet: Self-Supervised Hyperspectral SpatialSpectral Feature Understanding Network for Hyperspectral Change Detection"

import os
import torch
import numpy as np
from load_data import get_args_Hermiston,get_args_Barbara,get_args_Bay,get_args_river,get_args_farm
from utils import setup_seed,zz
from train_test import train_HyperNet_HBCD
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一块GPU
torch.set_num_threads(2)


if __name__ == "__main__":

    """ For (BCD)binary change detection of the HyperNet """
    dataset= 'farm'

    if dataset == 'Hermiston':
        SEED = np.arange(1, 6)
        for i in np.arange(0, 1):
            seed = SEED[i]
            print('\n')
            args = get_args_Hermiston(seed)
            # 假设args.seed是numpy.int32类型，这里进行转换
            if isinstance(args.seed, np.int32):
                args.seed = int(args.seed)
            setup_seed(args.seed)
            zz(seed)
            train_HyperNet_HBCD(args)

    if dataset == 'farm':
        SEED = np.arange(1, 6)
        for i in np.arange(0, 1):
            seed = SEED[i]
            print('\n')
            args = get_args_farm(seed)
            # 假设args.seed是numpy.int32类型，这里进行转换
            if isinstance(args.seed, np.int32):
                args.seed = int(args.seed)
            setup_seed(args.seed)
            zz(seed)
            train_HyperNet_HBCD(args)

    if dataset == 'river':
        SEED = np.arange(1, 6)
        for i in np.arange(0, 1):
            seed = SEED[i]
            print('\n')
            args = get_args_river(seed)
            # 假设args.seed是numpy.int32类型，这里进行转换
            if isinstance(args.seed, np.int32):
                args.seed = int(args.seed)
            setup_seed(args.seed)
            zz(seed)
            train_HyperNet_HBCD(args)

    if dataset== 'Bay':
        SEED = np.arange(1, 6)
        for i in np.arange(0, 1):
            seed = SEED[i]
            print('\n')
            args = get_args_Bay(seed)
            print(type(args.seed))
            # 假设args.seed是numpy.int32类型，这里进行转换
            if isinstance(args.seed, np.int32):
                args.seed = int(args.seed)
            setup_seed(args.seed)
            zz(seed)
            train_HyperNet_HBCD(args)

    if dataset== 'Barbara':
        SEED = np.arange(1, 11)
        Barbara_time = []
        for i in np.arange(0, 5):
            time1 = time.perf_counter()  # 使用time.perf_counter()替换time.clock()来记录开始时间
            seed = SEED[i]
            print('\n')
            args = get_args_Barbara(seed)
            #假设args.seed是numpy.int32类型，这里进行转换
            if isinstance(args.seed, np.int32):
                args.seed = int(args.seed)
            setup_seed(args.seed)
            zz(seed)
            train_HyperNet_HBCD(args)
            time2 = time.perf_counter()  # 使用time.perf_counter()替换time.clock()来记录结束时间
            Barbara_time.append(time2 - time1)










