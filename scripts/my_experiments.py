import os
import getpass
import sys

import numpy as np
import pickle
import pandas as pd
import sys
import os

import ipdb
import itertools
import tqdm

DATA_PATH = "Z:\Data" #Local
if not os.path.isdir(DATA_PATH): DATA_PATH = "/srv/local/data" #Workstation 1
if not os.path.isdir(DATA_PATH): DATA_PATH = "/srv/home/%s/data"%(getpass.getuser()) #Workstations and servers
__CUR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.join(__CUR_FILE_PATH, "Temp")
_PERSIST_PATH = os.path.join(WORKSPACE, 'cache')
LOG_OUTPUT_DIR = os.path.join(WORKSPACE, 'logs')
RANDOM_SEED = 7

ISRUC_NAME = 'ISRUC_SLEEP1'
CIFAR10_NAME = 'CIFAR10'
CIFAR100_NAME = 'CIFAR100'
IIICSup_NAME = "IIICSup_1109"
ImageNet1K_NAME = "ImageNet1K"
SVHN_NAME = 'SVHN'

NUM_CLASSES = {ISRUC_NAME: 5, IIICSup_NAME: 6, CIFAR100_NAME: 100, CIFAR10_NAME: 10, ImageNet1K_NAME: 1000, SVHN_NAME: 10}

from tune_dirichlet_nn_slim import tune_dir_nn

def get_default_hyperparams(nclass):
    if nclass <= 10:
        start_from = -5.0
    elif nclass >= 100:
        start_from = -2.0
    else:
        raise ValueError()
    lambdas = np.array([10 ** i for i in np.arange(start_from, 7)])
    lambdas = sorted(np.concatenate([lambdas, lambdas * 0.25, lambdas * 0.5]))
    mus = np.array([10 ** i for i in np.arange(start_from, 7)])
    return lambdas, mus



def main_Matrix_ODIR(dataset, filename='key_0.pkl'):
    lambdas, mus = get_default_hyperparams(NUM_CLASSES[dataset])

    cache_path = os.path.join(WORKSPACE, dataset, filename.replace(".pkl", "_res.pkl"))
    if os.path.isfile(cache_path): return cache_path
    if not os.path.isdir(os.path.dirname(cache_path)): os.makedirs(os.path.dirname(cache_path))
    # python -u tune_cal_odir.py -i 0 -kf 5 -d --comp_l2 --use_logits
    tune_dir_nn(os.path.join(WORKSPACE, dataset, filename),
                         lambdas=lambdas, mus=mus, verbose=False, k_folds=5, random_state=15, double_learning = True,
                         cache_path = cache_path,
                         loss_fn='sparse_categorical_crossentropy', comp_l2 = True, use_logits = True, use_scipy = False)
    return cache_path

if __name__ == '__main__':
    for i in range(10):
        #CIFAR-10
        main_Matrix_ODIR(CIFAR10_NAME, f'ViTB16_timm-20211226_013412_{i}.pkl'); break
        #main_Matrix_ODIR(CIFAR10_NAME, f'MixerB16_timm-20220102_222852_{i}.pkl')

        #CIFAR-100
        #main_Matrix_ODIR(CIFAR100_NAME, f'ViTB16_timm-20211215_013918_{i}.pkl')
        #main_Matrix_ODIR(CIFAR100_NAME, f'MixerB16_timm-20220102_191050_{i}.pkl')

        #SVHN ViT
        #main_Matrix_ODIR(SVHN_NAME, f'ViTB16_timm-20211226_235923_{i}.pkl')
        #main_Matrix_ODIR(SVHN_NAME, f'MixerB16_timm-20220103_085510_{i}.pkl')

        #IIIC
        #main_Matrix_ODIR(IIICSup_NAME, f'CNNEncoder2D_IIIC-20211226_131303_{i}.pkl')

        #ISRUC
        #main_Matrix_ODIR(ISRUC_NAME, f'CNNEncoder2D_ISRUC-20220102_225606_{i}.pkl')

        #ImageNet 1K
        #main_Matrix_ODIR(ImageNet1K_NAME, f'inception_resnet_v2_{i}.pkl')

        pass