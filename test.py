import os
import sys
import csv
import time
import random
import numpy as np
import importlib
import torch
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')

from utils.options import parse_args
from shutil import copyfile

from utils.dataset import TCGA_dataset
from torch.utils.data import DataLoader
from utils.loss_factory import Loss_factory
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from utils.engine import Engine

from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained

def set_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def main(args):    
    folds = list(map(int, args.fold.split(',')))
    
    dataset = TCGA_dataset(args)
    dataset.phase = 'eval'
    args.num_subsets = len(dataset.subsets)

    # 5-fold cross validation
    best_score = []
    best_splits = []
    train_list = []
    if args.stage == 'pretrain':
        train_list = [dataset.subsets, ]
    elif args.stage == 'finetune':
        train_list = [[set_name, ] for set_name in dataset.test_subsets]
    else:
        train_list = [[set_name, ] for set_name in dataset.test_subsets]
    
    model = importlib.import_module('models.{}.network'.format(args.model)).Network(args)
    model.eval()
    engine = Engine(args)
    criterion = Loss_factory(args)
    
    all_cindex = []
    for train_set in train_list:
        for fold in folds:
            args.current_fold = fold
        
            set_seed(args.seed)
            dataset.set_train_test(fold, trset=train_set)
            cindex_list = engine.deploy(model, dataset, criterion, train_set, fold)
            all_cindex.append(cindex_list)
    ss = np.array(all_cindex).reshape(-1, 5)

    return all_cindex

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Need model name!')
        exit()
        
    args = parse_args(sys.argv[1])
    print(args)
    results = main(args)
    print("Finished!")
