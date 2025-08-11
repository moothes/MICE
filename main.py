import os
import sys
import csv
import time
import random
import numpy as np
import pandas as pd
import importlib
import torch
    
from shutil import copyfile

from utils.options import parse_args
from utils.loss_factory import Loss_factory
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from utils.engine import Engine
from utils.dataset import TCGA_dataset
    
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
    args.num_subsets = len(dataset.subsets)

    # 5-fold cross validation
    best_score = []
    best_splits = []
    train_list = []
    if args.stage == 'pretrain':
        train_list = [dataset.subsets, ]
    elif args.stage == 'finetune':
        train_list = [[set_name, ] for set_name in dataset.test_subsets]
        args.num_epoch = 10
        args.loss = 'surv_1'
    else:
        train_list = [[set_name, ] for set_name in dataset.test_subsets]
        args.num_epoch = 20
        args.loss = 'surv_1'
        
    results_dir = "./results/{model}_{stage}_{time}".format(model=args.model, stage=args.stage, time=time.strftime("%m-%d-%H-%M"))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    args.results_dir = results_dir
    csv_path = os.path.join(results_dir, "results_all.csv")
    
    print(args)
    summary = pd.DataFrame(columns=['dataset', 'stage', 'fold', 'cindex', 'best_cindex'])
    for train_set in train_list:
        for fold in folds:
            args.current_fold = fold
            set_seed(args.seed)
            dataset.set_train_test(fold, trset=train_set)
            
            model = importlib.import_module('models.{}.network'.format(args.model)).Network(args)
            if args.stage == 'finetune':
                weight_fold = 'results/{}_pretrain_/fold_{}/'.format(args.model, fold)
                for file in os.listdir(weight_fold):
                    if file.endswith('.tar'):
                        weight_path = weight_fold + file
                
                weights = torch.load(weight_path)
                model.load_state_dict(weights['state_dict'], strict=False)
                print('Load weights from {}.'.format(weight_path))
                
            engine = Engine(args)
            criterion = Loss_factory(args)
            optimizer = define_optimizer(args, model)
            scheduler = define_scheduler(args, optimizer)
            
            cindex_list, best_split = engine.learning(model, dataset, criterion, optimizer, scheduler, train_set, fold)
            
            best_score.append(cindex_list)
            best_splits.append(best_split)
            print('Overall: {}.'.format(best_score))
            if args.stage == 'pretrain':
                print('Best_split: {}.'.format(best_splits))
            
            if args.stage == 'pretrain':
                save_set = dataset.test_subsets
            else:
                save_set = train_set
            for set_name, cindex, bsplit in zip(save_set, cindex_list, best_split):
                new_row = {'dataset': set_name, 'stage': args.stage, 'fold': fold, 'cindex': cindex, 'best_cindex': bsplit}
                summary.loc[len(summary)] = new_row

            summary.to_csv(csv_path, index=False)
    return csv_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Need model name!')
        exit()
        
    args = parse_args(sys.argv[1])
    results = main(args)
    print("Finished!")
