import os
import shutil
import torch
import numpy as np
from tqdm import tqdm
import time

from sksurv.metrics import concordance_index_censored
from sklearn.cluster import KMeans

import torch.optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

torch.set_num_threads(1)

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class Engine(object):
    def __init__(self, args):
        self.args = args        
        self.best_score = 0
        self.best_epoch = 0
        self.best_cindex_list = []
        self.set_list = []
        self.filename_best = None

    def learning(self, model, dataset, criterion, optimizer, scheduler, trset, fold):
        fold_dir = os.path.join(self.args.results_dir, 'fold_' + str(fold))
        if not os.path.isdir(fold_dir):
            os.mkdir(fold_dir)

        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            model = model.cuda()

        best_split = []
        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train one epoch
            self.run_epoch(dataset, model, criterion, phase='train', optimizer=optimizer, trset=trset)
            # test one epoch
            c_index = self.run_epoch(dataset, model, criterion, phase='eval', trset=trset)
            
            if best_split == []:
                best_split = c_index
            else:
                best_split = [max(a, b) for a, b in zip(best_split, c_index)]
            ave_cindex = np.mean(c_index)
            
            if ave_cindex >= self.best_score:
                self.best_score = ave_cindex
                self.best_epoch = epoch
                self.best_cindex_list = c_index
                self.set_list = dataset.test_subsets if self.args.stage == 'pretrain' else trset
                
                save_dict = {}
                for name, vec in model.state_dict().items():
                    if 'text_biobert' not in name:
                        save_dict[name] = vec
                        
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': save_dict,
                    'best_score': ave_cindex,
                    'trset': 'all' if len(trset) > 1 else trset[0],
                    'score': c_index,
                    'best_split': best_split,
                    'fold': fold,}, remove_last=False)
            print(' *** Current c-index={:.4f}, best c-index={:.4f} at epoch {}'.format(ave_cindex, self.best_score, self.best_epoch))
            print('>')
        return self.best_cindex_list, best_split 

    def run_epoch(self, dataset, model, criterion, phase='train', optimizer=None, trset='all'):
        eval('model.{}()'.format(phase))
        dataset.phase = phase
        if phase == 'train':
            train_loader = DataLoader(dataset=dataset, batch_size=4 if self.args.stage == 'pretrain' else 1, shuffle=True, num_workers=1, pin_memory=True, drop_last=True, collate_fn=collate_custom)
            return self.run_one_set(model, train_loader, len(train_loader), criterion, optimizer, phase='train', test_set=trset)
        else:
            c_index_set = []
            for test_set in trset:
                dataset.set_test_set(test_set)
                num_samples = dataset.test_len()
                if num_samples == 0:
                    continue
                test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_custom)
                
                index = self.run_one_set(model, test_loader, num_samples, criterion, optimizer, phase='eval', test_set=test_set)
                c_index_set.append(index)
            return c_index_set
    
    def run_one_set(self, model, dataset, num_samples, criterion, optimizer, phase='train', test_set='all'):
        eval('model.{}()'.format(phase))
        sum_loss = 0.0
        all_loss_dict = {}
        for k in criterion.loss_collection.keys():
            all_loss_dict[k] = 0
        all_risk_scores = np.zeros((0))
        all_censorships = np.zeros((0))
        all_event_times = np.zeros((0))
        set_prefix = test_set if phase == 'eval' else (len(test_set) if len(test_set) > 1 else test_set[0])
        progressbar = tqdm(range(num_samples), desc='{} {} samples from {} dataset(s) for epoch {}'.format(phase, num_samples, set_prefix, self.epoch), ncols=200)
        
        for index, (data_WSI, data_omic, data_report, cancer_id, label, event_time, c, pid) in zip(progressbar, dataset):
            wsis, omics, texts = [], [], []
            for wsi, omic, text in zip(data_WSI, data_omic, data_report):
                wsi = wsi.cuda()
                wsis.append(wsi)
                omics.append(omic.cuda())
                texts.append(text.cuda())
            
            cancer_id = cancer_id.long().cuda()
            label = label.float().cuda()
            event_time = event_time
            c = c.float().cuda()
            
            if phase == 'train':
                out = model(x_path=wsis, x_omic=omics, x_text=texts, cancer_id=cancer_id, phase=phase)
            else:
                with torch.no_grad():
                    out = model(x_path=wsis, x_omic=omics, x_text=texts, cancer_id=cancer_id, phase=phase)
            loss, loss_dict = criterion(out, {'label': label, 'event_time': event_time, 'c': c})
            
            risk = -torch.sum(out['S'][-1], dim=1).detach().cpu().numpy()
            all_risk_scores = np.concatenate((all_risk_scores, risk), axis=0)
            all_censorships = np.concatenate((all_censorships, c.cpu().numpy()), axis=0)
            all_event_times = np.concatenate((all_event_times, event_time), axis=0)
            
            pats = []
            for k, v in loss_dict.items():
                all_loss_dict[k] += v
                pats.append('{}: {:.4f}'.format(k, all_loss_dict[k] / (index + 1)))
            str_loss = ', '.join(pats)
            lr_str = 0
            sum_loss += loss.item()
            
            if phase == 'train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_str = optimizer.param_groups[-1]['lr']
                progressbar.set_postfix_str('LR: {:.1e}, {}'.format(lr_str, str_loss))
            
        # calculate loss and error for epoch
        sum_loss /= len(progressbar)
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(sum_loss, c_index))
        
        if phase == 'eval':
            n_bootstraps = 1000
            cindex_scores = []
            for _ in range(n_bootstraps):
                aa = [0, ]
                while np.sum(aa) == 0:
                    indices = np.random.choice(range(len(all_event_times)), len(all_event_times), replace=True)
                    aa = (1-all_censorships).astype(bool)[indices]
                bb = all_event_times[indices]
                cc = all_risk_scores[indices]
                cindex_bootstrap = concordance_index_censored(aa, bb, cc, tied_tol=1e-08)[0]
                cindex_scores.append(cindex_bootstrap)
            
        
            print(np.percentile(cindex_scores, [2.5, 97.5]))
        return c_index
    
    def deploy(self, model, dataset, criterion, trset, fold):
        c_index_set = []
        self.epoch = -1
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            model = model.cuda()

        for test_set in trset:
            dataset.set_test_set(test_set)
            num_samples = dataset.test_len()
            if num_samples == 0:
                continue
            test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_custom)
            
            base_weight_path = 'results/{}_{}_/fold_{}/'.format(self.args.model, self.args.stage, fold)
            
            file_list = os.listdir(base_weight_path)
            for file_name in file_list:
                if self.args.stage in ('finetune', 'split'):
                    if file_name.startswith(test_set):
                        weight_path = file_name
                else:
                    if file_name.endswith('.pth.tar'):
                        weight_path = file_name
            state_dict = torch.load(base_weight_path + weight_path)['state_dict']
            model.load_state_dict(torch.load(base_weight_path + weight_path)['state_dict'], strict=False)
            print('Load weights from {}.'.format(base_weight_path + weight_path))

            model.eval()
            index = self.run_one_set(model, test_loader, num_samples, criterion, None, phase='eval', test_set=test_set)
            c_index_set.append(index)
        return c_index_set

    def save_checkpoint(self, state, remove_last=False):
        if self.filename_best is not None and not remove_last:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.args.results_dir, 'fold_' + str(state['fold']), '{}_epoch{}_{:.4f}.pth.tar'.format(state['trset'], state['epoch'], state['best_score']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)

def collate_custom(batch):

    path_feat = [item[0].float() for item in batch] 
    gene_feat = [item[1].float() for item in batch] 
    text_feat = [item[2] for item in batch] 

    cancer_id = torch.LongTensor([item[3] for item in batch])
    label = torch.LongTensor([item[4] for item in batch])
    event_time = np.array([item[5] for item in batch])
    c = torch.FloatTensor([item[6] for item in batch])
    pid = np.array([item[7] for item in batch])

    return path_feat, gene_feat, text_feat, cancer_id, label, event_time, c, pid