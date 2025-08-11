from __future__ import print_function, division
import os
import math
import pickle
import random
import time
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset
import json
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer

eps = 1e-5
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def is_nan(s):
    try:
        num = float(s)
        return math.isnan(num)
    except ValueError:
        return False

class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    metadata: Optional[dict] = None

class TCGA_dataset(Dataset):
    def __init__(self, args='', phase='train'):
        self.anno_file = args.anno_file
        self.phase = phase
        ncls = args.n_classes
        self.modal = args.modal.split(',')
        self.text_encoder_name = args.text_encoder

        data_list = pd.read_csv(self.anno_file)
        self.data_list = data_list.dropna(subset=['Event'])
        self.data_list.dropna(subset=self.modal, how='all', inplace=True)
        print("Pre-training with missing-modality samples")
                
        self.subsets = np.unique(self.data_list['Study']).tolist()
        self.test_subsets = []
        self.augm_subsets = []
        for subset in self.subsets:
            sublist = self.data_list[self.data_list['Study'] == subset]
            if len(sublist) > 200 and len(sublist[sublist['Status'].values == 1]) > 50:
                self.test_subsets.append(subset)
            else:
                self.augm_subsets.append(subset)
        
        print('Totally {} subsets ({}/{}) loaded: {}.'.format(len(self.subsets), len(self.test_subsets), len(self.augm_subsets), self.subsets))

        self.get_label(args.lbl_type, args.n_classes)

        self.set_dict = {}
        for idx, subset in enumerate(self.subsets):
            self.set_dict[subset] = idx
            
        self.set_train_test(fold=0)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name, fast=True)
        self.tokenizer.max_length = 512
        
        if not "cls_token" in self.tokenizer.special_tokens_map:
            self.tokenizer.add_special_tokens({"cls_token": "[CLS]"})

        if not "sep_token" in self.tokenizer.special_tokens_map:
            self.tokenizer.add_special_tokens({"sep_token": "[SEP]"})
        
    def get_report(self, report_path=''):
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name, fast=True)
        self.tokenizer.max_length = 512
        
        if not "cls_token" in self.tokenizer.special_tokens_map:
            self.tokenizer.add_special_tokens({"cls_token": "[CLS]"})

        if not "sep_token" in self.tokenizer.special_tokens_map:
            self.tokenizer.add_special_tokens({"sep_token": "[SEP]"})
        
        rp_df = pd.read_csv(report_path)
        self.report_dict = {}
        for i, row in rp_df.iterrows():
            self.report_dict[row.iloc[0]] = str(row['text'])
    
    
    def trunk(self, texts, max_length=512, method="random"):
        text = texts.split(" ")
        if len(text) > max_length:
            if method == "random":
                start = np.random.randint(0, high=len(text) - max_length + 1)
                text = text[start : start + max_length]
            elif method == "head":
                text = text[: max_length]
            elif method == "tail":
                text = text[-max_length :]
            else:
                raise NotImplementedError("trunk method not implemented.")
                
        texts = " ".join(text)
        return texts
    
    def get_tokens(self, raw_text, max_seq_length=512):
    
        raw_text = self.trunk(raw_text, max_seq_length)
        tokens = self.tokenizer(raw_text, max_length=max_seq_length, add_special_tokens=True, truncation=True, padding="max_length", return_token_type_ids=False, return_attention_mask=True)
        text_ids, text_mask = tokens["input_ids"], tokens["attention_mask"]
        
        features = torch.stack([torch.tensor(text_ids), torch.tensor(text_mask)], dim=0)        
        return features
        
    def get_tokens_old(self, raw_text, max_seq_length=512):
    
        tokens = self.tokenizer.tokenize(raw_text)

        if len(tokens) > max_seq_length-2:
            tokens = tokens[:max_seq_length-2]
        
        tokens += ['[SEP]']
        segment_ids = [0] * len(tokens)

        # CLS token
        tokens = ['[CLS]'] + tokens
        segment_ids = [0] + segment_ids
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length

        #print()
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        
        features = torch.stack([torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids)], dim=0)
        
        return features

    def get_label(self, lbl_type='split', ncls=10):
        if lbl_type == 'split':
            base_df = []
            for study in self.subsets:
                study_df = self.data_list[self.data_list['Study'] == study].copy()
                uncensored_df = study_df[study_df['Status'] > 0]

                disc_labels, q_bins = pd.qcut(uncensored_df['Event'], q=ncls, retbins=True, labels=False)
                q_bins[-1] = self.data_list['Event'].max() + eps
                q_bins[0] = self.data_list['Event'].min() - eps

                disc_labels, q_bins = pd.cut(study_df['Event'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
                study_df['Label'] = disc_labels.values.astype(int)

                base_df.append(study_df)
            self.data_list = pd.concat(base_df, axis=0, ignore_index=True)
        elif lbl_type == 'joint':
            uncensored_df = self.data_list[self.data_list['Status'] > 0]

            disc_labels, q_bins = pd.qcut(uncensored_df['Event'], q=ncls, retbins=True, labels=False)
            q_bins[-1] = self.data_list['Event'].max() + eps
            q_bins[0] = self.data_list['Event'].min() - eps

            disc_labels, q_bins = pd.cut(self.data_list['Event'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
            self.data_list['Label'] = disc_labels.values.astype(int)

    def set_train_test(self, fold=0, trset='all'):
        self.fold = fold
        self.train_list = self.data_list[self.data_list['Split'] != self.fold]
        self.test_list = self.data_list[self.data_list['Split'] == self.fold]
        if trset != 'all':
            self.train_list = self.train_list[self.train_list['Study'].isin(trset)]
            self.test_list = self.test_list[self.test_list['Study'].isin(trset)]

    def __iter__(self):
        self.test_idx = 0
        return self
    
    def __next__(self):
        cur_idx = self.test_idx
        self.test_idx += 1
        return self.get_sample(cur_idx, 'test')

    def __getitem__(self, index, phase='train'):
        return self.get_sample(index)

    def set_test_set(self, subset):
        self.current_test_set = self.test_list[self.test_list['Study'] == subset]

    def test_len(self):
        return len(self.current_test_set)

    def get_sample(self, index):
        st = time.time()
        if self.phase == 'train':
            data_row = self.train_list.iloc[index]
        elif self.phase =='eval':
            data_row = self.current_test_set.iloc[index]
        else:
            return None    

        pid = data_row['ID']
        gene_file = data_row['rna']
        omic_file = data_row['omic']
        path_file = data_row['path']
        raw_text = data_row['text']

        if 'gene' in self.modal:
            if not str(gene_file) == 'nan':
                gene_feat = torch.load(gene_file)
            else:
                gene_feat = torch.zeros(1)
        elif 'omic' in self.modal: 
            if not str(omic_file) == 'nan':
                gene_feat = torch.load(omic_file)
            else:
                gene_feat = torch.zeros(1)
        else:
            gene_feat = torch.zeros(1)
            
        if 'path' in self.modal:
            if not str(path_file) == 'nan':
                path_files = path_file.split(';')
                path_feat = []
                for pfile in path_files:
                    pt = torch.load(pfile)
                    path_feat.append(pt)
                path_feat = torch.concat(path_feat, dim=0)
            else:
                path_feat = torch.zeros(1)
        else:
            path_feat = torch.zeros(1)
        
        if 'text' in self.modal:
            if pd.isna(raw_text): 
                text_feat = torch.zeros(1)
            else:
                text_feat = self.get_tokens(raw_text)
        else:
            text_feat = torch.zeros(1)
            
        cancer_id = torch.tensor(self.set_dict[data_row['Study']])
        label = torch.tensor([int(data_row['Label'])]) 
        event_time = float(data_row['Event'])
        status = 1 - torch.tensor(float(data_row['Status'])) # Code is based on MCAT, where censorship is reversed with status

        if torch.numel(path_feat) == 1 and torch.numel(gene_feat) == 1 and torch.numel(text_feat) == 1:
            print(pid)
            
        return path_feat.detach(), gene_feat.detach(), text_feat.detach(), cancer_id, label, event_time, status, pid

    def get_gene_embed(self, gene_list):
        pass

    def get_path_mean(self):
        self.path_mean = torch.load('/storage/ssd1/huajun/gpfm/tnbc_gpfm_mean.pt').view(1, -1)
        self.path_std = torch.load('/storage/ssd1/huajun/gpfm/tnbc_gpfm_std.pt').view(1, -1)

    def __len__(self):
        if self.phase == 'train':
            return len(self.train_list)
        elif self.phase == 'eval':
            return len(self.current_test_set)
        else:
            return 0
    