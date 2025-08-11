import torch
import numpy as np

import torch.nn as nn
from torch.nn import functional as F
import math
from .molre import MOEBlock
import warnings
from .biobert import get_biobert
from mamba_ssm import Mamba
warnings.filterwarnings("ignore")
from .vit import Transformer


custom_config = {'base'      : {'modal': 'path,omic,text',
                                'loss': 'surv_1,contra_2',
                                'lr': 1e-4, 
                                'optimizer': 'AdamW', 
                                'scheduler': 'None',
                                'num_epoch': 20,
                                'seed': 0,
                    },
                'customized': {'num_cluster': {'type': int, 'default': 6},
                    },
                }

def SNN_Block(dim1, dim2, dropout=0.2):
    return nn.Sequential(nn.Linear(dim1, dim2), nn.SELU(), nn.AlphaDropout(p=dropout, inplace=False))
    
def MLP_Block(dim1, dim2, dropout=0.2):
    return nn.Sequential(nn.Linear(dim1, dim2), nn.Dropout(dropout), nn.ReLU())

def conv1d_Block(dim1, dim2, dropout=0.2):
    return nn.Sequential(nn.Conv1d(dim1, dim2, 1), nn.InstanceNorm1d(dim2), nn.ReLU())
    
def conv2d_Block(dim1, dim2, dropout=0.2):
    return nn.Sequential(nn.Conv2d(dim1, dim2, 1), nn.InstanceNorm2d(dim2), nn.ReLU())

class MambaMIL(nn.Module):
    def __init__(self, path_dim=1024, feat_dim=64):
        super().__init__()
        self.feature = MLP_Block(path_dim, feat_dim) 
        
        self.mamba_block = Mamba(
            d_model=feat_dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.attention = nn.Sequential(nn.Linear(feat_dim, feat_dim//4), nn.Tanh(), nn.Linear(feat_dim//4, 1))
        
    def forward(self, x_path):
        feature = self.feature(x_path).unsqueeze(0)
        feat = self.mamba_block(feature) + feature
        A = self.attention(feat)
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.bmm(A, feature)  # KxL
        return M.squeeze(1), A

class Transformer_block(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim), requires_grad=True)
        self.trans = Transformer(dim=feat_dim, depth=2, heads=4, dim_head=64, mlp_dim=64)
        
    def forward(self, x):
        feat = torch.concat([self.cls_token, x], dim=1)
        feat = self.trans(feat)
        return feat[:, 0], feat[:, 1:]

class Network(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.feat_dim = 256
        self.n_classes = args.n_classes
        
        path_dim = 1024

        self.cancer_token = nn.Parameter(torch.randn(args.num_subsets, 1, self.feat_dim), requires_grad=True)
        
        # Pathology representation
        self.attmil = MambaMIL(path_dim, self.feat_dim)
        self.path_holder = MLP_Block(self.feat_dim, self.feat_dim)

        self.path_mean = torch.load('data/path/tcga_gpfm_mean.pt').cuda()
        self.path_std = torch.load('data/path/tcga_gpfm_std.pt').cuda()

        # Genomic representation
        self.omic_snn = nn.Sequential(MLP_Block(20245, self.feat_dim*2), MLP_Block(self.feat_dim*2, self.feat_dim))
        self.omic_holder = MLP_Block(self.feat_dim, self.feat_dim)

        # Report representation
        self.text_biobert = get_biobert()
        for param in self.text_biobert.parameters():
            param.requires_grad = False
        self.text_biobert.eval()
        self.text_snn = nn.Sequential(MLP_Block(768, self.feat_dim*2), MLP_Block(self.feat_dim*2, self.feat_dim))
        self.text_holder = MLP_Block(self.feat_dim, self.feat_dim)
        
        self.text_mean = torch.load('data/text/text_mean.pt').cuda()
        self.text_std = torch.load('data/text/text_std.pt').cuda()

        # Multimodal fusion
        self.moe_cls_token = MLP_Block(self.feat_dim, self.feat_dim)
        self.moe_block = MOEBlock(self.feat_dim, args.num_subsets, kernel_size=3, with_feat=False)
        
        self.transformer = Transformer_block(self.feat_dim)
        self.classifier = nn.Sequential(nn.Linear(self.feat_dim, self.n_classes))
        
    def forward(self, x_path, x_omic, x_text, cancer_id, phase='train', **kwargs):
        out_dict = {}
        batch_size = len(x_path)

        logits = []
        all_feats = []
        for idx, path_feat, omic_feat, text_feat, cid in zip(range(batch_size), x_path, x_omic, x_text, cancer_id):
            feats = []
            cpt = self.cancer_token[cid] # cancer-specific embedding
            
            if torch.numel(path_feat) == 1:
                path_feat = self.path_holder(cpt)
            else:
                path_feat = (path_feat - self.path_mean) / (self.path_std + 1e-10)
                path_feat, att = self.attmil(path_feat)
                out_dict['att'] = att
                feats.append(path_feat)

            if torch.numel(omic_feat) == 1:
                omic_feat = self.omic_holder(cpt)
            else:
                omic_feat = self.omic_snn(omic_feat).unsqueeze(0)
                feats.append(omic_feat)

            if torch.numel(text_feat) == 1:
                text_feat = self.text_holder(cpt)
            else:
                input_ids, input_mask = torch.unbind(text_feat, dim=0)
                with torch.no_grad():
                    text_pred = self.text_biobert(input_ids=input_ids.unsqueeze(0), attention_mask=input_mask.unsqueeze(0))
                    last_state = text_pred["last_hidden_state"]
                    tfeat = (last_state[:, 0] - self.text_mean) / (self.text_std + 1e-10)
                text_feat = self.text_snn(tfeat)
                
                feats.append(text_feat)
            
            all_feats.append(feats)
            feat = torch.stack([self.moe_cls_token(cpt), path_feat, omic_feat, text_feat], dim=1)
            knows, route_feat, route_1 = self.moe_block(feat, cid, phase=phase)
            
            feat = torch.stack(knows, dim=1)
            cls_out, feat = self.transformer(feat)
            feat = self.classifier(cls_out)
            logits.append(feat)

        logits = torch.concat(logits, dim=0)
        fuse_hazard = torch.sigmoid(logits)
        fuse_S = torch.cumprod(1 - fuse_hazard, dim=1)
        
        out_dict['contrastive'] = all_feats
        out_dict['hazards'] = [fuse_hazard]
        out_dict['S'] = [fuse_S]
        return out_dict
