import torch
import math 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def nll_loss(hazards, S, Y, c, alpha=0., eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1).long()  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    S_padded = torch.cat([torch.ones_like(c), S], 1) 
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def surv(out, gt, alpha=0.):
    loss = 0
    for haz, s in zip(out['hazards'], out['S']):
        loss += nll_loss(haz, s, gt['label'], gt['c'], alpha=alpha)
    return loss

def contra(out, gt, temperature=0.1):
    feat_list = []
    clabel = []
    for idx, feats in enumerate(out['contrastive']):
        feat = torch.concat(feats, dim=0)
        feat_list.append(feat)
        clabel.append(torch.ones((len(feat))) * idx)
    
    feats = torch.concat(feat_list, dim=0)
    labels = torch.concat(clabel, dim=0).cuda()

    feats = F.normalize(feats, dim=-1, p=2)
    logits_mask = torch.eye(feats.shape[0]).float().cuda()
    mask = torch.eq(labels.view(-1, 1), labels.contiguous().view(1, -1)).float() - logits_mask

    # compute logits
    logits = torch.matmul(feats, feats.T) / temperature
    logits = logits - logits_mask * 1e9

    # optional: minus the largest logit to stablize logits
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()

    # compute ground-truth distribution
    p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
    #loss = compute_cross_entropy(p, logits)
    logits = F.log_softmax(logits, dim=-1)
    loss = torch.sum(p * logits, dim=-1)

    return -loss.mean()

def know(out, gt):
    know_loss = 0
    for rfeat, ofeat in out['knowledge']:
        know_loss += torch.dist(rfeat, ofeat.detach(), p=2) 
    
    return know_loss.mean()

class Loss_factory(nn.Module):
    def __init__(self, args):
        super(Loss_factory, self).__init__()
        loss_item = args.loss.split(',')
        self.loss_collection = {}
        for loss_im in loss_item:
            tags = loss_im.split('_')
            self.loss_collection[tags[0]] = float(tags[1]) if len(tags) == 2 else 1.
            
        self.awl = args.awl
        self.awl_func = AutomaticWeightedLoss(len(loss_item)) if self.awl else None
        
    def forward(self, preds, target):
        ldict = {}
        for loss_name, weight in self.loss_collection.items():
            loss = eval(loss_name + '(preds, target) * weight')
            ldict[loss_name] = loss
            
        if self.awl:
            loss_sum = self.awl_func(ldict.values())
        else:
            loss_sum = sum(ldict.values())
        
        return loss_sum, ldict



class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x):
        
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum