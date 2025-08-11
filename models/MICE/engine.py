import os
import shutil
import torch
import numpy as np
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored
from sklearn.cluster import KMeans

import torch.optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

torch.set_num_threads(4)

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        # tensorboard
        '''
        if args.log_data:
            from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            if not os.path.isdir(writer_dir):
                os.mkdir(writer_dir)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
        '''
        self.best_score = 0
        self.best_epoch = 0
        self.filename_best = None

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler, subset):
        writer_dir = os.path.join(self.results_dir, subset + '_fold_' + str(self.fold))
        if not os.path.isdir(writer_dir):
            os.mkdir(writer_dir)
        self.writer = SummaryWriter(writer_dir, flush_secs=15)
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            model = model.cuda()

        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            self.run_epoch(val_loader, model, criterion, phase='eval')
            return

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            self.run_epoch(train_loader, model, criterion, phase='train', optimizer=optimizer)
            # evaluate on validation set
            c_index = self.run_epoch(val_loader, model, criterion, phase='eval')
            # remember best c-index and save checkpoint
            is_best = c_index >= self.best_score
            if is_best:
                self.best_score = c_index
                self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_score,
                    'subset': subset,})
            print(' *** best c-index={:.4f} at epoch {}'.format(self.best_score, self.best_epoch))
            #scheduler.step()
            print('>')
        return self.best_score, self.best_epoch

    def run_epoch(self, data_loader, model, criterion, phase='train', optimizer=None):
        eval('model.{}()'.format(phase))
        sum_loss = 0.0
        
        all_loss_dict = {}
        for k in criterion.loss_collection.keys():
            all_loss_dict[k] = 0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='{} Epoch: {}'.format(phase, self.epoch), ncols=150)
        
        for batch_idx, (data_WSI, data_omic, label, event_time, c, idx) in enumerate(dataloader):
            data_WSI = data_WSI.float().cuda()
            #print(data_WSI.shape)
            data_omic = data_omic.float().cuda()
            label = label.float().cuda()
            event_time = event_time.float().cuda()
            c = c.float().cuda()
            
            out = model(x_path=data_WSI, x_omic=data_omic, phase=phase, label=label)
            #print(out['S'], label.shape, event_time.shape, c.shape)
            loss, loss_dict = criterion(out, {'label': label, 'event_time': event_time, 'c': c})

            risk = -torch.sum(out['S'][-1], dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time
            
            print(out['text_emb'].shape)
            
            pats = []
            for k, v in loss_dict.items():
                all_loss_dict[k] += v
                pats.append('{}: {:.4f}'.format(k, all_loss_dict[k] / (batch_idx + 1)))
            str_loss = ', '.join(pats)
            lr_str = 0
            sum_loss += loss.item()
            
            if phase == 'train':
                loss.backward()
                #clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                lr_str = optimizer.param_groups[-1]['lr']
            
            dataloader.set_postfix_str('LR: {:.1e}, {}'.format(lr_str, str_loss))
            
        # calculate loss and error for epoch
        sum_loss /= len(dataloader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(sum_loss, c_index))

        if self.writer:
            self.writer.add_scalar('{}/loss'.format(phase), sum_loss, self.epoch)
            self.writer.add_scalar('{}/c_index'.format(phase), c_index, self.epoch)
        return c_index

    def deploy(self, data_loader, model, criterion):
        model = model.cuda()
        model.eval()
        val_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        visual_dict = {'all_knowledge': [], 'all_grad': [], 'all_id': []}
        dataloader = tqdm(data_loader, desc='Validating ', ncols=100)
        for batch_idx, (data_WSI, data_omic, label, event_time, c, idx) in enumerate(dataloader):
            #np_WSI = np.array(data_WSI)
            #kmeans = KMeans(n_clusters=6, init=model.path_know_memory.cpu().numpy()[0], random_state=0, n_init="auto").fit(np_WSI[0])
            #data_WSI = torch.tensor(kmeans.cluster_centers_).float().cuda()
            data_WSI = data_WSI.float().cuda()
            data_omic = data_omic.float().cuda()
            label = label.float().cuda()
            event_time = event_time.float().cuda()
            c = c.float().cuda()
            
            #with torch.no_grad():
                #out = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
            out = model(x_path=data_WSI, x_omic=data_omic, phase='test')
            #print(out['mask'][0][0])
            #print(out['S'], label.shape, event_time.shape, c.shape)
            #print(label, event_time)
            loss, loss_dict = criterion(out, {'label': label, 'event_time': event_time, 'c': c})

            risk = -torch.sum(out['S'][-1], dim=1).detach().cpu().numpy()
            #print(out['S'][-1], risk)
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time
            val_loss += loss.item()
                        
            case_id = data_loader.dataset.slide_data['case_id'][int(idx)]
            # Visualization of attention map
            #np.save('visual/att/{}/{}_com.npy'.format(self.args.dataset, case_id), out['att'][0][0, 0].cpu().numpy())
            #np.save('visual/att/{}/{}_syn.npy'.format(self.args.dataset, case_id), out['att'][1][0, 0].cpu().numpy())
            
            # Visualization of knowledge components
            #print(out['cohort'][1].shape)
            #np.save('visual/knowledge/{}/{}.npy'.format(self.args.dataset, case_id), out['cohort'][1][0].cpu().numpy())
            visual_dict['all_knowledge'].append(out['cohort'][1][0].detach().cpu().numpy())
            
            common, synergy, g_spec, p_spec = out['cohort'][1][0]
            model.zero_grad()
            pseudo_loss = torch.sum(common)
            pseudo_loss.backward(retain_graph=True)
            com_geno = model.geno_conv[0].weight.grad.detach().cpu().numpy()
            com_path = model.path_conv[0].weight.grad.detach().cpu().numpy()
            #print(geno_grad)
            #print(model.geno_conv[0].weight.grad)
            #print(model.path_conv[0].weight.grad.shape)
            
            model.zero_grad()
            pseudo_loss = torch.sum(synergy)
            pseudo_loss.backward(retain_graph=True)
            syn_geno = model.geno_conv[0].weight.grad.detach().cpu().numpy()
            syn_path = model.path_conv[0].weight.grad.detach().cpu().numpy()
            
            model.zero_grad()
            pseudo_loss = torch.sum(g_spec)
            pseudo_loss.backward(retain_graph=True)
            gs_geno = model.geno_conv[0].weight.grad.detach().cpu().numpy()
            gs_path = model.path_conv[0].weight.grad.detach().cpu().numpy()
            
            model.zero_grad()
            pseudo_loss = torch.sum(p_spec)
            pseudo_loss.backward()
            ps_geno = model.geno_conv[0].weight.grad.detach().cpu().numpy()
            ps_path = model.path_conv[0].weight.grad.detach().cpu().numpy()
            
            visual_dict['all_grad'].append([[com_geno, syn_geno, gs_geno, ps_geno], [com_path, syn_path, gs_path, ps_path]])
            
            visual_dict['all_id'].append(case_id)
            
        val_loss /= len(dataloader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(val_loss, c_index))
        return c_index, all_risk_scores, all_censorships, all_event_times, visual_dict
            
    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir,
                                          state['subset'] + '_fold_' + str(self.fold),
                                          'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'], epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)
    '''
    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir,
                                          'fold_' + str(self.fold),
                                          'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'], epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)
    '''
    
    
