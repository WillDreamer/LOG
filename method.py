
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

import torchvision.transforms as transforms
from loguru import logger
import torch.nn.functional as F
from data.data_helper import AdvDataset
from torch.utils.data.dataset import ConcatDataset
from model.model_loader import load_model
from evaluate import mean_average_precision
from model.labelmodel import *
from torch.nn import Parameter
from torch.autograd import Variable
from utils import *
import random
from PIL import ImageFilter
from collections import OrderedDict
# from apex import amp
torch.backends.cudnn.enabled = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from signal import signal, SIGPIPE, SIG_DFL 
#Ignore SIG_PIPE and don't throw exceptions on it... (http://docs.python.org/library/signal.html)
signal(SIGPIPE,SIG_DFL) 

def train(train_s_dataloader,
          query_dataloader,
          retrieval_dataloader,
          code_length,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          topk,
          num_class,
          evaluate_interval,
          tag,
          batch_size,
          knn,
          ):

    model = load_model(arch, code_length,num_class,num_class)
    # logger.info(model)
    model.to(device)
    #model = nn.DataParallel(model,device_ids=[0,1])
    # if isinstance(model,torch.nn.DataParallel):
    #     model = model.module
    parameter_list = model.get_parameters() 
    optimizer = optim.SGD(parameter_list, lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion_new = OrthoHashLoss()
    criterion = nn.CrossEntropyLoss()
    class_criterion = nn.CrossEntropyLoss()
    semantic_distance_criterion = nn.MSELoss()
    #model = nn.DataParallel(model,device_ids=[0,1])
    
    model.train()
    
    T_min = 100
    T_max = len(train_s_dataloader)
    K = 0
    for k in range(K):
        # print(len(train_s_dataloader)) ----> 185
        current_iter = 0
        for batch_idx, (data_s, _, target_s, index) in enumerate(train_s_dataloader):
            data_s = data_s.to(device)
            target_s = target_s.to(device)
            if current_iter == T_min:
                logger.info("Min-phase ended, %d-th Max-phase started!" % (k))
                max_optimizer = optim.SGD([data_s.requires_grad_()], lr=1)
                model.eval()

                init_feature = None
                for i in range(T_max):
                    max_optimizer.zero_grad()
                    logit_s, f_s, feature_s, code_s = model(data_s)
                    _, cls_pred = logit_s.max(dim=1)
                    last_features = feature_s
                    if i == 0:
                        init_feature = last_features.clone().detach()

                    class_loss = class_criterion(logit_s, target_s.argmax(1))
                    feature_loss = semantic_distance_criterion(last_features, init_feature)
                    adv_loss = 1*feature_loss - class_loss

                    adv_loss.backward()
                    max_optimizer.step()

                    del class_loss, feature_loss, last_features
                dataset_adv = AdvDataset(data_s.to('cpu').detach(), target_s.to('cpu').detach())
                datasets = train_s_dataloader.dataset
                dataset = ConcatDataset([datasets,dataset_adv])
                train_s_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, \
                    num_workers=4, pin_memory=True, drop_last=True)
                logger.info("%d-th Max-phase ended, Adv loss: %g" % (k, -1 * adv_loss))
                break
            
            optimizer.zero_grad()
            logit_s, f_s, feature_s, code_s = model(data_s)
            loss = criterion(logit_s, target_s.argmax(1))
            loss.backward()
            optimizer.step()
            current_iter += 1

    aug_num = (len(train_s_dataloader) - T_max)*batch_size
    logger.info('[Augmentation Sample Num:{}]'.format(aug_num))
    model.train()
    for epoch in range(max_iter):
        for batch_idx,((data_s, _, target_s, index),(data_s_ada, _, target_s_ada, index))\
             in enumerate(zip(train_s_dataloader, train_s_dataloader)):
           
            data_s = data_s.to(device)
            target_s = target_s.to(device)
            optimizer.zero_grad()
            logit_s, f_s, feature_s, code_s = model(data_s)
            loss = criterion(logit_s, target_s.argmax(1))

            data_s_ada = data_s_ada.to(device)
            target_s_ada = target_s_ada.to(device)
            logit_s_ada, _, _, _ = model(data_s_ada)
            min_ada, _ = torch.min(logit_s_ada,1)
            min_ada = min_ada.repeat(65,1).t().to(device)
            max_ada, _ = torch.max(logit_s_ada,1)
            max_ada = max_ada.repeat(65,1).t().to(device)
            out_norm_ada = (logit_s_ada - min_ada)/(max_ada - min_ada)
            zeros_ada = torch.zeros_like(out_norm_ada).to(device)
            out_norm_inac_ada = torch.where(out_norm_ada<0.001,out_norm_ada,zeros_ada).to(device)
            loss_cov_ada = -0.1 * out_norm_inac_ada.sum(1).mean()

            min, _ = torch.min(logit_s,1)
            min = min.repeat(65,1).t().to(device)
            max, _ = torch.max(logit_s,1)
            max = max.repeat(65,1).t().to(device)
            out_norm = (logit_s - min)/(max - min)
            
            zeros = torch.zeros_like(out_norm).to(device)
            out_norm_inac = torch.where(out_norm<0.001,out_norm,zeros).to(device)
            loss_cov = -0.1 * out_norm_inac.sum(1).mean()
            
            weights = list(OrderedDict(model.named_parameters()).values())
            weights = [i.requires_grad_() for i in weights]
            loss_sim = torch.norm(torch.autograd.grad(loss_cov,weights, create_graph=True, allow_unused=True)[0] - \
                torch.autograd.grad(loss_cov_ada,weights, create_graph=True, allow_unused=True)[0],p=2)
            
            loss += 0.1*loss_sim
            loss.backward(retain_graph=True)
            noiseLevel = 0.2
            num_gradual = 10
            clip_narry = np.linspace(1-noiseLevel, 1, num=num_gradual)
            clip_narry = clip_narry[::-1]
            if epoch < num_gradual:
                clip = clip_narry[epoch]
            clip = (1 - noiseLevel)
            to_concat_g = []
            to_concat_v = []
            for name, param in model.named_parameters():
                if param.dim() in [2, 4]:
                    try:
                        to_concat_g.append(param.grad.data.view(-1))
                        to_concat_v.append(param.data.view(-1))
                    except:
                        continue
            all_g = torch.cat(to_concat_g).cpu().detach()
            all_v = torch.cat(to_concat_v).cpu().detach()
            metric = torch.abs(all_g * all_v)
            num_params = all_v.size(0)
            nz = int(clip * num_params)
            top_values, _ = torch.topk(metric, nz)
            thresh = top_values[-1].to(device)

            for name, param in model.named_parameters():
                if param.dim() in [2, 4]:
                    try:
                        mask = (torch.abs(param.data * param.grad.data) >= thresh).type(torch.cuda.FloatTensor)
                        mask = mask * clip
                        param.grad.data = mask * param.grad.data
                    except:
                        continue
            optimizer.step() 
            optimizer.zero_grad()
        
        logger.info('[Epoch:{}/{}][loss:{:.4f}]'.format(epoch+1, max_iter, loss.item()))

        # Evaluate
        if (epoch % evaluate_interval == evaluate_interval-1):
            mAP = evaluate(model,
                            query_dataloader,
                            retrieval_dataloader,
                            code_length,
                            device,
                            topk,
                            save = True,
                            )
            logger.info('[iter:{}/{}][map:{:.4f}]'.format(
                epoch+1,
                max_iter,
                mAP,
            ))

    # Evaluate and save 
    mAP = evaluate(model,
                   query_dataloader,
                   retrieval_dataloader,
                   code_length,
                   device,
                   topk,
                   save=False,
                   )
    # torch.save({'iteration': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #         }, os.path.join('checkpoints', 'resume_{}.t'.format(code_length)))
    logger.info('Training finish, [iteration:{}][map:{:.4f}]'.format(epoch+1, mAP))


def evaluate(model, query_dataloader, retrieval_dataloader, code_length, device, topk, save):
    model.eval()

    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
    
    # One-hot encode targets

    onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
    onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)
   
    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )

    if save:
        np.save("code/query_code_{}_mAP_{}".format(code_length, mAP), query_code.cpu().detach().numpy())
        np.save("code/retrieval_code_{}_mAP_{}".format(code_length, mAP), retrieval_code.cpu().detach().numpy())
        np.save("code/query_target_{}_mAP_{}".format(code_length, mAP), onehot_query_targets.cpu().detach().numpy())
        np.save("code/retrieval_target_{}_mAP_{}".format(code_length, mAP), onehot_retrieval_targets.cpu().detach().numpy())
    
    model.train()

    return mAP


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, _,index in dataloader:
            data = data.to(device)
            _,_,_,outputs= model(data)
            code[index, :] = outputs.sign().cpu()

    return code

def print_image(data, name):
    from PIL import Image
    im = Image.fromarray(data)
    im.save('fig/topk/{}.png'.format(name))



class BaseClassificationLoss(nn.Module):
    def __init__(self):
        super(BaseClassificationLoss, self).__init__()
        self.losses = {}

    def forward(self, logits, code_logits, labels, onehot=True):
        raise NotImplementedError

def get_imbalance_mask(sigmoid_logits, labels, nclass, threshold=0.7, imbalance_scale=-1):
    if imbalance_scale == -1:
        imbalance_scale = 1 / nclass

    mask = torch.ones_like(sigmoid_logits) * imbalance_scale

    # wan to activate the output
    mask[labels == 1] = 1

    # if predicted wrong, and not the same as labels, minimize it
    correct = (sigmoid_logits >= threshold) == (labels == 1)
    mask[~correct] = 1

    multiclass_acc = correct.float().mean()

    # the rest maintain "imbalance_scale"
    return mask, multiclass_acc

class OrthoHashLoss(BaseClassificationLoss):
    def __init__(self,
                 ce=1,
                 s=8,
                 m=0.2,
                 m_type='cos',  # cos/arc
                 multiclass=False,
                 quan=0,
                 quan_type='cs',
                 multiclass_loss='label_smoothing',
                 **kwargs):
        super(OrthoHashLoss, self).__init__()
        self.ce = ce
        self.s = s
        self.m = m
        self.m_type = m_type
        self.multiclass = multiclass

        self.quan = quan
        self.quan_type = quan_type
        self.multiclass_loss = multiclass_loss
        assert multiclass_loss in ['bce', 'imbalance', 'label_smoothing']

    def compute_margin_logits(self, logits, labels):
        if self.m_type == 'cos':
            if self.multiclass:
                y_onehot = labels * self.m
                margin_logits = self.s * (logits - y_onehot)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                margin_logits = self.s * (logits - y_onehot)
        else:
            if self.multiclass:
                y_onehot = labels * self.m
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits

        return margin_logits

    def forward(self, logits, code_logits, labels, onehot=True):
        if self.multiclass:
            if not onehot:
                labels = F.one_hot(labels, logits.size(1))
            labels = labels.float()

            margin_logits = self.compute_margin_logits(logits, labels)

            if self.multiclass_loss in ['bce', 'imbalance']:
                loss_ce = F.binary_cross_entropy_with_logits(margin_logits, labels, reduction='none')
                if self.multiclass_loss == 'imbalance':
                    imbalance_mask, multiclass_acc = get_imbalance_mask(torch.sigmoid(margin_logits), labels,
                                                                        labels.size(1))
                    loss_ce = loss_ce * imbalance_mask
                    loss_ce = loss_ce.sum() / (imbalance_mask.sum() + 1e-7)
                    self.losses['multiclass_acc'] = multiclass_acc
                else:
                    loss_ce = loss_ce.mean()
            elif self.multiclass_loss in ['label_smoothing']:
                log_logits = F.log_softmax(margin_logits, dim=1)
                labels_scaled = labels / labels.sum(dim=1, keepdim=True)
                loss_ce = - (labels_scaled * log_logits).sum(dim=1)
                loss_ce = loss_ce.mean()
            else:
                raise NotImplementedError(f'unknown method: {self.multiclass_loss}')
        else:
            if onehot:
                labels = labels.argmax(1)
            margin_logits = self.compute_margin_logits(logits, labels)
            loss_ce = F.cross_entropy(margin_logits, labels)
            loss_ce_batch = F.cross_entropy(margin_logits, labels, reduction='none')


        if self.quan != 0:
            if self.quan_type == 'cs':
                quantization = (1. - F.cosine_similarity(code_logits, code_logits.detach().sign(), dim=1))
            elif self.quan_type == 'l1':
                quantization = torch.abs(code_logits - code_logits.detach().sign())
            else:  # l2
                quantization = torch.pow(code_logits - code_logits.detach().sign(), 2)
            quantization_batch = quantization
            quantization = quantization.mean()
        else:
            quantization_batch = torch.zeros_like(loss_ce_batch)
            quantization = torch.tensor(0.).to(logits.device)

        self.losses['ce'] = loss_ce
        self.losses['quan'] = quantization
        loss = self.ce * loss_ce + self.quan * quantization
        loss_batch = self.ce * loss_ce_batch + self.quan * quantization_batch
        return loss, loss_batch








    


    
