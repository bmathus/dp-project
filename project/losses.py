import torch
import torch.nn as nn
import torch.nn.functional as F
from run.config import Config
from torch.nn.modules.loss import CrossEntropyLoss
from project.utils import sharpening
from torch.autograd import Variable
import numpy as np

def entropy_loss(p,device, C=4):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).to(device)
    ent = torch.mean(y1)

    return ent

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def urpc_loss(cfg: Config, multi_scale_outputs, label_batch, dice_loss: DiceLoss, ce_loss: CrossEntropyLoss, kl_distance):
    outputs, outputs_aux1, outputs_aux2, outputs_aux3 = multi_scale_outputs

    #Outputs softmax
    outputs_soft = torch.softmax(outputs, dim=1)
    outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
    outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
    outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)

    #SUP CE loss (outputs <-> labels)
    loss_ce = ce_loss(outputs[:cfg.labeled_bs],label_batch[:cfg.labeled_bs][:].long()) # podla chatu možem dať preč [:] lebo .long() ajtak robi kopiu
    loss_ce_aux1 = ce_loss(outputs_aux1[:cfg.labeled_bs],label_batch[:cfg.labeled_bs][:].long())
    loss_ce_aux2 = ce_loss(outputs_aux2[:cfg.labeled_bs],label_batch[:cfg.labeled_bs][:].long())
    loss_ce_aux3 = ce_loss(outputs_aux3[:cfg.labeled_bs],label_batch[:cfg.labeled_bs][:].long())

    #SUP Dice loss (outputs_soft <-> labels)
    loss_dice = dice_loss(outputs_soft[:cfg.labeled_bs], label_batch[:cfg.labeled_bs].unsqueeze(1))
    loss_dice_aux1 = dice_loss(outputs_aux1_soft[:cfg.labeled_bs], label_batch[:cfg.labeled_bs].unsqueeze(1))
    loss_dice_aux2 = dice_loss(outputs_aux2_soft[:cfg.labeled_bs], label_batch[:cfg.labeled_bs].unsqueeze(1))
    loss_dice_aux3 = dice_loss(outputs_aux3_soft[:cfg.labeled_bs], label_batch[:cfg.labeled_bs].unsqueeze(1))

    #SUP loss
    supervised_loss = (loss_ce+loss_ce_aux1+loss_ce_aux2+loss_ce_aux3 +loss_dice+loss_dice_aux1+loss_dice_aux2+loss_dice_aux3)/8

    preds = (outputs_soft+outputs_aux1_soft +outputs_aux2_soft+outputs_aux3_soft)/4 #priemer pravdep predickie všetkých škal (labeled aj unlabeled)
    
    #weight = exp(KL (outputs_soft <-> preds_soft_avg))
    variance_main = torch.sum(kl_distance(torch.log(outputs_soft[cfg.labeled_bs:]), preds[cfg.labeled_bs:]), dim=1, keepdim=True)
    exp_variance_main = torch.exp(-variance_main)
    variance_aux1 = torch.sum(kl_distance(torch.log(outputs_aux1_soft[cfg.labeled_bs:]), preds[cfg.labeled_bs:]), dim=1, keepdim=True)
    exp_variance_aux1 = torch.exp(-variance_aux1)
    variance_aux2 = torch.sum(kl_distance(torch.log(outputs_aux2_soft[cfg.labeled_bs:]), preds[cfg.labeled_bs:]), dim=1, keepdim=True)
    exp_variance_aux2 = torch.exp(-variance_aux2)
    variance_aux3 = torch.sum(kl_distance(torch.log(outputs_aux3_soft[cfg.labeled_bs:]), preds[cfg.labeled_bs:]), dim=1, keepdim=True)
    exp_variance_aux3 = torch.exp(-variance_aux3)

    consistency_dist_main = (preds[cfg.labeled_bs:] - outputs_soft[cfg.labeled_bs:]) ** 2
    consistency_loss_main = torch.mean(consistency_dist_main * exp_variance_main) / (torch.mean(exp_variance_main) + 1e-8) 

    consistency_dist_aux1 = (preds[cfg.labeled_bs:] - outputs_aux1_soft[cfg.labeled_bs:]) ** 2
    consistency_loss_aux1 = torch.mean(consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8)

    consistency_dist_aux2 = (preds[cfg.labeled_bs:] - outputs_aux2_soft[cfg.labeled_bs:]) ** 2
    consistency_loss_aux2 = torch.mean(consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) 

    consistency_dist_aux3 = (preds[cfg.labeled_bs:] - outputs_aux3_soft[cfg.labeled_bs:]) ** 2
    consistency_loss_aux3 = torch.mean(consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) 

    consistency_loss = (consistency_loss_main + consistency_loss_aux1 + consistency_loss_aux2 + consistency_loss_aux3) / 4

    # uncertainty_min = (torch.mean(variance_main) + torch.mean(variance_aux1) + torch.mean(variance_aux2) + torch.mean(variance_aux3)) / 4

    return supervised_loss,loss_dice,loss_ce,consistency_loss,#uncertainty_min


def CPCR_loss_kd(outputs, label_batch,ce_loss,dice_loss: DiceLoss, consistency_criterion, cfg: Config, device: torch.device):
    outputs_d1,outputs_d2 = outputs

    loss_sup = supervised_loss(outputs_d1,outputs_d2, label_batch, ce_loss, dice_loss, cfg)

    loss_sup_deep = deep_supervised_loss(outputs_d2, label_batch, ce_loss, dice_loss, cfg)

    #print("outputs_d1[0].permute(0, 2, 3, 1).reshape(-1, 2)",outputs_d1[0].permute(0, 2, 3, 1).reshape(-1, 2).shape)
    #outputs_d1[0] torch.Size([24, 4, 256, 256])
    #outputs_d1[0].permute(0, 2, 3, 1) torch.Size([24, 256, 256, 4])
    #outputs_d1[0].permute(0, 2, 3, 1).reshape(-1, 2)

    #Uncertainty min
    outputs_avg_main_soft = (F.softmax(outputs_d1[0], dim=1) + F.softmax(outputs_d2[0], dim=1)) / 2
    en_loss = entropy_loss(outputs_avg_main_soft,device,C=4)
    
    #Unsup
    loss_consist_main = 0
    loss_consist_main += consistency_criterion(outputs_d1[0].permute(0, 2, 3, 1).reshape(-1, 4),outputs_d2[0].detach().permute(0, 2, 3, 1).reshape(-1, 4))
    loss_consist_main += consistency_criterion(outputs_d2[0].permute(0, 2, 3, 1).reshape(-1, 4),outputs_d1[0].detach().permute(0, 2, 3, 1).reshape(-1, 4))

    loss_consist_aux = 0
    for scale_num in range(1,4):
        loss_consist_aux += consistency_criterion(outputs_d1[scale_num].permute(0, 2, 3, 1).reshape(-1, 4),outputs_d2[scale_num].detach().permute(0, 2, 3, 1).reshape(-1, 4))
        loss_consist_aux += consistency_criterion(outputs_d2[scale_num].permute(0, 2, 3, 1).reshape(-1, 4),outputs_d1[scale_num].detach().permute(0, 2, 3, 1).reshape(-1, 4))

    loss_consist_aux = loss_consist_aux/3

    return loss_sup,loss_sup_deep, loss_consist_main, loss_consist_aux, en_loss


def supervised_loss(outputs_d1,outputs_d2, label_batch, ce_loss, dice_loss: DiceLoss, cfg: Config):
    loss_seg_dice = 0
    loss_seg_ce = 0

    # Sup loss decoder 1
    output_d1_main = outputs_d1[0][:cfg.labeled_bs]
    loss_seg_dice += dice_loss(F.softmax(output_d1_main, dim=1),label_batch[:cfg.labeled_bs].unsqueeze(1))
    loss_seg_ce += ce_loss(output_d1_main,label_batch[:cfg.labeled_bs][:].long())

    # Sup loss decoder 2
    output_d2_main = outputs_d2[0][:cfg.labeled_bs]
    loss_seg_dice += dice_loss(F.softmax(output_d2_main, dim=1),label_batch[:cfg.labeled_bs].unsqueeze(1))
    loss_seg_ce += ce_loss(output_d2_main,label_batch[:cfg.labeled_bs][:].long())

    return loss_seg_dice + loss_seg_ce

def deep_supervised_loss(outputs_d, label_batch, ce_loss, dice_loss: DiceLoss, cfg: Config):
    weights = [0.7,0.4,0.1]

    loss_sup_deep = 0
    for scale_num in range(0,3):
        output_scale = outputs_d[scale_num + 1][:cfg.labeled_bs]
        loss_seg_dice = dice_loss(F.softmax(output_scale, dim=1),label_batch[:cfg.labeled_bs].unsqueeze(1))
        loss_seg_ce = ce_loss(output_scale,label_batch[:cfg.labeled_bs][:].long())
        loss_sup_deep += (loss_seg_dice + loss_seg_ce) * weights[scale_num]
    
    return loss_sup_deep


def mtnet_loss(outputs, label_batch, ce_loss: CrossEntropyLoss, dice_loss: DiceLoss, consistency_criterion, cfg: Config):
    # Dec 1 output: torch.Size([24, 4, 256, 256])
    # Dec 2 output: torch.Size([24, 4, 256, 256])
    # Label: torch.Size([24, 256, 256])

    # num_outputs = len(outputs)
    # y_ori = torch.zeros((num_outputs,) + outputs[0].shape) # torch.Size([2, 24, 4, 256, 256])
    # y_pseudo_label = torch.zeros((num_outputs,) + outputs[0].shape) # torch.Size([2, 24, 4, 256, 256])

    loss_seg = 0
    loss_seg_dice = 0 

    y_d1 = outputs[0][:cfg.labeled_bs,...] # torch.Size([12, 4, 256, 256]) to iste ako outputs[idx][:cfg.labeled_bs]
    y_prob = F.softmax(y_d1, dim=1)
    # loss_seg += ce_loss(y_d1, label_batch[:cfg.labeled_bs][:].long())
    loss_seg_dice += dice_loss(y_prob, label_batch[:cfg.labeled_bs].unsqueeze(1))

    y_d2 = outputs[1][:cfg.labeled_bs,...] # torch.Size([12, 4, 256, 256]) to iste ako outputs[idx][:cfg.labeled_bs]
    y_prob = F.softmax(y_d2, dim=1)
    # loss_seg += ce_loss(y_d2, label_batch[:cfg.labeled_bs][:].long())
    loss_seg_dice += dice_loss(y_prob, label_batch[:cfg.labeled_bs].unsqueeze(1))


    y_prob_all_d1 = F.softmax(outputs[0], dim=1)
    y_pseudo_label_d1 = sharpening(y_prob_all_d1,cfg)

    y_prob_all_d2 = F.softmax(outputs[1], dim=1)
    y_pseudo_label_d2 = sharpening(y_prob_all_d2,cfg)
    
    loss_consist = 0
    loss_consist += consistency_criterion(y_prob_all_d1, y_pseudo_label_d2) + consistency_criterion(y_prob_all_d2, y_pseudo_label_d1)

    # for i in range(num_outputs):
    #     for j in range(num_outputs):
    #         if i != j:
    #             #print(f"i: {i} j: {j}")
    #             loss_consist += consistency_criterion(y_ori[i], y_pseudo_label[j])
    
    return loss_seg_dice,loss_seg,loss_consist


def msd_loss(outputs, label_batch, ce_loss: CrossEntropyLoss, dice_loss: DiceLoss, consistency_criterion, cfg: Config):
    outputs_d1,outputs_d2 = outputs

    # Sup
    loss_seg_dice = 0
    loss_seg_ce = 0

    output_d1_main = outputs_d1[0][:cfg.labeled_bs]
    loss_seg_dice += dice_loss(F.softmax(output_d1_main, dim=1),label_batch[:cfg.labeled_bs].unsqueeze(1))
    # loss_seg_ce += ce_loss(output_d1_main,label_batch[:cfg.labeled_bs][:].long())

    output_d2_main = outputs_d2[0][:cfg.labeled_bs]
    loss_seg_dice += dice_loss(F.softmax(output_d2_main, dim=1),label_batch[:cfg.labeled_bs].unsqueeze(1))
    # loss_seg_ce += ce_loss(output_d2_main,label_batch[:cfg.labeled_bs][:].long())
    
    # Unsup
    loss_consist_main = 0
    outmain_d1_soft = F.softmax(outputs_d1[0], dim=1)
    outmain_d2_soft = F.softmax(outputs_d2[0], dim=1)

    outmain_d1_pseudo = sharpening(outmain_d1_soft,cfg)
    outmain_d2_pseudo = sharpening(outmain_d2_soft,cfg)
    loss_consist_main += consistency_criterion(outmain_d1_soft,outmain_d2_pseudo) + consistency_criterion(outmain_d2_soft,outmain_d1_pseudo)

    loss_consist_aux = 0
    for scale_num in range(1,4):
        outscale_d1_soft = F.softmax(outputs_d1[scale_num][cfg.labeled_bs:], dim=1)
        outscale_d2_soft = F.softmax(outputs_d2[scale_num][cfg.labeled_bs:], dim=1)
        loss_consist_aux += consistency_criterion(outscale_d1_soft,outscale_d2_soft) 

    loss_consist_aux = loss_consist_aux/3
        
    return loss_seg_dice,loss_seg_ce,loss_consist_main,loss_consist_aux