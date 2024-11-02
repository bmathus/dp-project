import torch
import torch.nn as nn
import os
import numpy as np
import random
from tqdm import tqdm
from project.logging import Logger
from project.datamodule import BaseDataSets,RandomGenerator,TwoStreamBatchSampler, patients_to_slices
from project.utils import worker_init_fn,decide_device,sharpening,get_current_consistency_weight
from project.metrics import DiceLoss,mse_loss,test_single_volume_ds
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from pathlib import Path
from project.models import unet_mtnet,unet_hybrid
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import CrossEntropyLoss
from neptune import Run

class Trainer:
    def __init__(self, cfg, experiment_path ,resuming_run: bool):
        #Setup torch seed and device
        self.cfg = cfg
        self.start_epoch = 0
        self.resuming_run = resuming_run

        print("Is cudann available:",cudnn.is_available())
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.mps.manual_seed(cfg.seed)
        # os.environ['CUDA_VISIBLE_DEVICES'] = 0

        self.device = torch.device(decide_device()) # toto urpc nerobí
        print("| Setting up device:",self.device)

        # Create model
        # self.model: nn.Module = unet_urpc.UNet_URPC(in_chns=1,class_num=cfg.num_classes)
        # self.model: nn.Module = unet_mtnet.MCNet2d_v1(in_chns=1,class_num=cfg.num_classes)
        self.model: nn.Module = unet_hybrid.MSDNet(in_chns=1,class_num=cfg.num_classes)
        self.model = self.model.to(self.device)  # ani toto nerobí URPC dáva model.cuda()

    def setup(self, log: Logger):
        self.log = log

    def setup_dataloaders(self,cfg):
        print("| Setting up dataloaders:")
        db_train = BaseDataSets(base_dir=cfg.data_path, split="train", num=None, transform=transforms.Compose([
            RandomGenerator([cfg.patch_size, cfg.patch_size])
        ]))
        db_val = BaseDataSets(base_dir=cfg.data_path, split="val")

        print("| Total train samples:",len(db_train))
        print("| Total val samples:",len(db_val))
 
        labeled_slice = patients_to_slices("ACDC", cfg.labeled_num)

        labeled_idxs = list(range(0, labeled_slice))
        unlabeled_idxs = list(range(labeled_slice, len(db_train)))

        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, cfg.batch_size, cfg.batch_size-cfg.labeled_bs)

        trainloader = DataLoader(db_train, batch_sampler=batch_sampler,num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
        valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=1)

        return trainloader,valloader,len(db_train),len(db_val)

    def fit_mtnet(self,experiment_path: Path,run: Run):
        base_lr = self.cfg.base_lr
        cfg = self.cfg

        trainloader,valloader,total_samples,total_val_samples = self.setup_dataloaders(cfg)

        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0001)
        ce_loss = CrossEntropyLoss()
        consistency_criterion = mse_loss
        dice_loss = DiceLoss(cfg.num_classes)

        self.log.on_training_start()
        max_epoch = cfg.max_iter // len(trainloader) + 1
        iter_num = 0
        best_performance = 0.0
        iterator = tqdm(range(max_epoch), desc="| Training:")
        for _ in iterator:
            for _, sampled_batch in enumerate(trainloader):
                # Data to device
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.to(self.device), label_batch.to(self.device)

                outputs = self.model(volume_batch)

                loss_seg_dice,loss_seg,loss_consist = self.msd_loss(
                    outputs=outputs,
                    label_batch=label_batch,
                    ce_loss=ce_loss,
                    dice_loss=dice_loss,
                    consistency_criterion=consistency_criterion,
                    cfg=cfg
                )

                # loss_seg_dice,loss_seg,loss_consist = self.mtnet_loss(
                #     outputs=outputs,
                #     label_batch=label_batch,
                #     ce_loss=ce_loss,
                #     dice_loss=dice_loss,
                #     consistency_criterion=consistency_criterion,
                #     cfg=cfg
                # )

                # if i == 0:
                #     return
                
                # consistency_weight = get_current_consistency_weight(cfg,iter_num//150)

                loss = cfg.lamda * loss_seg_dice + 0.1 * loss_consist

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_num = iter_num + 1

                #Logging
                run["train/lr"].append(base_lr,step=iter_num)
                run["train/loss"].append(loss,step=iter_num)
                run["train/supervised_loss"].append(loss_seg_dice,step=iter_num)
                # run["train/consistency_weight"].append(consistency_weight,step=iter_num)    
                run["train/consistency_loss"].append(loss_consist,step=iter_num)
                iterator.set_postfix({"iter_num":iter_num,"loss":loss.item(),"loss_sup":loss_seg_dice.item(),"loss_consist":loss_consist.item()})

                # Validation
                if iter_num > 0 and iter_num % 200 == 0:
                    print(f" > Validating at iter: {iter_num}")

                    self.model.eval() #switch model to validation
                    performance = self.validation(valloader,total_val_samples,iter_num,cfg,run)

                    if performance > best_performance: #Save best bodel
                        best_performance = performance
                        best_model_path = os.path.join(experiment_path,'best_model.pth')
                        torch.save(self.model.state_dict(),best_model_path)
                        print(f" > Saving best model iter:{iter_num}, dice:{round(best_performance, 4)}")
                        run["val/best_model_dice"].append(best_performance,step=iter_num)
                    
                    self.model.train()  #switch to training
                
                # Save frequently
                if iter_num % 3000 == 0:
                    save_path = os.path.join(experiment_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save(self.model.state_dict(), save_path)
                    print(f" > Saving model to:{save_path}")
                
                if iter_num >= cfg.max_iter:
                    break

            if iter_num >= cfg.max_iter:
                iterator.close()
                break

        self.log.on_training_stop()


    def fit_urpc(self, experiment_path: Path,run: Run):
        base_lr = self.cfg.base_lr
        cfg = self.cfg

        trainloader,valloader,total_samples,total_val_samples = self.setup_dataloaders(cfg)

        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0001)
        ce_loss = CrossEntropyLoss()
        dice_loss = DiceLoss(cfg.num_classes)
        kl_distance = nn.KLDivLoss(reduction='none')

        self.log.on_training_start()
        max_epoch = cfg.max_iter // len(trainloader) + 1
        iter_num = 0
        best_performance = 0.0
        iterator = tqdm(range(max_epoch), desc="| Training:")
        for _ in iterator:
            for _, sampled_batch in enumerate(trainloader):

                # Data to device
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.to(self.device), label_batch.to(self.device)
                
                # Model inference
                outputs, outputs_aux1, outputs_aux2, outputs_aux3 = self.model(volume_batch)

                # Calculate loss terms
                supervised_loss,loss_dice,loss_ce,consistency_loss = self.urpc_loss(
                    cfg,
                    (outputs, outputs_aux1, outputs_aux2, outputs_aux3),
                    label_batch,
                    dice_loss,
                    ce_loss,
                    kl_distance
                )
            
                consistency_weight = get_current_consistency_weight(cfg,iter_num//150)
                
                # Overall loss
                loss = supervised_loss + consistency_weight * (0.5 * consistency_loss)
                # loss = supervised_loss + consistency_weight * consistency_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_ = base_lr * (1.0 - iter_num / cfg.max_iter) ** 0.9
                for param_group in optimizer.param_groups:
                     param_group['lr'] = lr_

                iter_num = iter_num + 1

                # Logging
                run["train/lr"].append(lr_,step=iter_num)
                run["train/loss"].append(loss,step=iter_num)
                run["train/supervised_loss"].append(supervised_loss,step=iter_num)
                run["train/loss_ce"].append(loss_ce,step=iter_num)
                run["train/loss_dice"].append(loss_dice,step=iter_num)
                run["train/consistency_loss"].append(consistency_loss,step=iter_num)  
                # run["train/uncertainty_min"].append(uncertainty_min,step=iter_num)
                run["train/consistency_weight"].append(consistency_weight,step=iter_num)    
                iterator.set_postfix({"iter_num":iter_num,"loss":loss.item(),"loss_dice":loss_dice.item()})
                
                # Validation
                if iter_num > 0 and iter_num % 200 == 0:
                    print(f" > Validating at iter: {iter_num}")

                    self.model.eval() #switch model to validation
                    performance = self.validation(valloader,total_val_samples,iter_num,cfg,run)

                    if performance > best_performance: #Save best bodel
                        best_performance = performance
                        best_model_path = os.path.join(experiment_path,'best_model.pth')
                        torch.save(self.model.state_dict(),best_model_path)
                        print(f" > Saving best model iter:{iter_num}, dice:{round(best_performance, 4)}")
                        run["val/best_model_dice"].append(best_performance,step=iter_num)
                    
                    self.model.train()  #switch to training
                
                # Save frequently
                if iter_num % 3000 == 0:
                    save_path = os.path.join(experiment_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save(self.model.state_dict(), save_path)
                    print(f" > Saving model to:{save_path}")
          
                    
                if iter_num >= cfg.max_iter:
                    break

            if iter_num >= cfg.max_iter:
                iterator.close()
                break
                
        self.log.on_training_stop()

    def validation(self,valloader,total_val_samples,iter_num, cfg, run: Run):
        metric_list = 0.0
        for _, sampled_batch in enumerate(valloader):
            metric_i = test_single_volume_ds(sampled_batch["image"], sampled_batch["label"], self.model,self.device,cfg)
            metric_list += np.array(metric_i)
        metric_list = metric_list / total_val_samples

        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        run["val/mean_dice"].append(performance,step=iter_num)
        run["val/mean_hd95"].append(mean_hd95,step=iter_num)
        return performance

    def urpc_loss(self,cfg, multi_scale_outputs, label_batch, dice_loss, ce_loss, kl_distance):
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

    def mtnet_loss(self,outputs,label_batch,ce_loss,dice_loss,consistency_criterion,cfg):
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
        loss_seg += ce_loss(y_d1, label_batch[:cfg.labeled_bs][:].long())
        loss_seg_dice += dice_loss(y_prob, label_batch[:cfg.labeled_bs].unsqueeze(1))

        y_d2 = outputs[1][:cfg.labeled_bs,...] # torch.Size([12, 4, 256, 256]) to iste ako outputs[idx][:cfg.labeled_bs]
        y_prob = F.softmax(y_d2, dim=1)
        loss_seg += ce_loss(y_d2, label_batch[:cfg.labeled_bs][:].long())
        loss_seg_dice += dice_loss(y_prob, label_batch[:cfg.labeled_bs].unsqueeze(1))


        y_prob_all_d1 = F.softmax(outputs[0], dim=1)
        y_pseudo_label_d1 = sharpening(y_prob_all_d1,cfg)

        y_prob_all_d2 = F.softmax(outputs[1], dim=1)
        y_pseudo_label_d2 = sharpening(y_prob_all_d2,cfg)
        
        loss_consist = 0
        loss_consist += (consistency_criterion(y_prob_all_d1, y_pseudo_label_d2) + consistency_criterion(y_prob_all_d2, y_pseudo_label_d1))

        # for i in range(num_outputs):
        #     for j in range(num_outputs):
        #         if i != j:
        #             #print(f"i: {i} j: {j}")
        #             loss_consist += consistency_criterion(y_ori[i], y_pseudo_label[j])
        
        return loss_seg_dice,loss_seg,loss_consist
    
    def msd_loss(self,outputs,label_batch,ce_loss,dice_loss,consistency_criterion,cfg):
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

        loss_consist = 0
        for scale_num in range(4):
            outscale_d1_soft = F.softmax(outputs_d1[scale_num], dim=1)
            outscale_d2_soft = F.softmax(outputs_d2[scale_num], dim=1)

            outscale_d1_pseudo = sharpening(outscale_d1_soft,cfg)
            outscale_d2_pseudo = sharpening(outscale_d2_soft,cfg)

            loss_consist += consistency_criterion(outscale_d1_soft,outscale_d2_pseudo) + consistency_criterion(outscale_d2_soft,outscale_d1_pseudo)

        loss_consist = loss_consist/4
            

        return loss_seg_dice,loss_seg_ce,loss_consist


        # pseudolabel sharpening











        

        pass


