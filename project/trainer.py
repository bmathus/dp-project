import torch
import torch.nn as nn
import os
import numpy as np
import random
from tqdm import tqdm
from project.logging import Logger
from run.config import Config
from project.datamodule import BaseDataSets,RandomGenerator,TwoStreamBatchSampler, patients_to_slices
from project.utils import worker_init_fn,decide_device, get_current_consistency_weight
from project.metrics import mse_loss, test_single_volume_ds, KDLoss
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from pathlib import Path
from project.models import unet_dbpnet, unet_mcnet,unet_urpc
from project.losses import urpc_loss, DiceLoss, CPCR_loss_kd, FocalLoss
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import CrossEntropyLoss
from neptune import Run

class Trainer:
    def __init__(self, cfg: Config ,resuming_run: bool):
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
        self.network_factory()

    def setup(self, log: Logger):
        self.log = log

    def network_factory(self):
        if self.cfg.network == "urpc":
            self.model: nn.Module = unet_urpc.UNet_URPC(in_chns=1,class_num=self.cfg.num_classes)
        elif self.cfg.network == "mcnet":
            self.model: nn.Module = unet_mcnet.MCNet2d_v1(in_chns=1,class_num=self.cfg.num_classes)
        elif self.cfg.network == "dbpnet":
            self.model: nn.Module = unet_dbpnet.DBPNet(in_chns=1,class_num=self.cfg.num_classes)
        self.model = self.model.to(self.device)  # ani toto nerobí URPC dáva model.cuda()

    def setup_dataloaders(self,cfg: Config):
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

    def fit_consistency_learning(self,experiment_path: Path,run: Run):
        base_lr = self.cfg.base_lr
        cfg = self.cfg

        trainloader, valloader, total_samples, total_val_samples = self.setup_dataloaders(cfg)

        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0001)
        ce_loss = FocalLoss(gamma=1.5, alpha=None, size_average=True)
        consistency_criterion = KDLoss(T=10)
        dice_loss = DiceLoss(cfg.num_classes)

        self.log.on_training_start()
        max_epoch = cfg.max_iter // len(trainloader) + 1
        iter_num = 0
        best_performance = 0.0
        iterator = tqdm(range(max_epoch), desc="| Training:")
        for epoch in iterator:
            for _, sampled_batch in enumerate(trainloader):
                # Data to device
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.to(self.device), label_batch.to(self.device)

                outputs = self.model(volume_batch)

                loss_seg_dice,loss_seg_ce,loss_consist_main, loss_consist_aux, en_loss = CPCR_loss_kd(
                    outputs=outputs,
                    label_batch=label_batch,
                    ce_loss=ce_loss,
                    dice_loss=dice_loss,
                    consistency_criterion= consistency_criterion,
                    cfg=cfg,
                    device=self.device
                )

                # loss_seg_dice,loss_seg,loss_consist = mtnet_loss(
                #     outputs=outputs,
                #     label_batch=label_batch,
                #     ce_loss=ce_loss,
                #     dice_loss=dice_loss,
                #     consistency_criterion=consistency_criterion,
                #     cfg=cfg
                # )
                
                consistency_weight = get_current_consistency_weight(cfg,iter_num//150)

                if epoch < 90:
                    loss_consist_main = torch.tensor((0,)).to(self.device)
                    loss_consist_aux = torch.tensor((0,)).to(self.device)
                    en_loss = torch.tensor((0,)).to(self.device)

                loss = cfg.lamda * (loss_seg_dice + loss_seg_ce) + (0.1 * loss_consist_main) + (0.1 * en_loss) + (consistency_weight * loss_consist_aux)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_num = iter_num + 1

                #Logging
                run["train/lr"].append(base_lr,step=iter_num)
                run["train/loss"].append(loss,step=iter_num)
                run["train/dice_loss"].append(loss_seg_dice,step=iter_num)
                run["train/ce_loss"].append(loss_seg_ce,step=iter_num)
                run["train/consistency_weight"].append(consistency_weight,step=iter_num)    
                # run["train/consistency_loss"].append(loss_consist,step=iter_num)
                run["train/consistency_loss_main"].append(loss_consist_main,step=iter_num)
                run["train/consistency_loss_aux"].append(loss_consist_aux,step=iter_num)
                run["train/en_loss"].append(en_loss,step=iter_num)
                iterator.set_postfix({"iter_num":iter_num,"loss":loss.item(),"dice_loss":loss_seg_dice.item(),"ce_loss":loss_seg_ce.item(),"loss_consist":loss_consist_main.item()})

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
                supervised_loss,loss_dice,loss_ce,consistency_loss = urpc_loss(
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

    def validation(self,valloader,total_val_samples,iter_num, cfg: Config, run: Run):
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

    

