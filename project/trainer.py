import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from project.logging import Statistics
from project.logging import Logger
from project.datamodule import BaseDataSets,RandomGenerator,TwoStreamBatchSampler, patients_to_slices
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from project.models.model import MultiLayerPerceptron
import torch.backends.cudnn as cudnn

class Trainer:
    def __init__(self, cfg, experiment_path ,resuming_run: bool):
        #Setup torch seed and device
        self.cfg = cfg
        self.start_epoch = 0
        self.resuming_run = resuming_run

        # if not cfg.deterministic:
        #     cudnn.benchmark = True
        #     cudnn.deterministic = False
        # else:
        #     cudnn.benchmark = False
        #     cudnn.deterministic = True

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.mps.manual_seed(cfg.seed)

        self.device = torch.device(decide_device()) # toto urpc nerobí
        print("Setting up device:",self.device)

        # Create model
        self.model: nn.Module = create_model(self.cfg)
        self.model: nn.Module = self.model.to(self.device)  # ani toto nerobí URPC

    def setup(self, log: Logger):
        self.log = log
        self.datamodule.setup(self.cfg)


    def fit(self, experiment_path: Path):
        db_train = BaseDataSets(base_dir="./project/ACDC", split="train", num=None, transform=transforms.Compose([
            RandomGenerator([256, 256])
        ]))

        db_val = BaseDataSets(base_dir="./project/ACDC", split="val")

        total_slices = len(db_train)
        labeled_slice = patients_to_slices("ACDC", self.cfg.labeled_num)
        labeled_idxs = list(range(0, labeled_slice))
        unlabeled_idxs = list(range(labeled_slice, total_slices))

        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, self.cfg.batch_size, self.cfg.batch_size-self.cfg.labeled_bs)

        def worker_init_fn(worker_id):
            random.seed(self.cfg.seed + worker_id)

        trainloader = DataLoader(db_train, batch_sampler=batch_sampler,num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

        self.model.train()
        valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=1)



        # if self.resuming_run:
        #     self._load_checkpoint(experiment_path)

        # Implement training loop
        self.log.on_training_start()
        for epoch in range(self.start_epoch,self.cfg.max_epochs):

            # Training phase
            train_stats = Statistics()
            self._train_epoch(
                epoch, 
                model=self.model, 
                dataloader=self.datamodule.dataloader_train, 
                train_stats=train_stats
            )

            # Validation phase
            valid_stats = Statistics()
            self._validate_epoch(
                epoch, 
                model=self.model, 
                dataloader=self.datamodule.dataloader_val, 
                valid_stats=valid_stats
            )

            #Log epoch metrics
            self.log.on_epoch_complete(
                epoch=epoch,
                stats=Statistics.merge(train_stats,valid_stats)
            )

            # Save latest models and create checkpoints
            self._save_checkpoint(epoch,experiment_path,file_name="latest.pt")
            if epoch % self.cfg.checkpoint_freq == 0:
                self._save_checkpoint(epoch,experiment_path, file_name=f"checkpoint-{epoch:04d}.pt")

        self.log.on_training_stop()

    def _train_epoch(self, epoch: int, model, dataloader, train_stats: Statistics):
        with tqdm(dataloader, desc=f"Train: {epoch}") as progress:
            for x,y in progress:

                # Move data to GPU
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass
                y_hat_logits = model(x)
                loss = self.loss_fun(y_hat_logits, y)

                # Backward pass & Update params
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # Update statistics
                train_stats.train_step(loss)
                progress.set_postfix(train_stats.get_averages())
       
       
    def _validate_epoch(self, epoch: int, model, dataloader, valid_stats: Statistics):        
        with torch.no_grad():       # We don't need gradients in validation
            with tqdm(dataloader, desc=f"Val: {epoch}") as progress:
                for x,y in progress:
                    # Move data to GPU
                    x = x.to(self.device)
                    y = y.to(self.device)

                    # Forward pass
                    y_hat_logits = model(x)
                    loss = self.loss_fun(y_hat_logits, y)

                    # Update statistics
                    valid_stats.valid_step(loss)
                    progress.set_postfix(valid_stats.get_averages())
    
    def _save_checkpoint(self,epoch: int, experiment_path: Path, file_name: str):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "epoch": epoch
        }

        #Create checkpoint foelder if does not exist
        checkpoint_path = experiment_path / "checkpoints" / file_name
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving checkpoint: {checkpoint_path.as_posix()}")
        torch.save(checkpoint, checkpoint_path.as_posix())
    
    def _load_checkpoint(self,experiment_path):
        checkpoint_path = experiment_path / "checkpoints" / "latest.pt"
        print(f" > Loading checkpoint: {checkpoint_path.as_posix()}")
        checkpoint = torch.load(
            checkpoint_path.as_posix(),
            map_location=torch.device("mps")
        )
        self.model.load_state_dict(checkpoint["model"])
        self.opt.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint["epoch"] + 1

def decide_device():
    if (torch.cuda.is_available()): 
        return "cuda"
    if (torch.backends.mps.is_available()):
        return "mps"
    return "cpu"

def create_model(cfg):
    # Create model
    model = MultiLayerPerceptron(
        nin = 28*28,                # Image size is 28x28
        nhidden = cfg.num_hidden,   # Larger hidden layer
        nout=10                     # 10 possible classes
    )
    return model

