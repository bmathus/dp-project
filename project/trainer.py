import torch
import torch.nn as nn
from tqdm import tqdm
from project.logging import Statistics
from project.logging import Log
from project.datamodule import DataModule
from pathlib import Path


def decide_device():
    if (torch.cuda.is_available()): 
        return "cuda"
    if (torch.backends.mps.is_available()):
        return "mps"
    return "cpu"

class Trainer:
    def __init__(self, cfg, model):
        # Select GPU device
        self.device = torch.device(decide_device())
        print("Setting up device:",self.device)

        # Setup model and config
        self.cfg = cfg
        self.start_epoch = 0
        self.model: nn.Module = model.to(self.device)  # Move parameters to GPU

        # Create optimizer
        self.opt = torch.optim.SGD(
            params=self.model.parameters(),
            lr=cfg.learning_rate
        )

        # Create loss function
        self.loss_fun = nn.CrossEntropyLoss()

    def setup(self, datamodule: DataModule, log: Log):
        self.log = log
        self.datamodule = datamodule
        self.datamodule.setup(self.cfg)


    def fit(self, experiment_path: Path):
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