from pathlib import Path
from project.model import MultiLayerPerceptron
from project.datamodule import DataModule
from project.trainer import Trainer
from project.logging import Log
import yaml
import torch

BASE_PATH = Path(".data/experiments")

class Experiment:
    def __init__(self,cfg):
        self.cfg = cfg

        #Create experiment path and save config
        self.experiment_path = BASE_PATH / cfg.name / cfg.ver
        print(f" > Created experiment : {cfg.name}/{cfg.ver}")

        self.experiment_path.mkdir(parents=True, exist_ok=True)
        self._save_config(cfg, self.experiment_path)

        # Create model & datamodule
        self.model = self._create_model(cfg)
        self.datamodule = DataModule()

        #Create trainer
        self.trainer = Trainer(cfg,self.model)

    def _create_model(self, cfg):
        # Create model
        model = MultiLayerPerceptron(
            nin = 28*28,                # Image size is 28x28
            nhidden = cfg.num_hidden,   # Larger hidden layer
            nout=10                     # 10 possible classes
        )
        return model

    def train(self):
        self.trainer.setup(
            datamodule=self.datamodule,
            log=Log(self)
        )
        self.trainer.fit()

    def save_checkpoint(self, file_name):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.trainer.opt.state_dict()
        }

        #Create checkpoint foelder if does not exist
        checkpoint_path = self.experiment_path / "checkpoints" / file_name
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving checkpoint: {checkpoint_path.as_posix()}")
        torch.save(checkpoint, checkpoint_path.as_posix())
    
    def _save_config(self, cfg, exp_path):
        config_yaml = yaml.dump(vars(cfg))

        print(" > Training Configuration:")
        print("----------------------------")
        print(config_yaml.strip())
        print("----------------------------")

        # Save configuration into YAML
        (exp_path / "config.yaml").write_text(config_yaml)


