from pathlib import Path
from project.models.model import MultiLayerPerceptron
from project.datamodule import DataModule
from project.trainer import Trainer
from project.logging import Log
from argparse import Namespace
import torch
import yaml
import neptune
import uuid

BASE_PATH = Path(".data/experiments")

class Experiment:
    def __init__(self,cfg):
        self.cfg = cfg
        
        resuming_run = False
        if self.cfg.ver != "auto": 
            resuming_run = True

        # Resuming experiment
        if resuming_run: 
            self.experiment_path = BASE_PATH / self.cfg.name / self.cfg.ver
            self._load_experiment_config()
            print(f" > Resuming experiment : {self.cfg.name}/{self.cfg.ver}")
        # Creating new experiment
        else:
            self.cfg.ver = "run-"+str(uuid.uuid4())[:8]
            self.experiment_path = BASE_PATH / self.cfg.name / self.cfg.ver
            self.experiment_path.mkdir(parents=True, exist_ok=True)
            self._save_config(self.cfg, self.experiment_path)
            print(f" > Created experiment : {self.cfg.name}/{self.cfg.ver}")

        # Create model & datamodule
        self.model = self._create_model(self.cfg)
        self.datamodule: DataModule = DataModule()

        #Create trainer
        self.trainer = Trainer(self.cfg,self.model)

        # Load model weight/optimizer
        if resuming_run:
            self._load_checkpoint()
            

    def _load_checkpoint(self):
        checkpoint_path = self.experiment_path / "checkpoints" / "latest.pt"
        print(f" > Loading checkpoint: {checkpoint_path.as_posix()}")
        checkpoint = torch.load(
            checkpoint_path.as_posix(),
            map_location=torch.device("mps")
        )
        self.model.load_state_dict(checkpoint["model"])
        self.trainer.opt.load_state_dict(checkpoint["optimizer"])
        self.trainer.start_epoch = checkpoint["epoch"] + 1

    def _load_experiment_config(self):
        config_filepath = self.experiment_path / "config.yaml"
        with config_filepath.open() as fp:
            config_dict = yaml.safe_load(fp)
            config_str = yaml.dump(config_dict)
            print(" > Loaded configuration: ")
            print("-------------------------")
            print(config_str.strip())
            print("----------------------------")
            self.cfg = Namespace(**config_dict)

    def _create_model(self, cfg):
        # Create model
        model = MultiLayerPerceptron(
            nin = 28*28,                # Image size is 28x28
            nhidden = cfg.num_hidden,   # Larger hidden layer
            nout=10                     # 10 possible classes
        )
        return model

    def train(self):
        # Initialize run
        self.run = neptune.init_run(
            project="dp-workspace/dp-segmentation",
            name=self.cfg.name,
            custom_run_id=self.cfg.ver,
            tags=["debug"],
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MGM2ZjZlNS0wOGVlLTRiN2UtYjYzNC1mNDJiZmU4MzlhYzIifQ==",
        )
        self.run["parameters"] = self.cfg

        #Start training
        self.trainer.setup(
            datamodule=self.datamodule,
            log=Log(self.run)
        )
        self.trainer.fit(self.experiment_path)

        #End run
        self.run.stop()
        
    
    def _save_config(self, cfg, exp_path):
        config_yaml = yaml.dump(vars(cfg))

        print(" > Training Configuration:")
        print("----------------------------")
        print(config_yaml.strip())
        print("----------------------------")

        # Save configuration into YAML
        (exp_path / "config.yaml").write_text(config_yaml)


