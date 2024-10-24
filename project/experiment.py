from pathlib import Path
from project.trainer import Trainer
from project.logging import Logger
from argparse import Namespace
import yaml
import neptune
import uuid

BASE_PATH = Path("data/experiments")


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg

        self.resuming_run = False
        if self.cfg.ver != "":
            self.resuming_run = True

        # Resuming experiment
        if self.resuming_run:
            self.experiment_path = BASE_PATH / self.cfg.name / self.cfg.ver
            self._load_param_config()
            print(f" > Resuming experiment : {self.cfg.name}/{self.cfg.ver}")
        # Creating new experiment
        else:
            self.cfg.ver = "run-" + str(uuid.uuid4())[:8]
            self.experiment_path = BASE_PATH / self.cfg.name / self.cfg.ver
            self.experiment_path.mkdir(parents=True, exist_ok=True)
            self._save_param_config(self.cfg, self.experiment_path)
            print(f" > Created experiment : {self.cfg.name}/{self.cfg.ver}")

        # Create trainer
        self.trainer = Trainer(self.cfg, self.experiment_path, self.resuming_run)

    def start_experiment(self):
        # Initialize run
        self.run = neptune.init_run(
            project="dp-workspace/dp-segmentation",
            name=self.cfg.name,
            custom_run_id=self.cfg.ver,
            tags=["debug"],
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MGM2ZjZlNS0wOGVlLTRiN2UtYjYzNC1mNDJiZmU4MzlhYzIifQ==",
        )
        self.run["parameters"] = self.cfg

        # Start training
        self.trainer.setup(log=Logger(self.run))
        if self.cfg.network == "urpc":
            self.trainer.fit_urpc(self.experiment_path, self.run)
        elif self.cfg.network == "mtnet":
            self.trainer.fit_mtnet(self.experiment_path,self.run)
        else:
            print("No fit dunction")

        # End run
        self.run.stop()

    def _load_param_config(self):
        config_filepath = self.experiment_path / "config.yaml"
        with config_filepath.open() as fp:
            config_dict = yaml.safe_load(fp)
            config_str = yaml.dump(config_dict)
            print(" > Loaded configuration: ")
            print("-------------------------")
            print(config_str.strip())
            print("----------------------------")
            self.cfg = Namespace(**config_dict)

    def _save_param_config(self, cfg, exp_path):
        config_yaml = yaml.dump(vars(cfg))

        print(" > Training Configuration:")
        print("----------------------------")
        print("|", config_yaml.strip())
        print("----------------------------")

        # Save configuration into YAML
        (exp_path / "config.yaml").write_text(config_yaml)
