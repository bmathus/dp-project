from dataclasses import dataclass
from typing import Literal

@dataclass
class Config:
    name: str
    ver: str
    data_path: str
    base_path: str
    network: Literal["urpc", "mcnet", "dbpnet"]

    seed: int
    deterministic: int

    max_iter: int
    base_lr: float
    num_classes: int

    batch_size: int
    labeled_bs: int
    labeled_num: int
    patch_size: int

    consistency: float
    consistency_rampup: float
    temperature: float
    lamda: float
    gamma: float