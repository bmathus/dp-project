import random
import torch
import numpy as np
from config.run_config import Config

def worker_init_fn(worker_id):
    random.seed(1337 + worker_id)
    
def decide_device():
    if (torch.cuda.is_available()): 
        return "cuda"
    if (torch.backends.mps.is_available()):
        return "mps"
    return "cpu"

def get_current_consistency_weight(cfg: Config,epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return cfg.consistency * sigmoid_rampup(epoch, cfg.consistency_rampup)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def sharpening(P,cfg: Config):
    T = 1/cfg.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen