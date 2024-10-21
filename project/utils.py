import random
import torch

def worker_init_fn(worker_id):
    random.seed(1337 + worker_id)


def decide_device():
    if (torch.cuda.is_available()): 
        return "cuda"
    if (torch.backends.mps.is_available()):
        return "mps"
    return "cpu"