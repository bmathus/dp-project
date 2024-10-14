import random

def worker_init_fn(worker_id):
    random.seed(1337 + worker_id)