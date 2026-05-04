import torch
import random
import numpy as np

config={
    'seed':42,
    'batch_size':64,
    'lr':1e-3,
    'num_epochs':10
}

def set_seed(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True