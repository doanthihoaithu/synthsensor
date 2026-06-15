import os
import random
import numpy as np
import torch

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_python_project_root_dir():
    return os.path.dirname(os.path.realpath(__file__))