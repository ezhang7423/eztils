"""
import often used modules
"""
import random
from pathlib import Path

import numpy as np
import torch
from einops import rearrange as rea
from einops import reduce

try:
    from jaxtyping import Bool, Float, Float16, Float32, Int
except RuntimeError:  # python < 3.9
    pass
from torch import einsum, nn, tensor
from torch.nn import functional as F
from torchvision.utils import save_image

"""
globals
"""
USE_GPU = False
DTYPE = torch.float
GPU_ID = 0
DEVICE = None


def set_gpu_mode(mode, gpu_id=0):
    global USE_GPU
    global DEVICE
    global GPU_ID
    GPU_ID = gpu_id
    USE_GPU = mode
    DEVICE = torch.device("cuda:" + str(gpu_id) if USE_GPU else "cpu")
    if USE_GPU:
        torch.cuda.set_device(gpu_id)


def activation_from_string(string):
    if string == "identity":
        return lambda x: x
    return getattr(nn, string)()


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


"""
helper functions
"""


def seed_everything(seed):
    """
    Set the seed for all the possible random number generators
    for global packages.
    :param seed:
    :return: None
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


from .distributions import *
from .lightning import *
from .math import *
from .modules import *
from .parameters import *
from .tensor_creators import *
from .to import *
