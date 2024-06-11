"""
import often used modules
"""

import random
from functools import partial

import numpy as np
import torch

from eztils.serialization import load, save

"""
globals
"""
USE_GPU = False
DTYPE = torch.float
GPU_ID = 0
DEVICE = None


def set_gpu_mode(mode, gpu_id=0):
    """
    Sets the mode and GPU ID for PyTorch.

    :param mode: A boolean indicating whether to use GPU or not.
    :type mode: bool
    :param gpu_id: The ID of the GPU to use, defaults to 0.
    :type gpu_id: int, optional
    """
    global USE_GPU
    global DEVICE
    global GPU_ID
    GPU_ID = gpu_id
    USE_GPU = mode
    DEVICE = torch.device("cuda:" + str(gpu_id) if USE_GPU else "cpu")
    if USE_GPU:
        torch.cuda.set_device(gpu_id)


def activation_from_string(string):
    """
    Returns the activation function corresponding to the given string.

    :param string: The name of the activation function to return.
    :type string: str
    :return: The activation function.
    :rtype: function
    """
    if string == "identity":
        return lambda x: x
    return getattr(nn, string)()


def soft_update_from_to(source, target, tau):
    """
    Update the target model parameters with a fraction of the source model parameters.

    :param source: The source model whose parameters will be used to update the target model.
    :type source: torch.nn.Module
    :param target: The target model whose parameters will be updated.
    :type target: torch.nn.Module
    :param tau: The fraction of the source model parameters to be used for updating the target model parameters.
    :type tau: float
    """
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


save = partial(save, save_fn=torch.save)
torch.load = partial(torch.load, map_location="cpu")
load = partial(load, load_fn=torch.load)


# def get_best_cuda() -> int:
#     import numpy as np
#     import pynvml

#     pynvml.nvmlInit()
#     deviceCount = pynvml.nvmlDeviceGetCount()
#     deviceMemory = []
#     for i in range(deviceCount):
#         handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#         mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         deviceMemory.append(mem_info.free)
#     deviceMemory = np.array(deviceMemory, dtype=np.int64)
#     best_device_index = np.argmax(deviceMemory)
#     print("best gpu:", best_device_index)
#     return best_device_index.item()


def freeze(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def identity(x):
    return x


from .distributions import *
from .lightning import *
from .math import *
from .modules import *
from .observable import *
from .parameters import *
from .tensor_creators import *
from .to import *
