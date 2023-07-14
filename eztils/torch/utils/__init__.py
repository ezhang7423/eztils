import random

import numpy as np
import torch

# helpers functions


def register_buffer(
    self, name, val
):  # register as same type as weights for lightning modules
    self.register_buffer(name, val.type(self.dtype))


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


from ...default.logging import *
from ...default.structures import *
from .arrays import *
from .model_wrappers import *
