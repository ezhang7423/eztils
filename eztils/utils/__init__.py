import inspect as inspect_
import os
import random
from inspect import getsourcefile, isfunction

import numpy as np
import torch

# helpers functions


def register_buffer(
    self, name, val
):  # register as same type as weights for lightning modules
    self.register_buffer(name, val.type(self.dtype))


def exists(x):
    """Check that x is not None"""
    return x is not None


def default(val, d):
    """If val exists, return it. Otherwise, return d"""
    if exists(val):
        return val
    return d() if isfunction(d) else d


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


def cycle(dl):
    while True:
        yield from dl


def abspath():
    # https://stackoverflow.com/questions/16771894/python-nameerror-global-name-file-is-not-defined
    # https://docs.python.org/3/library/inspect.html#inspect.FrameInfo
    # return os.path.dirname(inspect.stack()[1][1]) # type: ignore
    # return os.path.dirname(getsourcefile(lambda:0)) # type: ignore
    return os.path.dirname(getsourcefile(inspect_.stack()[1][0]))  # type: ignore


from .arrays import *
from .logging import *
from .model_wrappers import *
from .structures import *
