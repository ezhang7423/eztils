"""
GPU wrappers
"""
import torch

from eztils import default
from eztils.torch import DEVICE, DTYPE


def device_dtype_decorator(func):
    def wrapper(*args, torch_device=None, **kwargs):
        torch_device = default(torch_device, DEVICE)
        torch_dtype = default(kwargs.get("dtype"), DTYPE)
        return func(*args, **kwargs, device=torch_device, dtype=torch_dtype)

    return wrapper


@device_dtype_decorator
def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs)


@device_dtype_decorator
def tensor(*args, **kwargs):
    return torch.tensor(*args, **kwargs)


@device_dtype_decorator
def randint(*sizes, **kwargs):
    return torch.randint(*sizes, **kwargs)


@device_dtype_decorator
def zeros(*sizes, **kwargs):
    return torch.zeros(*sizes, **kwargs)


@device_dtype_decorator
def empty(*sizes, **kwargs):
    return torch.empty(*sizes, **kwargs)


@device_dtype_decorator
def ones(*sizes, **kwargs):
    return torch.ones(*sizes, **kwargs)


@device_dtype_decorator
def ones_like(*args, **kwargs):
    return torch.ones_like(*args, **kwargs)


@device_dtype_decorator
def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs)


@device_dtype_decorator
def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs)


@device_dtype_decorator
def randperm(*args, **kwargs):
    return torch.randperm(*args, **kwargs)


@device_dtype_decorator
def rand(*args, **kwargs):
    return torch.rand(*args, **kwargs)


@device_dtype_decorator
def tensor(*args, **kwargs):
    return torch.tensor(*args, **kwargs)


@device_dtype_decorator
def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs)
