from typing import Any, Union

import numpy as np
import torch

from eztils import normalize


def to_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def to_img(x: torch.Tensor):
    """converts a tensor to a numpy array suitable for saving as an image

    :param x:   a tensor of shape (C, H, W)
    :type x:   torch.Tensor
    :return:   a numpy array of shape (H, W, C) with values in [0, 255]
    :rtype:   np.ndarray
    """
    normalized = normalize(x)
    array = to_np(normalized)
    array = np.transpose(array, (1, 2, 0))
    return (array * 255).astype(np.uint8)


def to_torch(x, dtype=None, device=None):
    from eztils.torch import DEVICE, DTYPE

    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
        # import pdb; pdb.set_trace()
    return torch.tensor(x, dtype=dtype, device=device)


def to_device(
    input: Any, device: Union[str, torch.device, int] = None, inplace: bool = True
) -> Any:
    from eztils.torch import DEVICE

    device = device or DEVICE

    """Recursively places tensors on the appropriate device."""
    if input is None:
        return input
    elif isinstance(input, torch.Tensor):
        return input.to(device)
    elif isinstance(input, tuple):
        return tuple(
            to_device(input=subinput, device=device, inplace=inplace)
            for subinput in input
        )
    elif isinstance(input, list):
        if inplace:
            for i in range(len(input)):
                input[i] = to_device(input=input[i], device=device, inplace=inplace)
            return input
        else:
            return [
                to_device(input=subpart, device=device, inplace=inplace)
                for subpart in input
            ]
    elif isinstance(input, dict):
        if inplace:
            for key in input:
                input[key] = to_device(input=input[key], device=device, inplace=inplace)
            return input
        else:
            return {
                k: to_device(input=input[k], device=device, inplace=inplace)
                for k in input
            }
    elif isinstance(input, set):
        if inplace:
            for element in list(input):
                input.remove(element)
                input.add(to_device(element, device=device, inplace=inplace))
        else:
            return {to_device(k, device=device, inplace=inplace) for k in input}
    elif isinstance(input, np.ndarray) or np.isscalar(input) or isinstance(input, str):
        return input
    elif hasattr(input, "to"):
        # noinspection PyCallingNonCallable
        return input.to(device=device, inplace=inplace)
    else:
        raise NotImplementedError(
            f"Sorry, value of type {type(input)} is not supported."
        )
