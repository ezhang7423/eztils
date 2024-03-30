from typing import Any, Optional, Union

import functools

import numpy as np
import torch

from eztils.ezmath import normalize


def to_np(x):
    """
    Convert a tensor to a numpy array.

    :param x: Input tensor.
    :type x: torch.Tensor
    :return: Numpy array.
    :rtype: numpy.ndarray
    """
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
    """
    Converts a numpy array or a Python scalar to a PyTorch tensor.

    Args:
        x (numpy.ndarray, Python scalar): Input data to be converted to PyTorch tensor.
        dtype (torch.dtype, optional): Data type of the resulting tensor. Defaults to None.
        device (torch.device, optional): Device on which the tensor will be allocated. Defaults to None.

    Returns:
        torch.Tensor: A PyTorch tensor with the same data as the input.

    """
    from eztils.torch import DEVICE, DTYPE

    dtype = dtype or DTYPE
    device = device or DEVICE
    if isinstance(x, dict):
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
    return torch.tensor(x, dtype=dtype, device=device)


def to_device(
    input: Any,
    device: Union[str, torch.device, int] = None,
    inplace: Optional[bool] = True,
) -> Any:
    """
    Recursively places tensors on the appropriate device.

    Args:
        input (Any): The input tensor or collection of tensors to be moved to the device.
        device (Union[str, torch.device, int], optional): The device to move the tensors to. Defaults to None.
        inplace (bool, optional): Whether to perform the operation in place. Defaults to True.

    Returns:
        Any: The tensor or collection of tensors moved to the device.

    Raises:
        NotImplementedError: If the input is of a type that is not supported.
    """
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


"""
Recursively places tensors on the CPU.
"""
to_cpu = functools.partial(to_device, device="cpu")
