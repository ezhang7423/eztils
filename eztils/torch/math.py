import torch


def atanh(x):
    """
    Returns the inverse hyperbolic tangent of x.

    :param x: Input tensor.
    :type x: torch.Tensor
    :return: The inverse hyperbolic tangent of x.
    :rtype: torch.Tensor
    """
    # errors or instability at values near 1
    x = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * (torch.log(1 + x) - torch.log(1 - x))
