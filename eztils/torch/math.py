import torch


def atanh(x):
    # errors or instability at values near 1
    x = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * (torch.log(1 + x) - torch.log(1 - x))
