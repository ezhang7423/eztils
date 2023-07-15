from copy import deepcopy

import torch
from torch import nn


class EMA(nn.Module):
    """Model Exponential Moving Average V2 from timm"""

    def __init__(self, model, decay=0.9999):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)


class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """

    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output


class ParallelLayerNorm(LayerNorm):
    def __init__(self, num_heads, features, center=True, scale=False, eps=0.000001):
        super().__init__(features * num_heads, center, scale, eps)
        self.num_heads = num_heads
        self.features = features

    def forward(self, x_):
        x = x_.reshape(x_.shape[0], self.num_heads, self.features)

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)

        if self.scale:
            output = output * self.scale_param.reshape(self.num_heads, self.features)
        if self.center:
            output = output + self.center_param.reshape(self.num_heads, self.features)

        return output.reshape(*x_.shape)
