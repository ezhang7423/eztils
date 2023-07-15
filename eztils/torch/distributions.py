"""
Add custom distributions in addition to th existing ones
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F
from torch.distributions import Bernoulli as TorchBernoulli
from torch.distributions import Beta as TorchBeta
from torch.distributions import Distribution as TorchDistribution
from torch.distributions import Independent as TorchIndependent
from torch.distributions import Normal as TorchNormal
from torch.distributions import kl_divergence
from torch.distributions.utils import _sum_rightmost
from torchtyping import TensorType

from eztils.default.math import create_stats_ordered_dict
from eztils.torch.math import atanh
from eztils.torch.tensor_creators import ones, tensor, zeros
from eztils.torch.to import to_np


class Distribution(TorchDistribution):
    def sample_and_logprob(self):
        s = self.sample()
        log_p = self.log_prob(s)
        return s, log_p

    def rsample_and_logprob(self):
        s = self.rsample()
        log_p = self.log_prob(s)
        return s, log_p

    def mle_estimate(self):
        return self.mean

    def get_diagnostics(self):
        return {}


class TorchDistributionWrapper(Distribution):
    def __init__(self, distribution: TorchDistribution):
        self.distribution = distribution

    @property
    def batch_shape(self):
        return self.distribution.batch_shape

    @property
    def event_shape(self):
        return self.distribution.event_shape

    @property
    def arg_constraints(self):
        return self.distribution.arg_constraints

    @property
    def support(self):
        return self.distribution.support

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def variance(self):
        return self.distribution.variance

    @property
    def stddev(self):
        return self.distribution.stddev

    def sample(self, sample_size=torch.Size()):
        return self.distribution.sample(sample_shape=sample_size)

    def rsample(self, sample_size=torch.Size()):
        return self.distribution.rsample(sample_shape=sample_size)

    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def cdf(self, value):
        return self.distribution.cdf(value)

    def icdf(self, value):
        return self.distribution.icdf(value)

    def enumerate_support(self, expand=True):
        return self.distribution.enumerate_support(expand=expand)

    def entropy(self):
        return self.distribution.entropy()

    def perplexity(self):
        return self.distribution.perplexity()

    def __repr__(self):
        return "Wrapped " + self.distribution.__repr__()


class Delta(Distribution):
    """A deterministic distribution"""

    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value.detach()

    def rsample(self):
        return self.value

    @property
    def mean(self):
        return self.value

    @property
    def variance(self):
        return 0

    @property
    def entropy(self):
        return 0


class TanhDelta(Delta):
    def __init__(self, value):
        super().__init__(torch.tanh(value))
        self.pre_tanh_value = value


class Bernoulli(Distribution, TorchBernoulli):
    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(
            create_stats_ordered_dict(
                "probability",
                to_np(self.probs),
            )
        )
        return stats


class Independent(Distribution, TorchIndependent):
    def get_diagnostics(self):
        return self.base_dist.get_diagnostics()


class Beta(Distribution, TorchBeta):
    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(
            create_stats_ordered_dict(
                "alpha",
                to_np(self.concentration0),
            )
        )
        stats.update(
            create_stats_ordered_dict(
                "beta",
                to_np(self.concentration1),
            )
        )
        stats.update(
            create_stats_ordered_dict(
                "entropy",
                to_np(self.entropy()),
            )
        )
        return stats


class MultivariateDiagonalNormal(TorchDistributionWrapper):
    from torch.distributions import constraints

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

    def __init__(self, loc, scale_diag, reinterpreted_batch_ndims=1):
        dist = Independent(
            TorchNormal(loc, scale_diag),
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )
        super().__init__(dist)

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(
            create_stats_ordered_dict(
                "mean",
                to_np(self.mean),
                # exclude_max_min=True,
            )
        )
        stats.update(
            create_stats_ordered_dict(
                "std",
                to_np(self.distribution.stddev),
            )
        )
        return stats

    def __repr__(self):
        return self.distribution.base_dist.__repr__()


# Independent RV KL handling - https://github.com/pytorch/pytorch/issues/13545


@torch.distributions.kl.register_kl(TorchIndependent, TorchIndependent)
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)


class TanhGaussianMixture(TorchDistributionWrapper):
    """
    Construct a gaussian mixture model using gumbel softmax for backprop.
    https://bochang.me/blog/posts/pytorch-distributions/#table

    """

    def __init__(
        self,
        mean: TensorType["batch", "num_gaussians", "action_dim"],
        std: TensorType["batch", "num_gaussians", "action_dim"],
        weights: TensorType["batch", "num_gaussians"],
    ):
        if len(mean.shape) == 2:
            mean = mean.unsqueeze(dim=0)
        if len(std.shape) == 2:
            std = std.unsqueeze(dim=0)
        if len(weights.shape) == 1:
            weights = weights.unsqueeze(dim=0)

        weights_sum = weights.sum(dim=-1)
        assert torch.all(
            torch.isclose(weights_sum, torch.ones_like(weights_sum))
        ), "Weights must sum to 1"

        self.num_gaussians = weights.shape[-1]
        self.batch_size = weights.shape[0]
        self.normal_mean = mean
        self.weights = weights
        super().__init__(MultivariateDiagonalNormal(mean, std))

    def mle_estimate(self, return_std=False):
        """Return the mean of the most likely component.
        This often computes the mode of the distribution, but not always.
        """
        bz = torch.arange(self.batch_size)
        ind = torch.argmax(self.weights, dim=1)

        if not return_std:
            return self.mean[bz, ind]
        else:
            return self.mean[bz, ind], self.stddev[bz, ind]

    def sample(self, n=1):
        assert isinstance(n, int), f"{n} must be an int"
        return self.rsample(n=n, hard=True)

    def rsample(self, n=1, temperature=0.02, hard=False):
        """Differentiable sample with gumbel softmax

        Returns
        ----------
        tensor of size (n, num_feat)
        """
        assert isinstance(n, int), f"{n} must be an int"

        soft_categorical = F.gumbel_softmax(
            self.weights.log().unsqueeze(0).expand(n, -1, -1),
            tau=temperature,
            hard=hard,
        )
        soft_categorical = soft_categorical.unsqueeze(-1)
        sample = self.distribution.rsample([n])  # [n, batch, num_gaussian, dim]
        return torch.tanh(torch.einsum("hijk,hijk->hik", sample, soft_categorical))

    def _log_prob_from_pre_tanh(self, pre_tanh_value):
        if (
            len(pre_tanh_value.shape) == 3
        ):  # [n, batch, action_dim] -> Require [1, batch, action_dim]
            # this came from calling dist.sample()
            # assert pre_tanh_value.shape[0] == 1
            assert pre_tanh_value.shape[1] == self.batch_size
            if pre_tanh_value.shape[0] == 1:
                pre_tanh_value = pre_tanh_value.squeeze(0)

        # [batch_size, action_dim] -> [batch_size, num_gaussians, action_dim]
        pre_tanh_value_mg = pre_tanh_value.unsqueeze(-2)
        # [batch_size, num_gaussians]
        # equivalent to the jacobian, -log(1 - tanh(x)^2).
        correction = -2.0 * (
            tensor([2.0]).log() - pre_tanh_value - F.softplus(-2.0 * pre_tanh_value)
        ).sum(dim=-1)
        # [batch_size, num_gaussians]
        pi = torch.exp(self.distribution.log_prob(pre_tanh_value_mg)).clip(1e-40)
        ret = torch.log((self.weights * pi).sum(-1)) + correction
        return ret

    def log_prob(self, value, pre_tanh_value=None):
        assert (
            value.shape[-1] == self.mean.shape[-1]
        ), f"{value} must have shape [{self.batch_shape[0]}, {self.batch_shape[1]}, {self.event_shape[0]}]"

        if pre_tanh_value is None:
            # errors or instability at values near 1
            pre_tanh_value = atanh(value)
        else:
            assert pre_tanh_value.shape[-1] == self.mean.shape[-1]

        return self._log_prob_from_pre_tanh(pre_tanh_value)

    def __repr__(self):
        s = "TanhGaussianMixture(mean=%s, std=%s, weights=%s)"
        return s % (self.mean, self.stddev, self.weights)

    @property
    def mean(self):
        return torch.tanh(self.normal_mean)

    def perplexity(self):
        return NotImplementedError

    def entropy(self):
        return NotImplementedError

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(
            create_stats_ordered_dict(
                "mean",
                to_np(self.mean),
            )
        )
        stats.update(
            create_stats_ordered_dict(
                "stddev",
                to_np(self.stddev),
            )
        )
        stats.update(
            create_stats_ordered_dict(
                "weights",
                to_np(self.weights),
            )
        )
        return stats


class GaussianMixture(TanhGaussianMixture):
    def __init__(
        self,
        mean: TensorType["batch", "num_gaussians", "action_dim"],
        std: TensorType["batch", "num_gaussians", "action_dim"],
        weights: TensorType["batch", "num_gaussians"],
    ):
        super().__init__(mean, std, weights)

    def log_prob(self, value):
        assert (
            value.shape[-1] == self.mean.shape[-1]
        ), f"{value} must have shape [{self.batch_shape[0]}, {self.batch_shape[1]}, {self.event_shape[0]}]"
        if (
            len(value.shape) == 3
        ):  # [n, batch, action_dim] -> Require [1, batch, action_dim]
            # this came from calling dist.sample()
            assert value.shape[0] == 1
            assert value.shape[1] == self.batch_size
            value = value.squeeze(0)
        value = value.unsqueeze(1).expand(-1, self.num_gaussians, -1)
        pi = torch.exp(self.distribution.log_prob(value)).clip(1e-40)
        return torch.log(torch.einsum("ij,ij->i", self.weights, pi))

    def rsample(self, n=1, temperature=0.02, hard=False):
        return torch.atanh(super().rsample(n, temperature, hard))

    def __repr__(self):
        s = "GaussianMixture(mean=%s, std=%s, weights=%s)"
        return s % (self.mean, self.stddev, self.weights)

    @property
    def mean(self):
        return self.normal_mean


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = MultivariateDiagonalNormal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def _log_prob_from_pre_tanh(self, pre_tanh_value):
        """
        Adapted from
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73

        This formula is mathematically equivalent to log(1 - tanh(x)^2).

        Derivation:

        log(1 - tanh(x)^2)
         = log(sech(x)^2)
         = 2 * log(sech(x))
         = 2 * log(2e^-x / (e^-2x + 1))
         = 2 * (log(2) - x - log(e^-2x + 1))
         = 2 * (log(2) - x - softplus(-2x))

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        log_prob = self.normal.log_prob(pre_tanh_value)
        correction = -2.0 * (
            tensor([2.0]).log()
            - pre_tanh_value
            - torch.nn.functional.softplus(-2.0 * pre_tanh_value)
        ).sum(dim=-1)
        return log_prob + correction

    def log_prob(self, value, pre_tanh_value=None):
        if pre_tanh_value is None:
            # errors or instability at values near 1
            value = torch.clamp(value, -0.999999, 0.999999)
            pre_tanh_value = torch.log(1 + value) / 2 - torch.log(1 - value) / 2
        return self._log_prob_from_pre_tanh(pre_tanh_value)

    def rsample_with_pretanh(self, sample_shape=torch.Size()):
        z = self.normal_mean + self.normal_std * MultivariateDiagonalNormal(
            zeros(self.normal_mean.size()), ones(self.normal_std.size())
        ).sample(sample_shape)
        return torch.tanh(z), z

    def sample(self):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        return value.detach()

    def rsample(self, sample_shape=torch.Size()):
        """
        Sampling in the reparameterization case.
        """
        value, pre_tanh_value = self.rsample_with_pretanh(sample_shape)
        return value

    def sample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        value, pre_tanh_value = value.detach(), pre_tanh_value.detach()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    def rsample_and_logprob(self, sample_shape=torch.Size()):
        value, pre_tanh_value = self.rsample_with_pretanh(sample_shape)
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    @property
    def mean(self):
        return torch.tanh(self.normal_mean)

    @property
    def stddev(self):
        return self.normal_std

    def __repr__(self):
        return f"TanhNormal (shape: {self.normal_mean.shape})"

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(
            create_stats_ordered_dict(
                "mean",
                to_np(self.mean),
            )
        )
        stats.update(create_stats_ordered_dict("normal/std", to_np(self.normal_std)))
        stats.update(
            create_stats_ordered_dict(
                "normal/log_std",
                to_np(torch.log(self.normal_std)),
            )
        )
        return stats
