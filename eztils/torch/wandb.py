import torch
import wandb
from rich import print


def log_wandb_distribution(key, samples, quantiles: list = None):
    """
    https://github.com/ezhang7423/wandb-histogram-over-time/tree/main
    Log the distribution of sample data points in Weights & Biases (wandb).

    This function computes the specified quantiles of the provided samples and logs them to wandb.
    If Weights & Biases run is not active, it prints the distribution to the console using rich print.
    By default, quantiles are calculated for the 0.0001, 0.01, 0.05, 0.10, and 0.25 levels, as well as their
    corresponding levels above the median (0.5).

    :param key: The label under which to log the quantiles.
    :type key: str
    :param samples: An array of sample data points.
    :type samples: torch.FloatTensor or torch.DoubleTensor
    :param quantiles: A list of quantiles to calculate for the sample distribution, defaults to None.
                      If None, uses the default set of quantiles that focus on the lower half of the distribution.
    :type quantiles: list of float, optional

    example usage:
    >>> import wandb
    >>> wandb.init()
    >>> gaussian_process = []

    >>> for i in range(1, 100):
    >>>     data = torch.randn(100) * math.log(i)
    >>>     gaussian_process.append(data)
    >>>     log_wandb_distribution("gaussian_process", data)

    """
    assert len(samples.shape) == 1

    if quantiles is None:
        quantiles = [0.0001, 0.01, 0.05, 0.10, 0.25]

    # all values less than 0.5
    assert all([q < 0.5 for q in quantiles])
    quantiles = quantiles + [0.5] + [1 - q for q in quantiles[::-1]]  # add median

    dist_quantiles = torch.quantile(samples, torch.tensor(quantiles).to(samples))

    d = {key + f"/{q}": dist_quantiles[i].item() for i, q in enumerate(quantiles)}

    if wandb.run is None:
        print(d)
    else:
        wandb.log(d)
    # not a good solution since this is not editable over time
    # d = {
    #     key: wandb.plot.line_series(
    #         xs=list(range(dist_quantiles.shape[1])),
    #         ys=dist_quantiles,
    #         keys=quantiles,
    #         title="Distribution over time",
    #         xname="Time Step",
    #     )
    # }
