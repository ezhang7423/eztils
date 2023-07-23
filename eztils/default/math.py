from collections import OrderedDict
from numbers import Number

import numpy as np


def normalize(x):
    """
    scales `x` to [0, 1]
    """
    x = x - x.min()
    x = x / x.max()
    return x


def create_stats_ordered_dict(  # TODO support tensors
    name,
    data,
    stat_prefix=None,
    always_show_all_stats=True,
    exclude_max_min=False,
):
    if stat_prefix is not None:
        name = f"{stat_prefix}{name}"
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                f"{name}_{number}",
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if isinstance(data, np.ndarray) and data.size == 1 and not always_show_all_stats:
        return OrderedDict({name: float(data)})

    stats = OrderedDict(
        [
            (name + " Mean", np.mean(data)),
            (name + " Std", np.std(data)),
        ]
    )
    if not exclude_max_min:
        stats[name + " Max"] = np.max(data)
        stats[name + " Min"] = np.min(data)
    return stats
