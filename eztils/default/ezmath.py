from collections import OrderedDict
from numbers import Number

import numpy as np


def normalize(x):
    """
    Normalize an array by scaling its values between 0 and 1.

    :param x: The input array to be normalized.
    :type x: numpy.ndarray
    :return: The normalized array.
    :rtype: numpy.ndarray
    """
    x = x - x.min()
    x = x / x.max()
    return x


def create_stats_ordered_dict(
    name,
    data,
    stat_prefix=None,
    always_show_all_stats=True,
    exclude_max_min=False,
):
    """
    Create an ordered dictionary of statistics for the given data.

    :param name: The name of the data.
    :type name: str
    :param data: The data to compute statistics for.
    :type data: Union[Number, List[Number], Tuple[Number], np.ndarray]
    :param stat_prefix: A prefix to add to the name of the data, defaults to None.
    :type stat_prefix: Optional[str]
    :param always_show_all_stats: Whether to always show all statistics, defaults to True.
    :type always_show_all_stats: bool, optional
    :param exclude_max_min: Whether to exclude the maximum and minimum values, defaults to False.
    :type exclude_max_min: bool, optional
    :return: An ordered dictionary of statistics for the given data.
    :rtype: OrderedDict
    """
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
