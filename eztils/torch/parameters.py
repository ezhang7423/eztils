# -----------------------------------------------------------------------------#
# ----------------------------- parameter counting ----------------------------#
# -----------------------------------------------------------------------------#
# https://github.com/allenai/allenact/blob/f00445e4ae8724ccc001b3300af5c56a7f882614/allenact/utils/tensor_utils.py#L1


def param_to_module(param):
    """
    Extracts the module name from a parameter string.

    :param param: A string representing a parameter in the format "module.submodule.parameter".
    :type param: str
    :return: The name of the module containing the parameter.
    :rtype: str
    """
    module_name = param[::-1].split(".", maxsplit=1)[-1][::-1]
    return module_name


def report_parameters(model, topk=10):
    """
    Prints a summary of the total number of parameters in a PyTorch model, as well as the top-k parameters with the largest number of elements.

    :param model: The PyTorch model to analyze.
    :type model: torch.nn.Module
    :param topk: The number of parameters to display in the summary. Defaults to 10.
    :type topk: int, optional
    :return: The total number of parameters in the model.
    :rtype: int
    """

    def _to_str(num):
        if num >= 1e6:
            return f"{(num/1e6):.2f} M"
        else:
            return f"{(num/1e3):.2f} k"

    counts = {k: p.numel() for k, p in model.named_parameters()}
    n_parameters = sum(counts.values())
    print(f"[ utils/arrays ] Total parameters: {_to_str(n_parameters)}")

    modules = dict(model.named_modules())
    sorted_keys = sorted(counts, key=lambda x: -counts[x])
    # max_length = max([len(k) for k in sorted_keys])
    for i in range(topk):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        print(" " * 8, f"{key:10}: {_to_str(count)} | {modules[module]}")

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    print(
        " " * 8,
        f"... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters",
    )
    return n_parameters
