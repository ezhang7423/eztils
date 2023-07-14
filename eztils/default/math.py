def normalize(x):
    """
    scales `x` to [0, 1]
    """
    x = x - x.min()
    x = x / x.max()
    return x
