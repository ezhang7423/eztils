def map_recursive(fctn, x_or_iterable):
    """
    Apply `fctn` to each element in x_or_iterable.

    This is a generalization of the map function since this will work
    recursively for iterables.

    :param fctn: Function from element of iterable to something.
    :param x_or_iterable: An element or an Iterable of an element.
    :return: The same (potentially recursive) iterable but with
    all the elements transformed by fctn.
    """
    # if isinstance(x_or_iterable, Iterable):
    if isinstance(x_or_iterable, list) or isinstance(x_or_iterable, tuple):
        return type(x_or_iterable)(map_recursive(fctn, item) for item in x_or_iterable)
    else:
        return fctn(x_or_iterable)


def filter_recursive(x_or_iterable):
    """
    Filter out elements that are Falsy (where bool(x) is False) from
    potentially recursive lists.

    :param x_or_iterable: An element or a list.
    :return: If x_or_iterable is not an Iterable, then return x_or_iterable.
    Otherwise, return a filtered version of x_or_iterable.
    """
    if isinstance(x_or_iterable, list):
        new_items = []
        for sub_elem in x_or_iterable:
            filtered_sub_elem = filter_recursive(sub_elem)
            if filtered_sub_elem is not None and not (
                isinstance(filtered_sub_elem, list) and len(filtered_sub_elem) == 0
            ):
                new_items.append(filtered_sub_elem)
        return new_items
    else:
        return x_or_iterable


def batch(iterable, n=1):
    """
    Split an interable into batches of size `n`. If `n` does not evenly divide
    `iterable`, the last slice will be smaller.

    https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks

    Usage:
    ```
        for i in batch(range(0,10), 3):
            print i

        [0,1,2]
        [3,4,5]
        [6,7,8]
        [9]
    ```
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def find_key_recursive(obj, key):
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v, dict):
            result = find_key_recursive(v, key)
            if result is not None:
                return result
