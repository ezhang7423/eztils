import collections

"""
Dictionary methods
"""


def nested_dict_to_dot_map_dict(d, parent_key=""):
    """
    Convert a recursive dictionary into a flat, dot-map dictionary.

    :param d: e.g. {'a': {'b': 2, 'c': 3}}
    :param parent_key: Used for recursion
    :return: e.g. {'a.b': 2, 'a.c': 3}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + "." + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(nested_dict_to_dot_map_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def merge_recursive_dicts(a, b, path=None, ignore_duplicate_keys_in_second_dict=False):
    """
    Merge two dictionaries that may have nested dictionaries.

    :param a: The first dictionary to merge.
    :type a: dict
    :param b: The second dictionary to merge.
    :type b: dict
    :param path: The path to the current key being merged, used for error reporting. Defaults to None.
    :type path: list, optional
    :param ignore_duplicate_keys_in_second_dict: Whether to ignore duplicate keys in the second dictionary. Defaults to False.
    :type ignore_duplicate_keys_in_second_dict: bool, optional
    :raises Exception: If there are duplicate keys and ignore_duplicate_keys_in_second_dict is False.
    :return: The merged dictionary.
    :rtype: dict
    """

    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_recursive_dicts(
                    a[key],
                    b[key],
                    path + [str(key)],
                    ignore_duplicate_keys_in_second_dict=ignore_duplicate_keys_in_second_dict,
                )
            elif a[key] == b[key]:
                print(f"Same value for key: {key}")
            else:
                duplicate_key = ".".join(path + [str(key)])
                if ignore_duplicate_keys_in_second_dict:
                    print(f"duplicate key ignored: {duplicate_key}")
                else:
                    raise Exception(f"Duplicate keys at {duplicate_key}")
        else:
            a[key] = b[key]
    return a


def list_of_dicts__to__dict_of_lists(lst):
    """
    Convert a list of dictionaries into a dictionary of lists, where each key in the output dictionary
    corresponds to a key in the input dictionaries, and each value in the output dictionary is a list
    containing the values of that key in the input dictionaries.

    :param lst: A list of dictionaries.
    :type lst: list[dict]

    :return: A dictionary of lists.
    :rtype: dict

    Example:
    --------
    >>> x = [
    ...     {'foo': 3, 'bar': 1},
    ...     {'foo': 4, 'bar': 2},
    ...     {'foo': 5, 'bar': 3},
    ... ]
    >>> list_of_dicts__to__dict_of_lists(x)
    {'foo': [3, 4, 5], 'bar': [1, 2, 3]}
    """
    if len(lst) == 0:
        return {}
    keys = lst[0].keys()
    output_dict = collections.defaultdict(list)
    for d in lst:
        assert set(d.keys()) == set(keys), (d.keys(), keys)
        for k in keys:
            output_dict[k].append(d[k])
    return output_dict


def safe_json(data):
    """
    Check if a given data structure is JSON serializable.

    :param data: The data structure to check.
    :type data: Any
    :return: True if the data structure is JSON serializable, False otherwise.
    :rtype: bool
    """
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False


def dict_to_safe_json(d, sort=False):
    """
    Convert each value in the dictionary into a JSON'able primitive.

    :param d: The dictionary to convert.
    :type d: dict or collections.OrderedDict
    :param sort: Whether to sort the resulting dictionary by key. Defaults to False.
    :type sort: bool, optional
    :return: A new dictionary with all values converted to JSON'able primitives.
    :rtype: dict or collections.OrderedDict
    """
    if isinstance(d, collections.OrderedDict):
        new_d = collections.OrderedDict()
    else:
        new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict) or isinstance(item, collections.OrderedDict):
                new_d[key] = dict_to_safe_json(item, sort=sort)
            else:
                new_d[key] = str(item)
    if sort:
        return collections.OrderedDict(sorted(new_d.items()))
    else:
        return new_d


def recursive_items(dictionary):
    """
    Get all (key, item) recursively in a potentially recursive dictionary.
    Usage:

    ```
    x = {
        'foo' : {
            'bar' : 5
        }
    }
    recursive_items(x)
    # output:
    # ('foo', {'bar' : 5})
    # ('bar', 5)
    ```
    :param dictionary:
    :return:
    """
    for key, value in dictionary.items():
        yield key, value
        if isinstance(value, dict):
            yield from recursive_items(value)
