import inspect as inspect_
import os
from inspect import getsourcefile, isfunction


def default(val, d):
    """If val exists, return it. Otherwise, return d"""
    if val is not None:
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        yield from dl


def abspath():
    # https://stackoverflow.com/questions/16771894/python-nameerror-global-name-file-is-not-defined
    # https://docs.python.org/3/library/inspect.html#inspect.FrameInfo
    # return os.path.dirname(inspect.stack()[1][1]) # type: ignore
    # return os.path.dirname(getsourcefile(lambda:0)) # type: ignore
    return os.path.dirname(getsourcefile(inspect_.stack()[1][0]))  # type: ignore


def apply_dict(fn, d, *args, **kwargs):
    return {k: fn(v, *args, **kwargs) for k, v in d.items()}
