import inspect as inspect_
import os
import sys
from dataclasses import dataclass
from inspect import getsourcefile, isfunction
from pathlib import Path

"""
miscellaneous functions that are often used
"""


def default(val, d):
    """
    If val exists, return it. Otherwise, return d

    :param val: The value to check
    :param d: The default value to return if val is None
    :return: val if it exists, otherwise d
    """
    if val is not None:
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        yield from dl


def abspath():
    """
    Returns the absolute path of the directory containing the caller module.

    :return: The absolute path of the directory containing the caller module.
    :rtype: str

    # https://stackoverflow.com/questions/16771894/python-nameerror-global-name-file-is-not-defined
    # https://docs.python.org/3/library/inspect.html#inspect.FrameInfo
    # return os.path.dirname(inspect.stack()[1][1]) # type: ignore
    # return os.path.dirname(getsourcefile(lambda:0)) # type: ignore
    """
    return os.path.dirname(getsourcefile(inspect_.stack()[1][0]))  # type: ignore


from pathlib import Path


def setup_path(path: str) -> Path:
    """
    Create path if it does not exist.

    :param path: The path to be created.
    :type path: str
    :return: The created path.
    :rtype: Path
    """
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def apply_dict(fn, d, *args, **kwargs):
    """
    Apply a function to each value in a dictionary and return a new dictionary with the results.

    :param fn: The function to apply to each value in the dictionary.
    :type fn: function
    :param d: The dictionary to apply the function to.
    :type d: dict
    :param args: Additional positional arguments to pass to the function.
    :type args: tuple
    :param kwargs: Additional keyword arguments to pass to the function.
    :type kwargs: dict
    :return: A new dictionary with the results of applying the function to each value in the input dictionary.
    :rtype: dict
    """
    return {k: fn(v, *args, **kwargs) for k, v in d.items()}


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


"""
often used modules
"""
# beartype
from beartype import beartype
from beartype.door import die_if_unbearable  # <-- like "assert isinstance(...)"
from beartype.door import is_bearable  # <-------- like "isinstance(...)"
from beartype.vale import Is

enforced_dataclass = beartype(dataclass)
frozen_enforced_dataclass = beartype(dataclass(frozen=True))


from .dict_operations import *
from .ezitertools import *
from .ezlogging import *
from .ezmath import *
from .serialization import *
from .structures import *
