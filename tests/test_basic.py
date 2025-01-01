import pytest
import os
from eztils import abspath, apply_dict, cycle, default


def test_abspath_current_dir():
    path = abspath(".")
    assert os.path.isabs(path)
    assert os.path.exists(path)


def test_apply_dict():
    def adder(x, y=0):
        return x + y

    params = {"x": 5, "y": 10}
    assert apply_dict(adder, params) == 15


def test_cycle():
    items = [1, 2, 3]
    cycling = cycle(items, 2)
    assert isinstance(cycling, list)
    assert cycling == [1, 2, 3, 1, 2, 3]


def test_default():
    result = default(None, "fallback")
    assert result == "fallback"
    result2 = default("value", "fallback")
    assert result2 == "value"

