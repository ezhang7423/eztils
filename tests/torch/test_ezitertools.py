import pytest
from eztils.ezitertools import (
    filter_recursive,
    map_recursive,
    batch,
    find_key_recursive,
)


# Tests for filter_recursive
def test_filter_recursive_empty_list():
    assert filter_recursive([]) == []


def test_filter_recursive_no_filter_needed():
    assert filter_recursive([1, 2, 3]) == [1, 2, 3]


def test_filter_recursive_simple_filter():
    assert filter_recursive([0, 1, False, 3]) == [1, 3]


def test_filter_recursive_nested_lists():
    data = [[], [0, 2, [False, 4, []], 5], None, True]
    expected = [[2, [4], 5], None, True]
    assert filter_recursive(data) == expected


def test_filter_recursive_non_list():
    assert filter_recursive(0) == 0
    assert filter_recursive("test") == "test"


def test_filter_recursive_all_falsy():
    assert filter_recursive([0, False, None, [], 0.0]) == []


def test_filter_recursive_mixed_types():
    data = [0, "hello", [], 42, [False, "world"], None]
    expected = ["hello", 42, ["world"], None]
    assert filter_recursive(data) == expected


def test_filter_recursive_deeply_nested():
    data = [0, [False, [None, [[], [1]]]], 2]
    expected = [[[1]], 2]
    assert filter_recursive(data) == expected


def test_filter_recursive_only_empty_lists():
    data = [[], [[]], [[], [[], []]], None]
    assert filter_recursive(data) == [None]


def test_filter_recursive_large_numbers():
    data = [1000000, 0, 999999999, []]
    assert filter_recursive(data) == [1000000, 999999999]


def test_filter_recursive_nested_all_falsy():
    data = [[False], [0, None, []], 0, False]
    assert filter_recursive(data) == []


def test_filter_recursive_tuple():
    data = (0, 1, False, 2)
    assert filter_recursive(data) == (0, 1, False, 2)


def test_filter_recursive_list_with_dict():
    data = [False, {"a": 1}, [], None]
    expected = [{"a": 1}, None]
    assert filter_recursive(data) == expected


def test_filter_recursive_nested_strings():
    data = [["", "hello"], [False, "world"], " "]
    expected = [["hello"], ["world"], " "]
    assert filter_recursive(data) == expected


def test_filter_recursive_multiple_levels():
    data = [None, [0, [[], 1, False, 2], 3], 4]
    expected = [[1, 2, 3], 4]
    assert filter_recursive(data) == expected


def test_filter_recursive_single_element_list():
    data = [[[]]]
    assert filter_recursive(data) == []


def test_filter_recursive_non_recursive_list():
    data = [True]
    assert filter_recursive(data) == [True]


def test_filter_recursive_empty_string():
    data = [""]
    assert filter_recursive(data) == []


def test_extra_case_negative_numbers():
    data = [-1, -2, [0, -3, [False, -4]], []]
    expected = [-1, -2, [-3, [-4]]]
    assert filter_recursive(data) == expected


def test_extra_case_mixed_objects():
    data = [None, 0, {"key": None}, [True, {}], (False, 1)]
    expected = [{"key": None}, [True, {}], (False, 1)]
    assert filter_recursive(data) == expected


def test_extra_case_strings_with_spaces():
    data = [" ", "", "  ", [False, " ", [None, "   "]]]
    expected = [" ", "  ", [" ", ["   "]]]
    assert filter_recursive(data) == expected


def test_extra_case_float_values():
    data = [0.0, 3.14, [], [2.71, 0.0], None]
    expected = [3.14, [2.71], None]
    assert filter_recursive(data) == expected


def test_extra_case_deep_nested_variety():
    data = [[[[], [[]]], [[None, [False]]]], 1, ""]
    expected = [[[[None]]], 1]
    assert filter_recursive(data) == expected


# Tests for map_recursive
def test_map_recursive_empty_list():
    assert map_recursive(lambda x: x * 2, []) == []


def test_map_recursive_no_map_needed():
    assert map_recursive(lambda x: x, [1, 2, 3]) == [1, 2, 3]


def test_map_recursive_simple_map():
    assert map_recursive(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]


def test_map_recursive_nested_lists():
    data = [1, [2, [3, 4]], 5]
    expected = [2, [4, [6, 8]], 10]
    assert map_recursive(lambda x: x * 2, data) == expected


def test_map_recursive_non_list():
    assert map_recursive(lambda x: x * 2, 3) == 6
    assert map_recursive(lambda x: x.upper(), "test") == "TEST"


def test_map_recursive_mixed_types():
    data = [1, "hello", [2, "world"], 3]
    expected = [2, "HELLO", [4, "WORLD"], 6]
    assert (
        map_recursive(lambda x: x * 2 if isinstance(x, int) else x.upper(), data)
        == expected
    )


def test_map_recursive_deeply_nested():
    data = [1, [2, [3, [4, 5]]]]
    expected = [2, [4, [6, [8, 10]]]]
    assert map_recursive(lambda x: x * 2, data) == expected


def test_map_recursive_tuple():
    data = (1, 2, (3, 4))
    expected = (2, 4, (6, 8))
    assert map_recursive(lambda x: x * 2, data) == expected


def test_map_recursive_list_with_dict():
    data = [1, {"a": 2}, 3]
    expected = [2, {"a": 2}, 6]
    assert map_recursive(lambda x: x * 2 if isinstance(x, int) else x, data) == expected


def test_map_recursive_strings():
    data = ["hello", ["world"]]
    expected = ["HELLO", ["WORLD"]]
    assert map_recursive(lambda x: x.upper(), data) == expected


# Tests for batch
def test_batch_empty_iterable():
    assert list(batch([], 3)) == []


def test_batch_single_element():
    assert list(batch([1], 3)) == [[1]]


def test_batch_exact_division():
    assert list(batch([1, 2, 3, 4, 5, 6], 2)) == [[1, 2], [3, 4], [5, 6]]


def test_batch_non_exact_division():
    assert list(batch([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_batch_large_n():
    assert list(batch([1, 2, 3], 5)) == [[1, 2, 3]]


def test_batch_strings():
    assert list(batch("abcdef", 2)) == ["ab", "cd", "ef"]


def test_batch_tuples():
    assert list(batch((1, 2, 3, 4), 2)) == [(1, 2), (3, 4)]


def test_batch_mixed_types():
    assert list(batch([1, "a", 2, "b"], 2)) == [[1, "a"], [2, "b"]]


def test_batch_zero_n():
    with pytest.raises(ValueError):
        list(batch([1, 2, 3], 0))


def test_batch_negative_n():
    with pytest.raises(ValueError):
        list(batch([1, 2, 3], -1))


# Tests for find_key_recursive
def test_find_key_recursive_empty_dict():
    assert find_key_recursive({}, "key") is None


def test_find_key_recursive_key_not_found():
    assert find_key_recursive({"a": 1}, "key") is None


def test_find_key_recursive_key_found():
    assert find_key_recursive({"key": 1}, "key") == 1


def test_find_key_recursive_nested_dict():
    data = {"a": {"b": {"key": 1}}}
    assert find_key_recursive(data, "key") == 1


def test_find_key_recursive_deeply_nested():
    data = {"a": {"b": {"c": {"d": {"key": 1}}}}}
    assert find_key_recursive(data, "key") == 1


def test_find_key_recursive_multiple_keys():
    data = {"key": 1, "a": {"key": 2}}
    assert find_key_recursive(data, "key") == 1


def test_find_key_recursive_non_dict():
    with pytest.raises(AttributeError):
        find_key_recursive([1, 2, 3], "key")


def test_find_key_recursive_mixed_types():
    data = {"a": {"b": [{"key": 1}, {"c": 2}]}}
    assert find_key_recursive(data, "key") == 1


def test_find_key_recursive_key_in_list():
    data = {"a": [{"key": 1}, {"b": 2}]}
    assert find_key_recursive(data, "key") == 1


def test_find_key_recursive_key_in_tuple():
    data = {"a": ({"key": 1}, {"b": 2})}
    assert find_key_recursive(data, "key") == 1
