from typing import Any, Dict, MutableMapping


class AttriDict(dict):  # type: ignore
    """
    A dict which is accessible via attribute dot notation
    https://stackoverflow.com/a/41514848
    https://stackoverflow.com/a/14620633
    """

    DICT_RESERVED_KEYS = list(vars(dict).keys())

    def __init__(self, *args, **kwargs):
        """
        :param args: multiple dicts ({}, {}, ..)
        :param kwargs: arbitrary keys='value'
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, attr):
        if attr not in AttriDict.DICT_RESERVED_KEYS:
            return self.get(attr)
        return getattr(self, attr)

    def __setattr__(self, key, value):
        if key == "__dict__":
            super().__setattr__(key, value)
            return
        if key in AttriDict.DICT_RESERVED_KEYS:
            raise AttributeError("You cannot set a reserved name as attribute")
        self.__setitem__(key, value)

    def __copy__(self):
        return self.__class__(self)

    def copy(self):
        return self.__copy__()
