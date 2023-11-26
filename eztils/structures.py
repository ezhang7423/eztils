class AttriDict(dict):  # type: ignore
    """
    A dictionary subclass that allows accessing keys as attributes.

    This class inherits from the built-in `dict` class and overrides the `__getattr__` and `__setattr__` methods to
    allow accessing keys as attributes. It also provides a `copy` method that returns a shallow copy of the instance.

    Note that attribute names that conflict with built-in dictionary methods or attributes cannot be used.

    Example usage:
    ```
    my_dict = AttriDict({'foo': 'bar'})
    print(my_dict.foo)  # prints 'bar'
    my_dict.baz = 'qux'
    print(my_dict['baz'])  # prints 'qux'
    ```

    :param dict: Initial dictionary to populate the instance with.
    :type dict: dict
    :raises AttributeError: If an attribute name conflicts with a built-in dictionary method or attribute.
    :return: An instance of the `AttriDict` class.
    :rtype: AttriDict
    """

    DICT_RESERVED_KEYS = list(vars(dict).keys())

    def __init__(self, *args, **kwargs):
        """
        Initializes an instance of the `AttriDict` class.

        :param args: Dictionaries to merge into the instance. e.g. ({}, {}, ..)
        :type args: dict
        :param kwargs: Key-value pairs to add to the instance.
        :type kwargs: Any
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, attr):
        """
        Gets the value of a key as an attribute.

        :param attr: The name of the attribute to get.
        :type attr: str
        :return: The value of the key.
        :rtype: Any
        """
        if attr not in AttriDict.DICT_RESERVED_KEYS:
            return self.get(attr)
        return getattr(self, attr)

    def __setattr__(self, key, value):
        """
        Sets the value of a key as an attribute.

        :param key: The name of the key to set.
        :type key: str
        :param value: The value to set the key to.
        :type value: Any
        :raises AttributeError: If the key name conflicts with a built-in dictionary method or attribute.
        """
        if key == "__dict__":
            super().__setattr__(key, value)
            return
        if key in AttriDict.DICT_RESERVED_KEYS:
            raise AttributeError("You cannot set a reserved name as attribute")
        self.__setitem__(key, value)

    def __copy__(self):
        """
        Returns a shallow copy of the instance.

        :return: A shallow copy of the instance.
        :rtype: AttriDict
        """
        return self.__class__(self)

    def copy(self):
        """
        Returns a shallow copy of the instance.

        :return: A shallow copy of the instance.
        :rtype: AttriDict
        """
        return self.__copy__()
