from datetime import datetime

import numpy as np
from rich.console import Console
from rich.tree import Tree

console = Console()


def print_creator(color):
    """
    Returns a function that prints text in the specified color.

    :param color: The color to use for printing.
    :type color: str
    :return: A function that prints text in the specified color.
    :rtype: function
    """

    def print_creator_inner(*args, **kwargs):
        console.print(*args, **kwargs, style=color)

    return print_creator_inner


red = print_creator("red")
green = print_creator("green")
bold = print_creator("bold")
blue = print_creator("blue")
purple = print_creator("purple")
orange = print_creator("orange")


def inspect(obj, list_expand=3):
    """
    Inspects the given object and prints a tree representation of its attributes and values.

    :param obj: The object to inspect.
    :type obj: object
    :param list_expand: The maximum number of elements to expand in a list or numpy array, defaults to 3.
    :type list_expand: int, optional
    :return: None
    :rtype: None
    """
    try:
        import torch

        torch_exists = True
    except ImportError:
        torch_exists = False

    console = Console(record=True)

    class TreeWrapper:
        def __init__(self) -> None:
            self.tree = None

        def add(self, label):
            if self.tree is None:
                self.tree = Tree(label)  # type: ignore
                return self.tree
            else:
                return self.tree.add(label)  # type: ignore

    def type_fmt(obj):
        """
        Extracts the type name from an object's type and returns it as a string.

        :param obj: The object whose type name is to be extracted.
        :type obj: Any
        :return: The name of the object's type.
        :rtype: str
        """
        # get string within single quotes regex
        import re

        m = re.search(r"'(.*?)'", str(type(obj)))
        if m is not None:
            return m.group(1)
        return None

    def add_children(obj: object, root: Tree) -> None:
        """
        Recursively adds child nodes to a given root node based on the type of the input object.

        :param obj: The object to add child nodes for.
        :type obj: object
        :param root: The root node to add child nodes to.
        :type root: Tree
        """
        if isinstance(obj, dict):
            dict_node = root.add("dict")
            for key in obj.keys():
                add_children(obj[key], dict_node.add(f"[bold]{key}[/bold]"))
        elif isinstance(obj, list) or isinstance(obj, np.ndarray):
            list_node = root.add(f"{type_fmt(obj)}[{len(obj)}]")
            if len(obj) == 0:
                return
            for i in range(min(list_expand, len(obj))):
                add_children(obj[i], list_node)
            if len(obj) > list_expand:
                list_node.add("...")
        elif torch_exists and isinstance(obj, torch.Tensor):
            root.add(f"tensor[[blue]{tuple(obj.shape)}; {obj.dtype}[/blue]]")
        else:
            repr = str(obj)
            if len(repr) > 100:
                repr = repr[:100] + "..."
            root.add(f"{type_fmt(obj)}: [blue]{repr}[/blue]")

    treewrapper = TreeWrapper()
    add_children(obj, treewrapper)  # type: ignore
    console.print(treewrapper.tree)
