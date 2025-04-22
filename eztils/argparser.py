import dataclasses
import io
import json
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError
from copy import copy
from dataclasses import dataclass, field, fields, make_dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Literal,
    NewType,
    Optional,
    Tuple,
    Union,
    get_type_hints,
)

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


# Helper functions to identify types
def is_list_type(tp):
    origin = getattr(tp, "__origin__", None)
    return origin is list or origin is List


def is_dict_type(tp):
    origin = getattr(tp, "__origin__", None)
    return origin is dict or origin is Dict or tp is dict


def get_non_none_type(tp):
    if getattr(tp, "__origin__", None) is Union:
        args = tp.__args__
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0]
    return tp


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def string_to_dict(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        raise ArgumentTypeError(f"Could not parse '{s}' as JSON.")


def make_choice_type_function(choices: list) -> Callable[[str], Any]:
    """
    Creates a mapping function from each choices string representation to the actual value. Used to support multiple
    value types for a single argument.

    Args:
        choices (list): List of choices.

    Returns:
        Callable[[str], Any]: Mapping function from string representation to actual value for each choice.
    """
    str_to_choice = {str(choice): choice for choice in choices}
    return lambda arg: str_to_choice.get(arg, arg)


def update_dataclass_defaults(cls, instance):
    new_fields = []

    for f in fields(cls):
        if hasattr(instance, f.name):  # Update only if default is set
            instance_field = getattr(instance, f.name)

            # if the instance_field is already a Field, just set it directly
            if isinstance(instance_field, dataclasses.Field):
                f = instance_field
            else:
                # otherwise, set the default value to the primitive type
                f.default = instance_field

        new_fields.append((f.name, f.type, f))
    return make_dataclass(cls.__name__, new_fields)


class HfArgumentParser(ArgumentParser):
    """
    This subclass of `argparse.ArgumentParser` uses type hints on dataclasses to generate arguments.

    The class is designed to play well with the native argparse. In particular, you can add more (non-dataclass backed)
    arguments to the parser after initialization and you'll get the output back after parsing as an additional
    namespace. Optional: To create sub argument groups use the `_argument_group_name` attribute in the dataclass.
    """

    dataclass_types: Iterable[DataClassType]

    def __init__(
        self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs
    ):
        """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will "fill" instances with the parsed args.
            kwargs (`Dict[str, Any]`, *optional*):
                Passed to `argparse.ArgumentParser()` in the regular way.
        """
        # To make the default appear when using --help
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = ArgumentDefaultsHelpFormatter
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = list(dataclass_types)
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _parse_dataclass_field(
        self, parser: ArgumentParser, field: dataclasses.Field, prefix: str = ""
    ):
        field_name = f"--{prefix}{field.name}"
        dest = prefix + field.name
        kwargs = field.metadata.copy()
        # field.metadata is not used at all by Data Classes,
        # it is provided as a third-party extension mechanism.
        if isinstance(field.type, str):
            raise RuntimeError(
                "Unresolved type detected, which should have been done with the help of "
                "`typing.get_type_hints` method by default"
            )

        aliases = kwargs.pop("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]

        # Adjust aliases with prefix
        aliases = [f"--{prefix}{alias.lstrip('--')}" for alias in aliases]

        field_type = field.type
        origin_type = getattr(field_type, "__origin__", field_type)

        # Handle Optional types
        if origin_type is Union:
            args = field_type.__args__
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                field_type = non_none_args[0]
                origin_type = getattr(field_type, "__origin__", field_type)
            else:
                raise ValueError(f"Unsupported Union type in field '{field.name}'")

        # A variable to store kwargs for a boolean field, if needed
        # so that we can init a `no_*` complement argument (see below)
        bool_kwargs = {}

        if origin_type is Literal or (
            isinstance(field_type, type) and issubclass(field_type, Enum)
        ):
            # Handle Literal and Enum types
            if origin_type is Literal:
                kwargs["choices"] = field_type.__args__
            else:
                kwargs["choices"] = [x.value for x in field_type]

            kwargs["type"] = make_choice_type_function(kwargs["choices"])

            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            else:
                kwargs["required"] = True
        elif field_type is bool:
            # Handle bool types
            # Copy the current kwargs to use to instantiate a `no_*` complement argument below.
            # We do not initialize it here because the `no_*` alternative must be instantiated after the real argument
            bool_kwargs = copy(kwargs)

            # Hack because type=bool in argparse does not behave as we want.
            kwargs["type"] = string_to_bool
            if field.default is not None and field.default is not dataclasses.MISSING:
                # Default value is False if we have no default when of type bool.
                default = field.default
                # This is the value that will get picked if we don't include --field_name in any way
                kwargs["default"] = default
                # This tells argparse we accept 0 or 1 value after --field_name
                kwargs["nargs"] = "?"
                # This is the value that will get picked if we do --field_name (without value)
                kwargs["const"] = True
        elif is_list_type(field_type):
            # Handle lists
            kwargs["type"] = field_type.__args__[0]
            kwargs["nargs"] = "+"
            if field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            elif field.default is dataclasses.MISSING:
                kwargs["required"] = True
        elif is_dict_type(field_type):
            # Handle dicts
            kwargs["type"] = string_to_dict
            if field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            elif field.default is dataclasses.MISSING:
                kwargs["required"] = True
        else:
            # Handle other types
            kwargs["type"] = field_type
            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            else:
                kwargs["required"] = True

        parser.add_argument(field_name, *aliases, dest=dest, **kwargs)

        # Handle boolean --no_* flag
        if field.default is True and field_type is bool:
            bool_kwargs["default"] = False
            parser.add_argument(
                f"--no_{prefix}{field.name}",
                action="store_false",
                dest=dest,
                **bool_kwargs,
            )

    def _add_dataclass_arguments(self, dtype: DataClassType, prefix: str = ""):
        if hasattr(dtype, "_argument_group_name"):
            parser = self.add_argument_group(dtype._argument_group_name)
        else:
            parser = self
        self._add_dataclass_fields(parser, dtype, prefix=prefix)

    def _add_dataclass_fields(
        self, parser: ArgumentParser, dtype: DataClassType, prefix: str = ""
    ):
        try:
            type_hints: Dict[str, type] = get_type_hints(dtype)
        except NameError:
            raise RuntimeError(
                f"Type resolution failed for {dtype}. Try declaring the class in global scope or "
                "removing line of `from __future__ import annotations` which opts in Postponed "
                "Evaluation of Annotations (PEP 563)"
            )
        except TypeError as ex:
            # Remove this block when we drop Python 3.9 support
            if sys.version_info[:2] < (
                3,
                10,
            ) and "unsupported operand type(s) for |" in str(ex):
                python_version = ".".join(map(str, sys.version_info[:3]))
                raise RuntimeError(
                    f"Type resolution failed for {dtype} on Python {python_version}. Try removing "
                    "line of `from __future__ import annotations` which opts in union types as "
                    "`X | Y` (PEP 604) via Postponed Evaluation of Annotations (PEP 563). To "
                    "support Python versions that lower than 3.10, you need to use "
                    "`typing.Union[X, Y]` instead of `X | Y` and `typing.Optional[X]` instead of "
                    "`X | None`."
                ) from ex
            raise

        for dataclass_field in dataclasses.fields(dtype):
            if not dataclass_field.init:
                continue
            dataclass_field.type = type_hints[dataclass_field.name]
            field_name = f"{prefix}{dataclass_field.name}"

            # Check if the field.type is a dataclass
            if dataclasses.is_dataclass(dataclass_field.type):
                # Recursively add arguments for this dataclass
                self._add_dataclass_fields(
                    parser, dataclass_field.type, prefix=field_name + "_"
                )
            else:
                # Process the field as usual
                self._parse_dataclass_field(parser, dataclass_field, prefix=prefix)

    def parse_args_into_dataclasses(
        self,
        args=None,
        return_remaining_strings=False,
        look_for_args_file=True,
        args_filename=None,
        args_file_flag=None,
    ) -> Tuple[DataClass, ...]:
        """
        Parse command-line args into instances of the specified dataclass types.

        This relies on argparse's `ArgumentParser.parse_known_args`. See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args

        Args:
            args:
                List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser)
            return_remaining_strings:
                If true, also return a list of remaining argument strings.
            look_for_args_file:
                If true, will look for a ".args" file with the same base name as the entry point script for this
                process, and will append its potential content to the command line args.
            args_filename:
                If not None, will uses this file instead of the ".args" file specified in the previous argument.
            args_file_flag:
                If not None, will look for a file in the command-line args specified with this flag. The flag can be
                specified multiple times and precedence is determined by the order (last one wins).

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.abspath
                - if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args)
        """

        if args_file_flag or args_filename or (look_for_args_file and len(sys.argv)):
            args_files = []

            if args_filename:
                args_files.append(Path(args_filename))
            elif look_for_args_file and len(sys.argv):
                args_files.append(Path(sys.argv[0]).with_suffix(".args"))

            # args files specified via command line flag should overwrite default args files so we add them last
            if args_file_flag:
                # Create special parser just to extract the args_file_flag values
                args_file_parser = ArgumentParser()
                args_file_parser.add_argument(args_file_flag, type=str, action="append")

                # Use only remaining args for further parsing (remove the args_file_flag)
                cfg, args = args_file_parser.parse_known_args(args=args)
                cmd_args_file_paths = vars(cfg).get(args_file_flag.lstrip("-"), None)

                if cmd_args_file_paths:
                    args_files.extend([Path(p) for p in cmd_args_file_paths])

            file_args = []
            for args_file in args_files:
                if args_file.exists():
                    file_args += args_file.read_text().split()

            # in case of duplicate arguments the last one has precedence
            # args specified via the command line should overwrite args from files, so we add them last
            args = file_args + args if args is not None else file_args + sys.argv[1:]
        namespace, remaining_args = self.parse_known_args(args=args)
        data = vars(namespace)
        outputs = []
        for dtype in self.dataclass_types:
            obj = self._create_dataclass_instance(dtype, data)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(
                    f"Some specified arguments are not used by the HfArgumentParser: {remaining_args}"
                )

            return (*outputs,)

    def _create_dataclass_instance(self, dtype, data, prefix=""):
        kwargs = {}
        for dataclass_field in dataclasses.fields(dtype):
            field_name = prefix + dataclass_field.name
            if dataclasses.is_dataclass(dataclass_field.type):
                # Nested dataclass
                value = self._create_dataclass_instance(
                    dataclass_field.type, data, prefix=field_name + "_"
                )
                kwargs[dataclass_field.name] = value
            else:
                if field_name in data:
                    kwargs[dataclass_field.name] = data[field_name]
                else:
                    if dataclass_field.default is not dataclasses.MISSING:
                        kwargs[dataclass_field.name] = dataclass_field.default
                    elif dataclass_field.default_factory is not dataclasses.MISSING:
                        kwargs[dataclass_field.name] = dataclass_field.default_factory()
                    else:
                        raise ValueError(f"Missing value for field {field_name}")
        return dtype(**kwargs)

    def parse_dict(
        self, args: Dict[str, Any], allow_extra_keys: bool = False
    ) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead uses a dict and populating the dataclass
        types.

        Args:
            args (`dict`):
                dict containing config values
            allow_extra_keys (`bool`, *optional*, defaults to `False`):
                Defaults to False. If False, will raise an exception if the dict contains keys that are not parsed.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.
        """
        unused_keys = set(args.keys())
        outputs = []
        for dtype in self.dataclass_types:
            obj = self._create_dataclass_instance_from_dict(dtype, args)
            outputs.append(obj)
            used_keys = self._get_used_keys(dtype, args)
            unused_keys.difference_update(used_keys)
        if not allow_extra_keys and unused_keys:
            raise ValueError(
                f"Some keys are not used by the HfArgumentParser: {sorted(unused_keys)}"
            )
        return tuple(outputs)

    def _create_dataclass_instance_from_dict(self, dtype, data):
        kwargs = {}
        for dataclass_field in dataclasses.fields(dtype):
            if dataclasses.is_dataclass(dataclass_field.type):
                # Nested dataclass
                value = self._create_dataclass_instance_from_dict(
                    dataclass_field.type, data.get(dataclass_field.name, {})
                )
                kwargs[dataclass_field.name] = value
            else:
                if dataclass_field.name in data:
                    kwargs[dataclass_field.name] = data[dataclass_field.name]
                else:
                    if dataclass_field.default is not dataclasses.MISSING:
                        kwargs[dataclass_field.name] = dataclass_field.default
                    elif dataclass_field.default_factory is not dataclasses.MISSING:
                        kwargs[dataclass_field.name] = dataclass_field.default_factory()
                    else:
                        raise ValueError(
                            f"Missing value for field {dataclass_field.name}"
                        )
        return dtype(**kwargs)

    def _get_used_keys(self, dtype, data):
        used_keys = set()
        for field_ in dataclasses.fields(dtype):
            if dataclasses.is_dataclass(field_.type):
                if field_.name in data:
                    nested_keys = self._get_used_keys(field_.type, data[field_.name])
                    used_keys.update({f"{field_.name}.{k}" for k in nested_keys})
            else:
                if field_.name in data:
                    used_keys.add(field_.name)
        return used_keys

    def parse_json_file(
        self, json_file: str, allow_extra_keys: bool = False
    ) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.

        Args:
            json_file (`str` or `os.PathLike`):
                File name of the json file to parse
            allow_extra_keys (`bool`, *optional*, defaults to `False`):
                Defaults to False. If False, will raise an exception if the json file contains keys that are not
                parsed.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.
        """
        with open(Path(json_file), encoding="utf-8") as open_json_file:
            data = json.loads(open_json_file.read())
        outputs = self.parse_dict(data, allow_extra_keys=allow_extra_keys)
        return tuple(outputs)

    def _to(
        self,
        serializer: Callable[[Dict, io.TextIOWrapper], None],
        args: Tuple[DataClass, ...],
        file_path: str,
    ):
        ret = {}
        for arg in args:
            ret.update(dataclasses.asdict(arg))
        with open(file_path, "w") as fout:
            serializer(ret, fout)

    def to_json(self, args: Tuple[DataClass, ...], file_path: str):
        """
        Serializes the given dataclass instances into a JSON file.

        This method converts dataclass instances to a dictionary using `dataclasses.asdict`,
        and then serializes this dictionary to a JSON file specified by `file_path`.

        Args:
            args (Tuple[DataClass, ...]): A tuple containing instances of dataclasses. These instances
                are intended to be the ones that have been filled by parsing command-line arguments or by
                another method of initialization.
            file_path (str): The file path where the output JSON should be saved. If the file already
                exists, it will be overwritten.

        Returns:
            None. The output is written to a file specified by `file_path`.
        """
        return self._to(json.dump, args, file_path)


def HfArg(
    *,
    aliases: Union[str, List[str]] = None,
    help: str = None,
    default: Any = dataclasses.MISSING,
    default_factory: Callable[[], Any] = dataclasses.MISSING,
    metadata: dict = None,
    **kwargs,
) -> dataclasses.Field:
    """Argument helper enabling a concise syntax to create dataclass fields for parsing with `HfArgumentParser`.

    Example comparing the use of `HfArg` and `dataclasses.field`:
    ```
    @dataclass
    class Args:
        regular_arg: str = dataclasses.field(default="Huggingface", metadata={"aliases": ["--example", "-e"], "help": "This syntax could be better!"})
        hf_arg: str = HfArg(default="Huggingface", aliases=["--example", "-e"], help="What a nice syntax!")
    ```

    Args:
        aliases (Union[str, List[str]], optional):
            Single string or list of strings of aliases to pass on to argparse, e.g. `aliases=["--example", "-e"]`.
            Defaults to None.
        help (str, optional): Help string to pass on to argparse that can be displayed with --help. Defaults to None.
        default (Any, optional):
            Default value for the argument. If not default or default_factory is specified, the argument is required.
            Defaults to dataclasses.MISSING.
        default_factory (Callable[[], Any], optional):
            The default_factory is a 0-argument function called to initialize a field's value. It is useful to provide
            default values for mutable types, e.g. lists: `default_factory=list`. Mutually exclusive with `default=`.
            Defaults to dataclasses.MISSING.
        metadata (dict, optional): Further metadata to pass on to `dataclasses.field`. Defaults to None.

    Returns:
        Field: A `dataclasses.Field` with the desired properties.
    """
    if metadata is None:
        # Important, don't use as default param in function signature because dict is mutable and shared across function calls
        metadata = {}
    if aliases is not None:
        metadata["aliases"] = aliases
    if help is not None:
        metadata["help"] = help

    return dataclasses.field(
        metadata=metadata, default=default, default_factory=default_factory, **kwargs
    )


if __name__ == "__main__":
    # Assuming HfArgumentParser is defined as in your code
    # If it's imported from somewhere else, you might need to adjust the import
    # For this example, we will use the HfArgumentParser as defined in your code snippet.

    from dataclasses import dataclass, field
    from typing import ClassVar, Optional

    # ────────────────────────────────────────────────────────────────────────────────
    # 1.  NEW  nested structure
    # ────────────────────────────────────────────────────────────────────────────────
    @dataclass
    class OptimizerConfig:
        # (no group name → inherits the parent’s group)
        learning_rate: float = 5e-4
        beta1: float = 0.9
        beta2: float = 0.999

    @dataclass
    class TrainingArguments:
        _argument_group_name: ClassVar[str] = "Training Arguments"

        epochs: int = 3
        batch_size: int = 32
        # nested dataclass ➜ CLI flags will be --optimizer_learning_rate, --optimizer_beta1, …
        optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # ────────────────────────────────────────────────────────────────────────────────
    # 2.  Existing top‑level structures (unchanged)
    # ────────────────────────────────────────────────────────────────────────────────
    @dataclass
    class ModelArguments:
        _argument_group_name: ClassVar[str] = "Model Arguments"

        model_name: str = "bert-base-uncased"
        config_name: Optional[str] = None
        cache_dir: Optional[str] = None

    @dataclass
    class DataArguments:
        _argument_group_name: ClassVar[str] = "Data Arguments"

        dataset_name: str = "squad"
        max_seq_length: int = 384
        overwrite_cache: bool = False

    # ────────────────────────────────────────────────────────────────────────────────
    # 3.  Driver
    # ────────────────────────────────────────────────────────────────────────────────
    def main():
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, train_args, extras = parser.parse_args_into_dataclasses()

        print("\nModel Args:", model_args)
        print("Data  Args:", data_args)
        print("Train Args (with nested Optimizer):", train_args)

        # serialize everything
        parser.to_json((model_args, data_args, train_args), "config.json")
        
        print(parser.parse_json_file("config.json", allow_extra_keys=True))

    main()
