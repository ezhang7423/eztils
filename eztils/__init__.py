"""eds utilities"""

import functools
import json
import sys

# import often used modules
from pathlib import Path

from tqdm.auto import tqdm

"""
Get and set version of module
"""

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


""""""


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


json.dumps = functools.partial(
    json.dumps, cls=EnhancedJSONEncoder
)  # monkey patch json.dumps to support dataclasses
json.dump = functools.partial(json.dump, cls=EnhancedJSONEncoder)
""""""


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
__version__ = version


from .default import *
