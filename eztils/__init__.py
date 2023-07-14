"""eds utilities"""

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


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
__version__ = version


from .default import *
