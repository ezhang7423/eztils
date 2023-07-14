"""eds torch stuff"""

import sys

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata

from pathlib import Path

from einops import rearrange, reduce

# import often used modules
from torch import einsum, nn
from torch.nn import functional as F
from torchvision.utils import save_image
from tqdm.auto import tqdm

from .utils import *


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
