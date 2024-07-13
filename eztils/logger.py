""" 
This Python script defines a custom logger class Logger that extends from LoguruLogger. It includes a decorator requires_cfg to ensure the logger is configured before use.

The Logger class has methods to configure the logger (cfg), log data later (log_later), flush the logged data (flush), and close the logger (close).

The logger can log locally, to a file, or to Weights & Biases (wandb), depending on the configuration.

"""

from typing import Any, Dict, Optional

import dataclasses
import os
import sys
from collections import ChainMap
from io import StringIO

import torch
from loguru._logger import Core
from loguru._logger import Logger as LoguruLogger
from rich.console import Console
from rich.table import Table


@dataclasses.dataclass
class Config:
    wandb: bool = (
        False  # if toggled, this experiment will be tracked with Weights and Biases
    )
    wandb_project_name: str = "apple-picking-game"  # the wandb's project name
    wandb_entity: str = None  # the entity (team) of wandb's project
    log_locally: bool = True  # if toggled, log messages will be printed to stderr
    log_file: str = None  # the file to log to. relative to the Globals.LOG_DIR


def requires_cfg(func):
    def wrapper(self, *args, **kwargs):
        if not self.configured:
            raise ValueError(
                "Logger must be configured with `logger.cfg` before using this function"
            )
        return func(self, *args, **kwargs)

    return wrapper


def tensor_to_list(tensor):
    """
    Transforms a PyTorch tensor into a printable list.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        np.ndarray: The corresponding list.
    """
    if torch.is_tensor(tensor):
        # If tensor is on a GPU, move it to the CPU
        if tensor.is_cuda:
            tensor = tensor.cpu()
        # Detach the tensor from the computational graph and convert to list
        list_array = tensor.detach().tolist()
    else:
        raise TypeError("Input is not a PyTorch tensor.")

    return list_array


class Logger(LoguruLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configured = False

    def cfg(self, cfg: Config):
        self.log_wandb = cfg.wandb
        self.buffer = []
        self._flush_step = 0
        self.wandb_project = cfg.wandb_project_name
        self.log_locally = cfg.log_locally

        if self.log_wandb:
            import wandb

            self.wandb_module = wandb  # a hack to lazy load modules. this is unfortunate but necessary for faster imports
            if self.wandb_project is None:
                raise ValueError(
                    "wandb_project must be provided when log_wandb is True"
                )
            wandb.init(
                project=cfg.wandb_project_name, entity=cfg.wandb_entity, config=cfg
            )

        if not cfg.log_locally:
            self.remove()

        if cfg.log_file:
            if isinstance(cfg.log_file, bool):
                cfg.log_file = "output.log"

            log_directory = os.path.dirname(cfg.log_file)

            if not os.path.exists(log_directory):
                os.makedirs(log_directory)

            self.add(cfg.log_file, rotation="10 MB")  # what happens on rotation?
            print(os.path.abspath(cfg.log_file))

        self.configured = True

    @requires_cfg
    def log_later(
        self,
        data: Optional[Dict[str, Any]] = None,
        flush: bool = False,
    ):
        # Add to buffer
        self.buffer.append(data)
        if flush:
            self.flush(call_stack_depth=4)

    @requires_cfg
    def flush(self, call_stack_depth=2):
        data = dict(
            ChainMap(*self.buffer)
        )  # convert list of dicts to single dict, overwriting values with later dicts

        # Log to console by converting to table
        table = Table("Key", "Value", title=f"Log Step {self._flush_step}")

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = tensor_to_list(value)
            table.add_row(key, str(value))

        out = Console(file=StringIO())
        out.print(table)

        # need to log four levels above: flush, requires_cfg, log_later, requires_cfg
        self.opt(depth=call_stack_depth).info(f"\n{out.file.getvalue()}")

        if self.log_wandb:
            # Log to wandb
            self.wandb_module.log(data)

        # Clear the bufferk
        self.buffer.clear()
        self._flush_step += 1

    @requires_cfg
    def log_video(self, video_path):
        if self.log_wandb:
            self.wandb_module.log(
                {"video": self.wandb_module.Video(data_or_path=video_path)}
            )


# import wandb
# from loguru import Logger as LoguruLogger
# from loguru._logger import Core

# from harvest_sed.utils.__init__ import Config
logger = Logger(
    core=Core(),
    exception=None,
    depth=0,
    record=False,
    lazy=False,
    colors=False,
    raw=False,
    capture=True,
    patchers=[],
    extra={},
)

if sys.stderr:
    logger.add(sys.stderr)


if __name__ == "__main__":
    """
    Sample usage: logger is used to log a message, configure itself with a Config instance, and log another message with immediate flushing.
    """
    logger.info({"hi": "Logger cfg"})
    logger.cfg(Config())
    logger.log_later({"hi": "Logger cfg"}, flush=True)
