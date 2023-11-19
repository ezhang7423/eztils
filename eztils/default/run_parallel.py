# %%

import os
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
from art import tprint
from ezlogging import datestr
from rich import print


def get_user_defined_attrs(cls) -> list:
    return [
        attr
        for attr in dir(cls)
        if not callable(getattr(cls, attr)) and not attr.startswith("__")
    ]


class BaseHyperParameters:
    @classmethod
    def get_product(cls):
        return list(
            product(*[getattr(cls, attr) for attr in get_user_defined_attrs(cls)])
        )


# %%


def run_parallel(
    hparam_cls: BaseHyperParameters,
    use_cuda_visible_devices: bool = False,
    base_cmd: str = "python3 scripts/eval.py",
    typer_style: bool = True,
    data_path: str = "./",
    sleep_time: int = 5,
):
    """
    # example
    class HyperParameters(BaseHyperParameters):
        frame_offset = [-1]
        gen_steps = [-1]
        seeds = [12, 13, 42]
        epochs = [300000, 400000, 490000]



    :param hparam_cls: _description_
    :type hparam_cls: BaseHyperParameters
    :param use_cuda_visible_devices: _description_, defaults to False
    :type use_cuda_visible_devices: bool, optional
    :param base_cmd: _description_, defaults to 'python3 scripts/eval.py'
    :type base_cmd: str, optional
    """
    hparams = hparam_cls.get_product()
    attrs = get_user_defined_attrs(hparam_cls)
    tprint("Run Parallel", font="bigchief")
    print("Starting at", datestr(), "\n\n")

    # print a nice table
    from rich.table import Table

    table = Table(title="Hyperparameters")

    table.add_column("")
    for attr in attrs:
        table.add_column(attr)

    # Add rows to the table
    for i, values in enumerate(hparams):
        table.add_row(str(i), *[str(value) for value in values])

    # print(pd.DataFrame(hparams, columns=attrs))
    print(table)

    # Assuming d_count and base_cmd are defined elsewhere
    if use_cuda_visible_devices:
        import torch

        d_count = torch.cuda.device_count()

        base_cmd = f"CUDA_VISIBLE_DEVICES=(i) {base_cmd}"  # Placeholder

    for i, values in enumerate(hparams):
        args = {attr: value for attr, value in zip(attrs, values)}

        # Include additional command-specific options
        if use_cuda_visible_devices:
            cmd = base_cmd.replace("(i)", str(i % d_count))
        else:
            cmd = base_cmd

        # Append hyperparameters to the command
        for arg, value in args.items():
            if typer_style:
                if isinstance(value, bool):
                    cmd += f" --{'no-' if value else ''}{arg}"
                else:
                    cmd += f" --{arg} {value}"
            else:
                cmd += f" --{arg}={value}"

        fout = f'{datetime.now().strftime("%b_%d")}_{i}.out'
        data_path = Path(data_path)
        data_path.mkdir(exist_ok=True, parents=True)
        cmd += f" > {data_path / fout} 2>&1 &"

        print(f"Running {i}: {cmd}")
        time.sleep(sleep_time)
        os.system(cmd)


if __name__ == "__main__":

    class HyperParameters(BaseHyperParameters):
        frame_offset = [-1]
        gen_steps = [-1]
        seeds = [12, 13, 42]
        epochs = [300000, 400000, 490000]

    run_parallel(
        HyperParameters,
        use_cuda_visible_devices=False,
        typer_style=True,
        data_path="./data",
        sleep_time=0,
    )
