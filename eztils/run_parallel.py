# %%

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from subprocess import Popen

from art import tprint
from rich import print
from rich.table import Table


def get_user_defined_attrs(cls) -> list:
    return [
        attr
        for attr in dir(cls)
        if not callable(getattr(cls, attr)) and not attr.startswith("__")
    ]


def prod(*args, **kwargs):
    return list(product(*args, **kwargs))


class BaseHyperParameters:
    @classmethod
    def get_product(cls):
        return prod(*[getattr(cls, attr) for attr in get_user_defined_attrs(cls)])


# %%
def kill_processes(processes):
    for process in processes:
        try:
            # Sends SIGTERM to the entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            # Optionally, you can wait for the processes to ensure they have stopped
            process.wait(timeout=5)
        except Exception as e:
            print(f"Failed to terminate process group {process.pid}: {e}")


def calculate_split(total_splits, total_len, index):
    assert (
        0 <= index < total_splits
    ), f"Index {index} out of bounds for {total_splits} splits."
    assert total_splits > 0, "Total splits must be greater than 0."
    assert total_len > 0, "Total length must be greater than 0."

    if total_len < total_splits:  # if there are more splits than work
        if index < total_len:
            return index, index + 1
        return 0, 0  # give no work

    # divide as fairly as possible
    if (total_len / total_splits) % 1 > 0.5:
        # Calculate the length of each split by ceiling
        split_length = -(total_len // -total_splits)
    else:
        # Calculate the length of each split by floor
        split_length = total_len // total_splits

    # Calculate the start and end indices of the split
    start_index = index * split_length
    end_index = start_index + split_length

    if start_index >= total_len:
        return 0, 0

    # Adjust the end index if the split is not evenly divided
    if index == total_splits - 1 or end_index > total_len:
        end_index = total_len

    return start_index, end_index


def run_parallel(
    hparam_cls: BaseHyperParameters,
    use_cuda_visible_devices: bool = False,
    base_cmd: str = "python3 scripts/eval.py",
    typer_style: bool = True,
    data_path: str = "./runs",
    sleep_time: int = 5,
    debug: bool = False,
) -> list:
    """
    Run parallel processes with different hyperparameters.

    :return: List of spawned processes
    """
    from eztils import datestr

    """
    Handle signals
    """
    processes = []  # Store the process IDs

    # Define your signal handler function
    def signal_handler(signum, frame):
        print(f"Received signal: {signal.Signals(signum).name}")
        kill_processes(processes)
        sys.exit(1)  # or any appropriate exit code

    # Register the handlers within the function
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)
    """
    End handle signals
    """

    hparams = hparam_cls.get_product()
    attrs = get_user_defined_attrs(hparam_cls)
    tprint("Run Parallel", font="bigchief")
    print("Starting at", datestr(), "\n\n")

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

    for i, values in enumerate(hparams):
        args = {attr: value for attr, value in zip(attrs, values)}

        device_id = str(i % d_count) if use_cuda_visible_devices else ""
        cuda_cmd = f"CUDA_VISIBLE_DEVICES={device_id}" if device_id else ""

        # Append hyperparameters to the command
        full_cmd_args = []
        for arg, value in args.items():
            if typer_style:
                hyphen_arg = arg.replace("_", "-")
                if isinstance(value, bool):
                    arg_str = f"--{'no-' if not value else ''}{hyphen_arg}"
                else:
                    arg_str = f"--{hyphen_arg} {value}"
            else:
                arg_str = f"--{arg}={value}"

            full_cmd_args.append(arg_str)

        full_cmd = f"{cuda_cmd} {base_cmd} {' '.join(full_cmd_args)}"
        fout = f'{datetime.now().strftime("%b_%d")}_{i}.out'
        data_path = Path(data_path).resolve()
        data_path.mkdir(exist_ok=True, parents=True)
        output_file = data_path / fout
        print(f"Running {i}: {full_cmd} > {output_file}")

        if not debug:
            time.sleep(sleep_time)
            with open(output_file, "w") as fout:
                process = Popen(
                    full_cmd,
                    stdout=fout,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                    shell=True,
                )
            processes.append(process)  # this isn't working properly...TODO Fix
            print("pids", " ".join([str(p.pid) for p in processes]))

    return processes


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
