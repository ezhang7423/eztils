import os
import pickle
import re
from multiprocessing import shared_memory
from pathlib import Path

import loguru
from varname import ImproperUseError, argname


def remove_shm_from_resource_tracker():
    """
    Monkey patch multiprocessing.resource_tracker so SharedMemory won't be tracked
    More details at: https://bugs.python.org/issue38119
    """
    # pylint: disable=protected-access, import-outside-toplevel
    # Ignore linting errors in this bug workaround hack
    from multiprocessing import resource_tracker

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return None
        return resource_tracker._resource_tracker.register(name, rtype)

    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return None
        return resource_tracker._resource_tracker.unregister(name, rtype)

    resource_tracker.unregister = fix_unregister
    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


# More details at: https://bugs.python.org/issue38119
remove_shm_from_resource_tracker()


def make_valid_python_var(varStr):
    return re.sub(r"\W|^(?=\d)", "_", varStr)


def save(var, folder=None, memory=False, save_fn=pickle.dump):
    folder = Path(folder or ".").absolute().resolve()
    try:
        varname = make_valid_python_var(argname("var"))
    except ImproperUseError:
        varname = "tensor"

    if not memory:
        fname = str(folder / f"{varname}.pt")
        loguru.logger.info(f"Saving {varname} to {fname}")
        save_fn(var, fname)
    else:
        script = f"""
        from multiprocessing import shared_memory
        handle = shared_memory.SharedMemory(name=\\"{varname}\\", create=False)
        breakpoint()
        """.strip().replace(
            "\n", ";"
        )
        var_bytes = pickle.dumps(var)
        try:
            # see if exists
            s = shared_memory.SharedMemory(name=varname)
            s.close()
            s.unlink()
        except FileNotFoundError:
            pass

        shm = shared_memory.SharedMemory(create=True, size=len(var_bytes), name=varname)

        shm.buf[:] = var_bytes
        loguru.logger.info(f"Saved {varname} to shared memory at /dev/shm/{varname}")

        # so that the shared memory is not deleted when the script exits, as a handle still exits
        os.system(f"""tmux new-session -d -s {varname}""")
        os.system(f"""tmux send-keys -t {varname} "python -c '{script}'" Enter""")


def load(path, memory=False, load_fn=pickle.load, **kwargs):
    if not memory:
        return load_fn(path, **kwargs)
    else:
        shm = shared_memory.SharedMemory(name=path, create=False)
        return pickle.loads(shm.buf)
