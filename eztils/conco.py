import asyncio
import os
import pickle
import traceback

import asyncio
import os
import pickle
import traceback
import time
from datetime import datetime, timedelta
import inspect


def where_defined(fn):
    # unwrap decorators that used functools.wraps
    f = inspect.unwrap(fn)
    if hasattr(f, "__code__"):
        return {
            "file": f.__code__.co_filename,
            "line": f.__code__.co_firstlineno,
            "module": f.__module__,
        }
    # e.g. builtins or C extensions
    return {"file": None, "line": None, "module": getattr(f, "__module__", None)}


async def progress_watcher(
    fn,
    total: int,
    results: list,
    cache_dir: str = None,
    poll_interval: float = 1.0,
    stalled_timeout: float = 600.0,
    silent: bool = False,
):
    """
    Periodically checks completion progress, either by counting:
      - How many pickle files appear in cache_dir (if provided), or
      - How many entries in `results` are no longer 'in_progress' (if no cache_dir).

    Additionally:
      - Prints current local date/time in a pretty format
      - Prints elapsed time (HH:MM:SS)
      - Computes a naive ETA based on average completion rate

    Now also exits early if no progress (increase in completed count)
    occurs for longer than `stalled_timeout` seconds.
    """

    start_time = time.time()

    # Track when we last saw the completion count increase
    last_completed_count = 0
    last_progress_time = start_time

    while True:
        # Compute how many tasks are complete
        if cache_dir:
            if not os.path.isdir(cache_dir):
                completed_count = 0
            else:
                completed_count = sum(
                    f.startswith("result_") and f.endswith(".pkl") for f in os.listdir(cache_dir)
                )
        else:
            completed_count = sum(r != "in_progress" for r in results)

        # If completion count increased, update our "last progress" records
        if completed_count > last_completed_count:
            last_completed_count = completed_count
            last_progress_time = time.time()

        # Calculate elapsed time
        elapsed_seconds = time.time() - start_time
        elapsed_td = timedelta(seconds=int(elapsed_seconds))  # Convert to hh:mm:ss

        # Calculate tasks per second (avoid dividing by zero)
        if elapsed_seconds > 0 and completed_count > 0:
            tasks_per_sec = completed_count / elapsed_seconds
        else:
            tasks_per_sec = 0

        # Estimate time remaining
        remaining = total - completed_count
        if tasks_per_sec > 0:
            eta_seconds = remaining / tasks_per_sec
            eta_td = timedelta(seconds=int(eta_seconds))
        else:
            eta_td = timedelta(seconds=0)  # or "unknown"

        # Current local datetime in a nice format
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        fn_defn = where_defined(fn)
        if not silent:
            print(
                f"[{now_str}] "
                f"{fn_defn['file']}:{fn_defn['line']}, "
                f"Completed {completed_count}/{total} tasks, "
                f"Elapsed: {elapsed_td}, "
                f"ETA: {eta_td}"
            )

        # Check if all tasks completed
        if completed_count >= total:
            break

        # Check if we've stalled too long
        if stalled_timeout is not None:
            time_since_progress = time.time() - last_progress_time
            if time_since_progress > stalled_timeout:
                if not silent:
                    print(
                        f"Stalled for {time_since_progress:.1f}s (limit={stalled_timeout}s); exiting."
                    )
                break

        await asyncio.sleep(poll_interval)


async def async_concurrency(
    fn,  # The async function to call
    list_=None,  # The list of params to iterate over
    max_workers=int(1e5),  # Total number of tasks if list_ is not provided
    ignore_exceptions=True,
    timeout=480,
    sync_fn=False,
    cache_dir=None,  # Directory to cache results
    return_cache=False,  # just return what's in the cache
    filter_none=False,
    stop_after_frac: float | str | None = None,
    retry_none: bool = True,
    silent: bool = False,
):
    """
    Runs `fn` concurrently on items from list_.

    - If `cache_dir` is provided, results are stored and/or read from .pkl files named `result_{i}.pkl`.
    - If `cache_dir` is not provided, results remain in memory, but the same progress watcher will print progress.
    - This version also prints time elapsed and a naive ETA in the progress watcher.
    - Set `silent=True` to suppress console output.

    TODO: add a meetadata.pkl that captures args and "fn" code
    """
    if list_ is None:
        list_ = ["in_progress"] * max_workers

    total_items = len(list_)
    results = ["in_progress"] * total_items

    if stop_after_frac is not None:
        if stop_after_frac == "auto":
            if len(list_) < 1000:
                stop_after_frac = None
            if len(list_) < 10_000:
                stop_after_frac = 0.99
            else:
                stop_after_frac = 0.9  # could be 0.95 too

        if not (0.0 < stop_after_frac <= 1.0):
            raise ValueError("stop_after_frac must be in (0, 1].")
        if stop_after_frac == 1.0:
            stop_after_frac = None

    # Create the cache directory if needed
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_dir = os.path.abspath(cache_dir)
        if not silent:
            print(f"Results will be cached in {cache_dir}")

    semaphore = asyncio.Semaphore(max_workers)

    # Define an internal function that each task will run
    async def run_task(i, p):
        cache_file = None
        if cache_dir:
            cache_file = os.path.join(cache_dir, f"result_{i}.pkl")

        # If there's a cached result, try to load it
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    return i, pickle.load(f)
            except Exception as e:
                if not silent:
                    print(
                        f"Error loading cached result for index {i}: {e} - {'Recomputing' if not return_cache else ''}."
                    )

        if cache_dir and return_cache:
            return i, None

        async with semaphore:
            # if p is not iterable, make it a single tuple
            if not isinstance(p, (list, tuple)):
                p = (p,)
            try:
                if sync_fn:
                    # If fn is sync, run it in a thread
                    result = await asyncio.to_thread(fn, *p)
                else:
                    # Otherwise, call fn as an async function
                    result = await asyncio.wait_for(fn(*p), timeout=timeout)

                # Save to cache (if applicable)
                if cache_file:
                    try:
                        with open(cache_file, "wb") as f:
                            pickle.dump(result, f)
                    except Exception as e:
                        if not silent:
                            print(f"Error writing result to cache for index {i}: {e}")

            except asyncio.TimeoutError as e:
                if not silent:
                    print(f"TimeoutError occurred for index {i}: {e}")
                result = None
            except Exception as e:
                if ignore_exceptions:
                    if not silent:
                        print(f"Exception for index {i}:\n{traceback.format_exc()}")
                    result = None
                else:
                    # Raise for the top-level caller to handle
                    raise e

            return i, result

    # Kick off the progress watcher (works with or without cache_dir)
    progress_task = asyncio.create_task(
        progress_watcher(
            fn,
            total_items,
            results,
            cache_dir,
            silent=silent,
        )
    )

    # Create tasks for each item
    if retry_none and cache_dir:
        # go ahead and remove all None pkl results
        for i in range(total_items):
            cache_file = os.path.join(cache_dir, f"result_{i}.pkl")
            if cache_file and os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        res = pickle.load(f)
                    if res is None:
                        os.remove(cache_file)
                except Exception as e:
                    if not silent:
                        print(f"Error reading cached result for index {i}: {e} - removing file.")
                    os.remove(cache_file)
        
    tasks = [asyncio.create_task(run_task(i, p)) for i, p in enumerate(list_)]
    if not silent:
        print("Finished creating tasks. Waiting for all to complete...")

    try:
        # As tasks finish, store their results
        for completed in asyncio.as_completed(tasks):
            i, res = await completed
            results[i] = res

            # track completion (anything no longer "in_progress" counts)
            completed_so_far = sum(r != "in_progress" for r in results)  # minimal, readable

            # Check early-stop threshold  <-- NEW
            if stop_after_frac is not None:
                if completed_so_far / total_items >= stop_after_frac:
                    if not silent:
                        print(
                            f"Reached stop_after_frac={stop_after_frac:.2%} "
                            f"({completed_so_far}/{total_items}); cancelling remaining tasks."
                        )
                    # cancel everything still pending
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                        progress_task.cancel()
                    break

        if not return_cache:
            # Wait for the progress watcher to see all tasks completed
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
    finally:
        for task in tasks:
            task.cancel()
        progress_task.cancel()

    # change all in_progress to None
    results = [None if r == "in_progress" else r for r in results]

    if cache_dir and not return_cache:
        fpath = os.path.abspath(os.path.join(cache_dir, "_all_results.pkl"))
        with open(fpath, "wb") as f:
            pickle.dump(results, f)
        if not silent:
            print(f"Wrote all results to cache file {fpath}")

    if filter_none:
        results = [r for r in results if r is not None]
    return results
