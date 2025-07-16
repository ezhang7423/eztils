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


async def progress_watcher(
    total: int,
    results: list,
    cache_dir: str = None,
    poll_interval: float = 1.0,
    stalled_timeout: float = 600.0,
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
                    f.startswith("result_") and f.endswith(".pkl")
                    for f in os.listdir(cache_dir)
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

        print(
            f"[{now_str}] "
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
):
    """
    Runs `fn` concurrently on items from list_.

    - If `cache_dir` is provided, results are stored and/or read from .pkl files named `result_{i}.pkl`.
    - If `cache_dir` is not provided, results remain in memory, but the same progress watcher will print progress.
    - This version also prints time elapsed and a naive ETA in the progress watcher.

    TODO: add a meetadata.pkl that captures args and "fn" code
    """

    if list_ is None:
        list_ = ["in_progress"] * max_workers

    total_items = len(list_)
    results = ["in_progress"] * total_items

    # Create the cache directory if needed
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_dir = os.path.abspath(cache_dir)
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
                        print(f"Error writing result to cache for index {i}: {e}")

            except asyncio.TimeoutError as e:
                print(f"TimeoutError occurred for index {i}: {e}")
                result = None
            except Exception as e:
                if ignore_exceptions:
                    print(f"Exception for index {i}:\n{traceback.format_exc()}")
                    result = None
                else:
                    # Raise for the top-level caller to handle
                    raise e

            return i, result

    # Kick off the progress watcher (works with or without cache_dir)
    progress_task = asyncio.create_task(
        progress_watcher(total_items, results, cache_dir)
    )

    # Create tasks for each item
    tasks = [asyncio.create_task(run_task(i, p)) for i, p in enumerate(list_)]
    print("Finished creating tasks. Waiting for all to complete...")

    try:
        # As tasks finish, store their results
        for completed in asyncio.as_completed(tasks):
            i, res = await completed
            results[i] = res

        if not return_cache:
            # Wait for the progress watcher to see all tasks completed
            await progress_task
    finally:
        for task in tasks:
            task.cancel()
        progress_task.cancel()

    if cache_dir and not return_cache:
        fpath = os.path.abspath(os.path.join(cache_dir, "_all_results.pkl"))
        with open(fpath, "wb") as f:
            pickle.dump(results, f)
        print(f"Wrote all results to cache file {fpath}")

    if filter_none:
        results = [r for r in results if r is not None]
    return results
