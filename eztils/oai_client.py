import asyncio
from openai import Client, AsyncClient
from openai import DefaultAioHttpClient
import httpx
import atexit

BIG_RUN = True
MAX_CONNECTIONS = 100_000
MAX_KEEPALIVE_CONNECTIONS = 1_000
MAX_RETRIES = 5
DEFAULT_TIMEOUT = httpx.Timeout(timeout=6000, connect=5.0)

if BIG_RUN:
    async_client = AsyncClient(
        http_client=DefaultAioHttpClient(
            limits=httpx.Limits(
                max_connections=MAX_CONNECTIONS,
                max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
            ),
            timeout=DEFAULT_TIMEOUT,
        ),
        max_retries=MAX_RETRIES,
    )
    atexit.register(lambda: asyncio.run(async_client.close()))
else:
    async_client = AsyncClient()

# call async_client on system exit

models = {
    "o1":        "o1-2024-12-17",
    "o1_mini":   "o1-mini-2024-09-12",
    "o3_mini":   "o3-mini-2025-01-31",
    "o4_mini":   "o4-mini-2025-04-16",
    "o3":        "o3-2025-04-16",
    "4o":        "gpt-4o-2024-11-20",
    "4o_mini":   "gpt-4o-mini-2024-07-18",
    "4_1":       "gpt-4.1-2025-04-14",
    "4o_with_search": "gpt-4o-search-preview-2025-03-11",
    "3_5":       "gpt-3.5-turbo-0125",
}

def get_response(model_key, prompt, **kwargs):
    client = Client()
    return (
        client.chat.completions.create(
            model=models[model_key],
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        .choices[0]
        .message.content
    )


async def async_get_response(model_key, prompt, **kwargs):
    resp = await async_client.chat.completions.create(
        model=models[model_key],
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return resp.choices[0].message.content


def _make_sync_fn(key):
    def fn(prompt, **kwargs):
        return get_response(key, prompt, **kwargs)

    fn.__name__ = f"prompt_{key}"
    fn.__doc__ = f"Synchronously call model `{key}`"
    return fn


def _make_async_fn(key):
    async def fn(prompt, **kwargs):
        return await async_get_response(key, prompt, **kwargs)

    fn.__name__ = f"async_prompt_{key}"
    fn.__doc__ = f"Asynchronously call model `{key}`"
    return fn


# Inject everything into module namespace
for key in models:
    globals()[f"prompt_{key}"] = _make_sync_fn(key)
    globals()[f"async_prompt_{key}"] = _make_async_fn(key)

# Explicit exports for `from your_module import *`
__all__ = [
    *[f"prompt_{k}" for k in models],
    *[f"async_prompt_{k}" for k in models],
]

# make a factory for each model key
def main():

    from eztils.conco import async_concurrency
    import functools

    n = 1000
    prompt = "What is the capital of France?"


    res = async_concurrency(
        fn=lambda *args, **kwargs: async_prompt_4o(*args, **kwargs), list_=[prompt] * n, max_workers=n, stop_after_frac=0.99
    )  # this takes ~20 sec
    res = asyncio.run(res)
    
    print(res)
    print(len(res))
    # print num none
    print(sum(r is None for r in res))


if __name__ == "__main__":
    main()
