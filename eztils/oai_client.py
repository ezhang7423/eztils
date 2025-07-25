import asyncio
from openai import Client, AsyncClient

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
    return client.chat.completions.create(
        model=models[model_key],
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    ).choices[0].message.content

async def get_response_async(model_key, prompt, **kwargs):
    client = AsyncClient()
    resp = await client.chat.completions.create(
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
        return await get_response_async(key, prompt, **kwargs)
    fn.__name__ = f"async_prompt_{key}"
    fn.__doc__ = f"Asynchronously call model `{key}`"
    return fn

# Inject everything into module namespace
for key in models:    
    globals()[f"prompt_{key}"]       = _make_sync_fn(key)
    globals()[f"async_prompt_{key}"] = _make_async_fn(key)

# Explicit exports for `from your_module import *`
__all__ = [
    *[f"prompt_{k}"       for k in models],
    *[f"async_prompt_{k}" for k in models],
]

# make a factory for each model key


if __name__ == "__main__":
    from safesys.x.eddie.lib.conco import async_concurrency
    import functools

    n = 1000
    prompt = "What is the capital of France?"
    res = async_concurrency(fn=async_prompt_4o, list_=[prompt] * n, max_workers=n)  # this takes ~20 sec
    res = asyncio.run(res)
    print(res)
