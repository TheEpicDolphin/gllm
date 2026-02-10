import asyncio
from concurrent.futures import Future as ConcurrentFuture

def complete_future_threadsafe(fut, result=None, exception=None):
    if isinstance(fut, ConcurrentFuture) and not isinstance(fut, asyncio.Future):
        # concurrent.futures.Future is thread-safe
        if exception is None:
            fut.set_result(result)
        else:
            fut.set_exception(exception)
        return

    # asyncio.Future or Task — must schedule on its loop
    try:
        loop = fut.get_loop()
    except Exception:
        loop = None

    if loop is None:
        # Best effort fallback. Try current running loop, otherwise set directly.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

    if loop is None:
        if exception is None:
            fut.set_result(result)
        else:
            fut.set_exception(exception)
    else:
        if exception is None:
            loop.call_soon_threadsafe(fut.set_result, result)
        else:
            loop.call_soon_threadsafe(fut.set_exception, exception)