"""Limit raw asynchronous work without holding permits during output descent."""

from collections.abc import Coroutine
from inspect import iscoroutine


class OwnedAwaitable(Coroutine):
    """Keep unstarted batch wrappers responsible for their raw coroutines."""

    def __init__(self, awaitable, *owned):
        self.awaitable = awaitable
        self.owned = owned
        self.driver = self.run()

    def __await__(self):
        return self.driver.__await__()

    def send(self, value):
        return self.driver.send(value)

    def throw(self, *args):
        return self.driver.throw(*args)

    async def run(self):
        try:
            return await self.awaitable
        finally:
            self.close_owned()

    def close(self):
        self.driver.close()
        self.awaitable.close()
        self.close_owned()

    def close_owned(self):
        for awaitable in self.owned:
            if iscoroutine(awaitable) or isinstance(awaitable, OwnedAwaitable):
                awaitable.close()


def include_owned_awaitables(predicate):
    # Host predicates describe host values; engine-owned coroutine wrappers are
    # always awaitable, including with a strict inspect.iscoroutine predicate.
    return lambda value: isinstance(value, OwnedAwaitable) or predicate(value)


async def await_limited(awaitable, semaphore):
    if semaphore is None:
        return await awaitable

    acquired = False
    try:
        await semaphore.acquire()
        acquired = True
        return await awaitable
    finally:
        if acquired:
            semaphore.release()
        elif iscoroutine(awaitable) or isinstance(awaitable, OwnedAwaitable):
            # A cancelled waiter never started this coroutine. Close it so a
            # cancelled operation does not leave unawaited resolver/load work.
            awaitable.close()
