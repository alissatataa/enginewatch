"""In-process pub/sub used to fan events out to Server-Sent-Events clients.

``publish`` is thread-safe because sync FastAPI endpoints run in a worker thread
pool while subscribers live on the event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, event: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            return
        for q in list(self._subs):
            loop.call_soon_threadsafe(self._offer, q, event)

    @staticmethod
    def _offer(q: asyncio.Queue, event: dict[str, Any]) -> None:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass  # slow consumer: drop rather than block the ingest path

    async def subscribe(self, keepalive: float = 15.0) -> AsyncIterator[dict[str, Any] | None]:
        """Yield events; yields ``None`` every ``keepalive`` seconds of silence."""
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subs.add(q)
        try:
            while True:
                try:
                    yield await asyncio.wait_for(q.get(), keepalive)
                except TimeoutError:
                    yield None
        finally:
            self._subs.discard(q)
