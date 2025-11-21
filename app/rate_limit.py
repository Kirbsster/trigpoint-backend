from __future__ import annotations
import time, asyncio
from typing import Dict, List

class RateLimitExceeded(Exception):
    def __init__(self, retry_after: int):
        super().__init__("Too Many Requests")
        self.retry_after = retry_after

class SlidingWindowLimiter:
    """
    Simple in-memory sliding-window limiter.
    - limit: max hits within 'window' seconds per key.
    NOTE: For multi-process/instances, use Redis (see option below).
    """
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self._store: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def hit(self, key: str) -> tuple[int, int]:
        now = time.time()
        start = now - self.window
        async with self._lock:
            arr = self._store.get(key, [])
            # keep only events within the window
            arr = [t for t in arr if t > start]
            if len(arr) >= self.limit:
                # time until the oldest in-window event expires
                retry_after = int(arr[0] - start) + 1
                self._store[key] = arr
                raise RateLimitExceeded(retry_after)
            arr.append(now)
            self._store[key] = arr
            remaining = self.limit - len(arr)
        return remaining, self.window