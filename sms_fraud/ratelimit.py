"""In-memory, per-client sliding-window rate limiting.

State lives in each serverless instance, so this caps abuse per instance;
Vercel's firewall rate limiting is the global, stronger option.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable

from starlette.requests import Request


class RateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._max_clients = max_clients
        self._clock = clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def hit(self, key: str) -> float | None:
        """Record a request; return seconds to wait if the client is over its limit."""
        now = self._clock()
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = self._hits[key] = deque()
            else:
                self._hits.move_to_end(key)
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return self._window - (now - hits[0])
            hits.append(now)
            # Bounded memory: forget the least recently seen clients.
            while len(self._hits) > self._max_clients:
                self._hits.popitem(last=False)
            return None


def client_ip(request: Request, trust_proxy_headers: bool) -> str:
    """Client address; proxy headers are only trusted behind a known proxy (Vercel)."""
    if trust_proxy_headers:
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"
