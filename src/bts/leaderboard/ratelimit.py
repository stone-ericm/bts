"""Per-function rate-limit decorator.

Conservative posture: enforce a minimum gap between calls to the SAME
decorated callable. Each decorated function has its own cadence.
"""
from __future__ import annotations

import functools
import time
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def rate_limited(min_interval_s: float) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator: enforce >=min_interval_s between successive calls."""
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        last_called = 0.0

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal last_called
            now = time.monotonic()
            wait = (last_called + min_interval_s) - now
            if wait > 0:
                time.sleep(wait)
            last_called = time.monotonic()
            return fn(*args, **kwargs)

        return wrapper
    return decorator
