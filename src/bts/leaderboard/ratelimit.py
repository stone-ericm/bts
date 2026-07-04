"""Per-function rate-limit decorator.

Conservative posture: enforce a minimum gap between calls to the SAME
decorated callable. Each decorated function has its own cadence.

Optional jitter randomizes the gap so successive requests don't land on a
perfectly regular cadence — a machine-metronome 2.000s gap is one of the
clearest bot tells at scale. The gap becomes a random draw in
[min_interval_s, min_interval_s + jitter_s].
"""
from __future__ import annotations

import functools
import random
import time
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def next_gap(min_interval_s: float, jitter_s: float, rng: random.Random) -> float:
    """Pure: the target gap for the next call. Testable without a clock."""
    if jitter_s <= 0:
        return min_interval_s
    return min_interval_s + rng.uniform(0.0, jitter_s)


def rate_limited(
    min_interval_s: float, jitter_s: float = 0.0,
    rng: random.Random | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator: enforce a randomized gap in [min, min+jitter] between calls."""
    _rng = rng or random.Random()

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        last_called = 0.0

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal last_called
            gap = next_gap(min_interval_s, jitter_s, _rng)
            wait = (last_called + gap) - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last_called = time.monotonic()
            return fn(*args, **kwargs)

        return wrapper
    return decorator
