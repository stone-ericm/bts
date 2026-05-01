"""Tests for the rate-limit decorator."""
from __future__ import annotations

import time

from bts.leaderboard.ratelimit import rate_limited


class TestRateLimited:
    def test_first_call_is_immediate(self):
        @rate_limited(min_interval_s=1.0)
        def f():
            return time.monotonic()

        t0 = time.monotonic()
        f()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1, f"first call should not block; took {elapsed}s"

    def test_second_call_within_interval_is_delayed(self):
        @rate_limited(min_interval_s=0.2)
        def f():
            return None

        f()
        t1 = time.monotonic()
        f()
        elapsed = time.monotonic() - t1
        assert 0.15 <= elapsed <= 0.4, f"expected ~0.2s gap, got {elapsed}s"

    def test_third_call_after_interval_is_immediate(self):
        @rate_limited(min_interval_s=0.1)
        def f():
            return None

        f()
        time.sleep(0.2)
        t1 = time.monotonic()
        f()
        elapsed = time.monotonic() - t1
        assert elapsed < 0.05, f"call after interval should be immediate; took {elapsed}s"
