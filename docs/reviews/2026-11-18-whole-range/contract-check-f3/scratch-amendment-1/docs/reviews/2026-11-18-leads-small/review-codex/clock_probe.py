"""Verify the simulated year inside the pytest process before collecting tests."""

from datetime import date, datetime, timezone
import sys
import time


if __name__ == "__main__":
    requested = int(sys.argv[1])
    now = datetime.now(timezone.utc)
    assert date.today().year == now.year == requested, (requested, now, date.today())
    assert abs(now.timestamp() - time.time()) < 1
    start = time.monotonic()
    time.sleep(0.005)
    assert 0 < time.monotonic() - start < 1
    print(f"Verified realtime: {now.isoformat()} UTC, local date {date.today()}", flush=True)

    if sys.argv[2:]:
        import pytest

        raise SystemExit(pytest.main(sys.argv[2:]))
