"""The watchdog's clock seam: every check reads time through one injected clock (tz-aware ET)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(ET)


class FixedClock:
    """Tests: a fixed instant that can be moved explicitly."""

    def __init__(self, at: datetime):
        if at.tzinfo is None:
            raise ValueError("FixedClock needs a timezone-aware instant")
        self.at = at

    def now(self) -> datetime:
        return self.at.astimezone(ET)
