"""C1 rank 4a, T1: the fixed 2027 contest-date numbering (registration §3; checklist A3).

The numbered dates are the official 2027 contest calendar dates on which MLB games are scheduled. They are frozen,
before the first study capture, into a pinned calendar file:

    {"schema": "c1_r4a_calendar_v1", "season": 2027, "dates": ["2027-03-25", ...]}

The dates must be strictly increasing ISO dates in the season. Numbering starts at 1 on the first date and advances
whether or not a slate exists, so missing data never shifts or extends a window. Fit dates are 1-30 and test dates
31-90.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date

SCHEMA = "c1_r4a_calendar_v1"
FIT = range(1, 31)
TEST = range(31, 91)


class CalendarError(RuntimeError):
    pass


@dataclass(frozen=True)
class Calendar:
    dates: tuple
    sha256: str

    def number(self, d: date) -> int | None:
        try:
            return self.dates.index(d) + 1
        except ValueError:
            return None

    def date_of(self, n: int) -> date:
        return self.dates[n - 1]

    @staticmethod
    def window(n: int | None) -> str | None:
        return "fit" if n in FIT else "test" if n in TEST else None

    def fit_dates(self) -> list:
        return [d for d in self.dates if self.window(self.number(d)) == "fit"]

    def test_dates(self) -> list:
        return [d for d in self.dates if self.window(self.number(d)) == "test"]


def load(raw: bytes, pin: str) -> Calendar:
    """Parse the pinned calendar bytes (the bytes hashed are the bytes parsed)."""
    sha = hashlib.sha256(raw).hexdigest()
    if sha != pin:
        raise CalendarError(f"the calendar sha256 {sha[:12]} is not its pin {pin[:12]}")
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict) or obj.get("schema") != SCHEMA or type(obj.get("season")) is not int:
            raise CalendarError("not a c1_r4a_calendar_v1 object")
        days = tuple(date.fromisoformat(s) for s in obj["dates"])
    except (ValueError, TypeError, KeyError) as exc:
        raise CalendarError(f"unreadable calendar: {exc}") from None
    if not days:
        raise CalendarError("the calendar is empty")
    if any(d.year != obj["season"] for d in days):
        raise CalendarError("a date outside the season")
    if any(b <= a for a, b in zip(days, days[1:])):
        raise CalendarError("dates must be strictly increasing (sorted, no duplicate)")
    return Calendar(days, sha)
