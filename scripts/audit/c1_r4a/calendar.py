"""C1 rank 4a, T1: the fixed 2027 contest-date numbering (registration §3; checklist A3).

The numbered dates are the official 2027 contest calendar dates on which MLB games are scheduled. They are frozen,
before the first study capture, into a pinned calendar file:

    {"schema": "c1_r4a_calendar_v1", "season": 2027, "dates": ["2027-03-25", ...], "contest_end": "2027-09-26"}

- **The dates** must be strictly increasing ISO dates in season 2027 (the registered season; any other is refused).
- **The contest end** is the last day of the 2027 contest, recorded under checklist A3. It must be the last listed
  date. The C1 calendar stop (cycle proposal §5.6) closes the cycle when that day ends, and it takes precedence over
  an unfinished window (registration §3).
- **The windows need 90 dates.** A calendar with fewer cannot hold the registered fit and test windows, and is
  refused rather than failing later at date 40 or 90.

Numbering starts at 1 on the first date and advances whether or not a slate exists, so missing data never shifts or
extends a window. Fit dates are 1-30 and test dates 31-90.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date

SCHEMA = "c1_r4a_calendar_v1"
SEASON = 2027
FIT = range(1, 31)
TEST = range(31, 91)


class CalendarError(RuntimeError):
    pass


@dataclass(frozen=True)
class Calendar:
    dates: tuple
    sha256: str
    contest_end: date

    def number(self, d: date) -> int | None:
        try:
            return self.dates.index(d) + 1
        except ValueError:
            return None

    def date_of(self, n: int) -> date:
        if not 1 <= n <= len(self.dates):
            raise CalendarError(f"the calendar has no contest date {n}")
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
        raise CalendarError(f"the calendar sha256 {sha[:12]} is not its pin {str(pin)[:12]}")
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict) or obj.get("schema") != SCHEMA or not isinstance(obj.get("dates"), list) \
                or not all(isinstance(s, str) for s in obj["dates"]) or not isinstance(obj.get("contest_end"), str):
            raise CalendarError(f"not a {SCHEMA} object (schema, dates and contest_end)")
        days = tuple(date.fromisoformat(s) for s in obj["dates"])
        end = date.fromisoformat(obj["contest_end"])
    except (ValueError, TypeError, RecursionError) as exc:
        raise CalendarError(f"unreadable calendar: {exc}") from None
    if type(obj.get("season")) is not int or obj["season"] != SEASON:
        raise CalendarError(f"the calendar's season is not {SEASON}")
    if any(d.year != SEASON for d in days):
        raise CalendarError("a date outside the season")
    if any(b <= a for a, b in zip(days, days[1:])):
        raise CalendarError("dates must be strictly increasing (sorted, no duplicate)")
    if len(days) < TEST.stop - 1:
        raise CalendarError(f"the calendar has {len(days)} dates; the registered windows need {TEST.stop - 1}")
    if end != days[-1]:
        raise CalendarError("the contest end is not the calendar's last date")
    return Calendar(days, sha, end)
