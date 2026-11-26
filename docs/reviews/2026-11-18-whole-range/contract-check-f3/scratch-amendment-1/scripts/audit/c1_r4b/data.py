"""C1 4b inputs: outcome-free season calendars and validated, paired profile days (registration §§3–4, 7).

- **Calendar:** built from the public MLB regular-season schedule. A game day has at least one game that was played
  (not only postponed or cancelled); see `docs/sota_audit/2026-10-04-c1-r4b-calendar-coverage.md`. Raw days left is
  `exclusive_end − date` in calendar days, and no-opportunity days still consume time.
- **Profiles:** validated strictly; malformed, duplicate or ambiguous rows are refused, never skipped.
- **Pairing:** rank 1 is paired with the first lower-ranked candidate in a different game; without one the day is
  partnerless.
- **Coverage:** a game day with no profile rows is *unknown coverage* (counted; it makes acceptance inconclusive
  under §7), never an evidenced skip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from scripts.audit.c1_r3.acquire import played_status

REQUIRED = ("date", "rank", "batter_id", "game_pk", "p_game_hit", "actual_hit")


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Calendar:
    season: int
    opening: date
    final: date
    no_opportunity: frozenset

    @property
    def exclusive_end(self) -> date:
        return self.final + timedelta(days=1)

    @property
    def horizon(self) -> int:
        return (self.exclusive_end - self.opening).days

    def days_left(self, d: date) -> int:
        return max(0, (self.exclusive_end - d).days)

    def days(self) -> list[tuple[date, bool, int]]:
        """(date, opportunity, raw days left) for every calendar day opening..final."""
        out, d = [], self.opening
        while d <= self.final:
            out.append((d, d not in self.no_opportunity, self.days_left(d)))
            d += timedelta(days=1)
        return out


def calendar_from_schedule(sched: dict, season: int) -> Calendar:
    """Opportunity days come from the listings themselves (code review r1 F2): a schedule date is a game day when
    any of its listings was played. A suspended game's original day and its completion day both count; the unique
    feed inventory (which dedupes game ids) is a different object. Unsupported statuses are refused."""
    played = set()
    for day in sched.get("dates", []):
        states = [played_status(g["status"]["detailedState"]) for g in day.get("games", [])]
        if any(states):
            played.add(date.fromisoformat(day["date"]))
    played = {d for d in played if d.year == season}
    if not played:
        raise ValueError(f"{season}: no played regular-season games in the schedule")
    lo, hi = min(played), max(played)
    span = {lo + timedelta(days=i) for i in range((hi - lo).days + 1)}
    return Calendar(season=season, opening=lo, final=hi, no_opportunity=frozenset(span - played))


def exclude_dates(cal: Calendar, dates) -> Calendar:
    """Remove owner-ruled non-contest days (e.g. register row C1-4b-2023-10-02). A removed final day moves the
    final date back to the last remaining game day; days in between stay no-opportunity days."""
    drop = {date.fromisoformat(str(x)) for x in dates}
    opp = {dd for dd, o, _ in cal.days() if o} - drop
    if not opp:
        raise ValueError(f"{cal.season}: no contest days remain")
    lo, hi = min(opp), max(opp)
    span = {lo + timedelta(days=i) for i in range((hi - lo).days + 1)}
    return Calendar(season=cal.season, opening=lo, final=hi, no_opportunity=frozenset(span - opp))


def validate_profile(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ProfileError(f"missing columns {missing}")
    df = df[list(REQUIRED)].copy()
    if df[list(REQUIRED)].isna().any().any():
        raise ProfileError("null in a required column")
    for c in ("rank", "batter_id", "game_pk", "actual_hit"):
        if not pd.api.types.is_integer_dtype(df[c]):
            raise ProfileError(f"{c} must be integer")
    p = df["p_game_hit"].astype(float)
    if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
        raise ProfileError("p_game_hit must be finite in [0, 1]")
    if not df["actual_hit"].isin([0, 1]).all():
        raise ProfileError("actual_hit must be 0/1")
    if df.duplicated(["date", "rank"]).any():
        raise ProfileError("duplicate (date, rank)")
    if df.duplicated(["date", "batter_id", "game_pk"]).any():
        raise ProfileError("duplicate (date, batter_id, game_pk)")
    if (df["rank"] < 1).any():
        raise ProfileError("rank must be >= 1")
    if (df["batter_id"] <= 0).any() or (df["game_pk"] <= 0).any():
        raise ProfileError("batter_id and game_pk must be positive integers")
    firsts = df.groupby("date")["rank"].min()
    if (firsts != 1).any():
        raise ProfileError(f"no rank 1 on {list(firsts[firsts != 1].index)[:3]}")
    return df.sort_values(["date", "rank"]).reset_index(drop=True)


def season_days(df: pd.DataFrame, cal: Calendar) -> dict:
    """Per calendar day: opportunity, known coverage, raw days left, the primary's p and label, and the partner."""
    cal_days = cal.days()
    opp_dates = {d for d, opp, _ in cal_days if opp}
    by_date = {d: g for d, g in df.groupby("date", sort=True)}
    stray = sorted(set(by_date) - opp_dates)
    if stray:
        raise ProfileError(f"profile rows on dates that are not calendar game days: {stray[:3]}")
    n = len(cal_days)
    out = {"date": [d for d, _, _ in cal_days], "opp": np.zeros(n, bool), "known": np.zeros(n, bool),
           "d_raw": np.zeros(n, np.int64), "p1": np.full(n, np.nan), "hit1": np.zeros(n, bool),
           "partner": np.zeros(n, bool), "hit2": np.zeros(n, bool), "partner_rank": np.zeros(n, np.int64),
           "unknown_dates": []}
    for i, (d, opp, left) in enumerate(cal_days):
        out["opp"][i], out["d_raw"][i] = opp, left
        g = by_date.get(d)
        if not opp:
            continue
        if g is None:
            out["unknown_dates"].append(d)
            continue
        out["known"][i] = True
        top = g.iloc[0]
        out["p1"][i], out["hit1"][i] = float(top["p_game_hit"]), bool(top["actual_hit"])
        other = g[(g["rank"] > top["rank"]) & (g["game_pk"] != top["game_pk"])]
        if len(other):
            out["partner"][i], out["hit2"][i] = True, bool(other.iloc[0]["actual_hit"])
            out["partner_rank"][i] = int(other.iloc[0]["rank"])
    if not math.isfinite(float(np.nansum(out["p1"]))):
        raise ProfileError("non-finite primary probability")
    return out
