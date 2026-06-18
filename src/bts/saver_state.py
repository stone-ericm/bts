"""The Streak Saver manual flag: a sound, operator-controlled replacement for the unsound
ledger inference. Persisted at account_state/saver_state.json as one of {not_earned, active,
used}; the loader derives a fail-closed `uninitialized` for a missing/invalid/stale-season file.
See docs/superpowers/specs/2026-06-18-streak-saver-flag-design.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from bts.util import atomic_write_text   # NB: defined in bts.util, not bts.picks

_PERSISTED = {"not_earned", "active", "used"}


@dataclass(frozen=True)
class SaverState:
    state: str                     # not_earned | active | used | uninitialized (last never persisted)
    season: int | None
    source: str | None = None
    updated_at: str | None = None

    @property
    def is_available(self) -> bool:
        return self.state == "active"


def _path(picks_dir: Path) -> Path:
    return picks_dir / "account_state" / "saver_state.json"


def season_for(source_date: date | None, *, now_year: int) -> int:
    """Contest season = the observation's calendar year, else the current year."""
    return source_date.year if source_date is not None else now_year


def load_saver_state(picks_dir: Path, *, season: int) -> SaverState:
    """Read the saver flag for `season`. Returns state='uninitialized' (fail-closed, DISTINCT
    from not_earned) when the file is missing, invalid, or for another season. A stale-season
    file preserves its `season` so health can distinguish stale from missing."""
    path = _path(picks_dir)
    if not path.exists():
        return SaverState("uninitialized", None)
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return SaverState("uninitialized", None)
    st = d.get("state")
    fseason = d.get("season") if isinstance(d.get("season"), int) else None
    if st not in _PERSISTED or fseason != season:
        return SaverState("uninitialized", fseason)
    return SaverState(st, fseason, d.get("source"), d.get("updated_at"))


def _write_state(picks_dir: Path, *, state: str, season: int, source: str) -> None:
    atomic_write_text(_path(picks_dir), json.dumps({
        "season": season,
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }))
