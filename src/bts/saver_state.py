"""The Streak Saver manual flag: a sound, operator-controlled replacement for the unsound
ledger inference. Persisted at account_state/saver_state.json as one of {not_earned, active,
used}; the loader derives a fail-closed `uninitialized` for a missing/invalid/stale-season file.
See docs/superpowers/specs/2026-06-18-streak-saver-flag-design.md.
"""
from __future__ import annotations

import fcntl
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
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return SaverState("uninitialized", None)
    if not isinstance(d, dict):           # a JSON scalar/list ([], 123, "active") -> fail-closed
        return SaverState("uninitialized", None)
    st = d.get("state")
    fseason = d.get("season") if isinstance(d.get("season"), int) else None
    if not isinstance(st, str) or st not in _PERSISTED or fseason != season:
        return SaverState("uninitialized", fseason)   # isinstance guards an unhashable `state`
    return SaverState(st, fseason, d.get("source"), d.get("updated_at"))


def _write_state(picks_dir: Path, *, state: str, season: int, source: str) -> None:
    atomic_write_text(_path(picks_dir), json.dumps({
        "season": season,
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }))


# Allowed (prior -> new) transitions; anything else is REJECTED (so a scripted/cross-page POST
# can't do e.g. active -> not_earned). `force=True` (CLI --force only) bypasses the whitelist.
_ALLOWED = {
    ("uninitialized", "not_earned"), ("uninitialized", "active"), ("uninitialized", "used"),
    ("not_earned", "active"), ("active", "used"), ("used", "active"),
}


def _append_audit(picks_dir: Path, *, expected_prior: str, new_state: str, season: int,
                  source: str, peer: str | None, outcome: str) -> None:
    """Append-only transition audit trail (audit F7, accepted-risk detective control).

    saver_state.json keeps only the LAST write; this jsonl keeps every attempt
    — including rejected ones, which are exactly what a detective control for
    an unauthenticated tailnet endpoint wants to see. Lives inside data/picks,
    so the F5 ops backup carries it off-box. Best-effort: an audit write
    failure must never block the state transition itself.
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "expected_prior": expected_prior,
        "new_state": new_state,
        "season": season,
        "source": source,
        "outcome": outcome,
    }
    if peer is not None:
        entry["peer"] = peer
    try:
        log_path = _path(picks_dir).parent / "saver_transitions.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def transition_saver_state(picks_dir: Path, *, expected_prior: str, new_state: str,
                           season: int, source: str, force: bool = False,
                           peer: str | None = None) -> bool:
    """Guarded atomic transition: writes `new_state` ONLY if (a) `new_state` is valid, (b)
    `(expected_prior, new_state)` is an allowed transition (unless `force`), and (c) the current
    persisted state still equals `expected_prior` (re-read just before writing). Returns True iff
    written. The single monotonic-safe write path — auto-earn, CLI, and the dashboard all use it.
    Every attempt is appended to saver_transitions.jsonl; `peer` records the requesting client
    for network-originated mutations (audit F7)."""
    if new_state not in _PERSISTED:
        raise ValueError(f"invalid saver state: {new_state!r}")
    if not force and (expected_prior, new_state) not in _ALLOWED:
        _append_audit(picks_dir, expected_prior=expected_prior, new_state=new_state,
                      season=season, source=source, peer=peer, outcome="rejected_disallowed")
        return False
    lock_path = _path(picks_dir).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock:
        # Serialize concurrent writers (the 4x/day fetch auto-earn vs the CLI/dashboard): the
        # expected_prior guard must re-read and write UNDER the lock, else two callers from the
        # same prior could both pass the check before either writes (a lost update).
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if load_saver_state(picks_dir, season=season).state != expected_prior:
            _append_audit(picks_dir, expected_prior=expected_prior, new_state=new_state,
                          season=season, source=source, peer=peer,
                          outcome="rejected_state_mismatch")
            return False
        _write_state(picks_dir, state=new_state, season=season, source=source)
        _append_audit(picks_dir, expected_prior=expected_prior, new_state=new_state,
                      season=season, source=source, peer=peer, outcome="written")
        return True


def maybe_auto_earn_saver(picks_dir: Path, *, best_streak: int | None, season: int) -> None:
    """Fetch-path hook. Safe initialization + the only sound auto transition:
    - uninitialized + best_streak < 10  -> not_earned  (no save possible yet)
    - not_earned    + best_streak >= 10 -> active       (sound: best_streak is reliable)
    Never auto-inits `active` from uninitialized at >=10 (could be earned-and-used before we saw
    it -> fail-closed), and never overwrites active/used."""
    if best_streak is None:
        return
    current = load_saver_state(picks_dir, season=season).state
    if current == "uninitialized" and best_streak < 10:
        transition_saver_state(picks_dir, expected_prior="uninitialized",
                               new_state="not_earned", season=season, source="auto_earn")
    elif current == "not_earned" and best_streak >= 10:
        transition_saver_state(picks_dir, expected_prior="not_earned",
                               new_state="active", season=season, source="auto_earn")
