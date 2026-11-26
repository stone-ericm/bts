# Late-Pick Delivery Guard + Deadline-Aware Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make it impossible for the scheduler to deliver an unenterable pick, make the T−35 fallback deadline-aware (recompute after blocking work, defer only when a decision-relevant lineup window can actually finish before this pick's cutoff, deliver the enterable pick otherwise), and cut each prediction cascade from ~15.5 min to ~5 min by not sleeping on cached feeds / not re-refreshing the season intraday.

**Architecture:** Three layers, each independently testable. (1) A hard, fail-closed submission-cutoff guard at the single delivery chokepoint `_deliver_and_lock_pick`, plus live-only exclusion of past-cutoff games from candidate selection so a late cycle re-picks from later games. (2) A pure `plan_fallback_action` planner that the in-loop fallback calls AFTER the refresh with a fresh clock and freshly synced confirmations; it binds deferral to the contender's confirmation window and a measured cascade budget. (3) `pull_feeds` skips the inter-request delay on cache hits, and `_refresh_season_data` memoizes a successful same-day refresh via a marker file. A `late_delivery` health source backstops the guard at EOD.

**Tech Stack:** Python 3.12, pytest, pandas; scheduler is `src/bts/scheduler.py` (single sequential loop), picks persistence `src/bts/picks.py`, health sources `src/bts/health/*.py`.

**Spec:** Root-cause analysis + Codex adversarial review in `docs/audit/2026-08-30-late-pick-delivery.md` (written in Task 10; the RCA content is summarized in "Incident" below so this plan is self-contained).

## Global Constraints

- All `uv` commands: `UV_CACHE_DIR=/tmp/uv-cache uv run ...`.
- Fast regression suite (run after every task): `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest -m "not slow" --ignore=tests/simulate --ignore=tests/model --ignore=tests/experiment --ignore=tests/validate -q` (~1918 tests, ~30s).
- TDD: failing test first, then minimal implementation. One commit per task, message explains the non-obvious choice.
- BTS submission cutoff is **first pitch − 5 min** (single constant, Task 1). Never duplicate the literal `5` again.
- The scheduler's `_now_et()` is the only clock in scheduler.py; tests patch `bts.scheduler._now_et`.
- Never edit tracked files while the Codex herdr pane (`w5:p2`) is mid-review.
- Deploy = `git push origin main:deploy`, only inside the scheduler's idle window (after tonight's results scoring, i.e. after the journal prints "Idle until tomorrow"). Never manually restart the unit after pushing.

## Incident (2026-08-30, ET) — what the plan fixes

Kwan (CLE, first pitch 13:40, cutoff 13:35). 12:35 check → `should_lock=False` (gap 1.5% vs a PROJECTED contender, Arraez PHI 16:07 game). Next scheduled check 13:10 is after the T−35 fallback deadline (13:05) → sleep to 13:05 → fallback refresh took 15.5 min → at **13:20** `FALLBACK DEFERRED` using a `has_pending_future_window` boolean snapshotted at 12:50 (time-blind: the "4 future checks" were the overdue 13:10 and three post-pitch runs) → loop ran the overdue 13:10 check → another 15.5-min cascade → locked the same Kwan at **13:36:14**, one minute after the cutoff. Nothing in the lock/deliver path checks time-to-pitch. Each cascade spends ~10.6 min in `pull_feeds` sleeping 0.3 s × 2113 already-cached feeds plus ~1 min re-discovering the schedule and 28 s rebuilding the parquet.

---

### Task 1: One submission-cutoff constant + helpers

**Files:**
- Modify: `src/bts/picks.py` (add after `pick_was_delivered`, ~line 316)
- Modify: `src/bts/scheduler.py:837-850` (`_earliest_pick_game_et` → delegate)
- Modify: `src/bts/cli.py:1657` (`submit_cutoff_min = 5` → constant)
- Modify: `src/bts/health/pick_entry.py:23` (`SUBMIT_CUTOFF_MIN` → alias)
- Test: `tests/test_submission_cutoff.py`

**Interfaces:**
- Produces: `bts.picks.SUBMISSION_CUTOFF_MIN: int = 5`; `bts.picks.earliest_pick_game_et(daily) -> datetime` (ET-aware); `bts.picks.submission_cutoff_et(daily) -> datetime` (= earliest − 5 min). `bts.scheduler._earliest_pick_game_et` keeps its name (cli imports it) and delegates.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_submission_cutoff.py
"""The contest submission cutoff (first pitch − 5 min) has ONE definition."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _daily(primary_utc="2026-08-30T17:40:00Z", dd_utc=None):
    from bts.picks import DailyPick, Pick
    pick = Pick(batter_name="Kwan", batter_id=680757, team="CLE", lineup_position=1,
                pitcher_name="Lugo", pitcher_id=607625, p_game_hit=0.7566, flags=[],
                projected_lineup=False, game_pk=824393, game_time=primary_utc)
    dd = None
    if dd_utc:
        dd = Pick(batter_name="McNeil", batter_id=643446, team="ATH", lineup_position=2,
                  pitcher_name="Bassitt", pitcher_id=605135, p_game_hit=0.7428, flags=[],
                  projected_lineup=False, game_pk=824959, game_time=dd_utc)
    return DailyPick(date="2026-08-30", run_time="2026-08-30T17:36:12+00:00",
                     pick=pick, double_down=dd, runner_up=None)


def test_constant_is_five_minutes():
    from bts.picks import SUBMISSION_CUTOFF_MIN
    assert SUBMISSION_CUTOFF_MIN == 5


def test_cutoff_is_earliest_slot_minus_five():
    from bts.picks import submission_cutoff_et, earliest_pick_game_et
    d = _daily(primary_utc="2026-08-30T20:05:00Z", dd_utc="2026-08-30T17:40:00Z")
    assert earliest_pick_game_et(d) == datetime(2026, 8, 30, 13, 40, tzinfo=ET)
    assert submission_cutoff_et(d) == datetime(2026, 8, 30, 13, 35, tzinfo=ET)


def test_single_pick_cutoff():
    from bts.picks import submission_cutoff_et
    assert submission_cutoff_et(_daily()) == datetime(2026, 8, 30, 13, 35, tzinfo=ET)


def test_scheduler_and_health_reuse_the_constant():
    from bts.picks import SUBMISSION_CUTOFF_MIN
    from bts.health import pick_entry
    from bts.scheduler import _earliest_pick_game_et
    assert pick_entry.SUBMIT_CUTOFF_MIN is SUBMISSION_CUTOFF_MIN
    d = _daily(primary_utc="2026-08-30T20:05:00Z", dd_utc="2026-08-30T17:40:00Z")
    assert _earliest_pick_game_et(d) == datetime(2026, 8, 30, 13, 40, tzinfo=ET)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_submission_cutoff.py -q`
Expected: FAIL — `ImportError: cannot import name 'SUBMISSION_CUTOFF_MIN'`.

- [ ] **Step 3: Implement**

In `src/bts/picks.py` after `pick_was_delivered`:

```python
# BTS rejects submissions within 5 min of first pitch. The ONLY definition —
# cli check-pick-entered, health pick_entry/late_delivery and the scheduler's
# delivery guard all import it (2026-08-30 incident: a pick DM'd at 13:36 for a
# 13:40 first pitch was already unenterable).
SUBMISSION_CUTOFF_MIN = 5


def earliest_pick_game_et(daily: DailyPick) -> datetime:
    """Earliest first pitch among the committed slots, ET-aware.

    The BTS app locks BOTH picks at the first game to start, so every deadline
    (fallback, cutoff, entry nag) keys on the earlier of primary and double-down."""
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    times = [datetime.fromisoformat(daily.pick.game_time.replace("Z", "+00:00")).astimezone(et)]
    if daily.double_down:
        times.append(datetime.fromisoformat(
            daily.double_down.game_time.replace("Z", "+00:00")).astimezone(et))
    return min(times)


def submission_cutoff_et(daily: DailyPick) -> datetime:
    """Last instant the committed slots can still be entered (first pitch − 5 min)."""
    return earliest_pick_game_et(daily) - timedelta(minutes=SUBMISSION_CUTOFF_MIN)
```

(`datetime`/`timedelta` are already imported in picks.py; verify with `grep -n "^from datetime" src/bts/picks.py`.)

In `src/bts/scheduler.py` replace the body of `_earliest_pick_game_et` with a delegation (keep the name — `cli.py` imports it):

```python
def _earliest_pick_game_et(daily) -> datetime:
    """Earliest first pitch among primary + double-down (ET). Delegates to
    bts.picks.earliest_pick_game_et — kept under this name for cli/test imports."""
    from bts.picks import earliest_pick_game_et
    return earliest_pick_game_et(daily)
```

In `src/bts/cli.py` line ~1657: replace `submit_cutoff_min = 5` with

```python
    from bts.picks import SUBMISSION_CUTOFF_MIN
    submit_cutoff_min = SUBMISSION_CUTOFF_MIN
```

In `src/bts/health/pick_entry.py` line ~23: replace `SUBMIT_CUTOFF_MIN = 5  # ...` with

```python
from bts.picks import SUBMISSION_CUTOFF_MIN

SUBMIT_CUTOFF_MIN = SUBMISSION_CUTOFF_MIN  # single definition lives in bts.picks
```

- [ ] **Step 4: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_submission_cutoff.py tests/health/test_pick_entry_source.py tests/test_e3_missed_pick_alert.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/picks.py src/bts/scheduler.py src/bts/cli.py src/bts/health/pick_entry.py tests/test_submission_cutoff.py
git commit -m "refactor(picks): single SUBMISSION_CUTOFF_MIN + earliest/cutoff helpers

The 5-minute contest cutoff was a literal in cli.py and a constant in
health/pick_entry.py, and the scheduler never consulted it at all. One
definition in bts.picks so the delivery guard (next commit), the entry
nag and the EOD audits cannot drift."
```

---

### Task 2: Hard delivery guard at the chokepoint (+ `delivered_at`, refusal record, CRITICAL DM)

**Files:**
- Modify: `src/bts/picks.py:200-236` (`DailyPick` gets `delivered_at: str | None = None`; check `load_pick` tolerates missing keys — it builds via `.get`)
- Modify: `src/bts/scheduler.py:344-366` (`SchedulerState.delivery_refusals: list[dict] | None = None`)
- Modify: `src/bts/scheduler.py:658-805` (`_deliver_and_lock_pick`), `src/bts/scheduler.py:1712-1728` (`_defer_pick_at_fallback` → generalized archiver)
- Test: `tests/test_late_delivery_guard.py`

**Interfaces:**
- Consumes: `bts.picks.submission_cutoff_et`, `bts.picks.SUBMISSION_CUTOFF_MIN` (Task 1).
- Produces: `_archive_and_remove_pick(picks_dir, date, daily, *, prefix, key, reason, extra=None) -> Path` (writes `data/picks/<date>/<prefix>_<stamp>.json` with `payload[key] = {"reason": reason, "<key>_at": iso, **extra}` and unlinks `<date>.json`); `_defer_pick_at_fallback` becomes a thin wrapper (`prefix="deferred_fallback", key="deferred_fallback"`). `_refuse_late_delivery(...)` (internal). `DailyPick.delivered_at` stamped on DM/post success. `SchedulerState.delivery_refusals` entries `{"at", "label", "batter", "double_down", "cutoff_et", "late_min", "archive"}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_late_delivery_guard.py
"""Fail-closed delivery guard: never deliver a pick at/after its submission cutoff.

2026-08-30: the daemon DM'd Kwan at 13:36:14 for a 13:40 first pitch (cutoff 13:35).
The guard lives at the ONE delivery chokepoint (_deliver_and_lock_pick) so no
caller — lineup lock, in-loop fallback, final fallback — can send a dead pick."""
import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from bts.picks import DailyPick, Pick, load_pick, save_pick
from bts.scheduler import SchedulerState, _deliver_and_lock_pick

ET = ZoneInfo("America/New_York")
DATE = "2026-08-30"
CONFIG = {"scheduler": {"pick_delivery": "dm"}, "bluesky": {"dm_recipient": "eric.test"},
          "orchestrator": {"picks_dir": "PLACEHOLDER"}}


def _state():
    return SchedulerState(date=DATE, schedule_fetched_at="x", games=[], confirmed_game_pks=[],
                          runs_completed=[], pick_locked=False, pick_locked_at=None,
                          result_status=None, next_wakeup=None)


def _daily(dd_utc=None):
    pick = Pick(batter_name="Kwan", batter_id=680757, team="CLE", lineup_position=1,
                pitcher_name="Lugo", pitcher_id=607625, p_game_hit=0.7566, flags=[],
                projected_lineup=False, game_pk=824393, game_time="2026-08-30T17:40:00Z")
    dd = None
    if dd_utc:
        dd = Pick(batter_name="McNeil", batter_id=643446, team="ATH", lineup_position=2,
                  pitcher_name="Bassitt", pitcher_id=605135, p_game_hit=0.7428, flags=[],
                  projected_lineup=False, game_pk=824959, game_time=dd_utc)
    return DailyPick(date=DATE, run_time="2026-08-30T17:20:00+00:00", pick=pick,
                     double_down=dd, runner_up=None)


@pytest.fixture
def cfg(tmp_path):
    c = json.loads(json.dumps(CONFIG))
    c["orchestrator"]["picks_dir"] = str(tmp_path)
    return c


@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=0))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.dm.send_dm", return_value="msg-1")
@patch("bts.scheduler._now_et")
def test_one_second_before_cutoff_delivers(mock_now, mock_dm, _cap, _dss, cfg, tmp_path):
    mock_now.return_value = datetime(2026, 8, 30, 13, 34, 59, tzinfo=ET)
    daily = _daily(); save_pick(daily, tmp_path); state = _state()
    ok = _deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, "lineup")
    assert ok is True and state.pick_locked is True
    mock_dm.assert_called_once()
    assert load_pick(DATE, tmp_path).delivered_at == mock_now.return_value.isoformat()


@patch("bts.health.alert.dispatch_dm_for_health_alerts", return_value=True)
@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=0))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.dm.send_dm", return_value="msg-1")
@patch("bts.scheduler._now_et")
def test_at_cutoff_refuses_archives_and_alerts(mock_now, mock_dm, _cap, _dss, mock_alert, cfg, tmp_path, capsys):
    mock_now.return_value = datetime(2026, 8, 30, 13, 35, 0, tzinfo=ET)   # == cutoff
    daily = _daily(); save_pick(daily, tmp_path); state = _state()
    ok = _deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, "lineup")
    assert ok is False
    assert state.pick_locked is False
    mock_dm.assert_not_called()
    assert not (tmp_path / f"{DATE}.json").exists()          # removed so later cycles re-pick
    archives = list((tmp_path / DATE).glob("refused_delivery_*.json"))
    assert len(archives) == 1
    body = json.loads(archives[0].read_text())
    assert body["refused_delivery"]["reason"] == "past_submission_cutoff"
    assert state.delivery_refusals and state.delivery_refusals[0]["batter"] == "Kwan"
    alerts = mock_alert.call_args.args[0]
    assert alerts[0].level == "CRITICAL" and alerts[0].source == "late_delivery"
    assert "DELIVERY REFUSED" in capsys.readouterr().err


@patch("bts.health.alert.dispatch_dm_for_health_alerts", return_value=True)
@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=0))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.dm.send_dm", return_value="msg-1")
@patch("bts.scheduler._now_et")
def test_earlier_double_down_sets_the_cutoff(mock_now, mock_dm, _cap, _dss, _alert, cfg, tmp_path):
    # primary 16:05 ET, DD 13:40 ET → cutoff 13:35 from the DD
    daily = _daily(dd_utc="2026-08-30T17:40:00Z")
    daily.pick.game_time = "2026-08-30T20:05:00Z"
    save_pick(daily, tmp_path)
    mock_now.return_value = datetime(2026, 8, 30, 13, 36, tzinfo=ET)
    assert _deliver_and_lock_pick(daily, cfg, tmp_path, _state(), DATE, "fallback") is False
    mock_dm.assert_not_called()


@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=0))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.dm.send_dm", return_value="msg-1")
@patch("bts.scheduler._now_et")
def test_already_delivered_pick_still_locks_after_cutoff(mock_now, mock_dm, _cap, _dss, cfg, tmp_path):
    """Evidence path: a pick that WAS delivered earlier re-locks on restart even after cutoff."""
    mock_now.return_value = datetime(2026, 8, 30, 14, 0, tzinfo=ET)
    daily = _daily(); daily.notification_sent = True; daily.notification_id = "old"
    save_pick(daily, tmp_path); state = _state()
    assert _deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, "lineup") is True
    assert state.pick_locked is True
    mock_dm.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_late_delivery_guard.py -q`
Expected: `test_one_second_before_cutoff_delivers` fails on `delivered_at` (attribute missing); the refusal tests fail because the DM IS sent.

- [ ] **Step 3: Implement**

`src/bts/picks.py` — in `DailyPick`, after `delivery_attempted`:

```python
    # ET ISO timestamp of the successful DM/post (2026-08-30). run_time is the
    # SELECTION time; this is the DELIVERY time the late_delivery audit needs.
    delivered_at: str | None = None
```

`src/bts/scheduler.py`:

```python
# in SchedulerState, after committed_pick_written:
    # Fail-closed delivery guard refusals (2026-08-30): each entry
    # {at, label, batter, double_down, cutoff_et, late_min, archive}. The EOD
    # late_delivery health source turns any entry into a CRITICAL.
    delivery_refusals: list[dict] | None = None


def _archive_and_remove_pick(picks_dir: Path, date: str, daily, *, prefix: str, key: str,
                             reason: str, extra: dict | None = None) -> Path:
    """Archive the live <date>.json under <date>/<prefix>_<stamp>.json and remove it
    so later cycles re-select instead of reusing a candidate we chose not to (or
    could not) deliver."""
    source = picks_dir / f"{date}.json"
    archive_dir = picks_dir / date
    archive_dir.mkdir(parents=True, exist_ok=True)
    now = _now_et()
    archive = archive_dir / f"{prefix}_{now.strftime('%Y%m%dT%H%M%S%z')}.json"
    payload = asdict(daily)
    payload[key] = {"reason": reason, f"{key.rsplit('_', 1)[0]}_at": now.isoformat(), **(extra or {})}
    archive.write_text(json.dumps(payload, indent=2))
    if source.exists():
        source.unlink()
    return archive


def _defer_pick_at_fallback(picks_dir: Path, date: str, daily, reason: str) -> Path:
    """Archive and remove an unsafe fallback candidate so later checks refresh."""
    return _archive_and_remove_pick(picks_dir, date, daily, prefix="deferred_fallback",
                                    key="deferred_fallback", reason=reason)
```

NOTE on the `_at` key: the existing archive writes `deferred_fallback: {reason, deferred_at}` and `tests/test_scheduler.py` + `health/fallback_defer.py` read `deferred_fallback.reason` only. Keep the exact legacy key name by special-casing: write `"deferred_at"` when key == "deferred_fallback", else `"refused_at"`. Simplest: pass the timestamp key explicitly — signature `at_key: str` (`"deferred_at"` / `"refused_at"`). Use that instead of the `rsplit` trick.

```python
def _refuse_late_delivery(daily, config: dict, picks_dir: Path, state: SchedulerState,
                          date: str, label: str, cutoff, now) -> None:
    """Guard action: archive+remove the dead candidate, record it on state, DM a
    CRITICAL. Leaves pick_locked False so the next cycle re-picks from games that
    are still enterable (Task 3 excludes past-cutoff games from selection)."""
    from bts.health.alert import Alert, dispatch_dm_for_health_alerts
    late_min = (now - cutoff).total_seconds() / 60
    names = daily.pick.batter_name + (f" + {daily.double_down.batter_name}" if daily.double_down else "")
    print(f"  DELIVERY REFUSED ({label}) — {names}: submission cutoff "
          f"{cutoff.strftime('%H:%M ET')} passed {late_min:.1f} min ago; not delivering an "
          f"unenterable pick.", file=sys.stderr)
    archive = _archive_and_remove_pick(
        picks_dir, date, daily, prefix="refused_delivery", key="refused_delivery",
        at_key="refused_at", reason="past_submission_cutoff",
        extra={"label": label, "cutoff_et": cutoff.isoformat(), "late_min": round(late_min, 2)})
    state.delivery_refusals = (state.delivery_refusals or []) + [{
        "at": now.isoformat(), "label": label, "batter": daily.pick.batter_name,
        "double_down": daily.double_down.batter_name if daily.double_down else None,
        "cutoff_et": cutoff.isoformat(), "late_min": round(late_min, 2), "archive": archive.name}]
    save_state(state, picks_dir)
    msg = (f"LATE DELIVERY REFUSED: {names} would have been sent {late_min:.0f} min after the "
           f"{cutoff.strftime('%H:%M ET')} cutoff. Nothing delivered; the scheduler will re-pick "
           f"from later games if any remain. Enter a pick manually if you want one now.")
    dispatch_dm_for_health_alerts(
        [Alert("CRITICAL", "late_delivery", msg)],
        config.get("bluesky", {}).get("dm_recipient"),
        status_path=picks_dir.parent / "health_state" / "health_dm_delivery_status.json")
```

In `_deliver_and_lock_pick`, insert immediately AFTER the `if daily.delivery_attempted:` block and BEFORE `if mode == "private":`:

```python
    # Fail-closed submission-cutoff guard (2026-08-30 incident). Applies to every
    # delivery mode incl. private: a pick locked after the cutoff is unenterable.
    from bts.picks import submission_cutoff_et
    cutoff = submission_cutoff_et(daily)
    now = _now_et()
    if now >= cutoff:
        _refuse_late_delivery(daily, config, picks_dir, state, date, label, cutoff, now)
        return False
```

And in the DM success branch and the post success branch, before the first `save_pick(daily, picks_dir)` after success: `daily.delivered_at = _now_et().isoformat()`.

- [ ] **Step 4: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_late_delivery_guard.py tests/test_e2_delivery_idempotency.py tests/test_scheduler.py -q`
Expected: PASS (the scheduler suite's run_day tests use dates in the past relative to now? NO — they patch `_now_et` to the pick's day, so the guard sees the patched clock; verify none deliver after their cutoff; the `test_fallback_fires_when_pick_game_before_next_check` delivers at 15:55 for a 16:10 game → before 16:05 cutoff ✓).

- [ ] **Step 5: Commit**

```bash
git add src/bts/picks.py src/bts/scheduler.py tests/test_late_delivery_guard.py
git commit -m "fix(scheduler): fail-closed submission-cutoff guard at the delivery chokepoint

_deliver_and_lock_pick refuses to send at/after first pitch − 5 min for the
earliest committed slot, archives the dead candidate (refused_delivery_*.json,
<date>.json removed so later cycles re-pick), records the refusal on
scheduler state and DMs a CRITICAL. Also stamps DailyPick.delivered_at —
run_time is selection time, and the EOD audit needs the send time."
```

---

### Task 3: Exclude past-cutoff games from live candidate selection

**Files:**
- Modify: `src/bts/strategy.py:259-345` (`select_pick(..., unavailable_game_pks: set[int] | None = None)`)
- Modify: `src/bts/orchestrator.py:249-300` (`run_and_pick(..., unavailable_game_pks=None)` pass-through)
- Modify: `src/bts/scheduler.py` (`_games_past_cutoff` helper; `run_single_check` + `_refresh_pick_at_fallback_decision` accept and forward `unavailable_game_pks`; `run_day` computes it from `state.games` before each call)
- Test: `tests/test_cutoff_candidate_exclusion.py`

**Interfaces:**
- Produces: `bts.scheduler._games_past_cutoff(state_games: list[dict], now: datetime) -> set[int]` — game_pks whose `game_time_et − SUBMISSION_CUTOFF_MIN ≤ now`. `select_pick(..., unavailable_game_pks=...)` drops those rows from `available` and treats an existing `current` pick whose committed games intersect the set as stale (re-select).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cutoff_candidate_exclusion.py
"""Live-only: games whose submission cutoff has passed are not pick candidates.

Offline/backtest callers pass nothing and are unchanged (Codex review 2026-08-30 #7:
this is NOT a margin — only games that are already unenterable are excluded)."""
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")


def _preds():
    return pd.DataFrame([
        {"batter_name": "Kwan", "batter_id": 1, "team": "CLE", "game_pk": 100, "lineup": 1,
         "pitcher_name": "Lugo", "pitcher_id": 9, "p_game_hit": 0.76, "flags": "",
         "game_time": "2026-08-30T17:40:00Z"},
        {"batter_name": "McNeil", "batter_id": 2, "team": "ATH", "game_pk": 200, "lineup": 2,
         "pitcher_name": "Bassitt", "pitcher_id": 8, "p_game_hit": 0.74, "flags": "",
         "game_time": "2026-08-30T20:05:00Z"},
        {"batter_name": "Alvarez", "batter_id": 3, "team": "HOU", "game_pk": 300, "lineup": 2,
         "pitcher_name": "Pecko", "pitcher_id": 7, "p_game_hit": 0.70, "flags": "",
         "game_time": "2026-08-30T19:10:00Z"},
    ])


def test_games_past_cutoff_helper():
    from bts.scheduler import _games_past_cutoff
    games = [{"game_pk": 100, "game_time_et": "2026-08-30T13:40:00-04:00"},
             {"game_pk": 200, "game_time_et": "2026-08-30T16:05:00-04:00"}]
    assert _games_past_cutoff(games, datetime(2026, 8, 30, 13, 34, 59, tzinfo=ET)) == set()
    assert _games_past_cutoff(games, datetime(2026, 8, 30, 13, 35, tzinfo=ET)) == {100}
    assert _games_past_cutoff(games, datetime(2026, 8, 30, 16, 0, tzinfo=ET)) == {100, 200}


@patch("bts.strategy.get_game_statuses", return_value={100: "P", 200: "P", 300: "P"})
def test_select_pick_skips_unavailable_games(_st, tmp_path):
    from bts.strategy import select_pick
    sel = select_pick(_preds(), "2026-08-30", tmp_path, streak=0, saver_available=False,
                      allow_double=True, unavailable_game_pks={100})
    daily = sel.pick_result.daily
    assert daily.pick.batter_name == "McNeil"
    assert daily.double_down is None or daily.double_down.game_pk != 100


@patch("bts.strategy.get_game_statuses", return_value={100: "P", 200: "P", 300: "P"})
def test_select_pick_unchanged_without_the_argument(_st, tmp_path):
    from bts.strategy import select_pick
    sel = select_pick(_preds(), "2026-08-30", tmp_path, streak=0, saver_available=False,
                      allow_double=True)
    assert sel.pick_result.daily.pick.batter_name == "Kwan"


@patch("bts.strategy.get_game_statuses", return_value={100: "P", 200: "P", 300: "P"})
def test_existing_pick_in_unavailable_game_is_replaced(_st, tmp_path):
    from bts.picks import save_pick
    from bts.strategy import select_pick
    first = select_pick(_preds(), "2026-08-30", tmp_path, streak=0, saver_available=False)
    save_pick(first.pick_result.daily, tmp_path)
    again = select_pick(_preds(), "2026-08-30", tmp_path, streak=0, saver_available=False,
                        unavailable_game_pks={100})
    assert again.pick_result.daily.pick.batter_name == "McNeil"
```

- [ ] **Step 2: Run to verify failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cutoff_candidate_exclusion.py -q`
Expected: FAIL (`_games_past_cutoff` missing; `select_pick() got an unexpected keyword argument`).

- [ ] **Step 3: Implement**

`src/bts/strategy.py` `select_pick` signature: add `unavailable_game_pks: "set[int] | None" = None` after `require_detailed_statuses`. Right after `current = load_pick(date, picks_dir)` (inside `if not for_shadow:`), before the classification:

```python
        if current and unavailable_game_pks:
            committed = {current.pick.game_pk} | (
                {current.double_down.game_pk} if current.double_down else set())
            if committed & set(unavailable_game_pks):
                current = None   # a committed slot is past its cutoff: re-select
```

Right after `available = predictions[not_started]`:

```python
    if unavailable_game_pks:
        available = available[~available["game_pk"].astype(int).isin(list(unavailable_game_pks))]
```

`src/bts/orchestrator.py` `run_and_pick`: add `unavailable_game_pks: "set[int] | None" = None` kwarg and pass `unavailable_game_pks=unavailable_game_pks` into `select_pick(...)`.

`src/bts/scheduler.py`:

```python
def _games_past_cutoff(state_games: list[dict], now: datetime) -> set[int]:
    """game_pks whose submission cutoff (first pitch − SUBMISSION_CUTOFF_MIN) is ≤ now.
    Fed to run_and_pick so a late cycle re-picks from enterable games only."""
    from bts.picks import SUBMISSION_CUTOFF_MIN
    out = set()
    for g in state_games:
        try:
            start = datetime.fromisoformat(g["game_time_et"])
        except (KeyError, TypeError, ValueError):
            continue
        if start - timedelta(minutes=SUBMISSION_CUTOFF_MIN) <= now:
            out.add(int(g["game_pk"]))
    return out
```

`run_single_check(..., early_lock_gap, unavailable_game_pks: "set[int] | None" = None)` → pass to `run_and_pick(config, date, require_detailed_statuses=False, unavailable_game_pks=unavailable_game_pks)`. Same kwarg on `_refresh_pick_at_fallback_decision(config, date, cached_daily, early_lock_gap, unavailable_game_pks=None)`. In `run_day`, both call sites pass `unavailable_game_pks=_games_past_cutoff(state.games, _now_et())`; the post-loop final fallback call too.

- [ ] **Step 4: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cutoff_candidate_exclusion.py tests/test_scheduler.py tests/test_decide_action.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/strategy.py src/bts/orchestrator.py src/bts/scheduler.py tests/test_cutoff_candidate_exclusion.py
git commit -m "feat(scheduler): exclude past-cutoff games from live candidate selection

Only games that are ALREADY unenterable (first pitch − 5 ≤ now) are dropped —
no margin, so an enterable pick is never filtered away (Codex #7). Offline
callers pass nothing and are unchanged. Lets a late cycle re-pick from later
games instead of re-selecting the batter the guard just refused."
```

---

### Task 4: Lock decision carries the block reason + contender game

**Files:**
- Modify: `src/bts/scheduler.py:369-390` (`FallbackRefreshResult` + new `LockDecision`), `:982-1059` (`_lock_decision_from_predictions`), call sites in `run_single_check` (~1291) and `_refresh_pick_at_fallback_decision` (~1688)
- Test: `tests/test_lock_decision.py`; adjust `tests/test_scheduler.py` tests that assert on the tuple (grep `_lock_decision_from_predictions` in tests).

**Interfaces:**
- Produces:
```python
@dataclass(frozen=True)
class LockDecision:
    should_lock: bool
    best_projected: float | None
    should_lock_ungated: bool
    block_reason: str | None      # None | "status_failure" | "slot_unavailable" | "primary_projected" | "gap" | "dd_projected"
    contender_game_pk: int | None # game of the best projected contender when block_reason == "gap"
```
`_lock_decision_from_predictions(...) -> LockDecision`. `FallbackRefreshResult` gains `block_reason: str | None = None`, `contender_game_pk: int | None = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lock_decision.py
from unittest.mock import patch
import pandas as pd
from bts.picks import DailyPick, Pick

PREVIEW = {"abstract": "P", "detailed": "Scheduled"}


def _daily(primary_projected=False, dd=None):
    p = Pick(batter_name="Kwan", batter_id=1, team="CLE", lineup_position=1, pitcher_name="L",
             pitcher_id=2, p_game_hit=0.757, flags=[], projected_lineup=primary_projected,
             game_pk=100, game_time="2026-08-30T17:40:00Z")
    return DailyPick(date="2026-08-30", run_time="x", pick=p, double_down=dd, runner_up=None)


def _preds(contender_p=0.741, contender_flags="PROJECTED lineup"):
    return pd.DataFrame([
        {"batter_name": "Kwan", "game_pk": 100, "p_game_hit": 0.757, "flags": ""},
        {"batter_name": "Arraez", "game_pk": 300, "p_game_hit": contender_p, "flags": contender_flags},
    ])


@patch("bts.picks.get_game_statuses_detailed", return_value={100: PREVIEW, 200: PREVIEW, 300: PREVIEW})
def test_gap_block_names_the_contender_game(_st):
    from bts.scheduler import _lock_decision_from_predictions
    d = _lock_decision_from_predictions(_preds(), _daily(), "2026-08-30", 0.03)
    assert d.should_lock is False and d.block_reason == "gap" and d.contender_game_pk == 300
    assert d.should_lock_ungated is False and abs(d.best_projected - 0.741) < 1e-9


@patch("bts.picks.get_game_statuses_detailed", return_value={100: PREVIEW, 300: PREVIEW})
def test_gap_passed_locks(_st):
    from bts.scheduler import _lock_decision_from_predictions
    d = _lock_decision_from_predictions(_preds(contender_p=0.70), _daily(), "2026-08-30", 0.03)
    assert d.should_lock is True and d.block_reason is None and d.contender_game_pk is None


@patch("bts.picks.get_game_statuses_detailed", return_value={100: PREVIEW, 300: PREVIEW})
def test_primary_projected_block(_st):
    from bts.scheduler import _lock_decision_from_predictions
    d = _lock_decision_from_predictions(_preds(), _daily(primary_projected=True), "2026-08-30", 0.03)
    assert d.should_lock is False and d.block_reason == "primary_projected"


@patch("bts.picks.get_game_statuses_detailed", return_value={100: PREVIEW, 200: PREVIEW, 300: PREVIEW})
def test_dd_projected_is_gate_only(_st):
    from bts.scheduler import _lock_decision_from_predictions
    dd = Pick(batter_name="DD", batter_id=5, team="X", lineup_position=1, pitcher_name="P",
              pitcher_id=6, p_game_hit=0.72, flags=["PROJECTED lineup"], projected_lineup=True,
              game_pk=200, game_time="2026-08-30T20:05:00Z")
    d = _lock_decision_from_predictions(_preds(contender_p=0.70), _daily(dd=dd), "2026-08-30", 0.03)
    assert d.should_lock is False and d.should_lock_ungated is True and d.block_reason == "dd_projected"


@patch("bts.picks.get_game_statuses_detailed", side_effect=RuntimeError("down"))
def test_status_failure(_st):
    from bts.scheduler import _lock_decision_from_predictions
    d = _lock_decision_from_predictions(_preds(), _daily(), "2026-08-30", 0.03)
    assert d.should_lock is False and d.block_reason == "status_failure"
```

- [ ] **Step 2: Run to verify failure** — `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lock_decision.py -q` → FAIL (`tuple` has no attribute `should_lock`).

- [ ] **Step 3: Implement**

Add the dataclass next to `FallbackRefreshResult` and extend it:

```python
@dataclass(frozen=True)
class LockDecision:
    should_lock: bool
    best_projected: float | None
    should_lock_ungated: bool
    block_reason: str | None = None
    contender_game_pk: int | None = None


@dataclass
class FallbackRefreshResult:
    daily: object
    should_post: bool | None
    selection: "SelectionResult | None" = None
    should_post_ungated: bool | None = None
    # Why should_post is False (LockDecision.block_reason) and, for a gap block,
    # WHICH game's confirmation could change the pick — the planner (Task 5)
    # only defers for THAT window when it can finish before this pick's cutoff.
    block_reason: str | None = None
    contender_game_pk: int | None = None
```

Rewrite `_lock_decision_from_predictions` returns: status failure → `LockDecision(False, None, False, "status_failure")`; slot unavailable → `LockDecision(False, None, False, "slot_unavailable")`; track `best_projected_pk` alongside `best_projected` in the loop (`best_projected_pk = game_pk` when updating the max); then:

```python
    ungated = should_lock(pick_data, all_pick_data, early_lock_gap)
    gated = (should_lock(pick_data, all_pick_data, early_lock_gap, double_down=double_down_data)
             if double_down_data is not None else ungated)
    if pick_data["projected_lineup"]:
        reason, contender = "primary_projected", None
    elif not ungated:
        reason, contender = "gap", best_projected_pk
    elif not gated:
        reason, contender = "dd_projected", None
    else:
        reason, contender = None, None
    return LockDecision(gated, best_projected, ungated, reason, contender)
```

Call sites: `run_single_check`: `decision = _lock_decision_from_predictions(...)`; `do_post, best_projected = decision.should_lock, decision.best_projected`. `_refresh_pick_at_fallback_decision`: build `FallbackRefreshResult(fresh, decision.should_lock, selection=sel, should_post_ungated=decision.should_lock_ungated, block_reason=decision.block_reason, contender_game_pk=decision.contender_game_pk)`; the two early-return paths keep `block_reason=None`.

- [ ] **Step 4: Run** — `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lock_decision.py tests/test_scheduler.py -q` → PASS (fix any test that unpacked the old tuple).

- [ ] **Step 5: Commit** — `git commit -m "refactor(scheduler): LockDecision carries block_reason + contender game"`

---

### Task 5: Pure fallback planner + budget/reserve config

**Files:**
- Modify: `src/bts/scheduler.py` (add `FallbackPlan`, `plan_fallback_action`, `effective_cascade_budget_min`, `_run_has_pending_side`, near `_should_defer_at_fallback` ~line 385)
- Test: `tests/test_fallback_plan.py`

**Interfaces:**
- Produces:
```python
@dataclass(frozen=True)
class FallbackPlan:
    action: str                    # "deliver" | "defer"
    reason: str
    window_time_et: datetime | None = None

def plan_fallback_action(*, now: datetime, cutoff: datetime, should_post: bool | None,
                         should_post_ungated: bool | None, block_reason: str | None,
                         contender_game_pk: int | None, remaining_runs: list[dict],
                         confirmed_sides: set[tuple[int, str]], budget_min: float,
                         operator_reserve_min: float) -> FallbackPlan

def effective_cascade_budget_min(config_min: float, recent_durations_sec: list[float]) -> float
    # max(config_min, ceil(max(last two durations)/60) + 2)
```
Config keys (read in `run_day`): `cascade_budget_min` (default 12), `operator_reserve_min` (default 10). Effective fallback deadline floor: `max(resolved, SUBMISSION_CUTOFF_MIN + budget_eff + operator_reserve_min)`.

Rules (in order):
1. `should_post is True` → deliver `"should_lock_true"`.
2. `should_post is None` → deliver `"lock_decision_unknown"` (fail-closed, unchanged).
3. `should_post_ungated is True` → deliver `"dd_gate_only"` (Codex L1, unchanged).
4. `deliver_by = cutoff − operator_reserve_min`. A run is *pending* if any `(game_pk, side)` of its games is not in `confirmed_sides`; its `effective_start = max(run.time_et, now)`; *feasible* if `effective_start + budget_min ≤ deliver_by`.
5. `block_reason == "gap"`: relevant = pending runs containing `contender_game_pk` (if `contender_game_pk is None` → all pending runs). Any feasible relevant run → defer `"gap_contender_window_feasible"`; else deliver `"gap_no_feasible_window"` ← **policy change (approved 2026-08-30)**: previously abandoned the enterable pick whenever any pending window existed.
6. `block_reason == "primary_projected"`: any pending run (feasible or not) → defer `"primary_projected_pending_window"`; else deliver `"primary_projected_no_window"`. (Unchanged product choice from the 2026-07-06 audit: a projected PRIMARY is abandoned for the later slate.)
7. Anything else (`status_failure`, `slot_unavailable`, unknown): legacy rule — any pending run → defer `"legacy_pending_window"`, else deliver `"legacy_no_window"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fallback_plan.py
"""plan_fallback_action — the deadline-aware replacement for the 12:50-snapshot boolean."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
T = lambda h, m: datetime(2026, 8, 30, h, m, tzinfo=ET)


def _runs():
    return [{"time_et": T(13, 10), "game_pks": [823662, 823740, 823010]},
            {"time_et": T(14, 10), "game_pks": [823580]},
            {"time_et": T(15, 5), "game_pks": [824959, 823987]},
            {"time_et": T(18, 20), "game_pks": [824636]}]


def _confirmed(*pks):
    return {(pk, side) for pk in pks for side in ("away", "home")}


def _plan(**kw):
    from bts.scheduler import plan_fallback_action
    base = dict(now=T(13, 20), cutoff=T(13, 35), should_post=False, should_post_ungated=False,
                block_reason="gap", contender_game_pk=823987, remaining_runs=_runs(),
                confirmed_sides=_confirmed(823662, 823740, 823010, 823580, 824959),
                budget_min=20, operator_reserve_min=10)
    base.update(kw)
    return plan_fallback_action(**base)


def test_incident_2026_08_30_delivers_kwan_at_13_20():
    plan = _plan()
    assert plan.action == "deliver" and plan.reason == "gap_no_feasible_window"


def test_feasible_contender_window_defers():
    # contender's run at 12:30, now 12:00, cutoff 15:35 → 12:30+20 = 12:50 ≤ 15:25
    runs = [{"time_et": T(12, 30), "game_pks": [823987]}]
    plan = _plan(now=T(12, 0), cutoff=T(15, 35), remaining_runs=runs, confirmed_sides=set())
    assert plan.action == "defer" and plan.window_time_et == T(12, 30)


def test_overrun_run_uses_now_as_start():
    # run scheduled 13:10, now 13:20, budget 20 → finishes 13:40; cutoff 14:00 → deliver_by 13:50 → feasible
    runs = [{"time_et": T(13, 10), "game_pks": [823987]}]
    assert _plan(cutoff=T(14, 0), remaining_runs=runs, confirmed_sides=set()).action == "defer"
    # cutoff 13:45 → deliver_by 13:35 < 13:40 → infeasible → deliver
    assert _plan(cutoff=T(13, 45), remaining_runs=runs, confirmed_sides=set()).action == "deliver"


def test_unrelated_pending_windows_do_not_defer_a_gap_block():
    # contender game 823987 already confirmed; 824636 pending but irrelevant
    plan = _plan(now=T(12, 0), cutoff=T(15, 35),
                 confirmed_sides=_confirmed(823662, 823740, 823010, 823580, 824959, 823987))
    assert plan.action == "deliver"


def test_primary_projected_defers_even_when_infeasible():
    plan = _plan(block_reason="primary_projected", contender_game_pk=None)
    assert plan.action == "defer" and plan.reason == "primary_projected_pending_window"


def test_primary_projected_with_no_pending_window_delivers():
    plan = _plan(block_reason="primary_projected", contender_game_pk=None,
                 confirmed_sides=_confirmed(823662, 823740, 823010, 823580, 824959, 823987, 824636))
    assert plan.action == "deliver"


def test_gate_only_and_unknown_and_true_deliver():
    assert _plan(should_post_ungated=True, block_reason="dd_projected").reason == "dd_gate_only"
    assert _plan(should_post=None, should_post_ungated=None).reason == "lock_decision_unknown"
    assert _plan(should_post=True).reason == "should_lock_true"


def test_effective_budget_uses_last_two_measured_runs():
    from bts.scheduler import effective_cascade_budget_min
    assert effective_cascade_budget_min(12, []) == 12
    assert effective_cascade_budget_min(12, [300.0, 930.0, 320.0]) == 12   # last two: 15.5, 5.3 → 18? no: max(930,320)=930s→16+2=18
```

Fix the last assertion to the intended semantics before running: `effective_cascade_budget_min(12, [300.0, 930.0, 320.0]) == 18` (max of the LAST TWO = 930 s → ceil(15.5)=16 → +2 = 18) and `effective_cascade_budget_min(12, [930.0, 300.0, 320.0]) == 12` (last two = 300, 320 → 6+2 = 8 → floor 12).

- [ ] **Step 2: Run to verify failure** — `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_fallback_plan.py -q` → FAIL (ImportError).

- [ ] **Step 3: Implement** (next to `_should_defer_at_fallback`)

```python
@dataclass(frozen=True)
class FallbackPlan:
    action: str                      # "deliver" | "defer"
    reason: str
    window_time_et: datetime | None = None


def _run_has_pending_side(run: dict, confirmed_sides: set[tuple[int, str]]) -> bool:
    return any((pk, side) not in confirmed_sides
               for pk in run["game_pks"] for side in ("away", "home"))


def effective_cascade_budget_min(config_min: float, recent_durations_sec: list[float]) -> float:
    """Assumed cascade duration: the config floor or the slower of the last two
    measured cascades plus 2 min, whichever is larger. Self-calibrates within the
    day (the first cascade includes the once-a-day season refresh)."""
    import math
    recent = [d for d in recent_durations_sec[-2:] if d is not None]
    if not recent:
        return float(config_min)
    return float(max(config_min, math.ceil(max(recent) / 60) + 2))


def plan_fallback_action(*, now: datetime, cutoff: datetime, should_post: bool | None,
                         should_post_ungated: bool | None, block_reason: str | None,
                         contender_game_pk: int | None, remaining_runs: list[dict],
                         confirmed_sides: set[tuple[int, str]], budget_min: float,
                         operator_reserve_min: float) -> FallbackPlan:
    """Decide the in-loop fallback AFTER the refresh, against the live clock.

    Deferral is only justified when a lineup-confirmation window that can change
    THIS decision can also finish before this pick must be delivered. The
    2026-08-30 incident deferred on a boolean snapshotted 30 minutes earlier that
    counted windows after first pitch."""
    if should_post is True:
        return FallbackPlan("deliver", "should_lock_true")
    if should_post is None:
        return FallbackPlan("deliver", "lock_decision_unknown")
    if should_post_ungated is True:
        return FallbackPlan("deliver", "dd_gate_only")

    deliver_by = cutoff - timedelta(minutes=operator_reserve_min)
    pending = [r for r in remaining_runs if _run_has_pending_side(r, confirmed_sides)]

    def feasible(run: dict) -> bool:
        start = max(run["time_et"], now)
        return start + timedelta(minutes=budget_min) <= deliver_by

    if block_reason == "gap":
        relevant = ([r for r in pending if contender_game_pk in r["game_pks"]]
                    if contender_game_pk is not None else pending)
        for r in relevant:
            if feasible(r):
                return FallbackPlan("defer", "gap_contender_window_feasible", r["time_et"])
        return FallbackPlan("deliver", "gap_no_feasible_window")
    if block_reason == "primary_projected":
        if pending:
            return FallbackPlan("defer", "primary_projected_pending_window", pending[0]["time_et"])
        return FallbackPlan("deliver", "primary_projected_no_window")
    if pending:
        return FallbackPlan("defer", "legacy_pending_window", pending[0]["time_et"])
    return FallbackPlan("deliver", "legacy_no_window")
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(scheduler): deadline-aware fallback planner (pure) + cascade budget"`

---

### Task 6: Wire the planner into run_day (recompute after refresh, coalesce overrun checks, measure durations)

**Files:**
- Modify: `src/bts/scheduler.py:2085-2100` (config reads), `:2173-2200` (loop → `enumerate`, coalescing, durations), `:2290-2400` (in-loop fallback branch), `:2410-2440` (post-loop `fallback_min` floor), `SchedulerState.fallback_refreshes`
- Test: `tests/test_scheduler.py` — update `test_fallback_defers_when_should_lock_false_and_future_checks_remain` (→ now DELIVERS: gap block, window after cutoff), add `test_fallback_defers_when_primary_projected_and_window_pending`, keep DD-earlier test as `primary_projected`; add `tests/test_incident_2026_08_30.py` (advancing clock).

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `SchedulerState.fallback_refreshes: list[dict] | None` entries `{started, finished, duration_sec, action, reason}`; `runs_completed[*]["duration_sec"]`; coalesced runs recorded as `{"time", "skipped": True, "reason": "coalesced_after_fallback", ...}`.

- [ ] **Step 1: Write the failing tests**

(a) In `tests/test_scheduler.py`, change `test_fallback_defers_when_should_lock_false_and_future_checks_remain` to expect DELIVERY and rename it `test_fallback_delivers_gap_block_when_window_cannot_finish_before_cutoff`: set `mock_refresh.return_value = FallbackRefreshResult(daily=daily, should_post=False, should_post_ungated=False, block_reason="gap", contender_game_pk=200)`, `mock_post.return_value = "at://p"`, `mock_poll.return_value = "final"`; assert `mock_post.assert_called_once()`, `(tmp_path/"2026-04-06.json").exists()`, no `deferred_fallback_*` archive, and `"gap_no_feasible_window" in captured.err`. Add its sibling:

```python
    def test_fallback_defers_when_primary_projected_and_window_pending(self, ...same decorators...):
        # identical setup, but block_reason="primary_projected" → still deferred (2026-07-06 product choice)
        mock_refresh.return_value = FallbackRefreshResult(daily=daily, should_post=False,
            should_post_ungated=False, block_reason="primary_projected")
        ... run_day(...)
        mock_post.assert_not_called()
        assert not (tmp_path / "2026-04-06.json").exists()
        archives = list((tmp_path / "2026-04-06").glob("deferred_fallback_*.json"))
        assert len(archives) == 1
        assert "primary_projected_pending_window" in capsys.readouterr().err
```

In `test_fallback_defers_when_double_down_game_creates_early_deadline` set `block_reason="primary_projected"` on the mocked refresh (that IS the 7/06 scenario) — assertions unchanged.

(b) New `tests/test_incident_2026_08_30.py` — advancing fake clock:

```python
"""Replay of 2026-08-30 with an advancing clock: the fallback refresh finishes at
13:20 (T−20); the planner must DELIVER Kwan then — not defer and run the overdue
13:10 check into the 13:35 cutoff."""
import json
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from tests.test_scheduler import _game

ET = ZoneInfo("America/New_York")
DATE = "2026-08-30"
CASCADE = timedelta(minutes=15, seconds=30)


class Clock:
    def __init__(self, start): self.now = start
    def __call__(self): return self.now
    def advance(self, delta): self.now = self.now + delta


def test_kwan_delivered_at_13_20(tmp_path, capsys):
    from bts.picks import DailyPick, Pick, save_pick
    from bts.scheduler import FallbackRefreshResult, run_day
    from bts.strategy import PickResult

    clock = Clock(datetime(2026, 8, 30, 12, 35, tzinfo=ET))
    kwan = Pick(batter_name="Steven Kwan", batter_id=680757, team="CLE", lineup_position=1,
                pitcher_name="Lugo", pitcher_id=607625, p_game_hit=0.7566, flags=[],
                projected_lineup=False, game_pk=824393, game_time="2026-08-30T17:40:00Z")
    daily = DailyPick(date=DATE, run_time="2026-08-30T16:50:55+00:00", pick=kwan,
                      double_down=None, runner_up=None)
    save_pick(daily, tmp_path)

    def fake_check(**kw):            # the 12:35 lineup check: 15.5-min cascade, should_lock False
        clock.advance(CASCADE)
        return {"skipped": False, "new_lineups": 4, "should_post": False,
                "pick_result": PickResult(daily=daily, locked=False),
                "pick_name": "Steven Kwan", "pick_p": 0.7566}

    def fake_refresh(config, date, cached, gap, **kw):   # the 13:05 fallback refresh
        clock.advance(CASCADE)
        return FallbackRefreshResult(daily=daily, should_post=False, should_post_ungated=False,
                                     block_reason="gap", contender_game_pk=823987)

    def fake_sleep(secs): clock.advance(timedelta(seconds=secs))

    schedule = [_game(824393, "13:40", date=DATE), _game(823662, "14:10", date=DATE),
                _game(823987, "16:07", date=DATE), _game(824636, "19:20", date=DATE)]
    with patch("bts.scheduler.fetch_schedule", side_effect=[schedule, []]), \
         patch("bts.scheduler._now_et", side_effect=clock), \
         patch("bts.scheduler.time.sleep", side_effect=fake_sleep), \
         patch("bts.scheduler.run_single_check", side_effect=fake_check) as mock_check, \
         patch("bts.scheduler._refresh_pick_at_fallback_decision", side_effect=fake_refresh), \
         patch("bts.scheduler.count_new_confirmations", return_value=0), \
         patch("bts.scheduler.run_result_polling", return_value="final"), \
         patch("bts.scheduler._trigger_live_forward_capture_on_lock"), \
         patch("bts.dm.send_dm", return_value="dm-1") as mock_dm, \
         patch("bts.contest_state.load_decision_streak_state") as dss:
        dss.return_value.streak = 0
        run_day(date=DATE, config={
            "orchestrator": {"picks_dir": str(tmp_path)}, "tiers": [],
            "bluesky": {"dm_recipient": "eric"},
            "scheduler": {"pick_delivery": "dm", "early_lock_gap": 0.03,
                          "lineup_check_offset_min": 60, "cluster_min": 10,
                          "doubleheader_recheck_min": 15, "fallback_deadline_min": 35,
                          "fallback_deadline_min_morning": 25, "results_poll_interval_min": 15,
                          "results_cap_hour_et": 5, "cascade_budget_min": 12,
                          "operator_reserve_min": 10},
        })
    mock_dm.assert_called_once()
    assert mock_check.call_count == 1                       # the overdue 13:10 check never ran
    delivered = json.loads((tmp_path / f"{DATE}.json").read_text())
    assert delivered["delivered_at"].startswith("2026-08-30T13:20")
    err = capsys.readouterr().err
    assert "gap_no_feasible_window" in err and "LOCKED (fallback)" in err
```

- [ ] **Step 2: Run to verify failure** — `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_incident_2026_08_30.py tests/test_scheduler.py -q -k "fallback"` → the incident test fails (DM at 13:36 / `deferred`), the renamed test fails.

- [ ] **Step 3: Implement** in `run_day`:

Config reads (with the others):
```python
    cascade_budget_min_cfg = float(sched_config.get("cascade_budget_min", 12))
    operator_reserve_min = float(sched_config.get("operator_reserve_min", 10))
```
Add to `SchedulerState`: `fallback_refreshes: list[dict] | None = None`.

Helper inside scheduler.py:
```python
def _measured_cascade_durations(state: SchedulerState) -> list[float]:
    out = [r.get("duration_sec") for r in state.runs_completed if r.get("duration_sec") is not None]
    out += [f.get("duration_sec") for f in (state.fallback_refreshes or []) if f.get("duration_sec") is not None]
    return out


def _fallback_min_with_floor(resolved_min: int, budget_min: float, operator_reserve_min: float) -> int:
    """fallback_deadline_min is the LATEST CASCADE START before first pitch; it must leave
    room for cutoff + reserve + a full cascade. Raises (never lowers) the configured value."""
    from bts.picks import SUBMISSION_CUTOFF_MIN
    import math
    floor = math.ceil(SUBMISSION_CUTOFF_MIN + budget_min + operator_reserve_min)
    if floor > resolved_min:
        print(f"  fallback deadline raised {resolved_min}→{floor} min (cutoff 5 + budget "
              f"{budget_min:.0f} + reserve {operator_reserve_min:.0f}).", file=sys.stderr)
        return floor
    return resolved_min
```

Loop: `coalesced: set[int] = set()`; `for run_idx, run_info in enumerate(runs):` — at the top:
```python
        if run_idx in coalesced:
            print(f"  [{_now_et().strftime('%H:%M ET')}] Skipping overrun {target.strftime('%H:%M')} check — "
                  f"covered by the fallback refresh (no new lineups since).", file=sys.stderr)
            state.runs_completed.append({"time": _now_et().isoformat(), "new_lineups": 0,
                                         "skipped": True, "reason": "coalesced_after_fallback",
                                         "pick_name": None, "pick_p": None})
            save_state(state, picks_dir)
            continue
```
Around `run_single_check`: `t0 = time.monotonic()` … `duration_sec = round(time.monotonic() - t0, 1)`; include `"duration_sec": duration_sec` in the `runs_completed` entry; pass `unavailable_game_pks=_games_past_cutoff(state.games, _now_et())`. Replace `run_idx = runs.index(run_info)` with the enumerate index.

In-loop fallback branch — replace the block from `# Is there a later check...` through the defer/deliver:
```python
            budget_eff = effective_cascade_budget_min(cascade_budget_min_cfg, _measured_cascade_durations(state))
            fallback_min = _fallback_min_with_floor(fallback_min, budget_eff, operator_reserve_min)
            fallback_deadline = earliest_game_et - timedelta(minutes=fallback_min)
            now = _now_et()
            future_runs = runs[run_idx + 1:]
            has_earlier_check = any(r["time_et"] <= fallback_deadline for r in future_runs)

            if not has_earlier_check:
                ...sleep to fallback_deadline (unchanged)...
                daily = load_pick(date, picks_dir)
                if daily and not pick_was_delivered(daily):
                    try:
                        count_new_confirmations(all_game_pks, confirmed_sides)   # sync BEFORE
                    except Exception:
                        pass
                    before_refresh = set(confirmed_sides)
                    t0 = time.monotonic(); started = _now_et()
                    try:
                        refresh = _refresh_pick_at_fallback_decision(
                            config, date, daily, early_lock_gap,
                            unavailable_game_pks=_games_past_cutoff(state.games, started))
                    except ContestStateError as e: ...unchanged...
                    duration_sec = round(time.monotonic() - t0, 1)
                    daily = refresh.daily
                    ...standing-skip handling unchanged...
                    try:
                        count_new_confirmations(all_game_pks, confirmed_sides)   # sync AFTER
                    except Exception:
                        pass
                    new_since_refresh = confirmed_sides - before_refresh
                    now = _now_et()
                    budget_eff = effective_cascade_budget_min(
                        cascade_budget_min_cfg, _measured_cascade_durations(state) + [duration_sec])
                    plan = plan_fallback_action(
                        now=now, cutoff=submission_cutoff_et(daily),
                        should_post=refresh.should_post, should_post_ungated=refresh.should_post_ungated,
                        block_reason=refresh.block_reason, contender_game_pk=refresh.contender_game_pk,
                        remaining_runs=future_runs, confirmed_sides=confirmed_sides,
                        budget_min=budget_eff, operator_reserve_min=operator_reserve_min)
                    state.fallback_refreshes = (state.fallback_refreshes or []) + [{
                        "started": started.isoformat(), "finished": now.isoformat(),
                        "duration_sec": duration_sec, "action": plan.action, "reason": plan.reason}]
                    save_state(state, picks_dir)
                    if plan.action == "defer":
                        archive = _defer_pick_at_fallback(picks_dir, date, daily, reason=plan.reason)
                        print(f"  FALLBACK DEFERRED — {plan.reason}; window at "
                              f"{plan.window_time_et.strftime('%H:%M ET') if plan.window_time_et else '?'}; "
                              f"archived {archive.name}.", file=sys.stderr)
                        # Coalesce overrun scheduled checks the refresh already covered.
                        for j in range(run_idx + 1, len(runs)):
                            r = runs[j]
                            if r["time_et"] <= now and not any(
                                    (pk, s) in new_since_refresh or (pk, s) not in confirmed_sides
                                    for pk in r["game_pks"] for s in ("away", "home")):
                                coalesced.add(j)
                        continue
                    print(f"  FALLBACK — delivering ({plan.reason}) before "
                          f"{submission_cutoff_et(daily).strftime('%H:%M ET')} cutoff.", file=sys.stderr)
                    _deliver_and_lock_pick(daily, config, picks_dir, state, date, "fallback",
                                           selection=refresh.selection)
```
Keep the archive reason string for the health check: `fallback_defer.py` reads `deferred_fallback.reason` (any string OK; existing tests assert the literal `should_lock_false_future_checks_remain` — update those two assertions to the new reasons: `primary_projected_pending_window`).

Post-loop final fallback (step 5): apply `_fallback_min_with_floor` to its `fallback_min` and pass `unavailable_game_pks` to its refresh. The `time` module is already imported in scheduler.py (verify with grep; add `import time` if not).

- [ ] **Step 4: Run** — `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_incident_2026_08_30.py tests/test_scheduler.py tests/test_e3_missed_pick_alert.py tests/test_daily_decision.py tests/test_daily_decision_v2.py -q` → PASS. Then the full fast suite.

- [ ] **Step 5: Commit** — message must state the policy change:

```
feat(scheduler): deadline-aware in-loop fallback

After the fallback refresh the loop now re-syncs confirmations, re-reads the
clock and asks plan_fallback_action whether a lineup window that can change
THIS decision can still finish before this pick's deliver-by time
(cutoff − operator_reserve). A gap-rule block with no such window now
DELIVERS the enterable pick (policy change, approved 2026-08-30; previously
it was abandoned whenever any pending window existed — 8/27, 8/29, 8/30). A
projected PRIMARY still defers (2026-07-06 product choice). Overrun scheduled
checks the refresh already covered are coalesced instead of re-cascading.
Cascade durations are measured and feed a self-calibrating budget that also
floors the fallback deadline at cutoff + reserve + budget.
```

---

### Task 7: `pull_feeds` — no delay on cache hits

**Files:**
- Modify: `src/bts/data/pull.py:83-120`
- Test: `tests/data/test_pull.py` (append)

- [ ] **Step 1: Failing tests**

```python
def test_pull_feeds_does_not_sleep_on_cache_hits(tmp_path):
    games = [{"gamePk": 111, "date": "2025-06-01"}, {"gamePk": 222, "date": "2025-06-01"}]
    (tmp_path / "2025").mkdir()
    (tmp_path / "2025" / "111.json").write_text("{}")
    (tmp_path / "2025" / "222.json").write_text("{}")
    with patch("bts.data.pull.discover_games", return_value=games), \
         patch("bts.data.pull.urlopen") as mock_open, \
         patch("bts.data.pull.time.sleep") as mock_sleep:
        paths = pull_feeds("2025-06-01", "2025-06-01", tmp_path, delay=0.3)
    assert len(paths) == 2
    mock_open.assert_not_called()
    mock_sleep.assert_not_called()


def test_pull_feeds_sleeps_only_after_real_downloads(tmp_path):
    games = [{"gamePk": 111, "date": "2025-06-01"}, {"gamePk": 222, "date": "2025-06-01"},
             {"gamePk": 333, "date": "2025-06-01"}]
    (tmp_path / "2025").mkdir()
    (tmp_path / "2025" / "222.json").write_text("{}")          # cached; 111 and 333 fetched
    sample = {"gameData": {}, "liveData": {}}

    def _mock(url, **kwargs):
        resp = MagicMock(); resp.read.return_value = json.dumps(sample).encode(); return resp

    with patch("bts.data.pull.discover_games", return_value=games), \
         patch("bts.data.pull.urlopen", side_effect=_mock) as mock_open, \
         patch("bts.data.pull.time.sleep") as mock_sleep:
        pull_feeds("2025-06-01", "2025-06-01", tmp_path, delay=0.3)
    assert mock_open.call_count == 2
    assert mock_sleep.call_count == 1          # after 111 only; 333 is the last item
```

- [ ] **Step 2: Run** → FAIL (sleep called 1× / 2×).
- [ ] **Step 3: Implement** — in the loop:

```python
    for i, game in enumerate(games):
        season = game["date"][:4]
        output_dir = data_dir / season
        cached = (output_dir / f"{game['gamePk']}.json").exists()
        try:
            path = download_game_feed(game["gamePk"], output_dir)
            paths.append(path)
        except Exception as e:
            failed.append(game["gamePk"])
            print(f"  SKIP {game['gamePk']}: {e}", file=sys.stderr)
        # Throttle only between REAL requests: 2,100 cached feeds × 0.3 s was
        # ~10.6 min of sleep per intraday prediction run (2026-08-30 incident).
        if not cached and delay > 0 and i < len(games) - 1:
            time.sleep(delay)
```
- [ ] **Step 4: Run** `tests/data/test_pull.py` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "perf(pull): do not sleep between cached feeds"`

---

### Task 8: Memoize the intraday season refresh

**Files:**
- Modify: `src/bts/model/predict.py:783-810` (`_refresh_season_data`)
- Test: `tests/test_refresh_memo.py`

**Interfaces:**
- Produces: marker `data/processed/.refreshed_{season}_through_{yesterday}` written after a successful pull+rebuild; env `BTS_REFRESH_ALWAYS=1` forces a refresh.

- [ ] **Step 1: Failing tests**

```python
# tests/test_refresh_memo.py
"""_refresh_season_data runs the pull+rebuild once per (season, yesterday) — later
intraday cascades skip the ~12-minute no-op re-pull (2026-08-30)."""
import os
from unittest.mock import patch
import pandas as pd


def _run(date, tmp_path):
    from bts.model.predict import _refresh_season_data
    _refresh_season_data(date, raw_dir=str(tmp_path / "raw"), processed_dir=str(tmp_path / "proc"))


def test_second_call_same_day_skips(tmp_path, monkeypatch):
    monkeypatch.delenv("BTS_REFRESH_ALWAYS", raising=False)
    (tmp_path / "raw" / "2026").mkdir(parents=True)
    with patch("bts.data.pull.pull_feeds", return_value=[]) as pull, \
         patch("bts.data.build.build_season", return_value=pd.DataFrame({"a": [1]})) as build:
        (tmp_path / "proc").mkdir()
        _run("2026-08-30", tmp_path)
        _run("2026-08-30", tmp_path)
    assert pull.call_count == 1 and build.call_count == 1
    assert (tmp_path / "proc" / ".refreshed_2026_through_2026-08-29").exists()


def test_next_day_refreshes_again(tmp_path, monkeypatch):
    monkeypatch.delenv("BTS_REFRESH_ALWAYS", raising=False)
    (tmp_path / "raw" / "2026").mkdir(parents=True); (tmp_path / "proc").mkdir()
    with patch("bts.data.pull.pull_feeds", return_value=[]) as pull, \
         patch("bts.data.build.build_season", return_value=pd.DataFrame({"a": [1]})):
        _run("2026-08-30", tmp_path)
        _run("2026-08-31", tmp_path)
    assert pull.call_count == 2


def test_force_env_refreshes(tmp_path, monkeypatch):
    monkeypatch.setenv("BTS_REFRESH_ALWAYS", "1")
    (tmp_path / "raw" / "2026").mkdir(parents=True); (tmp_path / "proc").mkdir()
    with patch("bts.data.pull.pull_feeds", return_value=[]) as pull, \
         patch("bts.data.build.build_season", return_value=pd.DataFrame({"a": [1]})):
        _run("2026-08-30", tmp_path); _run("2026-08-30", tmp_path)
    assert pull.call_count == 2


def test_failed_build_does_not_write_marker(tmp_path, monkeypatch):
    monkeypatch.delenv("BTS_REFRESH_ALWAYS", raising=False)
    (tmp_path / "raw" / "2026").mkdir(parents=True); (tmp_path / "proc").mkdir()
    with patch("bts.data.pull.pull_feeds", return_value=[]), \
         patch("bts.data.build.build_season", side_effect=RuntimeError("boom")):
        try:
            _run("2026-08-30", tmp_path)
        except RuntimeError:
            pass
    assert not (tmp_path / "proc" / ".refreshed_2026_through_2026-08-29").exists()
```

- [ ] **Step 2: Run** → FAIL (pull called twice; no marker).
- [ ] **Step 3: Implement** — in `_refresh_season_data` after computing `yesterday`/paths:

```python
    marker = proc / f".refreshed_{season}_through_{yesterday}"
    output_path = proc / f"pa_{season}.parquet"
    if (marker.exists() and output_path.exists()
            and os.environ.get("BTS_REFRESH_ALWAYS", "0") != "1"):
        print(f"  Season data already refreshed through {yesterday} "
              f"({marker.name}); skipping intraday re-pull.", file=sys.stderr)
        return
    print(f"  Refreshing {season} data through {yesterday}...", file=sys.stderr)
    paths = pull_feeds(season_start, yesterday, raw, delay=0.3)
    print(f"  {len(paths)} game feeds ...", file=sys.stderr)
    df = build_season(raw, output_path, season)
    print(f"  Rebuilt {output_path.name}: {len(df)} PAs", file=sys.stderr)
    proc.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.now().isoformat())
```
(`import os` at module top if absent.) Known limitation to document: a game from `yesterday` that finalizes later today (suspended/resumed) is picked up by tomorrow's first refresh, not intraday.

- [ ] **Step 4: Run** → PASS. Also run `tests/test_cli_preview_date.py -q`.
- [ ] **Step 5: Commit** — `git commit -m "perf(predict): refresh season data once per day, not per cascade"`

---

### Task 9: `late_delivery` health source + `fallback_defer` compares both slots

**Files:**
- Create: `src/bts/health/late_delivery.py`
- Modify: `src/bts/health/__init__.py` (export), `src/bts/health/runner.py:141` (register after `fallback_defer`), `src/bts/health/fallback_defer.py:169-181` (`same_pick` over primary AND double_down)
- Test: `tests/health/test_late_delivery.py`, extend `tests/health/test_fallback_defer.py`

**Interfaces:**
- Produces: `late_delivery.check(picks_dir, today=None, now=None, operator_reserve_min=10) -> list[Alert]`: CRITICAL when today's delivered pick has `delivered_at ≥ submission_cutoff` (fallback for `delivered_at is None`: `scheduler_state.pick_locked_at`), CRITICAL when `scheduler_state.delivery_refusals` is non-empty, WARN when delivered inside the operator reserve; `[]` when undelivered or file missing.

- [ ] **Step 1: Failing tests**

```python
# tests/health/test_late_delivery.py
import json
from datetime import date, datetime
from zoneinfo import ZoneInfo
from bts.picks import DailyPick, Pick, save_pick

ET = ZoneInfo("America/New_York")
D = date(2026, 8, 30)


def _delivered(tmp_path, delivered_at):
    pick = Pick(batter_name="Kwan", batter_id=1, team="CLE", lineup_position=1, pitcher_name="L",
                pitcher_id=2, p_game_hit=0.75, flags=[], projected_lineup=False, game_pk=100,
                game_time="2026-08-30T17:40:00Z")
    d = DailyPick(date="2026-08-30", run_time="x", pick=pick, double_down=None, runner_up=None,
                  notification_sent=True, notification_id="m", notification_channel="bluesky_dm",
                  delivered_at=delivered_at)
    save_pick(d, tmp_path)


def test_after_cutoff_is_critical(tmp_path):
    from bts.health import late_delivery
    _delivered(tmp_path, "2026-08-30T13:36:14-04:00")
    a = late_delivery.check(tmp_path, today=D)
    assert a and a[0].level == "CRITICAL" and "13:35" in a[0].message


def test_inside_reserve_is_warn(tmp_path):
    from bts.health import late_delivery
    _delivered(tmp_path, "2026-08-30T13:30:00-04:00")
    a = late_delivery.check(tmp_path, today=D)
    assert a and a[0].level == "WARN"


def test_comfortable_delivery_is_silent(tmp_path):
    from bts.health import late_delivery
    _delivered(tmp_path, "2026-08-30T12:50:00-04:00")
    assert late_delivery.check(tmp_path, today=D) == []


def test_refusal_on_state_is_critical(tmp_path):
    from bts.health import late_delivery
    (tmp_path / "2026-08-30").mkdir()
    (tmp_path / "2026-08-30" / "scheduler_state.json").write_text(json.dumps({
        "date": "2026-08-30", "schedule_fetched_at": "x", "games": [], "confirmed_game_pks": [],
        "runs_completed": [], "pick_locked": False, "pick_locked_at": None, "result_status": None,
        "next_wakeup": None, "delivery_refusals": [{"at": "2026-08-30T13:36:14-04:00",
        "label": "lineup", "batter": "Kwan", "cutoff_et": "2026-08-30T13:35:00-04:00", "late_min": 1.2}]}))
    a = late_delivery.check(tmp_path, today=D)
    assert a and a[0].level == "CRITICAL" and "refused" in a[0].message.lower()


def test_no_pick_is_silent(tmp_path):
    from bts.health import late_delivery
    assert late_delivery.check(tmp_path, today=D) == []
```

For `fallback_defer`, add to `tests/health/test_fallback_defer.py` a case where the primary matches but the archived DD differs → message contains `same_pick=false`.

- [ ] **Step 2: Run** → FAIL (module missing).
- [ ] **Step 3: Implement** `src/bts/health/late_delivery.py`:

```python
"""EOD backstop for the delivery-cutoff guard (2026-08-30 incident).

CRITICAL when the day's pick was sent at/after first pitch − 5 min (unenterable),
or when the scheduler refused a late delivery (nothing was sent). WARN when the
send landed inside the operator reserve — the pick was enterable but Eric had
less than `operator_reserve_min` to act on it."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from bts.health.alert import Alert
from bts.picks import load_pick, pick_was_delivered, submission_cutoff_et

SOURCE = "late_delivery"
ET = ZoneInfo("America/New_York")


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=ET)


def _state(picks_dir: Path, day: date) -> dict:
    try:
        return json.loads((Path(picks_dir) / day.isoformat() / "scheduler_state.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def check(picks_dir: Path, today: date | None = None, now: datetime | None = None,
          operator_reserve_min: float = 10) -> list[Alert]:
    day = today or (now or datetime.now(ET)).astimezone(ET).date()
    alerts: list[Alert] = []
    state = _state(picks_dir, day)
    for r in state.get("delivery_refusals") or []:
        alerts.append(Alert("CRITICAL", SOURCE, (
            f"late delivery REFUSED for {day.isoformat()}: {r.get('batter')} ({r.get('label')}) "
            f"was {r.get('late_min')} min past the {r.get('cutoff_et')} cutoff; nothing delivered "
            f"by that path")))
    try:
        daily = load_pick(day.isoformat(), picks_dir)
    except Exception:
        daily = None
    if daily is None or not pick_was_delivered(daily):
        return alerts
    delivered_at = _parse(getattr(daily, "delivered_at", None)) or _parse(state.get("pick_locked_at"))
    if delivered_at is None:
        return alerts
    cutoff = submission_cutoff_et(daily)
    names = daily.pick.batter_name + (f" + {daily.double_down.batter_name}" if daily.double_down else "")
    if delivered_at >= cutoff:
        alerts.append(Alert("CRITICAL", SOURCE, (
            f"late delivery for {day.isoformat()}: {names} sent {delivered_at.astimezone(ET):%H:%M} ET, "
            f"cutoff was {cutoff:%H:%M} ET — unenterable when delivered")))
    elif delivered_at > cutoff - timedelta(minutes=operator_reserve_min):
        alerts.append(Alert("WARN", SOURCE, (
            f"tight delivery for {day.isoformat()}: {names} sent {delivered_at.astimezone(ET):%H:%M} ET, "
            f"only {(cutoff - delivered_at).total_seconds() / 60:.0f} min before the {cutoff:%H:%M} ET cutoff")))
    return alerts
```
Register in `runner.py` right after `fallback_defer`: `alerts.extend(_safe_run("late_delivery", lambda: late_delivery.check(picks_dir, today=today)))` and add `late_delivery` to the `from bts.health import (...)` list + `health/__init__.py`. In `fallback_defer.check`, compute `same_pick = _same_pick(delivered, deferred) and _same_pick(_slot(vars(final_daily.double_down)) if final_daily.double_down else {}, _slot(latest_payload.get("double_down")))`.

- [ ] **Step 4: Run** `tests/health -q` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(health): late_delivery source; fallback_defer compares both slots"`

---

### Task 10: Docs, audit records, script

**Files:**
- Create: `docs/audit/2026-08-30-late-pick-delivery.md` (incident timeline, three-layer RCA, Codex review triage, fix summary, config keys, counterfactuals)
- Create: `docs/audit/2026-08-30-same-game-pair-correlation.md` + `scripts/audit/same_game_pair_lift.py` (from the scratchpad script; results table)
- Modify: `ARCHITECTURE.md` scheduler section (~line 247: fallback semantics; health sources table: add `late_delivery`; config keys `cascade_budget_min`, `operator_reserve_min`; refresh memo + `BTS_REFRESH_ALWAYS`), `CLAUDE.md` Architecture bullet on the fallback (one line), `docs/optimization-ideas.md` (note the singleton-slate backlog item now composes with the planner)

- [ ] **Step 1:** Write the audit doc (sections: Summary · Timeline · Root cause (3 layers) · What Codex corrected · Fix · Policy change and its scope · Counterfactuals (8/27, 8/29, 7/06) · Config · Follow-ups).
- [ ] **Step 2:** Copy the pair-correlation script, add the doc with the results table from the session (R values + CIs, matched controls, decision translation, caveats).
- [ ] **Step 3:** ARCHITECTURE/CLAUDE edits.
- [ ] **Step 4:** Commit — `git commit -m "docs: 2026-08-30 late-pick incident audit, planner semantics, pair-correlation backtest"`

---

### Task 11: Regression, adversarial review, deploy

- [ ] **Step 1:** Full fast suite green: `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest -m "not slow" --ignore=tests/simulate --ignore=tests/model --ignore=tests/experiment --ignore=tests/validate -q`.
- [ ] **Step 2:** Codex review round 2 via herdr (`w5:p2`): prompt file in scratchpad, assume defects, diff = `git diff main...HEAD`; triage findings (real / false-flag / over-engineered) with evidence; fix real ones; re-run suite; commit.
- [ ] **Step 3:** Merge to main (fast-forward), push `main`.
- [ ] **Step 4:** Wait for the idle window (journal on bts-hetzner prints "Idle until tomorrow's wakeup"), then `git push origin main:deploy`; watch the canary; confirm `systemctl --user is-active bts-scheduler`; next morning confirm the journal shows `already refreshed through … skipping` on the second cascade and cascade durations ~5 min.
- [ ] **Step 5:** If the deploy cannot happen before tomorrow 10:00 ET, apply the interim mitigation on the box: `fallback_deadline_min = 50` in `~/.bts-orchestrator.toml` (picked up at the 10:00 restart), and revert it after the deploy.
