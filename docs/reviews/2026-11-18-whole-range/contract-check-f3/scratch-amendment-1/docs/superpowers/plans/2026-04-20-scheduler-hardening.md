# Scheduler Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the BTS scheduler daemon against hangs, stalls, and tight morning-game windows so the system behaves correctly for once-per-season edge cases (Patriots' Day ~11:10 AM, London Series ~9:10 AM, July 4 morning games) AND silently-broken scheduler processes.

**Architecture:** Additive hardening layers — none change pick-making logic for normal cases. (1) External heartbeat-staleness cron pings healthchecks.io on daemon stall; (2) systemd `WatchdogSec` + `sd_notify` for in-kernel hang detection with auto-restart; (3) dynamic `fallback_deadline_min` that extends the lineup-wait window for pre-11-AM first pitches; (4) TZ round-trip test for London games.

**Tech Stack:** Python 3.12, systemd (user), healthchecks.io, pytest, bash.

---

## Background — what exists today

**Already built** (confirmed via code + production state 2026-04-20):

1. **Heartbeat module** `src/bts/heartbeat.py`: `write_heartbeat(path, state, sleeping_until=...)` writes JSON atomically with state (`running`/`sleeping`/`waiting_for_games`/`idle_end_of_day`), timestamp (UTC), optional `sleeping_until`. Currently written by the daemon at all major transitions.

2. **Systemd unit** `/home/bts/.config/systemd/user/bts-scheduler.service`: `Restart=always`, `RestartSec=300`, `Type=simple`. Catches process crashes — but NOT hangs where the process is alive but stuck. No `WatchdogSec` configured.

3. **Early-game wake-up** (`scheduler.py:compute_wakeup_time`): `early_game_buffer_min=60` in config wakes the daemon 60 min before earliest game. Works correctly for today's 11:10 AM Patriots' Day start.

4. **TZ handling**: clean `zoneinfo.ZoneInfo("America/New_York")` / `ZoneInfo("UTC")` everywhere. MLB StatsAPI `gameDate` is UTC; scheduler converts via `.astimezone(ET)`. No hardcoded offsets anywhere in `scheduler.py`.

5. **Fallback deadline** (`scheduler.py:623`): `fallback_deadline_min = sched_config.get("fallback_deadline_min", 15)`, production config is 35. Computes `earliest_game_et - fallback_deadline_min` as the "force-a-pick-even-without-confirmed-lineups" deadline.

**What's missing**:
- No external liveness watcher → daemon hangs are invisible until a missed pick.
- No `WatchdogSec` → daemon can deadlock for hours without systemd knowing.
- Fallback deadline is a fixed 35 min — for a 9:10 AM London first pitch, that's an 8:35 AM lock vs typical 8:10-AM lineup drops. Tight.
- No regression test for London-series TZ handling.

## Threat model

| Failure mode | Current detection | Gap | Task |
|---|---|---|---|
| Daemon process crashes | ✅ systemd `Restart=always` | — | — |
| Daemon hangs (stuck in a loop, deadlocked HTTP call, etc.) | ❌ None | **HIGH** | 1 + 3 |
| Daemon writes heartbeat but logic is wedged | ⚠️ Partial (heartbeat is fresh even if pick logic stuck) | MEDIUM | 1 (state-aware check) |
| Morning first-pitch lineup drops late (8:10+ for 9:10 AM game) | ❌ Force-picks with projected lineup at 8:35 | MEDIUM | 2 |
| London Series DST/TZ arithmetic wrong | ❌ No regression test | LOW (code looks right, but untested) | 4 |

## File Structure

- **Create**: `scripts/check_heartbeat.py` — staleness checker used by cron. Reads `data/.heartbeat`, compares vs state-specific thresholds, pings healthchecks.io fail endpoint if stale.
- **Modify**: `scripts/cron-setup-hetzner.sh` — add the `*/5` heartbeat check cron.
- **Modify**: `src/bts/scheduler.py:623` (fallback_deadline logic) — make `fallback_deadline_min` a function of earliest-game-time when in morning-game regime.
- **Modify**: `/home/bts/.bts-orchestrator.toml` — add `fallback_deadline_min_morning` config key.
- **Modify**: `/home/bts/.config/systemd/user/bts-scheduler.service` — add `WatchdogSec=600`, `Type=notify`. Requires code change in `scheduler.py` to call `sd_notify(WATCHDOG=1)` from main loop.
- **Create**: `tests/test_heartbeat_staleness.py` — unit tests for the staleness checker.
- **Create**: `tests/test_scheduler_fallback_deadline.py` — unit tests for dynamic deadline.
- **Create**: `tests/test_london_series_tz.py` — round-trip TZ test using MLB-API-shaped gameDate strings.

No files deleted. Production scheduler keeps running during all changes; cut-over is done piecewise with systemd reload.

---

## Task 1: Heartbeat staleness monitor cron

**Files:**
- Create: `scripts/check_heartbeat.py`
- Create: `tests/test_heartbeat_staleness.py`
- Modify: `scripts/cron-setup-hetzner.sh`

The scheduler's heartbeat file is already written — we just need an external watcher. The watcher runs every 5 min, compares heartbeat state + timestamp to expected thresholds:

- `state == "running"` → fresh if timestamp age < 5 min
- `state == "sleeping"` with `sleeping_until` in the future → fresh always (daemon is intentionally asleep)
- `state == "sleeping"` with `sleeping_until` in the past → must have woken recently; fresh if timestamp age < 10 min past `sleeping_until`, otherwise stale
- `state == "waiting_for_games"` → fresh if timestamp age < 10 min
- `state == "idle_end_of_day"` → fresh until midnight ET; stale if still in this state after 01:00 ET
- File missing → stale

On stale: curl healthchecks.io `/fail` endpoint using the `healthchecks-bts-ping-url` keychain creds. That triggers email/notification alert.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_heartbeat_staleness.py
"""Tests for scripts/check_heartbeat.py staleness decision logic."""
from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import sys
sys.path.insert(0, "scripts")
from check_heartbeat import is_stale


def _write_hb(path: Path, **kv) -> None:
    path.write_text(json.dumps(kv))


def test_fresh_running_state(tmp_path):
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    _write_hb(hb_path, state="running", timestamp=now.isoformat())
    stale, _ = is_stale(hb_path, now=now)
    assert not stale


def test_stale_running_state(tmp_path):
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    old = now - timedelta(minutes=8)
    _write_hb(hb_path, state="running", timestamp=old.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
    assert "running" in reason.lower()


def test_sleeping_with_future_wakeup_is_fresh(tmp_path):
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    wake = now + timedelta(hours=2)
    _write_hb(hb_path, state="sleeping",
              timestamp=(now - timedelta(hours=1)).isoformat(),
              sleeping_until=wake.isoformat())
    stale, _ = is_stale(hb_path, now=now)
    assert not stale


def test_sleeping_past_wakeup_is_stale(tmp_path):
    """If sleeping_until is in the past by >10 min, daemon should have woken."""
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    wake = now - timedelta(minutes=15)  # past due
    _write_hb(hb_path, state="sleeping",
              timestamp=(now - timedelta(hours=2)).isoformat(),
              sleeping_until=wake.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
    assert "sleeping" in reason.lower()


def test_missing_file_is_stale(tmp_path):
    hb_path = tmp_path / "missing.json"
    stale, reason = is_stale(hb_path)
    assert stale
    assert "not found" in reason.lower()


def test_corrupt_file_is_stale(tmp_path):
    hb_path = tmp_path / "hb.json"
    hb_path.write_text("{not valid json")
    stale, reason = is_stale(hb_path)
    assert stale


def test_idle_end_of_day_after_1am_is_stale(tmp_path):
    """Daemon should transition off idle_end_of_day by 1 AM ET."""
    from zoneinfo import ZoneInfo
    hb_path = tmp_path / "hb.json"
    # Construct a "now" at 01:30 ET
    et = ZoneInfo("America/New_York")
    now = datetime(2026, 4, 21, 1, 30, tzinfo=et).astimezone(timezone.utc)
    ts = datetime(2026, 4, 20, 23, 30, tzinfo=timezone.utc)  # 2h old
    _write_hb(hb_path, state="idle_end_of_day", timestamp=ts.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
```

- [ ] **Step 2: Run tests to verify failure (missing module)**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_heartbeat_staleness.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'check_heartbeat'`

- [ ] **Step 3: Implement the staleness checker**

```python
# scripts/check_heartbeat.py
"""Heartbeat staleness checker — runs every 5 min via cron on bts-hetzner.

Usage:
    python3 scripts/check_heartbeat.py [--heartbeat-path PATH] [--ping-url URL]

Returns:
    Exit code 0 if fresh. Exit code 1 + POST to hc-ping /fail if stale.
    Exit code 2 on internal error (unreadable heartbeat file etc).

Integration: invoke from cron like
    */5 * * * * cd /home/bts/projects/bts && /home/bts/.local/bin/uv run \\
        python scripts/check_heartbeat.py --heartbeat-path data/.heartbeat \\
        --ping-url "$(security find-generic-password -s healthchecks-bts-scheduler-ping-url -w)" \\
        >> /home/bts/logs/heartbeat.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# State -> staleness thresholds (seconds)
RUNNING_MAX_AGE = 5 * 60          # running: fresh = timestamp age < 5 min
WAITING_MAX_AGE = 10 * 60         # waiting_for_games: 10 min
SLEEPING_OVERRUN = 10 * 60        # sleeping: if past sleeping_until, fresh = <10 min overshoot
IDLE_END_HOUR_ET = 1              # idle_end_of_day state is stale after 01:00 ET next day


def is_stale(
    path: Path,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Return (is_stale, reason). `now` is optional for tests; defaults to datetime.now(UTC)."""
    if now is None:
        now = datetime.now(timezone.utc)

    if not path.exists():
        return True, f"heartbeat file not found: {path}"

    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return True, f"heartbeat unreadable: {e}"

    ts_str = raw.get("timestamp")
    state = raw.get("state", "unknown")
    try:
        ts = datetime.fromisoformat(ts_str)
    except (TypeError, ValueError):
        return True, f"heartbeat timestamp invalid: {ts_str}"
    age_s = (now - ts).total_seconds()

    if state == "running":
        if age_s > RUNNING_MAX_AGE:
            return True, f"running state but timestamp {age_s:.0f}s old (>{RUNNING_MAX_AGE}s)"
        return False, "fresh running"

    if state == "waiting_for_games":
        if age_s > WAITING_MAX_AGE:
            return True, f"waiting_for_games but timestamp {age_s:.0f}s old"
        return False, "fresh waiting"

    if state == "sleeping":
        wake_str = raw.get("sleeping_until")
        if not wake_str:
            # No wake target at all is suspicious
            return True, "sleeping state without sleeping_until"
        try:
            wake = datetime.fromisoformat(wake_str)
        except ValueError:
            return True, f"sleeping_until invalid: {wake_str}"
        overshoot = (now - wake).total_seconds()
        if overshoot > SLEEPING_OVERRUN:
            return True, f"sleeping past sleeping_until by {overshoot:.0f}s (>{SLEEPING_OVERRUN}s)"
        return False, "fresh sleeping"

    if state == "idle_end_of_day":
        # Check: is current time past 01:00 ET of the next calendar day after timestamp?
        ts_et = ts.astimezone(ET)
        now_et = now.astimezone(ET)
        cutoff = ts_et.replace(hour=IDLE_END_HOUR_ET, minute=0, second=0, microsecond=0) + timedelta(days=1)
        if now_et > cutoff:
            return True, f"idle_end_of_day past {IDLE_END_HOUR_ET:02d}:00 ET cutoff"
        return False, "fresh idle_end_of_day"

    return True, f"unknown state: {state}"


def ping(url: str, suffix: str = "") -> None:
    full = url + suffix
    req = urllib.request.Request(full, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
    except Exception as e:
        print(f"  ping failed: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heartbeat-path", type=Path, required=True)
    ap.add_argument("--ping-url", default=None,
                    help="Healthchecks.io base URL (without /fail suffix)")
    args = ap.parse_args()

    stale, reason = is_stale(args.heartbeat_path)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] stale={stale}  reason={reason}")

    if stale:
        if args.ping_url:
            ping(args.ping_url, "/fail")
        sys.exit(1)

    if args.ping_url:
        ping(args.ping_url)  # success ping keeps hc-ping "up"
    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — all should pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_heartbeat_staleness.py -v`

Expected: 7 PASS.

- [ ] **Step 5: Create the healthchecks.io scheduler check**

Manually via healthchecks.io UI OR via the API (user's preference). The check should:
- Name: `bts-scheduler-heartbeat`
- Schedule: cron `*/5 * * * *` with grace 5 min
- Alert: email to `stone.ericm@gmail.com` on fail

Save the resulting ping URL to macOS keychain as `healthchecks-bts-scheduler-ping-url`:

Run on Mac:
```bash
security add-generic-password -a "claude-cli" -s "healthchecks-bts-scheduler-ping-url" -w "https://hc-ping.com/XXXXXXXX"
```

Also create the equivalent secret on bts-hetzner (either via `security` equivalent or by writing to `~/.bts-secrets.env` gitignored).

- [ ] **Step 6: Add the cron line to `scripts/cron-setup-hetzner.sh`**

In the Hetzner cron section (the one that writes the `# BTS-HETZNER` tagged lines), append:

```bash
*/5 * * * * cd /home/bts/projects/bts && set -a && . ./.env && set +a && /home/bts/.local/bin/uv run python scripts/check_heartbeat.py --heartbeat-path data/.heartbeat --ping-url "$BTS_SCHEDULER_HEARTBEAT_PING_URL" >> /home/bts/logs/heartbeat.log 2>&1 # BTS-HETZNER
```

Add `BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.com/XXX` to `/home/bts/projects/bts/.env` (NOT committed; .env is gitignored).

- [ ] **Step 7: Install the cron + verify healthcheck succeeds**

Run on Mac (deploys to Hetzner via existing cron-setup-hetzner.sh workflow):

```bash
ssh bts-hetzner 'cd /home/bts/projects/bts && bash scripts/cron-setup-hetzner.sh install'
```

Then wait 6 min and verify:
```bash
ssh bts-hetzner 'tail /home/bts/logs/heartbeat.log'
# Expected: "stale=False  reason=fresh sleeping"
```

Visit healthchecks.io dashboard — the `bts-scheduler-heartbeat` check should show "up" after first success ping.

- [ ] **Step 8: Commit Task 1**

```bash
git add scripts/check_heartbeat.py tests/test_heartbeat_staleness.py scripts/cron-setup-hetzner.sh
git commit -m "feat(scheduler): heartbeat staleness monitor cron

Runs every 5 min. Reads data/.heartbeat, validates state-specific freshness:
  running        -> fresh if timestamp < 5m old
  waiting_games  -> fresh if timestamp < 10m old
  sleeping       -> fresh if now < sleeping_until + 10m
  idle_end_of_day-> fresh until 01:00 ET next day

On stale: hc-ping /fail -> email alert."
```

---

## Task 2: Dynamic `fallback_deadline_min` for morning games

**Files:**
- Modify: `src/bts/scheduler.py:623` — change fixed `fallback_deadline_min` lookup to dynamic function
- Modify: `/home/bts/.bts-orchestrator.toml` — add new config key
- Create: `tests/test_scheduler_fallback_deadline.py`

**The current logic**: `fallback_deadline = earliest_game_et - timedelta(minutes=fallback_deadline_min)` where `fallback_deadline_min=35` is fixed. For a 9:10 AM London game, lock = 8:35 AM ET. Lineups typically drop 60-90 min before first pitch (8:10-8:40 AM for a 9:10 start), so at 8:35 lineups may not have dropped yet, forcing a projected-lineup pick.

**Proposed logic**: for morning games (first-pitch before 11 AM ET), use a shorter fallback buffer (25 min). For 9:10 AM, lock = 8:45 AM — 10 more minutes of lineup-wait tolerance. Normal-time games (first-pitch after 11 AM) keep the 35 min buffer unchanged.

- [ ] **Step 1: Write the failing unit test**

```python
# tests/test_scheduler_fallback_deadline.py
"""Unit tests for scheduler's dynamic fallback_deadline_min logic."""
from __future__ import annotations
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def test_normal_evening_game_uses_standard_buffer():
    from bts.scheduler import resolve_fallback_deadline_min

    game_et = datetime(2026, 4, 20, 19, 10, tzinfo=ET)  # 7:10 PM
    m = resolve_fallback_deadline_min(
        earliest_game_et=game_et,
        standard_min=35,
        morning_min=25,
        morning_cutoff_hour=11,
    )
    assert m == 35


def test_patriots_day_morning_game_uses_morning_buffer():
    from bts.scheduler import resolve_fallback_deadline_min

    game_et = datetime(2026, 4, 20, 11, 10, tzinfo=ET)  # 11:10 AM Patriots Day
    m = resolve_fallback_deadline_min(
        earliest_game_et=game_et,
        standard_min=35,
        morning_min=25,
        morning_cutoff_hour=11,
    )
    # Cutoff is "before 11 AM" — 11:10 is NOT before 11 AM so standard applies
    assert m == 35


def test_london_morning_game_uses_morning_buffer():
    from bts.scheduler import resolve_fallback_deadline_min

    game_et = datetime(2026, 6, 20, 9, 10, tzinfo=ET)  # London 9:10 AM ET
    m = resolve_fallback_deadline_min(
        earliest_game_et=game_et,
        standard_min=35,
        morning_min=25,
        morning_cutoff_hour=11,
    )
    assert m == 25


def test_exactly_cutoff_hour_uses_standard():
    from bts.scheduler import resolve_fallback_deadline_min

    game_et = datetime(2026, 4, 20, 11, 0, tzinfo=ET)  # 11:00 AM exactly
    m = resolve_fallback_deadline_min(
        earliest_game_et=game_et,
        standard_min=35,
        morning_min=25,
        morning_cutoff_hour=11,
    )
    assert m == 35  # cutoff is strict "before 11"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler_fallback_deadline.py -v`

Expected: FAIL with `ImportError: cannot import name 'resolve_fallback_deadline_min' from 'bts.scheduler'`

- [ ] **Step 3: Add the function to scheduler.py**

Add near `compute_wakeup_time` (around line 106 in scheduler.py):

```python
def resolve_fallback_deadline_min(
    earliest_game_et: "datetime",
    standard_min: int = 35,
    morning_min: int = 25,
    morning_cutoff_hour: int = 11,
) -> int:
    """Return fallback_deadline_min adjusted for morning games.

    For games with first pitch strictly before morning_cutoff_hour (in ET),
    use morning_min instead of standard_min. This gives morning games
    (Patriots' Day, London Series, July 4) more lineup-wait tolerance
    before force-picking with projected lineups.
    """
    if earliest_game_et.hour < morning_cutoff_hour:
        return morning_min
    return standard_min
```

- [ ] **Step 4: Run the unit tests — should pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler_fallback_deadline.py -v`

Expected: 4 PASS.

- [ ] **Step 5: Wire it into the scheduler body**

Modify `src/bts/scheduler.py` around line 623:

```python
# BEFORE:
fallback_deadline_min = sched_config.get("fallback_deadline_min", 15)

# AFTER:
fallback_deadline_min_standard = sched_config.get("fallback_deadline_min", 15)
fallback_deadline_min_morning = sched_config.get("fallback_deadline_min_morning", 25)
morning_cutoff_hour = sched_config.get("morning_cutoff_hour", 11)
```

And at line 771 where fallback_deadline is computed:

```python
# BEFORE:
fallback_deadline = earliest_game_et - timedelta(minutes=fallback_deadline_min)

# AFTER:
fallback_min = resolve_fallback_deadline_min(
    earliest_game_et,
    standard_min=fallback_deadline_min_standard,
    morning_min=fallback_deadline_min_morning,
    morning_cutoff_hour=morning_cutoff_hour,
)
fallback_deadline = earliest_game_et - timedelta(minutes=fallback_min)
```

- [ ] **Step 6: Add integration test using scheduler state load**

```python
# Append to tests/test_scheduler_fallback_deadline.py
def test_scheduler_body_uses_dynamic_deadline_for_morning_game(tmp_path):
    """Integration: a full scheduler pass on a morning-game day uses 25 not 35."""
    # This is a smoke test — mock the config + one game at 9:10 ET, assert
    # the computed fallback_deadline is 8:45 (not 8:35).
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from bts.scheduler import resolve_fallback_deadline_min

    ET = ZoneInfo("America/New_York")
    game = datetime(2026, 6, 22, 9, 10, tzinfo=ET)
    m = resolve_fallback_deadline_min(game)
    expected_lock = game - timedelta(minutes=m)
    assert expected_lock == datetime(2026, 6, 22, 8, 45, tzinfo=ET)
```

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler_fallback_deadline.py -v`

Expected: 5 PASS.

- [ ] **Step 7: Update orchestrator config on bts-hetzner**

Manually edit `/home/bts/.bts-orchestrator.toml` (ssh bts-hetzner, then vim):

```toml
[scheduler]
# ... existing ...
fallback_deadline_min = 35
fallback_deadline_min_morning = 25   # NEW: for games before 11 AM ET
morning_cutoff_hour = 11              # NEW
```

- [ ] **Step 8: Commit + deploy**

```bash
git add src/bts/scheduler.py tests/test_scheduler_fallback_deadline.py
git commit -m "feat(scheduler): dynamic fallback_deadline_min for morning games

For games with first pitch before 11 AM ET (Patriots' Day ~11:10 AM excluded
since >= 11, London Series ~9:10 AM, July 4 morning games), use a shorter
fallback_deadline_min (default 25) to give 10 extra minutes of lineup-wait
tolerance before force-picking projected lineups.

Orchestrator config on bts-hetzner must also add:
  fallback_deadline_min_morning = 25
  morning_cutoff_hour = 11"
```

Deploy: `git push origin main` triggers the auto-deploy workflow on bts-hetzner (per existing deploy reference). Then manually add the two config lines to `/home/bts/.bts-orchestrator.toml` and restart:

```bash
ssh bts-hetzner 'systemctl --user restart bts-scheduler'
```

---

## Task 3: Systemd `WatchdogSec` for in-kernel hang detection

**Files:**
- Modify: `/home/bts/.config/systemd/user/bts-scheduler.service`
- Modify: `src/bts/scheduler.py` — add `sd_notify` heartbeat call in main loop

Systemd's `WatchdogSec=N` feature: if the daemon doesn't call `sd_notify(WATCHDOG=1)` within N seconds, systemd considers it hung and restarts it per `Restart=` policy. Hardcore hang-detection at the kernel IPC level. Complements Task 1 (which detects stalls from OUTSIDE) with one that fires from systemd itself.

Requires the daemon to call `sd_notify()` periodically. Python has `sdnotify` via `systemd-python` or a trivial socket write. Prefer the explicit socket-write path — fewer dependencies.

**Cadence**: scheduler wakes every ~60 sec when there's work pending, and sleeps longer between games. We need sd_notify pings at <WatchdogSec cadence during both. `WatchdogSec=600` (10 min) is a reasonable balance — covers sleep periods up to 10 min and requires ping at least once per wake cycle.

### Phase B watchdog cadence — known gotcha

`src/bts/scheduler.py`'s `run_result_polling` loop contains a `time.sleep(poll_interval_min * 60)` — by default `results_poll_interval_min = 15`, meaning 900s (15 min) between iterations. Each iteration fires `notify_watchdog()`, but `WatchdogSec=600` (10 min) would false-kill the daemon during mid-game result polling.

**Recommended**: Phase B should set `WatchdogSec=1800` (30 min) — comfortable headroom above the 15-min poll interval. Alternatively, refactor the poll sleep to `time.sleep(min(poll_interval_min * 60, 60))` in a loop that emits `notify_watchdog()` on each iteration — cleaner but touches reviewed scheduler logic.

The YAGNI-consistent choice for Phase B: go with `WatchdogSec=1800`.

- [ ] **Step 1: Write a test for the sd_notify helper**

```python
# Append to tests/test_heartbeat_staleness.py — OR new file
# tests/test_sd_notify.py
"""Test the sd_notify helper handles missing NOTIFY_SOCKET gracefully."""
import os
import socket
import tempfile

from bts.sd_notify import notify_watchdog


def test_no_socket_is_noop(monkeypatch):
    """If NOTIFY_SOCKET env is missing, notify_watchdog() should silently no-op."""
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    # Should not raise
    notify_watchdog()


def test_sends_watchdog_message_when_socket_set(monkeypatch, tmp_path):
    """When NOTIFY_SOCKET is set to a valid unix socket, send WATCHDOG=1."""
    sock_path = str(tmp_path / "notify.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(sock_path)
    srv.settimeout(2)
    monkeypatch.setenv("NOTIFY_SOCKET", sock_path)

    notify_watchdog()

    data, _ = srv.recvfrom(64)
    assert data == b"WATCHDOG=1"
```

- [ ] **Step 2: Implement the sd_notify helper**

```python
# src/bts/sd_notify.py
"""Minimal sd_notify client for systemd watchdog integration.

No dependencies beyond stdlib. Silently no-ops when not running under systemd
(NOTIFY_SOCKET env var not set).
"""
from __future__ import annotations

import os
import socket


def notify_raw(message: str) -> None:
    """Send a raw sd_notify message to $NOTIFY_SOCKET. No-op if unset."""
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return
    # @ prefix = abstract namespace on Linux
    if sock_path.startswith("@"):
        sock_path = "\0" + sock_path[1:]
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.sendto(message.encode(), sock_path)
    except OSError:
        pass  # not fatal — heartbeat/staleness cron still covers us
    finally:
        sock.close()


def notify_watchdog() -> None:
    """Reset systemd's watchdog timer."""
    notify_raw("WATCHDOG=1")


def notify_ready() -> None:
    """Tell systemd the daemon is ready (for Type=notify units)."""
    notify_raw("READY=1")
```

- [ ] **Step 3: Run tests to verify passing**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sd_notify.py -v`

Expected: 2 PASS.

- [ ] **Step 4: Wire into scheduler main loop**

In `src/bts/scheduler.py`, near the top of the main loop (around the main `while True:` / work-tick):

```python
from bts.sd_notify import notify_ready, notify_watchdog

# At startup, once:
notify_ready()

# At every heartbeat write in the main loop:
write_heartbeat(heartbeat_path, state=HeartbeatState.RUNNING, ...)
notify_watchdog()
```

Every `write_heartbeat` should be paired with `notify_watchdog()`. The scheduler already calls `write_heartbeat` frequently in the main loop — pair them.

- [ ] **Step 5: Update the systemd unit file**

Edit `/home/bts/.config/systemd/user/bts-scheduler.service` on bts-hetzner:

```ini
[Unit]
Description=BTS Dynamic Lineup Scheduler
After=network-online.target
Wants=network-online.target

[Service]
Type=notify              # CHANGED from 'simple'
WatchdogSec=600          # NEW: 10-min watchdog
NotifyAccess=main        # NEW: only main process sends notifications
WorkingDirectory=/home/bts/projects/bts
Environment=UV_CACHE_DIR=/tmp/uv-cache
Environment=PATH=/home/bts/.local/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=/home/bts/projects/bts/.env
ExecStart=/home/bts/.local/bin/uv run bts schedule --config /home/bts/.bts-orchestrator.toml
Restart=always
RestartSec=30            # CHANGED: was 300; faster restart now watchdog can catch hangs

[Install]
WantedBy=default.target
```

- [ ] **Step 6: Reload + restart + verify**

```bash
ssh bts-hetzner
systemctl --user daemon-reload
systemctl --user restart bts-scheduler
systemctl --user status bts-scheduler --no-pager
# Expected:
#   Active: active (running)
#   Status: "..."
#   Watchdog: 10min ...
```

Verify systemd is receiving watchdog pings:
```bash
journalctl --user -u bts-scheduler -n 50 --no-pager | grep -i "watchdog\|notify"
```

Monitor for 15 min — daemon should stay up and not trigger a watchdog restart.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/bts/sd_notify.py tests/test_sd_notify.py src/bts/scheduler.py
git commit -m "feat(scheduler): systemd sd_notify + WatchdogSec hang detection

Scheduler now calls sd_notify(WATCHDOG=1) on every heartbeat write.
Systemd unit: Type=notify, WatchdogSec=600, RestartSec=30.

If daemon hangs (stuck in HTTP call, deadlock, etc.) and misses a watchdog
ping for >10 min, systemd kills + restarts it within 30s. Complements the
external heartbeat-staleness cron from Task 1."
```

Also commit the updated unit file to the repo (if tracked) so future re-provisions apply it automatically. Check: `ls config/systemd/` or `ls deploy/` — if there's a canonical unit file location in the repo, update it.

---

## Task 4: London Series TZ regression test

**Files:**
- Create: `tests/test_london_series_tz.py`

TZ handling in scheduler.py uses `zoneinfo` consistently. Low risk of bugs, but London is the one case where DST matters differently (UK switches DST on different dates than US) AND a morning-ET game combined with the British-based UTC offset creates an unusual time-of-day ambiguity. Add a regression test that feeds in a realistic London-game `gameDate` (UTC string from MLB StatsAPI) and verifies we compute the correct ET first-pitch, wake-up, and fallback-deadline.

- [ ] **Step 1: Write the test**

```python
# tests/test_london_series_tz.py
"""Regression test: London Series (first pitch 2:10 PM BST) TZ round-trip.

Ensures our scheduler converts the MLB StatsAPI UTC gameDate to ET correctly
during British Summer Time and handles the morning-game fallback_deadline_min
dispatch properly.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from bts.scheduler import compute_wakeup_time, resolve_fallback_deadline_min

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def test_london_series_first_pitch_converts_correctly():
    """A 2:10 PM London (BST, UTC+1) first pitch = 13:10 UTC = 09:10 ET (during EDT)."""
    # Simulate MLB StatsAPI output: gameDate is UTC
    utc_first_pitch = datetime(2026, 6, 20, 13, 10, tzinfo=UTC)
    et_first_pitch = utc_first_pitch.astimezone(ET)
    assert et_first_pitch.hour == 9
    assert et_first_pitch.minute == 10


def test_london_series_uses_morning_fallback_buffer():
    """London 9:10 AM ET first pitch should trigger the morning fallback_deadline_min."""
    game_et = datetime(2026, 6, 20, 9, 10, tzinfo=ET)
    m = resolve_fallback_deadline_min(
        earliest_game_et=game_et,
        standard_min=35, morning_min=25, morning_cutoff_hour=11,
    )
    assert m == 25


def test_london_series_wakeup_at_8_10_et():
    """Scheduler should wake 60 min before earliest game = 8:10 AM ET for 9:10 AM first pitch."""
    # compute_wakeup_time takes a list of game dicts. Shape matches
    # what fetch_schedule returns. Construct minimal fake.
    games = [{"game_time_et": datetime(2026, 6, 20, 9, 10, tzinfo=ET),
              "game_pk": 999999,
              "is_doubleheader_game2": False,
              "lineup_confirmed": False}]
    wake = compute_wakeup_time(
        games,
        default_hour_et=10,
        early_buffer_min=60,
    )
    assert wake == datetime(2026, 6, 20, 8, 10, tzinfo=ET)


def test_dst_boundary_march_game():
    """On DST transition day (US spring-forward), 9:10 AM ET is correctly computed."""
    # March 8, 2026 is US spring-forward Sunday
    # UK doesn't spring-forward until March 29, 2026
    # So a London game on March 15 would be: London 2:10 PM GMT = 14:10 UTC = 10:10 AM ET (EDT)
    utc_first_pitch = datetime(2026, 3, 15, 14, 10, tzinfo=UTC)
    et_first_pitch = utc_first_pitch.astimezone(ET)
    assert et_first_pitch.hour == 10
    assert et_first_pitch.minute == 10
    # Morning buffer should still apply at 10:10 AM
    m = resolve_fallback_deadline_min(
        earliest_game_et=et_first_pitch,
        standard_min=35, morning_min=25, morning_cutoff_hour=11,
    )
    assert m == 25
```

- [ ] **Step 2: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_london_series_tz.py -v`

Expected: 4 PASS.

If FAIL on the `compute_wakeup_time` test, inspect the actual return shape vs. the test's expected structure — that function's argument format might differ from our test fixture. Adjust the test, not the function.

- [ ] **Step 3: Commit Task 4**

```bash
git add tests/test_london_series_tz.py
git commit -m "test(scheduler): London Series TZ regression tests

Verifies:
  - MLB UTC gameDate -> ET conversion correct for BST/EDT overlap
  - 9:10 AM ET London first pitch triggers morning fallback_deadline_min
  - Wake-up computed at 8:10 AM ET (60min early_buffer)
  - DST-transition boundary (US spring-forward before UK) handled correctly"
```

---

## Out of scope (deferred)

- **Alerting richness**: currently relies on healthchecks.io email. Could add Bluesky DM on stale heartbeat. Not urgent.
- **Lineup-drop monitoring**: detect when BOS/DET-style early games have NOT dropped lineups by game_time-60min and warn. Useful but only after Task 1 surfaces real patterns.
- **Separate Type=forking complication**: our `uv run` wrapper spawns a child — `Type=notify` works because the main PID (uv's child Python) sends the notify. If systemd considers uv's parent PID as the main, notifications might not register. Confirm during Task 3 Step 6.
- **Journal log rotation**: if the daemon is restarted hundreds of times in a loop (e.g., WatchdogSec too tight), we'd fill the journal. Mitigated by `RestartSec=30s` rate-limit.

## Success criteria

This plan is complete when:
1. `scripts/check_heartbeat.py` runs every 5 min on bts-hetzner, pinging hc-ping on stale heartbeat.
2. Simulated hang test: stop the scheduler with `kill -STOP <pid>`, wait 7 min, verify hc-ping `/fail` triggers. Restart with `kill -CONT <pid>`.
3. `morning_cutoff_hour=11` + `fallback_deadline_min_morning=25` active in production config. Unit tests cover the dispatch logic.
4. Systemd `WatchdogSec=600` active; daemon survives a real 24h period without spurious restart.
5. London Series TZ tests pass. If a real London game is on the schedule, first pick runs cleanly through the new morning-buffer path.
