# Dynamic Lineup Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed 3x/day cron with a game-time-aware scheduler that checks lineups 45 min before each game, commits picks only on confirmed lineups (with gap threshold), and polls for results until resolution.

**Architecture:** A long-running scheduler daemon on Pi5 fetches today's schedule, computes lineup check times (`game_time - 45min`), invokes the existing orchestrator cascade when new confirmed lineups appear, and manages the full daily lifecycle including Bluesky posting and result checking. The existing `orchestrate()` function becomes a callable building block; `should_post_now()` is replaced by the scheduler's confirmation-based logic.

**Tech Stack:** Python 3.12, Click, asyncio.sleep (for timing), tomllib (config), existing orchestrator/strategy/posting modules.

---

## File Structure

| File | Responsibility |
|---|---|
| **Create:** `src/bts/scheduler.py` | Schedule computation, lineup monitoring, daily lifecycle loop, state persistence, result polling |
| **Create:** `tests/test_scheduler.py` | Unit tests for schedule computation, clustering, gap logic, doubleheader detection, result polling |
| **Modify:** `src/bts/strategy.py` | Add `should_lock()` with early lock gap logic |
| **Modify:** `src/bts/picks.py` | Add `suspended` and `unresolved` result types to `check_hit()` and `get_game_statuses()` |
| **Modify:** `src/bts/orchestrator.py` | Extract `run_and_pick()` from `orchestrate()` — cascade + strategy, no posting |
| **Modify:** `src/bts/cli.py` | Add `bts schedule` CLI command |
| **Modify:** `src/bts/posting.py` | Remove `should_post_now()` (replaced by scheduler), keep posting mechanics |
| **Create:** `scripts/bts-scheduler.service` | systemd unit for Pi5 |
| **Modify:** `scripts/cron-setup-pi5.sh` | Remove 3 prediction crons, keep 1am check-results as safety net |

---

### Task 1: Extend picks.py — Game Status and Result Types

**Files:**
- Modify: `src/bts/picks.py:142-156` (get_game_statuses)
- Modify: `src/bts/picks.py:37` (DailyPick.result docstring)
- Test: `tests/test_picks.py`

- [ ] **Step 1: Write failing test for extended game statuses**

```python
# In tests/test_picks.py — add to existing file

class TestGetGameStatusesExtended:
    @patch("bts.picks.retry_urlopen")
    def test_returns_suspended_status(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps({
            "dates": [{"games": [
                {"gamePk": 111, "status": {"abstractGameCode": "F"}},
                {"gamePk": 222, "status": {
                    "abstractGameCode": "L",
                    "detailedState": "Suspended",
                }},
            ]}],
        }).encode()
        from bts.picks import get_game_statuses_detailed
        statuses = get_game_statuses_detailed("2026-04-03")
        assert statuses[111] == {"abstract": "F", "detailed": "Final"}
        assert statuses[222] == {"abstract": "L", "detailed": "Suspended"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_picks.py::TestGetGameStatusesExtended -v`
Expected: FAIL — `get_game_statuses_detailed` not defined

- [ ] **Step 3: Implement `get_game_statuses_detailed()`**

Add to `src/bts/picks.py` after the existing `get_game_statuses()` (line 156):

```python
def get_game_statuses_detailed(date: str) -> dict[int, dict[str, str]]:
    """Get detailed game statuses for all games on a date.

    Returns {game_pk: {"abstract": code, "detailed": state}} where:
        abstract: P = Preview, L = Live, F = Final
        detailed: e.g. "Suspended", "Delayed Start", "Final", "In Progress"
    """
    resp = json.loads(retry_urlopen(
        f"{API_BASE}/api/v1/schedule?sportId=1&date={date}",
        timeout=15,
    ).read())
    statuses = {}
    for d in resp.get("dates", []):
        for g in d.get("games", []):
            statuses[g["gamePk"]] = {
                "abstract": g["status"]["abstractGameCode"],
                "detailed": g["status"].get("detailedState", ""),
            }
    return statuses
```

- [ ] **Step 4: Update DailyPick.result type annotation**

In `src/bts/picks.py`, update line 37:

```python
    result: str | None = None  # "hit", "miss", "suspended", "unresolved", or None (pending)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_picks.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/bts/picks.py tests/test_picks.py
git commit -m "feat(picks): add detailed game statuses and suspended/unresolved result types"
```

---

### Task 2: Extract `run_and_pick()` from Orchestrator

**Files:**
- Modify: `src/bts/orchestrator.py:99-200`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing test for `run_and_pick()`**

```python
# Add to tests/test_orchestrator.py

class TestRunAndPick:
    @patch("bts.orchestrator.run_cascade")
    @patch("bts.strategy.get_game_statuses", return_value={778899: "P"})
    @patch("bts.strategy._load_mdp", return_value=None)
    def test_returns_predictions_and_result(self, _mdp, _statuses, mock_cascade, tmp_path):
        import pandas as pd
        from bts.orchestrator import run_and_pick

        mock_cascade.return_value = (
            pd.DataFrame(json.loads(SAMPLE_PREDICTIONS)),
            "mac",
        )
        config = {
            "orchestrator": {"picks_dir": str(tmp_path)},
            "tiers": [{"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5}],
        }
        predictions, pick_result, tier = run_and_pick(config, "2026-04-01")

        assert predictions is not None
        assert len(predictions) == 1
        assert tier == "mac"
        assert pick_result is not None
        assert pick_result.daily.pick.batter_name == "Jacob Wilson"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py::TestRunAndPick -v`
Expected: FAIL — `run_and_pick` not importable

- [ ] **Step 3: Extract `run_and_pick()` from `orchestrate()`**

In `src/bts/orchestrator.py`, add a new function and refactor `orchestrate()` to call it:

```python
def run_and_pick(
    config: dict,
    date: str,
) -> tuple[pd.DataFrame | None, "PickResult | None", str | None]:
    """Run cascade and apply strategy. No posting, no DMs.

    Returns (predictions, pick_result, tier_name).
    predictions is None if all tiers fail.
    pick_result is None if skip or no games.
    """
    from bts.picks import load_streak
    from bts.strategy import select_pick

    picks_dir = Path(config["orchestrator"]["picks_dir"])

    predictions, tier_name = run_cascade(config["tiers"], date)
    if predictions is None or predictions.empty:
        return predictions, None, tier_name

    streak = load_streak(picks_dir)
    result = select_pick(predictions, date, picks_dir, streak=streak)

    return predictions, result, tier_name


def orchestrate(config_path: Path, date: str) -> bool:
    """Run the full orchestration: cascade -> strategy -> save -> post.

    Returns True if a pick was made, False otherwise.
    """
    from bts.dm import send_dm
    from bts.picks import save_pick, load_streak
    from bts.posting import format_post, format_skip_post, post_to_bluesky, should_post_now

    config = load_config(config_path)
    picks_dir = Path(config["orchestrator"]["picks_dir"])
    dm_recipient = config["bluesky"]["dm_recipient"]

    predictions, result, tier_name = run_and_pick(config, date)

    if predictions is None:
        msg = f"BTS {date}: All compute tiers failed. No pick made."
        print(msg, file=sys.stderr)
        try:
            send_dm(dm_recipient, msg)
            print(f"  DM sent to {dm_recipient}", file=sys.stderr)
        except Exception as e:
            print(f"  DM failed: {e}", file=sys.stderr)
        return False

    if predictions.empty:
        print(f"No games found for {date}.", file=sys.stderr)
        return False

    if result is None:
        top = predictions.iloc[0] if not predictions.empty else None
        if top is not None:
            streak = load_streak(picks_dir)
            print(f"Skipping — {top['batter_name']} at {top['p_game_hit']:.1%} "
                  f"below threshold. Streak holds at {streak}.", file=sys.stderr)
            if should_post_now(top.get("game_time", ""), False):
                text = format_skip_post(top["batter_name"], top.get("team", "?"),
                                        top["p_game_hit"], streak)
                try:
                    uri = post_to_bluesky(text)
                    print(f"  Posted skip to Bluesky: {uri}", file=sys.stderr)
                except Exception as e:
                    print(f"  Bluesky skip post failed: {e}", file=sys.stderr)
        else:
            print(f"No valid picks. Streak holds at {load_streak(picks_dir)}.", file=sys.stderr)
        return False

    if result.locked:
        print(f"Pick locked: {result.daily.pick.batter_name}", file=sys.stderr)
        if not result.daily.bluesky_posted:
            streak = load_streak(picks_dir)
            text = format_post(
                result.daily.pick.batter_name, result.daily.pick.team,
                result.daily.pick.pitcher_name, result.daily.pick.p_game_hit,
                streak,
                result.daily.double_down.batter_name if result.daily.double_down else None,
                result.daily.double_down.p_game_hit if result.daily.double_down else None,
            )
            try:
                uri = post_to_bluesky(text)
                result.daily.bluesky_posted = True
                result.daily.bluesky_uri = uri
                save_pick(result.daily, picks_dir)
                print(f"  Posted to Bluesky (catch-up): {uri}", file=sys.stderr)
            except Exception as e:
                print(f"  Bluesky catch-up failed: {e}", file=sys.stderr)
        return True

    daily = result.daily
    save_pick(daily, picks_dir)
    print(
        f"Pick ({tier_name}): {daily.pick.batter_name} "
        f"({daily.pick.p_game_hit:.1%})",
        file=sys.stderr,
    )

    streak = load_streak(picks_dir)
    if should_post_now(daily.pick.game_time, daily.bluesky_posted):
        text = format_post(
            daily.pick.batter_name, daily.pick.team,
            daily.pick.pitcher_name, daily.pick.p_game_hit, streak,
            daily.double_down.batter_name if daily.double_down else None,
            daily.double_down.p_game_hit if daily.double_down else None,
        )
        try:
            uri = post_to_bluesky(text)
            daily.bluesky_posted = True
            daily.bluesky_uri = uri
            save_pick(daily, picks_dir)
            print(f"  Posted to Bluesky: {uri}", file=sys.stderr)
        except Exception as e:
            print(f"  Bluesky post failed: {e}", file=sys.stderr)

    return True
```

- [ ] **Step 4: Run all orchestrator tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py -v`
Expected: All PASS (existing tests still pass + new test passes)

- [ ] **Step 5: Commit**

```bash
git add src/bts/orchestrator.py tests/test_orchestrator.py
git commit -m "refactor(orchestrator): extract run_and_pick() for scheduler reuse"
```

---

### Task 3: Add `should_lock()` to Strategy

**Files:**
- Modify: `src/bts/strategy.py`
- Test: `tests/test_strategy.py`

- [ ] **Step 1: Write failing tests for `should_lock()`**

```python
# Add to tests/test_strategy.py

class TestShouldLock:
    def test_locks_when_all_confirmed(self):
        from bts.strategy import should_lock

        top_pick = {"p_game_hit": 0.82, "projected_lineup": False, "game_pk": 100}
        all_picks = [
            {"p_game_hit": 0.82, "projected_lineup": False, "game_pk": 100},
            {"p_game_hit": 0.79, "projected_lineup": False, "game_pk": 200},
        ]
        assert should_lock(top_pick, all_picks, early_lock_gap=0.03) is True

    def test_locks_when_gap_exceeds_threshold(self):
        from bts.strategy import should_lock

        top_pick = {"p_game_hit": 0.85, "projected_lineup": False, "game_pk": 100}
        all_picks = [
            {"p_game_hit": 0.85, "projected_lineup": False, "game_pk": 100},
            {"p_game_hit": 0.80, "projected_lineup": True, "game_pk": 200},
        ]
        # Gap is 0.05, threshold is 0.03 — lock
        assert should_lock(top_pick, all_picks, early_lock_gap=0.03) is True

    def test_waits_when_gap_below_threshold(self):
        from bts.strategy import should_lock

        top_pick = {"p_game_hit": 0.83, "projected_lineup": False, "game_pk": 100}
        all_picks = [
            {"p_game_hit": 0.83, "projected_lineup": False, "game_pk": 100},
            {"p_game_hit": 0.82, "projected_lineup": True, "game_pk": 200},
        ]
        # Gap is 0.01, threshold is 0.03 — wait
        assert should_lock(top_pick, all_picks, early_lock_gap=0.03) is False

    def test_waits_when_top_pick_is_projected(self):
        from bts.strategy import should_lock

        top_pick = {"p_game_hit": 0.85, "projected_lineup": True, "game_pk": 100}
        all_picks = [
            {"p_game_hit": 0.85, "projected_lineup": True, "game_pk": 100},
        ]
        assert should_lock(top_pick, all_picks, early_lock_gap=0.03) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_strategy.py::TestShouldLock -v`
Expected: FAIL — `should_lock` not defined

- [ ] **Step 3: Implement `should_lock()`**

Add to `src/bts/strategy.py` after the `_apply_densest_bucket()` function (after line 138):

```python
def should_lock(
    top_pick: dict,
    all_picks: list[dict],
    early_lock_gap: float,
) -> bool:
    """Decide if the current top pick should be locked (posted to Bluesky).

    Locks when:
    1. The top pick has a confirmed (not projected) lineup, AND
    2. Either all picks have confirmed lineups, OR the gap between
       the top pick and the best projected-lineup pick exceeds early_lock_gap.
    """
    if top_pick.get("projected_lineup", True):
        return False

    # Find the best projected-lineup pick (excluding the top pick's game)
    best_projected = None
    for p in all_picks:
        if p.get("projected_lineup", False) and p["game_pk"] != top_pick["game_pk"]:
            if best_projected is None or p["p_game_hit"] > best_projected["p_game_hit"]:
                best_projected = p

    if best_projected is None:
        # All confirmed — safe to lock
        return True

    return (top_pick["p_game_hit"] - best_projected["p_game_hit"]) >= early_lock_gap
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_strategy.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/bts/strategy.py tests/test_strategy.py
git commit -m "feat(strategy): add should_lock() with early lock gap threshold"
```

---

### Task 4: Schedule Computation — Core Scheduler Logic

**Files:**
- Create: `src/bts/scheduler.py`
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: Write failing tests for schedule computation**

```python
# tests/test_scheduler.py
"""Tests for the dynamic lineup scheduler."""

import json
import pytest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _game(game_pk: int, time_et: str, team_away: str = "NYM", team_home: str = "ATL"):
    """Build a mock MLB schedule game entry."""
    # Convert ET time string to UTC ISO format
    et_dt = datetime.strptime(f"2026-04-03 {time_et}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    utc_iso = et_dt.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
    return {
        "gamePk": game_pk,
        "gameDate": utc_iso,
        "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
        "teams": {
            "away": {"team": {"name": team_away}},
            "home": {"team": {"name": team_home}},
        },
    }


class TestComputeRunTimes:
    def test_single_game(self):
        from bts.scheduler import compute_run_times

        games = [_game(100, "19:05")]
        runs = compute_run_times(games, offset_min=45, cluster_min=10)

        assert len(runs) == 1
        # 19:05 - 45min = 18:20
        assert runs[0]["time_et"].hour == 18
        assert runs[0]["time_et"].minute == 20
        assert runs[0]["game_pks"] == [100]

    def test_clusters_nearby_games(self):
        from bts.scheduler import compute_run_times

        games = [
            _game(100, "19:05"),
            _game(200, "19:10"),
            _game(300, "19:15"),
        ]
        runs = compute_run_times(games, offset_min=45, cluster_min=10)

        # 18:20, 18:25, 18:30 — all within 10min of 18:20 → one cluster
        assert len(runs) == 1
        assert sorted(runs[0]["game_pks"]) == [100, 200, 300]

    def test_separates_distant_games(self):
        from bts.scheduler import compute_run_times

        games = [
            _game(100, "13:10"),  # check at 12:25
            _game(200, "19:05"),  # check at 18:20
        ]
        runs = compute_run_times(games, offset_min=45, cluster_min=10)

        assert len(runs) == 2
        assert runs[0]["game_pks"] == [100]
        assert runs[1]["game_pks"] == [200]


class TestDetectDoubleheaderGame2:
    def test_finds_doubleheader(self):
        from bts.scheduler import detect_doubleheader_game2s

        games = [
            _game(100, "13:10", "NYM", "ATL"),
            _game(200, "19:05", "NYM", "ATL"),  # same teams = DH game 2
            _game(300, "19:10", "LAD", "SF"),
        ]
        dh2s = detect_doubleheader_game2s(games)
        assert dh2s == {200}

    def test_no_doubleheader(self):
        from bts.scheduler import detect_doubleheader_game2s

        games = [
            _game(100, "19:05", "NYM", "ATL"),
            _game(200, "19:10", "LAD", "SF"),
        ]
        dh2s = detect_doubleheader_game2s(games)
        assert dh2s == set()


class TestComputeWakeUpTime:
    def test_default_when_no_early_games(self):
        from bts.scheduler import compute_wakeup_time

        games = [_game(100, "19:05")]
        wakeup = compute_wakeup_time(games, default_hour_et=10, early_buffer_min=60)
        assert wakeup.hour == 10
        assert wakeup.minute == 0

    def test_early_wakeup_for_international_game(self):
        from bts.scheduler import compute_wakeup_time

        games = [_game(100, "06:10"), _game(200, "19:05")]
        wakeup = compute_wakeup_time(games, default_hour_et=10, early_buffer_min=60)
        # 6:10 - 60min = 5:10
        assert wakeup.hour == 5
        assert wakeup.minute == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler.py -v`
Expected: FAIL — `bts.scheduler` not found

- [ ] **Step 3: Implement schedule computation functions**

Create `src/bts/scheduler.py`:

```python
"""Dynamic lineup scheduler for BTS.

Replaces fixed cron runs with game-time-aware lineup checks.
Checks lineups 45 min before each game, clusters nearby checks,
and commits picks only when confirmed lineup + gap threshold met.
"""

import json
import sys
import time
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
from zoneinfo import ZoneInfo

from bts.util import retry_urlopen
from bts.picks import API_BASE

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def fetch_schedule(date: str) -> list[dict]:
    """Fetch today's MLB schedule. Returns list of game dicts."""
    resp = json.loads(retry_urlopen(
        f"{API_BASE}/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher",
        timeout=15,
    ).read())
    games = []
    for d in resp.get("dates", []):
        games.extend(d.get("games", []))
    return games


def _game_time_et(game: dict) -> datetime:
    """Extract game time as ET datetime."""
    utc = datetime.fromisoformat(game["gameDate"].replace("Z", "+00:00"))
    return utc.astimezone(ET)


def compute_run_times(
    games: list[dict],
    offset_min: int = 45,
    cluster_min: int = 10,
) -> list[dict]:
    """Compute clustered lineup check times from game schedule.

    For each game, the check time is game_time - offset_min.
    Checks within cluster_min of each other are merged into one run.

    Returns list of {"time_et": datetime, "game_pks": [int, ...]}
    sorted by time.
    """
    if not games:
        return []

    checks = []
    for g in games:
        et = _game_time_et(g)
        check_time = et - timedelta(minutes=offset_min)
        checks.append({"time_et": check_time, "game_pk": g["gamePk"]})

    checks.sort(key=lambda c: c["time_et"])

    # Cluster
    clusters = []
    current = {"time_et": checks[0]["time_et"], "game_pks": [checks[0]["game_pk"]]}

    for c in checks[1:]:
        if (c["time_et"] - current["time_et"]) <= timedelta(minutes=cluster_min):
            current["game_pks"].append(c["game_pk"])
        else:
            clusters.append(current)
            current = {"time_et": c["time_et"], "game_pks": [c["game_pk"]]}

    clusters.append(current)
    return clusters


def detect_doubleheader_game2s(games: list[dict]) -> set[int]:
    """Detect game 2 of doubleheaders (fluid start time).

    Returns set of game_pks that are doubleheader game 2s.
    Detected by finding two games with the same away+home team pair.
    """
    team_games = {}
    for g in games:
        away = g["teams"]["away"]["team"]["name"]
        home = g["teams"]["home"]["team"]["name"]
        key = (away, home)
        team_games.setdefault(key, []).append(g)

    game2s = set()
    for key, team_g in team_games.items():
        if len(team_g) >= 2:
            # Sort by game time — later one is game 2
            team_g.sort(key=lambda x: _game_time_et(x))
            for g in team_g[1:]:
                game2s.add(g["gamePk"])

    return game2s


def compute_wakeup_time(
    games: list[dict],
    default_hour_et: int = 10,
    early_buffer_min: int = 60,
) -> datetime:
    """Compute scheduler wake-up time based on earliest game.

    If any game starts before the default init hour, wakes up
    early_buffer_min before the earliest game.
    """
    today_et = datetime.now(ET).replace(hour=default_hour_et, minute=0, second=0, microsecond=0)

    if not games:
        return today_et

    earliest = min(_game_time_et(g) for g in games)
    early_wake = earliest - timedelta(minutes=early_buffer_min)

    if early_wake < today_et:
        return early_wake

    return today_et
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/bts/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): schedule computation — run times, clustering, doubleheader detection"
```

---

### Task 5: Lineup Confirmation Detection

**Files:**
- Modify: `src/bts/scheduler.py`
- Modify: `tests/test_scheduler.py`

- [ ] **Step 1: Write failing tests for lineup confirmation**

```python
# Add to tests/test_scheduler.py

class TestCheckConfirmedLineups:
    @patch("bts.scheduler.retry_urlopen")
    def test_detects_confirmed_lineup(self, mock_urlopen):
        from bts.scheduler import check_confirmed_lineups

        # Mock a game feed where boxscore has battingOrder
        feed = {
            "liveData": {"boxscore": {"teams": {
                "away": {"players": {
                    "ID123": {"battingOrder": "100", "person": {"fullName": "A"}},
                    "ID456": {"battingOrder": "200", "person": {"fullName": "B"}},
                }},
                "home": {"players": {
                    "ID789": {"battingOrder": "100", "person": {"fullName": "C"}},
                }},
            }}},
        }
        mock_urlopen.return_value.read.return_value = json.dumps(feed).encode()

        result = check_confirmed_lineups([111])
        assert result == {111: True}

    @patch("bts.scheduler.retry_urlopen")
    def test_detects_no_lineup(self, mock_urlopen):
        from bts.scheduler import check_confirmed_lineups

        feed = {
            "liveData": {"boxscore": {"teams": {
                "away": {"players": {}},
                "home": {"players": {}},
            }}},
        }
        mock_urlopen.return_value.read.return_value = json.dumps(feed).encode()

        result = check_confirmed_lineups([111])
        assert result == {111: False}

    @patch("bts.scheduler.retry_urlopen")
    def test_counts_new_confirmations(self, mock_urlopen):
        from bts.scheduler import count_new_confirmations

        feed_confirmed = {
            "liveData": {"boxscore": {"teams": {
                "away": {"players": {"ID1": {"battingOrder": "100", "person": {"fullName": "A"}}}},
                "home": {"players": {"ID2": {"battingOrder": "100", "person": {"fullName": "B"}}}},
            }}},
        }
        mock_urlopen.return_value.read.return_value = json.dumps(feed_confirmed).encode()

        previously_confirmed = set()
        new_count = count_new_confirmations([111], previously_confirmed)
        assert new_count == 1
        assert 111 in previously_confirmed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler.py::TestCheckConfirmedLineups -v`
Expected: FAIL — functions not defined

- [ ] **Step 3: Implement lineup confirmation functions**

Add to `src/bts/scheduler.py`:

```python
def check_confirmed_lineups(game_pks: list[int]) -> dict[int, bool]:
    """Check which games have confirmed lineups posted.

    A lineup is confirmed if the boxscore has players with battingOrder set.
    Returns {game_pk: has_confirmed_lineup}.
    """
    results = {}
    for pk in game_pks:
        try:
            resp = json.loads(retry_urlopen(
                f"{API_BASE}/api/v1.1/game/{pk}/feed/live",
                timeout=15,
            ).read())
            has_lineup = False
            for side in ("away", "home"):
                players = resp["liveData"]["boxscore"]["teams"][side]["players"]
                for pid, pdata in players.items():
                    if pdata.get("battingOrder"):
                        has_lineup = True
                        break
                if has_lineup:
                    break
            results[pk] = has_lineup
        except Exception:
            results[pk] = False

    return results


def count_new_confirmations(
    game_pks: list[int],
    previously_confirmed: set[int],
) -> int:
    """Check for new lineup confirmations since last check.

    Updates previously_confirmed in place. Returns count of newly confirmed games.
    """
    statuses = check_confirmed_lineups(game_pks)
    new_count = 0
    for pk, confirmed in statuses.items():
        if confirmed and pk not in previously_confirmed:
            previously_confirmed.add(pk)
            new_count += 1
    return new_count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/bts/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): lineup confirmation detection via boxscore batting order"
```

---

### Task 6: State Persistence

**Files:**
- Modify: `src/bts/scheduler.py`
- Modify: `tests/test_scheduler.py`

- [ ] **Step 1: Write failing tests for state save/load**

```python
# Add to tests/test_scheduler.py

class TestSchedulerState:
    def test_save_and_load_roundtrip(self, tmp_path):
        from bts.scheduler import SchedulerState, save_state, load_state

        state = SchedulerState(
            date="2026-04-03",
            schedule_fetched_at="2026-04-03T10:00:00-04:00",
            games=[{"game_pk": 100, "game_time_et": "2026-04-03T19:05:00-04:00",
                     "lineup_confirmed": False, "is_doubleheader_game2": False}],
            confirmed_game_pks=[],
            runs_completed=[],
            pick_locked=False,
            pick_locked_at=None,
            result_status=None,
            next_wakeup=None,
        )
        save_state(state, tmp_path)

        loaded = load_state("2026-04-03", tmp_path)
        assert loaded is not None
        assert loaded.date == "2026-04-03"
        assert len(loaded.games) == 1
        assert loaded.pick_locked is False

    def test_load_returns_none_when_missing(self, tmp_path):
        from bts.scheduler import load_state

        assert load_state("2026-04-03", tmp_path) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler.py::TestSchedulerState -v`
Expected: FAIL — `SchedulerState` not defined

- [ ] **Step 3: Implement state dataclass and persistence**

Add to `src/bts/scheduler.py`:

```python
from dataclasses import dataclass, asdict, field


@dataclass
class SchedulerState:
    """Daily scheduler state, persisted to JSON."""
    date: str
    schedule_fetched_at: str
    games: list[dict]  # [{game_pk, game_time_et, lineup_confirmed, is_doubleheader_game2}]
    confirmed_game_pks: list[int]
    runs_completed: list[dict]  # [{time, new_lineups, skipped}]
    pick_locked: bool
    pick_locked_at: str | None
    result_status: str | None  # "final", "suspended", "unresolved", None
    next_wakeup: str | None  # ISO for next day's wake-up


def save_state(state: SchedulerState, picks_dir: Path) -> Path:
    """Save scheduler state to JSON."""
    date_dir = picks_dir / state.date
    date_dir.mkdir(parents=True, exist_ok=True)
    path = date_dir / "scheduler_state.json"
    path.write_text(json.dumps(asdict(state), indent=2))
    return path


def load_state(date: str, picks_dir: Path) -> SchedulerState | None:
    """Load scheduler state from JSON. Returns None if not found."""
    path = picks_dir / date / "scheduler_state.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return SchedulerState(**data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/bts/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): state persistence — save/load daily scheduler state"
```

---

### Task 7: Main Scheduler Loop

**Files:**
- Modify: `src/bts/scheduler.py`
- Modify: `tests/test_scheduler.py`

- [ ] **Step 1: Write failing test for the scheduler loop**

```python
# Add to tests/test_scheduler.py

class TestSchedulerRun:
    @patch("bts.scheduler.fetch_schedule")
    @patch("bts.scheduler.check_confirmed_lineups")
    @patch("bts.scheduler.time.sleep")
    @patch("bts.scheduler._now_et")
    def test_skips_run_when_no_new_lineups(
        self, mock_now, mock_sleep, mock_lineups, mock_schedule, tmp_path
    ):
        from bts.scheduler import run_single_check

        mock_lineups.return_value = {100: False}

        result = run_single_check(
            date="2026-04-03",
            all_game_pks=[100],
            confirmed_game_pks=set(),
            config={"orchestrator": {"picks_dir": str(tmp_path)}, "tiers": []},
            early_lock_gap=0.03,
        )
        assert result["skipped"] is True
        assert result["new_lineups"] == 0

    @patch("bts.scheduler.fetch_schedule")
    @patch("bts.scheduler.check_confirmed_lineups")
    @patch("bts.orchestrator.run_cascade")
    @patch("bts.strategy.get_game_statuses", return_value={100: "P"})
    @patch("bts.strategy._load_mdp", return_value=None)
    def test_triggers_prediction_on_new_lineup(
        self, _mdp, _statuses, mock_cascade, mock_lineups, mock_schedule, tmp_path
    ):
        import pandas as pd
        from bts.scheduler import run_single_check

        mock_lineups.return_value = {100: True}
        mock_cascade.return_value = (
            pd.DataFrame([{
                "batter_name": "Test", "batter_id": 1, "team": "NYM",
                "lineup": 1, "pitcher_name": "P", "pitcher_id": 2,
                "game_pk": 100, "game_time": "2026-04-03T23:05:00Z",
                "p_hit_pa": 0.30, "p_game_hit": 0.82, "flags": "",
            }]),
            "mac",
        )

        result = run_single_check(
            date="2026-04-03",
            all_game_pks=[100],
            confirmed_game_pks=set(),
            config={
                "orchestrator": {"picks_dir": str(tmp_path)},
                "tiers": [{"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5}],
            },
            early_lock_gap=0.03,
        )
        assert result["skipped"] is False
        assert result["new_lineups"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler.py::TestSchedulerRun -v`
Expected: FAIL — `run_single_check` not defined

- [ ] **Step 3: Implement `run_single_check()` and `_now_et()`**

Add to `src/bts/scheduler.py`:

```python
def _now_et() -> datetime:
    """Current time in ET. Extracted for testability."""
    return datetime.now(ET)


def run_single_check(
    date: str,
    all_game_pks: list[int],
    confirmed_game_pks: set[int],
    config: dict,
    early_lock_gap: float,
) -> dict:
    """Run a single lineup check cycle.

    1. Check for new confirmed lineups.
    2. If new confirmations, run prediction cascade.
    3. Evaluate should_lock().

    Returns {"skipped": bool, "new_lineups": int, "should_post": bool,
             "pick_result": PickResult | None}.
    """
    from bts.orchestrator import run_and_pick
    from bts.picks import save_pick
    from bts.strategy import should_lock

    new_count = count_new_confirmations(all_game_pks, confirmed_game_pks)

    if new_count == 0:
        return {"skipped": True, "new_lineups": 0, "should_post": False, "pick_result": None}

    print(f"  {new_count} new confirmed lineup(s). Running predictions...", file=sys.stderr)

    predictions, pick_result, tier = run_and_pick(config, date)

    if predictions is None or pick_result is None:
        return {"skipped": False, "new_lineups": new_count, "should_post": False,
                "pick_result": pick_result}

    if pick_result.locked:
        return {"skipped": False, "new_lineups": new_count, "should_post": False,
                "pick_result": pick_result}

    # Save candidate pick
    picks_dir = Path(config["orchestrator"]["picks_dir"])
    save_pick(pick_result.daily, picks_dir)

    # Check if we should lock
    pick_data = {
        "p_game_hit": pick_result.daily.pick.p_game_hit,
        "projected_lineup": pick_result.daily.pick.projected_lineup,
        "game_pk": pick_result.daily.pick.game_pk,
    }
    all_pick_data = []
    for _, row in predictions.iterrows():
        if row.get("p_game_hit") and row["p_game_hit"] == row["p_game_hit"]:  # not NaN
            all_pick_data.append({
                "p_game_hit": float(row["p_game_hit"]),
                "projected_lineup": "PROJECTED" in str(row.get("flags", "")),
                "game_pk": int(row["game_pk"]),
            })

    do_post = should_lock(pick_data, all_pick_data, early_lock_gap)

    return {"skipped": False, "new_lineups": new_count, "should_post": do_post,
            "pick_result": pick_result}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/bts/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): run_single_check — lineup check + prediction + lock decision"
```

---

### Task 8: Result Polling Loop

**Files:**
- Modify: `src/bts/scheduler.py`
- Modify: `tests/test_scheduler.py`

- [ ] **Step 1: Write failing tests for result polling**

```python
# Add to tests/test_scheduler.py

class TestPollResults:
    @patch("bts.scheduler.retry_urlopen")
    def test_returns_final(self, mock_urlopen):
        from bts.scheduler import poll_game_result

        mock_urlopen.return_value.read.return_value = json.dumps({
            "gameData": {"status": {
                "abstractGameCode": "F",
                "detailedState": "Final",
            }},
        }).encode()

        status = poll_game_result(12345)
        assert status == "final"

    @patch("bts.scheduler.retry_urlopen")
    def test_returns_live(self, mock_urlopen):
        from bts.scheduler import poll_game_result

        mock_urlopen.return_value.read.return_value = json.dumps({
            "gameData": {"status": {
                "abstractGameCode": "L",
                "detailedState": "In Progress",
            }},
        }).encode()

        status = poll_game_result(12345)
        assert status == "live"

    @patch("bts.scheduler.retry_urlopen")
    def test_returns_suspended(self, mock_urlopen):
        from bts.scheduler import poll_game_result

        mock_urlopen.return_value.read.return_value = json.dumps({
            "gameData": {"status": {
                "abstractGameCode": "L",
                "detailedState": "Suspended",
            }},
        }).encode()

        status = poll_game_result(12345)
        assert status == "suspended"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler.py::TestPollResults -v`
Expected: FAIL — `poll_game_result` not defined

- [ ] **Step 3: Implement result polling**

Add to `src/bts/scheduler.py`:

```python
def poll_game_result(game_pk: int) -> str:
    """Check a game's current status.

    Returns one of: "final", "live", "suspended", "preview", "unknown".
    """
    try:
        resp = json.loads(retry_urlopen(
            f"{API_BASE}/api/v1.1/game/{game_pk}/feed/live",
            timeout=15,
        ).read())
    except Exception:
        return "unknown"

    abstract = resp["gameData"]["status"]["abstractGameCode"]
    detailed = resp["gameData"]["status"].get("detailedState", "")

    if abstract == "F":
        return "final"
    if "suspend" in detailed.lower():
        return "suspended"
    if abstract == "L":
        return "live"
    if abstract == "P":
        return "preview"
    return "unknown"


def run_result_polling(
    game_pk: int,
    date: str,
    picks_dir: Path,
    poll_interval_min: int = 15,
    cap_hour_et: int = 5,
) -> str:
    """Poll a game until it reaches a terminal state.

    Returns the final status: "final", "suspended", or "unresolved".
    On "final", runs check-results logic to update streak.
    """
    from bts.picks import load_pick, check_hit, update_streak, save_pick

    while True:
        now = _now_et()
        if now.hour >= cap_hour_et and now.hour < 10:
            # Past the cap — give up
            print(f"  Result polling capped at {cap_hour_et}am ET. Flagging as unresolved.",
                  file=sys.stderr)
            daily = load_pick(date, picks_dir)
            if daily:
                daily.result = "unresolved"
                save_pick(daily, picks_dir)
            return "unresolved"

        status = poll_game_result(game_pk)
        print(f"  [{now.strftime('%H:%M ET')}] Game {game_pk}: {status}", file=sys.stderr)

        if status == "final":
            # Run check-results logic
            daily = load_pick(date, picks_dir)
            if daily:
                primary_result = check_hit(
                    daily.pick.game_pk, daily.pick.batter_id,
                    batter_name=daily.pick.batter_name,
                    date=date, team=daily.pick.team,
                )
                if primary_result is None:
                    daily.result = "unresolved"
                    save_pick(daily, picks_dir)
                    return "unresolved"

                results = [primary_result]
                if daily.double_down:
                    double_result = check_hit(
                        daily.double_down.game_pk, daily.double_down.batter_id,
                        batter_name=daily.double_down.batter_name,
                        date=date, team=daily.double_down.team,
                    )
                    if double_result is not None:
                        results.append(double_result)

                update_streak(results, picks_dir)
                daily.result = "hit" if all(results) else "miss"
                save_pick(daily, picks_dir)
                print(f"  Result: {daily.result}. Streak updated.", file=sys.stderr)
            return "final"

        if status == "suspended":
            daily = load_pick(date, picks_dir)
            if daily:
                daily.result = "suspended"
                save_pick(daily, picks_dir)
            return "suspended"

        # Still live — wait and retry
        time.sleep(poll_interval_min * 60)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/bts/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): result polling with suspended/unresolved handling"
```

---

### Task 9: Main Daemon Loop and CLI Command

**Files:**
- Modify: `src/bts/scheduler.py`
- Modify: `src/bts/cli.py`
- Modify: `tests/test_scheduler.py`

- [ ] **Step 1: Write failing test for `run_day()`**

```python
# Add to tests/test_scheduler.py

class TestRunDay:
    @patch("bts.scheduler.fetch_schedule")
    @patch("bts.scheduler._now_et")
    @patch("bts.scheduler.time.sleep")
    @patch("bts.scheduler.run_single_check")
    @patch("bts.scheduler.run_result_polling")
    def test_dry_run_shows_schedule(
        self, mock_poll, mock_check, mock_sleep, mock_now, mock_schedule,
        tmp_path, capsys
    ):
        from bts.scheduler import run_day

        mock_schedule.return_value = [
            _game(100, "13:10"),
            _game(200, "19:05"),
            _game(300, "19:10"),
        ]
        # Set time past all checks so loop exits immediately
        mock_now.return_value = datetime(2026, 4, 3, 22, 0, tzinfo=ET)

        run_day(
            date="2026-04-03",
            config={"orchestrator": {"picks_dir": str(tmp_path)}, "tiers": [],
                    "scheduler": {"early_lock_gap": 0.03, "lineup_check_offset_min": 45,
                                  "cluster_min": 10, "doubleheader_recheck_min": 15,
                                  "results_poll_interval_min": 15, "results_cap_hour_et": 5}},
            dry_run=True,
        )
        # Should not have called run_single_check in dry_run mode
        mock_check.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler.py::TestRunDay -v`
Expected: FAIL — `run_day` not defined

- [ ] **Step 3: Implement `run_day()` main loop**

Add to `src/bts/scheduler.py`:

```python
def run_day(
    date: str,
    config: dict,
    dry_run: bool = False,
) -> None:
    """Run the scheduler for a single day.

    Fetches schedule, computes run times, sleeps between checks,
    triggers predictions, posts to Bluesky when ready, polls results.
    """
    from bts.picks import save_pick, load_streak
    from bts.posting import format_post, format_skip_post, post_to_bluesky

    sched_config = config.get("scheduler", {})
    offset_min = sched_config.get("lineup_check_offset_min", 45)
    cluster_min = sched_config.get("cluster_min", 10)
    dh_recheck_min = sched_config.get("doubleheader_recheck_min", 15)
    early_lock_gap = sched_config.get("early_lock_gap", 0.03)
    poll_interval_min = sched_config.get("results_poll_interval_min", 15)
    cap_hour_et = sched_config.get("results_cap_hour_et", 5)
    picks_dir = Path(config["orchestrator"]["picks_dir"])

    # 1. Fetch schedule
    print(f"[{_now_et().strftime('%H:%M ET')}] Fetching schedule for {date}...", file=sys.stderr)
    games = fetch_schedule(date)
    if not games:
        print(f"No games scheduled for {date}.", file=sys.stderr)
        return

    all_game_pks = [g["gamePk"] for g in games]
    dh_game2s = detect_doubleheader_game2s(games)

    # 2. Compute run times
    runs = compute_run_times(games, offset_min=offset_min, cluster_min=cluster_min)

    print(f"  {len(games)} games, {len(runs)} scheduled checks:", file=sys.stderr)
    for r in runs:
        print(f"    {r['time_et'].strftime('%H:%M ET')} — {len(r['game_pks'])} game(s)", file=sys.stderr)
    if dh_game2s:
        print(f"  Doubleheader game 2s (fluid time): {dh_game2s}", file=sys.stderr)

    if dry_run:
        print("  (--dry-run: not executing checks)", file=sys.stderr)
        return

    # 3. Initialize state
    confirmed_pks: set[int] = set()
    state = SchedulerState(
        date=date,
        schedule_fetched_at=_now_et().isoformat(),
        games=[{
            "game_pk": g["gamePk"],
            "game_time_et": _game_time_et(g).isoformat(),
            "lineup_confirmed": False,
            "is_doubleheader_game2": g["gamePk"] in dh_game2s,
        } for g in games],
        confirmed_game_pks=[],
        runs_completed=[],
        pick_locked=False,
        pick_locked_at=None,
        result_status=None,
        next_wakeup=None,
    )
    save_state(state, picks_dir)

    # 4. Main loop — sleep until each check time, then run
    for run_info in runs:
        target = run_info["time_et"]
        now = _now_et()

        if now < target:
            wait_secs = (target - now).total_seconds()
            print(f"  Sleeping until {target.strftime('%H:%M ET')} "
                  f"({wait_secs / 60:.0f} min)...", file=sys.stderr)
            time.sleep(wait_secs)

        now = _now_et()
        if now < target:
            continue  # Shouldn't happen, but guard

        print(f"\n[{_now_et().strftime('%H:%M ET')}] Running lineup check...", file=sys.stderr)
        result = run_single_check(
            date=date,
            all_game_pks=all_game_pks,
            confirmed_game_pks=confirmed_pks,
            config=config,
            early_lock_gap=early_lock_gap,
        )

        state.runs_completed.append({
            "time": _now_et().isoformat(),
            "new_lineups": result["new_lineups"],
            "skipped": result["skipped"],
        })
        state.confirmed_game_pks = list(confirmed_pks)
        # Update per-game confirmation status
        for g in state.games:
            g["lineup_confirmed"] = g["game_pk"] in confirmed_pks
        save_state(state, picks_dir)

        if result["should_post"] and result["pick_result"] and not result["pick_result"].locked:
            daily = result["pick_result"].daily
            streak = load_streak(picks_dir)
            text = format_post(
                daily.pick.batter_name, daily.pick.team,
                daily.pick.pitcher_name, daily.pick.p_game_hit, streak,
                daily.double_down.batter_name if daily.double_down else None,
                daily.double_down.p_game_hit if daily.double_down else None,
            )
            try:
                uri = post_to_bluesky(text)
                daily.bluesky_posted = True
                daily.bluesky_uri = uri
                save_pick(daily, picks_dir)
                state.pick_locked = True
                state.pick_locked_at = _now_et().isoformat()
                save_state(state, picks_dir)
                print(f"  LOCKED — Posted to Bluesky: {uri}", file=sys.stderr)
            except Exception as e:
                print(f"  Bluesky post failed: {e}", file=sys.stderr)

        if state.pick_locked:
            print(f"  Pick locked. Stopping lineup checks.", file=sys.stderr)
            break

    # 5. Fallback — if not yet locked, check for deadline
    if not state.pick_locked:
        from bts.picks import load_pick
        daily = load_pick(date, picks_dir)
        if daily and not daily.bluesky_posted:
            game_et = datetime.fromisoformat(daily.pick.game_time).astimezone(ET)
            now = _now_et()
            mins_to_game = (game_et - now).total_seconds() / 60
            if mins_to_game <= 15:
                print(f"  FALLBACK — 15min to first pitch, posting on projected data.",
                      file=sys.stderr)
                streak = load_streak(picks_dir)
                text = format_post(
                    daily.pick.batter_name, daily.pick.team,
                    daily.pick.pitcher_name, daily.pick.p_game_hit, streak,
                    daily.double_down.batter_name if daily.double_down else None,
                    daily.double_down.p_game_hit if daily.double_down else None,
                )
                try:
                    uri = post_to_bluesky(text)
                    daily.bluesky_posted = True
                    daily.bluesky_uri = uri
                    save_pick(daily, picks_dir)
                    state.pick_locked = True
                    state.pick_locked_at = _now_et().isoformat()
                    save_state(state, picks_dir)
                except Exception as e:
                    print(f"  Bluesky fallback post failed: {e}", file=sys.stderr)

    # 6. Doubleheader game 2 re-checks
    for pk in dh_game2s:
        if pk in confirmed_pks:
            continue
        if state.pick_locked:
            break
        print(f"  DH game 2 ({pk}): re-checking every {dh_recheck_min}min...", file=sys.stderr)
        for _ in range(10):  # Max 10 re-checks (~2.5 hours)
            time.sleep(dh_recheck_min * 60)
            new = count_new_confirmations([pk], confirmed_pks)
            if new > 0:
                print(f"  DH game 2 ({pk}): lineup confirmed.", file=sys.stderr)
                break

    # 7. Next-day lookahead for wake-up time
    tomorrow = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        tomorrow_games = fetch_schedule(tomorrow)
        wakeup = compute_wakeup_time(
            tomorrow_games,
            default_hour_et=sched_config.get("default_init_hour_et", 10),
            early_buffer_min=sched_config.get("early_game_buffer_min", 60),
        )
        state.next_wakeup = wakeup.isoformat()
        save_state(state, picks_dir)
        print(f"  Tomorrow's wake-up: {wakeup.strftime('%H:%M ET')}", file=sys.stderr)
    except Exception as e:
        print(f"  Failed to fetch tomorrow's schedule: {e}", file=sys.stderr)

    # 8. Result polling (wait until 1am ET, then poll)
    if state.pick_locked:
        from bts.picks import load_pick
        daily = load_pick(date, picks_dir)
        if daily and daily.result is None:
            # Sleep until 1am ET
            now = _now_et()
            target_1am = now.replace(hour=1, minute=0, second=0, microsecond=0)
            if now.hour >= 1:
                target_1am += timedelta(days=1)
            wait = (target_1am - now).total_seconds()
            if wait > 0:
                print(f"  Waiting until 1am ET for result check ({wait / 3600:.1f}h)...",
                      file=sys.stderr)
                time.sleep(wait)

            game_pk = daily.pick.game_pk
            status = run_result_polling(
                game_pk, date, picks_dir,
                poll_interval_min=poll_interval_min,
                cap_hour_et=cap_hour_et,
            )
            state.result_status = status
            save_state(state, picks_dir)
            print(f"  Day complete. Result: {status}", file=sys.stderr)
```

- [ ] **Step 4: Add CLI command to `src/bts/cli.py`**

Add after the `orchestrate` command (after line 473):

```python
@cli.command()
@click.option("--date", default=None, help="Date to schedule (YYYY-MM-DD, default: today)")
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True), help="Orchestrator config TOML file")
@click.option("--dry-run", is_flag=True, help="Show schedule without executing")
def schedule(date: str | None, config_path: str, dry_run: bool):
    """Run the dynamic lineup scheduler for a day.

    Fetches the MLB schedule, computes lineup check times (game_time - 45min),
    sleeps between checks, runs predictions when new lineups confirm, and
    posts to Bluesky when lock conditions are met.
    """
    from datetime import date as date_type, datetime, timezone
    from bts.orchestrator import load_config
    from bts.scheduler import run_day

    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    config = load_config(Path(config_path))
    run_day(date=date, config=config, dry_run=dry_run)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler.py -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/bts/scheduler.py src/bts/cli.py tests/test_scheduler.py
git commit -m "feat(scheduler): main daemon loop with CLI command, fallback, DH re-checks, result polling"
```

---

### Task 10: systemd Service and Cron Update

**Files:**
- Create: `scripts/bts-scheduler.service`
- Modify: `scripts/cron-setup-pi5.sh`

- [ ] **Step 1: Create systemd service unit**

```ini
# scripts/bts-scheduler.service
[Unit]
Description=BTS Dynamic Lineup Scheduler
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=stonehengee
WorkingDirectory=/home/stonehengee/projects/bts
Environment=UV_CACHE_DIR=/tmp/uv-cache
Environment=PATH=/home/stonehengee/.local/bin:/home/stonehengee/.cargo/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=/home/stonehengee/projects/bts/.env
ExecStart=/home/stonehengee/.local/bin/uv run bts schedule --config /home/stonehengee/.bts-orchestrator.toml
Restart=on-failure
RestartSec=300

[Install]
WantedBy=default.target
```

- [ ] **Step 2: Update cron script — remove 3 prediction crons, keep check-results safety net**

Replace the `CRON_LINES` in `scripts/cron-setup-pi5.sh`:

```bash
CRON_LINES="$MARKER
0 1 * * * $PATH_PREFIX; $ENV_SOURCE; cd $BTS_DIR && $UV_PREFIX uv run bts check-results --date $YESTERDAY_CMD >> $LOG_DIR/orchestrator.log 2>&1 $MARKER"
```

- [ ] **Step 3: Add scheduler config section to TOML example**

Create `config/orchestrator.example.toml` (or update if it exists elsewhere):

```toml
[orchestrator]
picks_dir = "data/picks"

[bluesky]
dm_recipient = "did:plc:your-did-here"

[scheduler]
early_lock_gap = 0.03          # TBD — derive from backtesting
lineup_check_offset_min = 45
cluster_min = 10
doubleheader_recheck_min = 15
results_poll_interval_min = 15
results_cap_hour_et = 5
default_init_hour_et = 10
early_game_buffer_min = 60

[[tiers]]
name = "mac"
ssh_host = "mac"
bts_dir = "/Users/stone/projects/bts"
timeout_min = 5

[[tiers]]
name = "alienware"
ssh_host = "alienware"
bts_dir = "C:\\Users\\stone\\projects\\bts"
timeout_min = 10
platform = "windows"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/bts-scheduler.service scripts/cron-setup-pi5.sh config/orchestrator.example.toml
git commit -m "feat(deploy): systemd service, updated cron, example config with scheduler section"
```

---

### Task 11: Backtest `early_lock_gap` Threshold

**Files:**
- Create: `scripts/backtest_early_lock_gap.py`

- [ ] **Step 1: Write the backtesting script**

```python
#!/usr/bin/env python3
"""Backtest early_lock_gap threshold against historical seasons.

For each historical day, simulates what the scheduler would have seen at
each lineup check time: which lineups were confirmed (game started vs not),
and whether waiting would have changed the top pick.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/backtest_early_lock_gap.py \
        --profiles-dir data/simulation --seasons 2021,2022,2023,2024,2025
"""

import argparse
from pathlib import Path

import pandas as pd


def simulate_day(day_df: pd.DataFrame, gap_threshold: float) -> dict:
    """Simulate the scheduler's lock decision for a single day.

    Returns {"would_lock_early": bool, "early_pick": str, "final_pick": str,
             "early_hit": bool, "final_hit": bool, "gap": float}.
    """
    # The day_df has columns: batter_name, p_game_hit, game_time, is_hit, flags
    # Sort by p_game_hit descending
    ranked = day_df.sort_values("p_game_hit", ascending=False).reset_index(drop=True)

    if len(ranked) < 2:
        return None

    top = ranked.iloc[0]
    second = ranked.iloc[1]
    gap = top["p_game_hit"] - second["p_game_hit"]

    return {
        "would_lock_early": gap >= gap_threshold,
        "early_pick": top["batter_name"],
        "final_pick": top["batter_name"],  # Same in backtest (all confirmed)
        "early_hit": bool(top.get("is_hit", False)),
        "gap": gap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles-dir", default="data/simulation")
    parser.add_argument("--seasons", default="2021,2022,2023,2024,2025")
    args = parser.parse_args()

    profiles_dir = Path(args.profiles_dir)
    seasons = [int(s) for s in args.seasons.split(",")]

    all_days = []
    for season in seasons:
        path = profiles_dir / f"backtest_{season}.parquet"
        if not path.exists():
            print(f"Skipping {season} — no backtest file")
            continue
        df = pd.read_parquet(path)
        all_days.append(df)

    if not all_days:
        print("No data found.")
        return

    profiles = pd.concat(all_days, ignore_index=True)
    print(f"Loaded {len(profiles)} daily profiles across {len(all_days)} seasons")

    # Test multiple gap thresholds
    for gap in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        lock_count = 0
        lock_hit = 0
        wait_count = 0
        wait_hit = 0

        for date, day_df in profiles.groupby("date"):
            result = simulate_day(day_df, gap)
            if result is None:
                continue
            if result["would_lock_early"]:
                lock_count += 1
                if result["early_hit"]:
                    lock_hit += 1
            else:
                wait_count += 1
                if result["early_hit"]:
                    wait_hit += 1

        total = lock_count + wait_count
        lock_pct = lock_count / total * 100 if total else 0
        lock_acc = lock_hit / lock_count * 100 if lock_count else 0
        wait_acc = wait_hit / wait_count * 100 if wait_count else 0
        print(f"  gap={gap:.2f}: lock {lock_count}/{total} ({lock_pct:.0f}%), "
              f"lock_acc={lock_acc:.1f}%, wait_acc={wait_acc:.1f}%")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the backtest**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/backtest_early_lock_gap.py --profiles-dir data/simulation`

Review output and choose the threshold that balances lock frequency vs accuracy.

- [ ] **Step 3: Update config with empirical threshold**

Update `early_lock_gap` in `~/.bts-orchestrator.toml` and `config/orchestrator.example.toml` with the chosen value.

- [ ] **Step 4: Commit**

```bash
git add scripts/backtest_early_lock_gap.py config/orchestrator.example.toml
git commit -m "analysis: backtest early_lock_gap threshold — empirical derivation"
```

---

### Task 12: Update ARCHITECTURE.md and CLAUDE.md

**Files:**
- Modify: `ARCHITECTURE.md`
- Modify: `CLAUDE.md` (if scheduler commands needed)

- [ ] **Step 1: Update ARCHITECTURE.md Orchestration section**

Replace the existing Orchestration section with updated diagram and description reflecting the scheduler daemon, dynamic run times, and confirmation-based posting.

- [ ] **Step 2: Add scheduler commands to CLAUDE.md Quick Start**

Add:
```bash
# Scheduler (Pi5 — replaces 3x/day cron)
UV_CACHE_DIR=/tmp/uv-cache uv run bts schedule --config ~/.bts-orchestrator.toml
UV_CACHE_DIR=/tmp/uv-cache uv run bts schedule --config ~/.bts-orchestrator.toml --dry-run
```

- [ ] **Step 3: Commit**

```bash
git add ARCHITECTURE.md CLAUDE.md
git commit -m "docs: update architecture and quickstart for dynamic lineup scheduler"
```
