# BTS Cloud Migration — Plan 01: Lineup Data Collection

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build CLI commands to collect lineup posting times from the MLB Stats API, analyze the resulting distribution, and backfill coarse data from existing Pi5 scheduler state files. Deploy the collection script on Pi5 so data starts accumulating immediately while other migration work proceeds.

**Architecture:** A lightweight poller runs every 5 minutes during an "active window" (3 hours before first game of the day through start of last game). For each game, it polls `/api/v1.1/game/{pk}/feed/live` and records the first time `battingOrder` is populated for each team. Results are written to JSONL logs that can be analyzed for percentile distributions.

**Tech Stack:** Python 3.12, uv, Click for CLI, MLB Stats API (v1.1), stdlib `urllib`/`json`, systemd timer for scheduling on Pi5.

**Dependencies on other plans:** None. This plan is independently executable and testable.

**Parent spec:** `docs/superpowers/specs/2026-04-09-bts-cloud-migration-design.md` (§ Lineup posting time data collection)

---

## File Structure

- Create `src/bts/data/lineup_collect.py` — polling + JSONL logging, ~120 lines
- Create `src/bts/data/lineup_analyze.py` — distribution + backfill, ~180 lines
- Modify `src/bts/cli.py` — register three new commands under `bts data`
- Create `tests/test_lineup_collect.py` — unit tests for polling logic, ~150 lines
- Create `tests/test_lineup_analyze.py` — unit tests for distribution + backfill, ~150 lines
- Create `scripts/systemd/bts-lineup-collect.service` — systemd user unit for Pi5
- Create `scripts/systemd/bts-lineup-collect.timer` — systemd timer (every 5 min)

---

### Task 1: Data structures and `poll_game_lineup()`

**Files:**
- Create: `src/bts/data/lineup_collect.py`
- Create: `tests/test_lineup_collect.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_lineup_collect.py`:

```python
"""Tests for lineup collection polling logic."""
import json
from unittest.mock import patch, MagicMock

import pytest

from bts.data.lineup_collect import poll_game_lineup, LineupPollResult


def test_poll_returns_no_lineup_when_battingorder_empty():
    fake_response = {
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"players": {"ID123": {"battingOrder": ""}}},
                    "home": {"players": {"ID456": {"battingOrder": ""}}},
                }
            }
        }
    }
    with patch("bts.data.lineup_collect.retry_urlopen") as mock_fetch:
        mock_fetch.return_value.read.return_value = json.dumps(fake_response).encode()
        result = poll_game_lineup(game_pk=12345)
    assert result == LineupPollResult(game_pk=12345, away_confirmed=False, home_confirmed=False)


def test_poll_returns_away_confirmed_when_away_has_battingorder():
    fake_response = {
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"players": {"ID123": {"battingOrder": "100"}}},
                    "home": {"players": {"ID456": {}}},
                }
            }
        }
    }
    with patch("bts.data.lineup_collect.retry_urlopen") as mock_fetch:
        mock_fetch.return_value.read.return_value = json.dumps(fake_response).encode()
        result = poll_game_lineup(game_pk=12345)
    assert result.away_confirmed is True
    assert result.home_confirmed is False


def test_poll_returns_both_confirmed():
    fake_response = {
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"players": {"ID1": {"battingOrder": "100"}}},
                    "home": {"players": {"ID2": {"battingOrder": "200"}}},
                }
            }
        }
    }
    with patch("bts.data.lineup_collect.retry_urlopen") as mock_fetch:
        mock_fetch.return_value.read.return_value = json.dumps(fake_response).encode()
        result = poll_game_lineup(game_pk=12345)
    assert result.away_confirmed is True
    assert result.home_confirmed is True


def test_poll_returns_both_false_on_api_error():
    with patch("bts.data.lineup_collect.retry_urlopen", side_effect=Exception("network down")):
        result = poll_game_lineup(game_pk=12345)
    assert result.game_pk == 12345
    assert result.away_confirmed is False
    assert result.home_confirmed is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lineup_collect.py -v
```

Expected: All four tests FAIL with `ModuleNotFoundError: No module named 'bts.data.lineup_collect'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/bts/data/lineup_collect.py`:

```python
"""Poll MLB API for lineup confirmation times.

Runs as a periodic cron/timer. For each game scheduled on a given date,
polls the feed endpoint and records the first time both sides have
confirmed lineups (battingOrder populated for at least one player).
"""
import json
from dataclasses import dataclass

from bts.picks import API_BASE
from bts.util import retry_urlopen


@dataclass
class LineupPollResult:
    """Result of polling one game's current lineup status."""
    game_pk: int
    away_confirmed: bool
    home_confirmed: bool


def poll_game_lineup(game_pk: int) -> LineupPollResult:
    """Fetch one game's feed and check lineup confirmation for each side.

    A side is 'confirmed' if at least one player has battingOrder set to
    a non-empty value. Returns both=False on any API error.
    """
    try:
        raw = retry_urlopen(
            f"{API_BASE}/api/v1.1/game/{game_pk}/feed/live",
            timeout=15,
        ).read()
        data = json.loads(raw)
    except Exception:
        return LineupPollResult(game_pk=game_pk, away_confirmed=False, home_confirmed=False)

    away_confirmed = False
    home_confirmed = False
    try:
        players_by_side = data["liveData"]["boxscore"]["teams"]
        for player in players_by_side.get("away", {}).get("players", {}).values():
            if player.get("battingOrder"):
                away_confirmed = True
                break
        for player in players_by_side.get("home", {}).get("players", {}).values():
            if player.get("battingOrder"):
                home_confirmed = True
                break
    except (KeyError, TypeError):
        pass

    return LineupPollResult(
        game_pk=game_pk,
        away_confirmed=away_confirmed,
        home_confirmed=home_confirmed,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lineup_collect.py -v
```

Expected: All four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/data/lineup_collect.py tests/test_lineup_collect.py
git commit -m "feat(lineup): add poll_game_lineup for MLB feed lineup check"
```

---

### Task 2: Collection loop state tracking

**Files:**
- Modify: `src/bts/data/lineup_collect.py`
- Modify: `tests/test_lineup_collect.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lineup_collect.py`:

```python
from datetime import datetime, timezone
from pathlib import Path

from bts.data.lineup_collect import CollectionState, run_collection_tick


def test_collection_state_records_first_confirmation():
    state = CollectionState(date="2026-04-10")
    now = datetime(2026, 4, 10, 17, 30, tzinfo=timezone.utc)

    state.record_poll(
        game_pk=12345,
        game_time_et="2026-04-10T19:05:00-04:00",
        poll_time_utc=now,
        away_confirmed=True,
        home_confirmed=False,
    )

    entry = state.games[12345]
    assert entry.first_away_confirmed_utc == now.isoformat()
    assert entry.first_home_confirmed_utc is None
    assert entry.poll_count == 1


def test_collection_state_does_not_overwrite_first_confirmation():
    state = CollectionState(date="2026-04-10")
    first = datetime(2026, 4, 10, 17, 30, tzinfo=timezone.utc)
    second = datetime(2026, 4, 10, 17, 35, tzinfo=timezone.utc)

    state.record_poll(12345, "2026-04-10T19:05:00-04:00", first, True, False)
    state.record_poll(12345, "2026-04-10T19:05:00-04:00", second, True, True)

    entry = state.games[12345]
    assert entry.first_away_confirmed_utc == first.isoformat()
    assert entry.first_home_confirmed_utc == second.isoformat()
    assert entry.poll_count == 2


def test_collection_state_serializes_to_jsonl(tmp_path: Path):
    state = CollectionState(date="2026-04-10")
    now = datetime(2026, 4, 10, 17, 30, tzinfo=timezone.utc)
    state.record_poll(12345, "2026-04-10T19:05:00-04:00", now, True, True)

    state.write_jsonl(tmp_path)

    out_file = tmp_path / "2026-04-10.jsonl"
    assert out_file.exists()
    line = json.loads(out_file.read_text().strip())
    assert line["game_pk"] == 12345
    assert line["first_away_confirmed_utc"] == now.isoformat()
    assert line["first_home_confirmed_utc"] == now.isoformat()
    assert line["poll_count"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lineup_collect.py -v
```

Expected: Three new tests FAIL with `ImportError: cannot import name 'CollectionState'`.

- [ ] **Step 3: Write implementation**

Append to `src/bts/data/lineup_collect.py`:

```python
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class GameCollectionEntry:
    """Per-game collection state within one day."""
    game_pk: int
    game_time_et: str
    first_away_confirmed_utc: Optional[str] = None
    first_home_confirmed_utc: Optional[str] = None
    poll_count: int = 0


class CollectionState:
    """Stateful tracker for one day of lineup-time collection."""

    def __init__(self, date: str):
        self.date = date
        self.games: dict[int, GameCollectionEntry] = {}

    def record_poll(
        self,
        game_pk: int,
        game_time_et: str,
        poll_time_utc: datetime,
        away_confirmed: bool,
        home_confirmed: bool,
    ) -> None:
        """Update state with one poll result. First confirmation is sticky."""
        entry = self.games.get(game_pk)
        if entry is None:
            entry = GameCollectionEntry(game_pk=game_pk, game_time_et=game_time_et)
            self.games[game_pk] = entry

        if away_confirmed and entry.first_away_confirmed_utc is None:
            entry.first_away_confirmed_utc = poll_time_utc.isoformat()
        if home_confirmed and entry.first_home_confirmed_utc is None:
            entry.first_home_confirmed_utc = poll_time_utc.isoformat()
        entry.poll_count += 1

    def write_jsonl(self, out_dir: Path) -> Path:
        """Write all known entries to {date}.jsonl (one JSON object per line)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.date}.jsonl"
        lines = []
        for entry in sorted(self.games.values(), key=lambda e: e.game_pk):
            lines.append(json.dumps({
                "game_pk": entry.game_pk,
                "game_time_et": entry.game_time_et,
                "first_away_confirmed_utc": entry.first_away_confirmed_utc,
                "first_home_confirmed_utc": entry.first_home_confirmed_utc,
                "poll_count": entry.poll_count,
            }))
        out_path.write_text("\n".join(lines) + "\n" if lines else "")
        return out_path
```

- [ ] **Step 4: Run test to verify it passes**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lineup_collect.py -v
```

Expected: All seven tests PASS (4 from Task 1 + 3 from Task 2).

- [ ] **Step 5: Commit**

```bash
git add src/bts/data/lineup_collect.py tests/test_lineup_collect.py
git commit -m "feat(lineup): add CollectionState for per-day tracking"
```

---

### Task 3: `run_collection_tick` — one iteration of the loop

**Files:**
- Modify: `src/bts/data/lineup_collect.py`
- Modify: `tests/test_lineup_collect.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lineup_collect.py`:

```python
def test_run_collection_tick_polls_only_games_needing_confirmation():
    state = CollectionState(date="2026-04-10")
    now_confirmed = datetime(2026, 4, 10, 17, 30, tzinfo=timezone.utc)
    # Game 1 already has both sides confirmed from a previous tick
    state.record_poll(1, "2026-04-10T19:05:00-04:00", now_confirmed, True, True)
    # Game 2 needs confirmation
    state.games[2] = GameCollectionEntry(
        game_pk=2,
        game_time_et="2026-04-10T19:10:00-04:00",
    )

    mock_poll = MagicMock()
    mock_poll.return_value = LineupPollResult(game_pk=2, away_confirmed=True, home_confirmed=False)

    now = datetime(2026, 4, 10, 17, 45, tzinfo=timezone.utc)
    with patch("bts.data.lineup_collect.poll_game_lineup", mock_poll):
        run_collection_tick(state, now_utc=now)

    # Game 1 should NOT be polled (both confirmed)
    # Game 2 should be polled once
    assert mock_poll.call_count == 1
    mock_poll.assert_called_with(2)
    # Game 2 now has away confirmed
    assert state.games[2].first_away_confirmed_utc == now.isoformat()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lineup_collect.py::test_run_collection_tick_polls_only_games_needing_confirmation -v
```

Expected: FAIL with `ImportError: cannot import name 'run_collection_tick'`.

- [ ] **Step 3: Write implementation**

Append to `src/bts/data/lineup_collect.py`:

```python
def run_collection_tick(
    state: CollectionState,
    now_utc: datetime,
) -> None:
    """Poll games that still need confirmation. Updates state in place.

    Skips games where both sides are already confirmed (no work to do).
    """
    for game_pk, entry in list(state.games.items()):
        if entry.first_away_confirmed_utc and entry.first_home_confirmed_utc:
            continue
        result = poll_game_lineup(game_pk)
        state.record_poll(
            game_pk=game_pk,
            game_time_et=entry.game_time_et,
            poll_time_utc=now_utc,
            away_confirmed=result.away_confirmed,
            home_confirmed=result.home_confirmed,
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lineup_collect.py -v
```

Expected: All eight tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/data/lineup_collect.py tests/test_lineup_collect.py
git commit -m "feat(lineup): add run_collection_tick for single poll iteration"
```

---

### Task 4: CLI command `bts data collect-lineup-times`

**Files:**
- Modify: `src/bts/data/lineup_collect.py`
- Modify: `src/bts/cli.py`
- Modify: `tests/test_lineup_collect.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lineup_collect.py`:

```python
from bts.data.lineup_collect import collect_for_date


def test_collect_for_date_initializes_state_from_schedule(tmp_path):
    fake_schedule = [
        {"gamePk": 1, "gameDate": "2026-04-10T23:05:00Z"},
        {"gamePk": 2, "gameDate": "2026-04-10T23:10:00Z"},
    ]
    with patch("bts.data.lineup_collect.fetch_schedule", return_value=fake_schedule), \
         patch("bts.data.lineup_collect.poll_game_lineup") as mock_poll:
        mock_poll.return_value = LineupPollResult(game_pk=1, away_confirmed=False, home_confirmed=False)
        state = collect_for_date(date="2026-04-10", out_dir=tmp_path)

    assert set(state.games.keys()) == {1, 2}
    # Each game was polled once (one tick only in this test)
    assert mock_poll.call_count == 2
    # JSONL should exist even if nothing confirmed
    assert (tmp_path / "2026-04-10.jsonl").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lineup_collect.py::test_collect_for_date_initializes_state_from_schedule -v
```

Expected: FAIL with `ImportError: cannot import name 'collect_for_date'`.

- [ ] **Step 3: Write implementation**

Add to `src/bts/data/lineup_collect.py`:

```python
from datetime import timezone
from bts.scheduler import fetch_schedule, _game_time_et


def collect_for_date(date: str, out_dir: Path) -> CollectionState:
    """Run one collection pass for all games on a given date.

    Initializes state from the MLB schedule, polls each game exactly once,
    and writes the current state to JSONL before returning. This function
    is designed to be called repeatedly by a cron/timer (every 5 minutes),
    accumulating more confirmations with each call.
    """
    state = _load_or_create_state(date, out_dir)

    games = fetch_schedule(date)
    for g in games:
        game_pk = g["gamePk"]
        if game_pk not in state.games:
            state.games[game_pk] = GameCollectionEntry(
                game_pk=game_pk,
                game_time_et=_game_time_et(g).isoformat(),
            )

    run_collection_tick(state, now_utc=datetime.now(timezone.utc))
    state.write_jsonl(out_dir)
    return state


def _load_or_create_state(date: str, out_dir: Path) -> CollectionState:
    """Reload existing JSONL if present so re-runs are incremental."""
    state = CollectionState(date=date)
    existing = out_dir / f"{date}.jsonl"
    if not existing.exists():
        return state

    for line in existing.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        state.games[row["game_pk"]] = GameCollectionEntry(
            game_pk=row["game_pk"],
            game_time_et=row["game_time_et"],
            first_away_confirmed_utc=row.get("first_away_confirmed_utc"),
            first_home_confirmed_utc=row.get("first_home_confirmed_utc"),
            poll_count=row.get("poll_count", 0),
        )
    return state
```

- [ ] **Step 4: Run test to verify it passes**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lineup_collect.py -v
```

Expected: All nine tests PASS.

- [ ] **Step 5: Register the CLI command**

Add to `src/bts/cli.py` (in the `data` group alongside existing `data pull`, `data build`):

```python
@data.command(name="collect-lineup-times")
@click.option("--date", default=None, help="Date (YYYY-MM-DD, default today ET)")
@click.option("--out-dir", default="data/lineup_posting_times", type=click.Path())
def data_collect_lineup_times(date, out_dir):
    """Poll MLB API once for lineup confirmation times on the given date.

    Designed to be called every 5 minutes via systemd timer or cron.
    Each call is a single poll pass across all games that still need
    confirmation. JSONL file is updated in place with accumulating data.
    """
    from datetime import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo
    from bts.data.lineup_collect import collect_for_date

    if date is None:
        date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    state = collect_for_date(date=date, out_dir=Path(out_dir))
    n_both = sum(
        1 for g in state.games.values()
        if g.first_away_confirmed_utc and g.first_home_confirmed_utc
    )
    click.echo(f"{date}: {n_both}/{len(state.games)} games fully confirmed")
```

- [ ] **Step 6: Smoke test the CLI command**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts data collect-lineup-times --date 2026-04-09 --out-dir /tmp/lineup-smoke
cat /tmp/lineup-smoke/2026-04-09.jsonl
```

Expected: JSONL file exists with one line per game scheduled on 2026-04-09. Each line has `poll_count >= 1`. For games where lineups are already confirmed (i.e., games in progress or completed), `first_away_confirmed_utc` and `first_home_confirmed_utc` will be populated with the current timestamp (not historically accurate but a reasonable smoke test).

- [ ] **Step 7: Commit**

```bash
git add src/bts/data/lineup_collect.py src/bts/cli.py tests/test_lineup_collect.py
git commit -m "feat(lineup): add 'bts data collect-lineup-times' CLI"
```

---

### Task 5: Analysis — distribution computation

**Files:**
- Create: `src/bts/data/lineup_analyze.py`
- Create: `tests/test_lineup_analyze.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_lineup_analyze.py`:

```python
"""Tests for lineup-time distribution analysis."""
import json
from pathlib import Path

import pytest

from bts.data.lineup_analyze import (
    compute_minutes_before_first_pitch,
    compute_distribution,
    Distribution,
)


def test_compute_minutes_before_first_pitch_positive_when_lineup_before_game():
    # Game at 19:05 ET (23:05 UTC), lineup confirmed at 18:00 ET (22:00 UTC)
    # → 65 minutes before first pitch
    result = compute_minutes_before_first_pitch(
        lineup_time_utc="2026-04-10T22:00:00+00:00",
        game_time_et="2026-04-10T19:05:00-04:00",
    )
    assert result == 65


def test_compute_minutes_before_first_pitch_negative_when_after():
    # Lineup 'confirmed' 10 min after first pitch (anomaly, should still handle)
    result = compute_minutes_before_first_pitch(
        lineup_time_utc="2026-04-10T23:15:00+00:00",
        game_time_et="2026-04-10T19:05:00-04:00",
    )
    assert result == -10


def test_compute_distribution_percentiles():
    # 10 games, minutes-before-first-pitch values from 30 to 120
    samples = [30, 45, 60, 70, 80, 90, 100, 110, 115, 120]
    dist = compute_distribution(samples)
    assert dist.n == 10
    assert dist.p10 == pytest.approx(36, abs=1)
    assert dist.p50 == pytest.approx(85, abs=1)
    assert dist.p90 == pytest.approx(116, abs=1)
    assert dist.p95 == pytest.approx(118, abs=1)


def test_compute_distribution_empty():
    dist = compute_distribution([])
    assert dist.n == 0
    assert dist.p50 is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lineup_analyze.py -v
```

Expected: All four tests FAIL with `ModuleNotFoundError: No module named 'bts.data.lineup_analyze'`.

- [ ] **Step 3: Write implementation**

Create `src/bts/data/lineup_analyze.py`:

```python
"""Analyze lineup posting time distributions from collected JSONL logs.

Reads files written by `bts data collect-lineup-times` and computes
statistics on how many minutes before first pitch each lineup was
confirmed. Used to inform scheduler timing configuration.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class Distribution:
    """Percentile summary of minutes-before-first-pitch."""
    n: int
    p10: Optional[float]
    p25: Optional[float]
    p50: Optional[float]
    p75: Optional[float]
    p90: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
    mean: Optional[float]
    minutes: list[int]


def compute_minutes_before_first_pitch(
    lineup_time_utc: str,
    game_time_et: str,
) -> int:
    """Compute minutes between a lineup confirmation and first pitch.

    Positive = confirmed before first pitch (normal).
    Negative = confirmed after (anomaly, should be rare).
    """
    lineup_dt = datetime.fromisoformat(lineup_time_utc)
    game_dt = datetime.fromisoformat(game_time_et)
    delta_sec = (game_dt - lineup_dt).total_seconds()
    return round(delta_sec / 60)


def compute_distribution(samples: Iterable[int]) -> Distribution:
    """Compute percentile distribution from a collection of minute values."""
    data = sorted(samples)
    n = len(data)
    if n == 0:
        return Distribution(
            n=0, p10=None, p25=None, p50=None, p75=None,
            p90=None, p95=None, p99=None, mean=None, minutes=[],
        )

    def percentile(p: float) -> float:
        # Linear interpolation between closest ranks
        k = (n - 1) * (p / 100)
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return float(data[f])
        return data[f] + (data[c] - data[f]) * (k - f)

    return Distribution(
        n=n,
        p10=percentile(10),
        p25=percentile(25),
        p50=percentile(50),
        p75=percentile(75),
        p90=percentile(90),
        p95=percentile(95),
        p99=percentile(99),
        mean=sum(data) / n,
        minutes=data,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lineup_analyze.py -v
```

Expected: All four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/data/lineup_analyze.py tests/test_lineup_analyze.py
git commit -m "feat(lineup): add distribution computation for lineup times"
```

---

### Task 6: `load_samples_from_jsonl` and CLI `analyze-lineup-times`

**Files:**
- Modify: `src/bts/data/lineup_analyze.py`
- Modify: `src/bts/cli.py`
- Modify: `tests/test_lineup_analyze.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lineup_analyze.py`:

```python
def test_load_samples_reads_both_sides(tmp_path: Path):
    jsonl = tmp_path / "2026-04-10.jsonl"
    jsonl.write_text(
        json.dumps({
            "game_pk": 1,
            "game_time_et": "2026-04-10T19:05:00-04:00",
            "first_away_confirmed_utc": "2026-04-10T22:00:00+00:00",
            "first_home_confirmed_utc": "2026-04-10T21:45:00+00:00",
            "poll_count": 5,
        }) + "\n"
    )

    from bts.data.lineup_analyze import load_samples_from_jsonl
    samples = load_samples_from_jsonl(tmp_path, from_date="2026-04-10", to_date="2026-04-10")
    # Two samples per game (away + home)
    assert len(samples) == 2
    # 23:05 UTC is first pitch; 22:00 UTC is 65 min before; 21:45 is 80 min before
    assert sorted(samples) == [65, 80]


def test_load_samples_ignores_null_confirmations(tmp_path: Path):
    jsonl = tmp_path / "2026-04-10.jsonl"
    jsonl.write_text(
        json.dumps({
            "game_pk": 1,
            "game_time_et": "2026-04-10T19:05:00-04:00",
            "first_away_confirmed_utc": None,
            "first_home_confirmed_utc": "2026-04-10T21:45:00+00:00",
            "poll_count": 5,
        }) + "\n"
    )
    from bts.data.lineup_analyze import load_samples_from_jsonl
    samples = load_samples_from_jsonl(tmp_path, from_date="2026-04-10", to_date="2026-04-10")
    assert samples == [80]


def test_load_samples_respects_date_range(tmp_path: Path):
    for date, minutes_before in [("2026-04-08", 60), ("2026-04-10", 75)]:
        jsonl = tmp_path / f"{date}.jsonl"
        game_time_et = f"{date}T19:05:00-04:00"
        lineup_utc = (datetime.fromisoformat(game_time_et)
                      - timedelta(minutes=minutes_before)).astimezone(timezone.utc).isoformat()
        jsonl.write_text(
            json.dumps({
                "game_pk": 1,
                "game_time_et": game_time_et,
                "first_away_confirmed_utc": lineup_utc,
                "first_home_confirmed_utc": lineup_utc,
                "poll_count": 1,
            }) + "\n"
        )

    from bts.data.lineup_analyze import load_samples_from_jsonl
    samples = load_samples_from_jsonl(tmp_path, from_date="2026-04-10", to_date="2026-04-10")
    assert samples == [75, 75]  # Only 2026-04-10's two sides
```

Also add import at top of test file: `from datetime import datetime, timezone, timedelta`.

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lineup_analyze.py -v
```

Expected: New tests FAIL with `ImportError: cannot import name 'load_samples_from_jsonl'`.

- [ ] **Step 3: Write implementation**

Append to `src/bts/data/lineup_analyze.py`:

```python
import json


def load_samples_from_jsonl(
    in_dir: Path,
    from_date: str,
    to_date: str,
) -> list[int]:
    """Load all lineup-time samples from JSONL files in a date range.

    Reads both first_away_confirmed_utc and first_home_confirmed_utc per
    game as separate samples. Skips nulls. Returns a list of
    minutes-before-first-pitch integers ready to feed into compute_distribution.
    """
    from datetime import datetime as _dt

    samples: list[int] = []
    for jsonl in sorted(in_dir.glob("*.jsonl")):
        date_str = jsonl.stem
        if date_str < from_date or date_str > to_date:
            continue
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            game_time_et = row["game_time_et"]
            for field in ("first_away_confirmed_utc", "first_home_confirmed_utc"):
                confirmed = row.get(field)
                if confirmed is None:
                    continue
                samples.append(
                    compute_minutes_before_first_pitch(confirmed, game_time_et)
                )
    return samples
```

- [ ] **Step 4: Register CLI command**

Add to `src/bts/cli.py` inside the `data` group:

```python
@data.command(name="analyze-lineup-times")
@click.option("--in-dir", default="data/lineup_posting_times", type=click.Path())
@click.option("--from-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--to-date", required=True, help="End date (YYYY-MM-DD)")
def data_analyze_lineup_times(in_dir, from_date, to_date):
    """Report lineup-posting-time distribution for a date range.

    Prints percentiles and a short histogram-style summary. Use to inform
    scheduler timing configuration (lineup_check_offset_min, fallback_deadline_min).
    """
    from pathlib import Path
    from bts.data.lineup_analyze import load_samples_from_jsonl, compute_distribution

    samples = load_samples_from_jsonl(Path(in_dir), from_date, to_date)
    dist = compute_distribution(samples)

    click.echo(f"Lineup posting time distribution ({from_date} to {to_date})")
    click.echo(f"  n = {dist.n} samples")
    if dist.n == 0:
        click.echo("  (no samples — check data/lineup_posting_times/ has data for this range)")
        return
    click.echo(f"  mean   = {dist.mean:.0f} min before first pitch")
    click.echo(f"  p10    = {dist.p10:.0f}")
    click.echo(f"  p25    = {dist.p25:.0f}")
    click.echo(f"  p50    = {dist.p50:.0f}")
    click.echo(f"  p75    = {dist.p75:.0f}")
    click.echo(f"  p90    = {dist.p90:.0f}")
    click.echo(f"  p95    = {dist.p95:.0f}")
    click.echo(f"  p99    = {dist.p99:.0f}")
    click.echo("")
    click.echo("Interpretation:")
    click.echo(f"  To capture p95 of lineups at lock time, use lineup_check_offset_min >= {int(dist.p95) + 5}")
    click.echo(f"  For fallback_deadline_min, accept up to p90 ({int(dist.p90)}) loss of confirmed data")
```

- [ ] **Step 5: Run tests and smoke-test CLI**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lineup_analyze.py -v
UV_CACHE_DIR=/tmp/uv-cache uv run bts data analyze-lineup-times --from-date 2026-04-01 --to-date 2026-04-09
```

Expected: tests pass; CLI output either prints "(no samples ...)" if you haven't collected any, or shows percentiles if you have.

- [ ] **Step 6: Commit**

```bash
git add src/bts/data/lineup_analyze.py src/bts/cli.py tests/test_lineup_analyze.py
git commit -m "feat(lineup): add 'bts data analyze-lineup-times' CLI"
```

---

### Task 7: Backfill from existing Pi5 scheduler state

**Files:**
- Modify: `src/bts/data/lineup_analyze.py`
- Modify: `tests/test_lineup_analyze.py`
- Modify: `src/bts/cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lineup_analyze.py`:

```python
def test_backfill_from_scheduler_state(tmp_path: Path):
    # Fake Pi5 scheduler state: two runs on one day
    picks_dir = tmp_path / "picks"
    date_dir = picks_dir / "2026-04-05"
    date_dir.mkdir(parents=True)
    scheduler_state = {
        "date": "2026-04-05",
        "games": [
            {"game_pk": 1, "game_time_et": "2026-04-05T19:05:00-04:00",
             "lineup_confirmed": True, "is_doubleheader_game2": False},
            {"game_pk": 2, "game_time_et": "2026-04-05T19:10:00-04:00",
             "lineup_confirmed": True, "is_doubleheader_game2": False},
        ],
        "confirmed_game_pks": [1, 2],
        "runs_completed": [
            {"time": "2026-04-05T22:20:00-04:00", "new_lineups": 1, "skipped": False,
             "pick_name": None, "pick_p": None},  # 45 min before 19:05 ET
            {"time": "2026-04-05T22:25:00-04:00", "new_lineups": 1, "skipped": False,
             "pick_name": None, "pick_p": None},  # 40 min before 19:05 ET
        ],
        "pick_locked": True,
        "pick_locked_at": "2026-04-05T22:25:00-04:00",
        "result_status": "final",
        "next_wakeup": None,
            "schedule_fetched_at": "2026-04-05T10:00:00-04:00",
    }
    (date_dir / "scheduler_state.json").write_text(json.dumps(scheduler_state))

    from bts.data.lineup_analyze import backfill_from_scheduler_state
    samples = backfill_from_scheduler_state(picks_dir)

    # Two runs, each with 1 new_lineup → 2 samples
    # First run 45 min before first pitch (of game 1 at 19:05)
    # Second run 40 min before
    assert sorted(samples) == [40, 45]


def test_backfill_returns_empty_when_no_state(tmp_path: Path):
    picks_dir = tmp_path / "empty"
    picks_dir.mkdir()
    from bts.data.lineup_analyze import backfill_from_scheduler_state
    assert backfill_from_scheduler_state(picks_dir) == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lineup_analyze.py -v
```

Expected: New tests FAIL with `ImportError: cannot import name 'backfill_from_scheduler_state'`.

- [ ] **Step 3: Write implementation**

Append to `src/bts/data/lineup_analyze.py`:

```python
def backfill_from_scheduler_state(picks_dir: Path) -> list[int]:
    """Extract coarse lineup-time samples from existing Pi5 scheduler state.

    This is a bootstrap data source for the forward-looking collection.
    Each completed run in scheduler_state.json records a timestamp and
    'new_lineups' count. We attribute each new confirmation to that
    run's time minus first_pitch of the earliest game, producing
    coarse (5-15 min resolution) samples.

    Returns minutes-before-first-pitch integers.
    """
    samples: list[int] = []
    for date_dir in sorted(picks_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        state_file = date_dir / "scheduler_state.json"
        if not state_file.exists():
            continue
        state = json.loads(state_file.read_text())

        # Earliest game of the day is first pitch
        game_times = [g["game_time_et"] for g in state.get("games", [])]
        if not game_times:
            continue
        earliest_game_et = min(game_times)
        game_dt = datetime.fromisoformat(earliest_game_et)

        for run in state.get("runs_completed", []):
            n_new = run.get("new_lineups", 0)
            if n_new <= 0:
                continue
            run_dt = datetime.fromisoformat(run["time"])
            minutes_before = round((game_dt - run_dt).total_seconds() / 60)
            # Attribute the minutes to each new confirmation
            samples.extend([minutes_before] * n_new)

    return samples
```

Add `from datetime import datetime` at the top of `lineup_analyze.py` if not already imported.

- [ ] **Step 4: Register CLI command**

Add to `src/bts/cli.py` inside the `data` group:

```python
@data.command(name="backfill-lineup-times")
@click.option("--picks-dir", default="data/picks", type=click.Path(exists=True))
def data_backfill_lineup_times(picks_dir):
    """Extract coarse lineup-time samples from existing Pi5 scheduler state.

    Coarse (5-15 min resolution) but real data to bootstrap the distribution
    analysis before the collection script has accumulated a week of data.
    Combine output with results from 'bts data analyze-lineup-times'.
    """
    from pathlib import Path
    from bts.data.lineup_analyze import backfill_from_scheduler_state, compute_distribution

    samples = backfill_from_scheduler_state(Path(picks_dir))
    dist = compute_distribution(samples)
    click.echo(f"Bootstrap from Pi5 scheduler state: n={dist.n}")
    if dist.n:
        click.echo(f"  p50={dist.p50:.0f}, p90={dist.p90:.0f}, p95={dist.p95:.0f}")
```

- [ ] **Step 5: Run tests and smoke-test**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_lineup_analyze.py -v
```

Expected: All tests PASS.

If you have Pi5 scheduler state locally (via rsync or shared volume), test:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts data backfill-lineup-times --picks-dir /path/to/pi5/data/picks
```

- [ ] **Step 6: Commit**

```bash
git add src/bts/data/lineup_analyze.py src/bts/cli.py tests/test_lineup_analyze.py
git commit -m "feat(lineup): add backfill-lineup-times from Pi5 scheduler state"
```

---

### Task 8: Systemd unit files for Pi5 deployment

**Files:**
- Create: `scripts/systemd/bts-lineup-collect.service`
- Create: `scripts/systemd/bts-lineup-collect.timer`
- Create: `scripts/systemd/README.md`

- [ ] **Step 1: Create systemd unit files**

Create `scripts/systemd/bts-lineup-collect.service`:

```ini
[Unit]
Description=BTS lineup time collection (one-shot)
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=%h/projects/bts
Environment="PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin"
Environment="UV_CACHE_DIR=/tmp/uv-cache"
ExecStart=%h/.local/bin/uv run bts data collect-lineup-times
StandardOutput=journal
StandardError=journal
```

Create `scripts/systemd/bts-lineup-collect.timer`:

```ini
[Unit]
Description=BTS lineup time collection (every 5 min)
Requires=bts-lineup-collect.service

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
AccuracySec=30s
Persistent=true

[Install]
WantedBy=timers.target
```

Create `scripts/systemd/README.md`:

```markdown
# Systemd units for Pi5 deployment

## Lineup time collection

Install as user units on Pi5:

    mkdir -p ~/.config/systemd/user
    cp scripts/systemd/bts-lineup-collect.service ~/.config/systemd/user/
    cp scripts/systemd/bts-lineup-collect.timer ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now bts-lineup-collect.timer

Verify it's running:

    systemctl --user list-timers bts-lineup-collect
    journalctl --user -u bts-lineup-collect -f

Collected data accumulates under `~/projects/bts/data/lineup_posting_times/`.
To pull the data back to Mac for analysis:

    rsync -az pi5:~/projects/bts/data/lineup_posting_times/ \
        ~/projects/bts/data/lineup_posting_times/
```

- [ ] **Step 2: Validate unit files locally**

```bash
systemd-analyze verify scripts/systemd/bts-lineup-collect.service
systemd-analyze verify scripts/systemd/bts-lineup-collect.timer
```

Expected: no errors. Note: on macOS this command may not be available — if so, validation happens at Pi5 deploy time.

- [ ] **Step 3: Commit**

```bash
git add scripts/systemd/bts-lineup-collect.service \
        scripts/systemd/bts-lineup-collect.timer \
        scripts/systemd/README.md
git commit -m "feat(lineup): add systemd timer for Pi5 lineup collection"
```

---

### Task 9: Deploy to Pi5 and verify

**Files:** (runbook, no code changes)

- [ ] **Step 1: Push changes to main**

```bash
git push origin main
```

- [ ] **Step 2: SSH to Pi5 and pull**

```bash
ssh stonehengee@pi5.local 'cd ~/projects/bts && git pull origin main'
```

Expected: pull succeeds with the new commits.

- [ ] **Step 3: Install systemd units on Pi5**

```bash
ssh stonehengee@pi5.local 'mkdir -p ~/.config/systemd/user && \
    cp ~/projects/bts/scripts/systemd/bts-lineup-collect.service ~/.config/systemd/user/ && \
    cp ~/projects/bts/scripts/systemd/bts-lineup-collect.timer ~/.config/systemd/user/ && \
    systemctl --user daemon-reload && \
    systemctl --user enable --now bts-lineup-collect.timer'
```

- [ ] **Step 4: Verify the timer is active**

```bash
ssh stonehengee@pi5.local 'systemctl --user list-timers bts-lineup-collect'
```

Expected: output shows the timer with next run within 5 minutes.

- [ ] **Step 5: Trigger one run manually and check output**

```bash
ssh stonehengee@pi5.local 'systemctl --user start bts-lineup-collect.service'
ssh stonehengee@pi5.local 'journalctl --user -u bts-lineup-collect --since "5 minutes ago" --no-pager'
```

Expected: log shows `{date}: N/M games fully confirmed` where N and M are game counts for today.

- [ ] **Step 6: Verify JSONL file exists and has content**

```bash
ssh stonehengee@pi5.local 'ls -la ~/projects/bts/data/lineup_posting_times/ && head ~/projects/bts/data/lineup_posting_times/*.jsonl'
```

Expected: today's JSONL file exists with one entry per scheduled game.

- [ ] **Step 7: Commit the deployment marker (optional)**

There is no code to commit for deployment — just note the deployment date in your memory or migration tracker. The lineup data collection is now running continuously on Pi5, accumulating the distribution needed for Phase 2 of the cloud migration.

---

## Completion criteria for Plan 01

All of the following must be true before considering this plan complete:

- [ ] All tests pass: `uv run pytest tests/test_lineup_collect.py tests/test_lineup_analyze.py -v`
- [ ] `bts data collect-lineup-times` works locally on Mac
- [ ] `bts data analyze-lineup-times --from-date X --to-date Y` produces a distribution report (may be empty if no data yet)
- [ ] `bts data backfill-lineup-times` produces coarse data from existing Pi5 scheduler state
- [ ] Systemd timer is installed and active on Pi5
- [ ] Data is accumulating in `~/projects/bts/data/lineup_posting_times/` on Pi5, verified by checking the directory after 30+ minutes
- [ ] No regression: existing Pi5 bts-scheduler.service is still running normally (the new collect timer is independent)

**Next plan:** `02-r2.md` — R2 canonical data layer. Can be executed in parallel with this one (no dependencies between them).
