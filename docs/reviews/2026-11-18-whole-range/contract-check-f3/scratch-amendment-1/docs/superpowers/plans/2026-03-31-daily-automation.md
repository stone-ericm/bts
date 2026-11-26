# BTS Daily Automation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automate the daily BTS workflow — predict, save picks, post to Bluesky, track streak — with three runs per day and smart posting.

**Architecture:** Two new modules (`picks.py` for pick persistence/MLB API helpers, `posting.py` for Bluesky posting/timing), a `run_pipeline()` extraction in `predict.py`, and two new CLI commands (`bts run`, `bts check-results`). Mac cron for scheduling.

**Tech Stack:** Python 3.12, Click, stdlib JSON/urllib, zoneinfo, dataclasses, pytest + unittest.mock

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/bts/picks.py` | Pick/DailyPick dataclasses, JSON file I/O, streak tracking, MLB schedule/boxscore API helpers |
| `src/bts/posting.py` | Bluesky auth + posting, post text formatting, post timing logic |
| `tests/test_picks.py` | Tests for picks module (file I/O, streak, MLB API) |
| `tests/test_posting.py` | Tests for posting module (formatting, timing, API calls) |

### Modified Files
| File | Changes |
|------|---------|
| `src/bts/model/predict.py` | Add `game_time` to slot dicts in `_fetch_game_slots()`, add `run_pipeline()` |
| `src/bts/cli.py` | Add `bts run` and `bts check-results` commands, refactor `bts predict` to use `run_pipeline()`, refactor `bts post` to use `posting.py` |

---

### Task 1: Pick data model, file I/O, and streak tracking

**Files:**
- Create: `src/bts/picks.py`
- Create: `tests/test_picks.py`

- [ ] **Step 1: Write failing tests for Pick data model and file I/O**

```python
# tests/test_picks.py
import json
import pytest
from bts.picks import Pick, DailyPick, save_pick, load_pick


def _sample_pick(**overrides):
    defaults = dict(
        batter_name="Jacob Wilson",
        batter_id=700363,
        team="ATH",
        lineup_position=1,
        pitcher_name="Jose Suarez",
        pitcher_id=660761,
        p_game_hit=0.763,
        flags=[],
        projected_lineup=False,
        game_pk=778899,
        game_time="2026-04-01T23:10:00Z",
    )
    defaults.update(overrides)
    return Pick(**defaults)


def _sample_daily(pick=None, **overrides):
    defaults = dict(
        date="2026-04-01",
        run_time="2026-04-01T15:00:00+00:00",
        pick=pick or _sample_pick(),
        double_down=None,
        runner_up={"batter_name": "Jake Mangum", "p_game_hit": 0.726},
        streak=3,
        bluesky_posted=False,
        bluesky_uri=None,
    )
    defaults.update(overrides)
    return DailyPick(**defaults)


class TestPickFileIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        daily = _sample_daily()
        save_pick(daily, tmp_path)

        loaded = load_pick("2026-04-01", tmp_path)
        assert loaded is not None
        assert loaded.pick.batter_name == "Jacob Wilson"
        assert loaded.pick.p_game_hit == pytest.approx(0.763)
        assert loaded.pick.game_pk == 778899
        assert loaded.streak == 3

    def test_load_nonexistent_returns_none(self, tmp_path):
        assert load_pick("2099-01-01", tmp_path) is None

    def test_save_creates_directory(self, tmp_path):
        subdir = tmp_path / "nested" / "picks"
        daily = _sample_daily()
        save_pick(daily, subdir)
        assert (subdir / "2026-04-01.json").exists()

    def test_save_overwrites_existing(self, tmp_path):
        daily = _sample_daily(streak=3)
        save_pick(daily, tmp_path)

        updated = _sample_daily(streak=4, bluesky_posted=True, bluesky_uri="at://did:plc:xxx/post/yyy")
        save_pick(updated, tmp_path)

        loaded = load_pick("2026-04-01", tmp_path)
        assert loaded.streak == 4
        assert loaded.bluesky_posted is True

    def test_roundtrip_with_double_down(self, tmp_path):
        double = _sample_pick(batter_name="Shohei Ohtani", batter_id=660271, p_game_hit=0.741)
        daily = _sample_daily(double_down=double)
        save_pick(daily, tmp_path)

        loaded = load_pick("2026-04-01", tmp_path)
        assert loaded.double_down is not None
        assert loaded.double_down.batter_name == "Shohei Ohtani"

    def test_roundtrip_preserves_flags(self, tmp_path):
        pick = _sample_pick(flags=["IL? (8d rest)", "PROJECTED lineup"], projected_lineup=True)
        daily = _sample_daily(pick=pick)
        save_pick(daily, tmp_path)

        loaded = load_pick("2026-04-01", tmp_path)
        assert loaded.pick.flags == ["IL? (8d rest)", "PROJECTED lineup"]
        assert loaded.pick.projected_lineup is True

    def test_json_format_matches_spec(self, tmp_path):
        daily = _sample_daily()
        save_pick(daily, tmp_path)

        raw = json.loads((tmp_path / "2026-04-01.json").read_text())
        assert raw["date"] == "2026-04-01"
        assert raw["pick"]["batter_name"] == "Jacob Wilson"
        assert raw["pick"]["game_pk"] == 778899
        assert raw["double_down"] is None
        assert raw["runner_up"]["batter_name"] == "Jake Mangum"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_picks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bts.picks'`

- [ ] **Step 3: Implement Pick data model and file I/O**

```python
# src/bts/picks.py
"""Pick persistence, streak tracking, and MLB API helpers for BTS automation."""

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Pick:
    batter_name: str
    batter_id: int
    team: str
    lineup_position: int
    pitcher_name: str
    pitcher_id: int | None
    p_game_hit: float
    flags: list[str]
    projected_lineup: bool
    game_pk: int
    game_time: str  # ISO 8601 UTC


@dataclass
class DailyPick:
    date: str
    run_time: str
    pick: Pick
    double_down: Pick | None
    runner_up: dict | None  # {"batter_name": str, "p_game_hit": float}
    streak: int
    bluesky_posted: bool = False
    bluesky_uri: str | None = None


def pick_from_row(row) -> Pick:
    """Create a Pick from a prediction DataFrame row."""
    flags_str = row.get("flags", "")
    flags = [f.strip() for f in flags_str.split(",") if f.strip()] if flags_str else []
    return Pick(
        batter_name=row["batter_name"],
        batter_id=int(row["batter_id"]),
        team=row["team"],
        lineup_position=int(row["lineup"]),
        pitcher_name=row["pitcher_name"],
        pitcher_id=int(row["pitcher_id"]) if row.get("pitcher_id") else None,
        p_game_hit=float(row["p_game_hit"]),
        flags=flags,
        projected_lineup="PROJECTED" in flags_str,
        game_pk=int(row["game_pk"]),
        game_time=row["game_time"],
    )


def save_pick(daily: DailyPick, picks_dir: Path) -> Path:
    """Save daily pick to JSON file."""
    picks_dir.mkdir(parents=True, exist_ok=True)
    path = picks_dir / f"{daily.date}.json"
    path.write_text(json.dumps(asdict(daily), indent=2))
    return path


def load_pick(date: str, picks_dir: Path) -> DailyPick | None:
    """Load daily pick from JSON file. Returns None if not found."""
    path = picks_dir / f"{date}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return DailyPick(
        date=data["date"],
        run_time=data["run_time"],
        pick=Pick(**data["pick"]),
        double_down=Pick(**data["double_down"]) if data["double_down"] else None,
        runner_up=data["runner_up"],
        streak=data["streak"],
        bluesky_posted=data.get("bluesky_posted", False),
        bluesky_uri=data.get("bluesky_uri"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_picks.py::TestPickFileIO -v`
Expected: 7 tests PASS

- [ ] **Step 5: Write failing tests for streak tracking**

Add to `tests/test_picks.py`:

```python
from bts.picks import load_streak, save_streak, update_streak


class TestStreak:
    def test_load_empty_returns_zero(self, tmp_path):
        assert load_streak(tmp_path) == 0

    def test_save_and_load_roundtrip(self, tmp_path):
        save_streak(5, tmp_path)
        assert load_streak(tmp_path) == 5

    def test_update_single_hit_increments(self, tmp_path):
        save_streak(3, tmp_path)
        new = update_streak([True], tmp_path)
        assert new == 4
        assert load_streak(tmp_path) == 4

    def test_update_single_miss_resets(self, tmp_path):
        save_streak(10, tmp_path)
        new = update_streak([False], tmp_path)
        assert new == 0
        assert load_streak(tmp_path) == 0

    def test_update_double_both_hit_adds_two(self, tmp_path):
        save_streak(5, tmp_path)
        new = update_streak([True, True], tmp_path)
        assert new == 7

    def test_update_double_one_miss_resets(self, tmp_path):
        save_streak(5, tmp_path)
        new = update_streak([True, False], tmp_path)
        assert new == 0

    def test_update_double_both_miss_resets(self, tmp_path):
        save_streak(5, tmp_path)
        new = update_streak([False, False], tmp_path)
        assert new == 0
```

- [ ] **Step 6: Run streak tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_picks.py::TestStreak -v`
Expected: FAIL — `ImportError: cannot import name 'load_streak'`

- [ ] **Step 7: Implement streak tracking**

Add to `src/bts/picks.py`:

```python
def load_streak(picks_dir: Path) -> int:
    """Load current streak count. Returns 0 if no streak file."""
    path = picks_dir / "streak.json"
    if not path.exists():
        return 0
    return json.loads(path.read_text()).get("streak", 0)


def save_streak(streak: int, picks_dir: Path) -> None:
    """Save current streak count."""
    picks_dir.mkdir(parents=True, exist_ok=True)
    path = picks_dir / "streak.json"
    path.write_text(json.dumps({
        "streak": streak,
        "updated": datetime.now(timezone.utc).isoformat(),
    }))


def update_streak(results: list[bool], picks_dir: Path) -> int:
    """Update streak based on pick results.

    Single pick: [True] -> +1, [False] -> 0
    Double-down: [True, True] -> +2, anything else -> 0
    """
    current = load_streak(picks_dir)
    new = current + len(results) if all(results) else 0
    save_streak(new, picks_dir)
    return new
```

- [ ] **Step 8: Run all picks tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_picks.py -v`
Expected: 14 tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/bts/picks.py tests/test_picks.py
git commit -m "feat: add pick persistence and streak tracking (picks.py)"
```

---

### Task 2: MLB API helpers — game status and result checking

**Files:**
- Modify: `src/bts/picks.py`
- Modify: `tests/test_picks.py`

- [ ] **Step 1: Write failing tests for game status and result checking**

Add to `tests/test_picks.py`:

```python
from unittest.mock import patch, MagicMock
from bts.picks import get_game_statuses, check_hit


def _mock_schedule_response(games):
    """Build a mock MLB schedule API response."""
    return {"dates": [{"games": games}]}


def _mock_feed_response(batter_id, hits, status_code="F"):
    """Build a mock MLB game feed with boxscore stats."""
    return {
        "gameData": {"status": {"abstractGameCode": status_code}},
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"players": {}},
                    "home": {
                        "players": {
                            f"ID{batter_id}": {
                                "stats": {"batting": {"hits": hits}},
                            }
                        }
                    },
                }
            }
        },
    }


class TestGameStatuses:
    @patch("bts.picks.urlopen")
    def test_returns_status_map(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            _mock_schedule_response([
                {"gamePk": 100, "status": {"abstractGameCode": "P"}},
                {"gamePk": 200, "status": {"abstractGameCode": "L"}},
                {"gamePk": 300, "status": {"abstractGameCode": "F"}},
            ])
        ).encode()

        result = get_game_statuses("2026-04-01")
        assert result == {100: "P", 200: "L", 300: "F"}

    @patch("bts.picks.urlopen")
    def test_empty_schedule(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            {"dates": []}
        ).encode()
        assert get_game_statuses("2026-04-01") == {}


class TestCheckHit:
    @patch("bts.picks.urlopen")
    def test_batter_got_hit(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            _mock_feed_response(batter_id=700363, hits=2, status_code="F")
        ).encode()
        assert check_hit(778899, 700363) is True

    @patch("bts.picks.urlopen")
    def test_batter_no_hit(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            _mock_feed_response(batter_id=700363, hits=0, status_code="F")
        ).encode()
        assert check_hit(778899, 700363) is False

    @patch("bts.picks.urlopen")
    def test_game_not_final_returns_none(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            _mock_feed_response(batter_id=700363, hits=0, status_code="L")
        ).encode()
        assert check_hit(778899, 700363) is None

    @patch("bts.picks.urlopen")
    def test_batter_not_in_game(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            _mock_feed_response(batter_id=999999, hits=1, status_code="F")
        ).encode()
        # Batter 700363 not in boxscore (only 999999 is)
        assert check_hit(778899, 700363) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_picks.py::TestGameStatuses tests/test_picks.py::TestCheckHit -v`
Expected: FAIL — `ImportError: cannot import name 'get_game_statuses'`

- [ ] **Step 3: Implement MLB API helpers**

Add to `src/bts/picks.py` (at top, add the import; then add functions):

```python
# Add to imports at top of picks.py
from urllib.request import urlopen

API_BASE = "https://statsapi.mlb.com"


def get_game_statuses(date: str) -> dict[int, str]:
    """Get game statuses for all games on a date.

    Returns {game_pk: abstractGameCode} where codes are:
        P = Preview (not started), L = Live, F = Final
    """
    resp = json.loads(urlopen(
        f"{API_BASE}/api/v1/schedule?sportId=1&date={date}",
        timeout=15,
    ).read())
    statuses = {}
    for d in resp.get("dates", []):
        for g in d.get("games", []):
            statuses[g["gamePk"]] = g["status"]["abstractGameCode"]
    return statuses


def check_hit(game_pk: int, batter_id: int) -> bool | None:
    """Check if a batter got a hit in a game.

    Returns True (hit), False (no hit), or None (game not final).
    """
    resp = json.loads(urlopen(
        f"{API_BASE}/api/v1.1/game/{game_pk}/feed/live",
        timeout=15,
    ).read())
    status = resp["gameData"]["status"]["abstractGameCode"]
    if status != "F":
        return None

    # Check boxscore batting stats
    for side in ("away", "home"):
        players = resp["liveData"]["boxscore"]["teams"][side]["players"]
        key = f"ID{batter_id}"
        if key in players:
            hits = players[key].get("stats", {}).get("batting", {}).get("hits", 0)
            return hits > 0
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_picks.py -v`
Expected: 20 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/bts/picks.py tests/test_picks.py
git commit -m "feat: add MLB API helpers for game status and result checking"
```

---

### Task 3: Add game_time to prediction pipeline + extract run_pipeline

**Files:**
- Modify: `src/bts/model/predict.py:259-367` — add `game_time` to slots
- Modify: `src/bts/model/predict.py` — add `run_pipeline()` function
- Modify: `src/bts/cli.py:76-158` — refactor `predict` command to use `run_pipeline()`

- [ ] **Step 1: Add game_time to `_fetch_game_slots`**

In `src/bts/model/predict.py`, inside `_fetch_game_slots()`, capture `gameDate` from the schedule and include it in each slot:

In the loop at line ~276, after `status = g["status"]["detailedState"]` add:

```python
            game_time = g.get("gameDate", "")
```

Then in the slot dict construction (line ~347-358), add `game_time`:

```python
                    slot = {
                        "batter_id": lp["batter_id"],
                        "batter_name": lp["batter_name"],
                        "team": team_abbr,
                        "lineup": lp["lineup"],
                        "pitcher_id": opp_pitcher_id,
                        "pitcher_name": opp_pitcher_name,
                        "pitcher_hand": opp_pitcher_hand,
                        "venue_id": venue_id,
                        "weather_temp": temp,
                        "game_pk": pk,
                        "game_time": game_time,
                        "status": status,
                    }
```

- [ ] **Step 2: Add `run_pipeline()` function to predict.py**

Add at the end of `src/bts/model/predict.py`:

```python
def run_pipeline(
    date: str,
    data_dir: str = "data/processed",
    check_openers: bool = True,
) -> pd.DataFrame:
    """Run the full prediction pipeline for a date.

    Loads historical data, computes features, trains the 12-model blend,
    and returns ranked picks sorted by P(game hit).
    """
    from bts.features.compute import compute_all_features

    proc = Path(data_dir)
    dfs = []
    for parquet in sorted(proc.glob("pa_*.parquet")):
        dfs.append(pd.read_parquet(parquet))
    if not dfs:
        raise RuntimeError("No Parquet files found. Run 'bts data build' first.")

    df = pd.concat(dfs, ignore_index=True)
    df = compute_all_features(df)
    df["date"] = pd.to_datetime(df["date"])

    model = train_model(df)
    blend = train_blend(df)
    lookups = _build_feature_lookups(df)

    return predict(
        date, df, model, lookups,
        check_openers=check_openers,
        blend=blend,
    )
```

- [ ] **Step 3: Refactor `bts predict` CLI to use `run_pipeline()`**

Replace the body of the `predict` command in `src/bts/cli.py` (lines 77-158):

```python
@cli.command()
@click.option("--date", required=True, help="Date to predict (YYYY-MM-DD)")
@click.option("--data-dir", default="data/processed", type=click.Path(), help="Processed data directory")
@click.option("--top", default=15, type=int, help="Number of picks to show")
@click.option("--no-opener-check", is_flag=True, help="Skip opener detection (faster)")
def predict(date: str, data_dir: str, top: int, no_opener_check: bool):
    """Generate ranked BTS picks for a date."""
    import pandas as pd
    from bts.model.predict import run_pipeline

    click.echo(f"Running prediction pipeline for {date}...")
    picks = run_pipeline(date, data_dir, check_openers=not no_opener_check)

    if picks.empty:
        click.echo("No games found for this date.")
        return

    click.echo(f"\n{'='*80}")
    click.echo(f"BTS PICKS — {date}")
    click.echo(f"{'='*80}")
    click.echo(f"{'#':<4} {'Batter':<22} {'Team':<5} {'Pos':>3} {'vs Pitcher':<22} {'P(PA)':>6} {'P(Game)':>7}  {'Flags'}")
    click.echo(f"{'-'*80}")

    shown = 0
    for _, row in picks.iterrows():
        if shown >= top:
            break
        if pd.isna(row.get("p_game_hit")):
            continue
        flags = row.get("flags", "")
        click.echo(
            f"{shown+1:<4} {row['batter_name']:<22} {row['team']:<5} "
            f"{int(row['lineup']):>3} {row['pitcher_name']:<22} "
            f"{row['p_hit_pa']:>5.1%} {row['p_game_hit']:>6.1%}  {flags}"
        )
        shown += 1

    # Recommendation: 1 or 2 picks based on P(both hit)
    DOUBLE_THRESHOLD = 0.65
    best = picks.iloc[0]
    valid_picks = picks[picks["p_game_hit"].notna()]

    if len(valid_picks) >= 2:
        second = valid_picks.iloc[1]
        p_both = best["p_game_hit"] * second["p_game_hit"]

        if p_both >= DOUBLE_THRESHOLD:
            click.echo(f"\nDOUBLE DOWN: {best['batter_name']} ({best['p_game_hit']:.1%}) "
                        f"+ {second['batter_name']} ({second['p_game_hit']:.1%})")
            click.echo(f"  P(both hit): {p_both:.1%}")
        else:
            click.echo(f"\nSingle pick: {best['batter_name']} ({best['p_game_hit']:.1%})")
            click.echo(f"  P(both hit) with #{second['batter_name']}: {p_both:.1%} (below {DOUBLE_THRESHOLD:.0%} threshold)")
    else:
        click.echo(f"\nSingle pick: {best['batter_name']} ({best['p_game_hit']:.1%})")

    if best.get("flags"):
        click.echo(f"  WARNING: {best['flags']}")
```

- [ ] **Step 4: Verify existing `bts predict` still works**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v`
Expected: All existing tests pass (no regressions)

Manual check (if games are available):
```bash
cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run bts predict --date 2026-03-31
```

- [ ] **Step 5: Commit**

```bash
git add src/bts/model/predict.py src/bts/cli.py
git commit -m "refactor: extract run_pipeline(), add game_time to prediction slots"
```

---

### Task 4: Bluesky posting module

**Files:**
- Create: `src/bts/posting.py`
- Create: `tests/test_posting.py`
- Modify: `src/bts/cli.py:160-232` — refactor `bts post` to use `posting.py`

- [ ] **Step 1: Write failing tests for post formatting**

```python
# tests/test_posting.py
from bts.posting import format_post


class TestFormatPost:
    def test_single_pick(self):
        text = format_post(
            batter="Jacob Wilson", team="ATH", pitcher="Jose Suarez",
            p_game=0.763, streak=3,
        )
        assert text == (
            "Today's pick: Jacob Wilson (ATH)\n"
            "vs Jose Suarez | 76.3%\n\n"
            "Streak: 3"
        )

    def test_double_down(self):
        text = format_post(
            batter="Jacob Wilson", team="ATH", pitcher="Jose Suarez",
            p_game=0.763, streak=3,
            double="Shohei Ohtani", double_p_game=0.741,
        )
        assert "Today's picks: Jacob Wilson (ATH) + Shohei Ohtani" in text
        assert "P(both): 56.5%" in text
        assert "Streak: 3" in text

    def test_streak_zero(self):
        text = format_post(
            batter="Mike Trout", team="LAA", pitcher="Max Scherzer",
            p_game=0.875, streak=0,
        )
        assert "Streak: 0" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_posting.py::TestFormatPost -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bts.posting'`

- [ ] **Step 3: Implement posting module**

```python
# src/bts/posting.py
"""Bluesky posting for BTS picks."""

import json
import subprocess
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen, Request
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def format_post(
    batter: str,
    team: str,
    pitcher: str,
    p_game: float,
    streak: int,
    double: str | None = None,
    double_p_game: float | None = None,
) -> str:
    """Format Bluesky post text for a BTS pick."""
    if double and double_p_game is not None:
        p_both = p_game * double_p_game
        return (
            f"Today's picks: {batter} ({team}) + {double}\n"
            f"vs {pitcher} | P(both): {p_both:.1%}\n\n"
            f"Streak: {streak}"
        )
    return (
        f"Today's pick: {batter} ({team})\n"
        f"vs {pitcher} | {p_game:.1%}\n\n"
        f"Streak: {streak}"
    )


def get_bluesky_password() -> str:
    """Get Bluesky app password from macOS Keychain."""
    result = subprocess.run(
        ["security", "find-generic-password", "-a", "claude-cli",
         "-s", "bluesky-bts-app-password", "-w"],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def post_to_bluesky(text: str) -> str:
    """Post text to Bluesky. Returns post URI.

    Raises RuntimeError if auth or posting fails.
    """
    password = get_bluesky_password()
    if not password:
        raise RuntimeError("No Bluesky app password found in keychain")

    # Authenticate
    auth_data = json.dumps({
        "identifier": "beatthestreakbot.bsky.social",
        "password": password,
    }).encode()
    req = Request("https://bsky.social/xrpc/com.atproto.server.createSession",
                  data=auth_data, headers={"Content-Type": "application/json"})
    session = json.loads(urlopen(req, timeout=15).read())

    # Post
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    record = {
        "repo": session["did"],
        "collection": "app.bsky.feed.post",
        "record": {
            "$type": "app.bsky.feed.post",
            "text": text,
            "createdAt": now,
        },
    }
    req = Request(
        "https://bsky.social/xrpc/com.atproto.repo.createRecord",
        data=json.dumps(record).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {session['accessJwt']}"},
    )
    resp = json.loads(urlopen(req, timeout=15).read())
    return resp["uri"]


def should_post_now(game_time_utc: str, already_posted: bool) -> bool:
    """Decide if we should post the pick to Bluesky now.

    Posts if:
    - Not already posted for today
    - Game starts within 3 hours, OR
    - It's the evening run (after 7pm ET)
    """
    if already_posted:
        return False

    now_et = datetime.now(ET)
    game_dt = datetime.fromisoformat(game_time_utc).astimezone(ET)

    # Post if game starts within 3 hours
    if game_dt - now_et <= timedelta(hours=3):
        return True

    # Post on the final run (after 7pm ET)
    if now_et.hour >= 19:
        return True

    return False
```

- [ ] **Step 4: Run format tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_posting.py::TestFormatPost -v`
Expected: 3 tests PASS

- [ ] **Step 5: Write failing tests for post timing logic**

Add to `tests/test_posting.py`:

```python
from unittest.mock import patch
from datetime import datetime
from zoneinfo import ZoneInfo
from bts.posting import should_post_now

ET = ZoneInfo("America/New_York")


class TestShouldPostNow:
    @patch("bts.posting.datetime")
    def test_game_within_3_hours_posts(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 4, 1, 14, 0, tzinfo=ET)
        mock_dt.fromisoformat = datetime.fromisoformat
        # Game at 4:30pm ET = within 2.5 hours
        assert should_post_now("2026-04-01T20:30:00Z", already_posted=False) is True

    @patch("bts.posting.datetime")
    def test_game_far_away_skips(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 4, 1, 11, 0, tzinfo=ET)
        mock_dt.fromisoformat = datetime.fromisoformat
        # Game at 7:10pm ET = 8+ hours away
        assert should_post_now("2026-04-01T23:10:00Z", already_posted=False) is False

    @patch("bts.posting.datetime")
    def test_evening_run_always_posts(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 4, 1, 19, 30, tzinfo=ET)
        mock_dt.fromisoformat = datetime.fromisoformat
        # Game at 10pm ET — but it's after 7pm so post anyway
        assert should_post_now("2026-04-02T02:00:00Z", already_posted=False) is True

    def test_already_posted_skips(self):
        assert should_post_now("2026-04-01T23:10:00Z", already_posted=True) is False
```

- [ ] **Step 6: Run timing tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_posting.py -v`
Expected: 7 tests PASS

Note: The `datetime` mock needs care. The mock replaces `datetime.now()` and `datetime.fromisoformat()`. If the mock setup doesn't work correctly with `zoneinfo`, fall back to patching `bts.posting.datetime.now` specifically or freezing time differently. Adjust mocking approach if tests fail.

- [ ] **Step 7: Refactor `bts post` CLI to use posting module**

Replace the `post` command body in `src/bts/cli.py`:

```python
@cli.command()
@click.option("--date", required=True, help="Date of the pick (YYYY-MM-DD)")
@click.option("--batter", required=True, help="Batter name")
@click.option("--team", required=True, help="Team abbreviation")
@click.option("--pitcher", required=True, help="Opposing pitcher name")
@click.option("--pct", required=True, type=float, help="P(game hit) percentage")
@click.option("--streak", required=True, type=int, help="Current streak count")
@click.option("--double", default=None, help="Second pick name for double down (optional)")
@click.option("--double-pct", default=None, type=float, help="Second pick P(game hit)")
@click.option("--dry-run", is_flag=True, help="Print post text without posting")
def post(date: str, batter: str, team: str, pitcher: str, pct: float,
         streak: int, double: str, double_pct: float, dry_run: bool):
    """Post today's pick to Bluesky."""
    from bts.posting import format_post, post_to_bluesky

    p_game = pct / 100
    double_p = double_pct / 100 if double_pct else None
    text = format_post(batter, team, pitcher, p_game, streak, double, double_p)

    if dry_run:
        click.echo(f"Would post:\n{text}")
        return

    try:
        uri = post_to_bluesky(text)
        click.echo(f"Posted: {uri}")
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
```

- [ ] **Step 8: Run all tests**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 9: Commit**

```bash
git add src/bts/posting.py tests/test_posting.py src/bts/cli.py
git commit -m "feat: extract Bluesky posting into module with post timing logic"
```

---

### Task 5: `bts run` command — daily automation orchestrator

**Files:**
- Modify: `src/bts/cli.py` — add `run` command

- [ ] **Step 1: Write the `bts run` command**

Add to `src/bts/cli.py`, after the existing `post` command:

```python
@cli.command()
@click.option("--date", required=True, help="Date to predict (YYYY-MM-DD)")
@click.option("--data-dir", default="data/processed", type=click.Path(), help="Processed data directory")
@click.option("--picks-dir", default="data/picks", type=click.Path(), help="Picks output directory")
@click.option("--dry-run", is_flag=True, help="Skip Bluesky posting")
def run(date: str, data_dir: str, picks_dir: str, dry_run: bool):
    """Run daily BTS automation: predict, save pick, optionally post.

    Designed to run via cron at 11am, 4pm, and 7:30pm ET.
    Each run picks the best available batter whose game hasn't started.
    Posts to Bluesky when the game is within 3 hours or after 7pm ET.
    """
    import pandas as pd
    from pathlib import Path
    from datetime import datetime, timezone
    from bts.model.predict import run_pipeline
    from bts.picks import (
        pick_from_row, save_pick, load_pick, load_streak,
        get_game_statuses, DailyPick,
    )
    from bts.posting import format_post, post_to_bluesky, should_post_now

    picks_path = Path(picks_dir)
    DOUBLE_THRESHOLD = 0.65

    # Step 1: Run prediction pipeline
    click.echo(f"[{datetime.now(timezone.utc).strftime('%H:%M UTC')}] Running predictions for {date}...")
    try:
        predictions = run_pipeline(date, data_dir)
    except RuntimeError as e:
        click.echo(f"ERROR: {e}", err=True)
        return
    except Exception as e:
        click.echo(f"ERROR: Pipeline failed — {e}", err=True)
        return

    if predictions.empty:
        click.echo("No games found for this date.")
        return

    # Step 2: Load current state
    current = load_pick(date, picks_path)
    streak = load_streak(picks_path)

    # Step 3: Filter to games not yet started
    statuses = get_game_statuses(date)
    not_started = predictions["game_pk"].map(lambda pk: statuses.get(pk) == "P")
    available = predictions[not_started]

    if available.empty:
        if current:
            click.echo(f"All games started. Pick locked: {current.pick.batter_name}")
        else:
            click.echo("All games started, no pick was made.")
        return

    # Step 4: Check if current pick is locked
    if current and statuses.get(current.pick.game_pk) != "P":
        click.echo(f"Pick locked: {current.pick.batter_name} (game started)")
        return

    # Step 5: Select best available
    best_row = available.iloc[0]
    new_pick = pick_from_row(best_row)

    if current and current.pick.batter_id == new_pick.batter_id:
        click.echo(f"Confirmed: {new_pick.batter_name} ({new_pick.p_game_hit:.1%})")
    elif current:
        click.echo(f"Upgraded: {current.pick.batter_name} -> {new_pick.batter_name} "
                    f"({current.pick.p_game_hit:.1%} -> {new_pick.p_game_hit:.1%})")
    else:
        click.echo(f"Pick: {new_pick.batter_name} ({new_pick.p_game_hit:.1%}) "
                    f"vs {new_pick.pitcher_name}")

    # Step 6: Check for double-down
    double_pick = None
    valid = available[available["p_game_hit"].notna()]
    if len(valid) >= 2:
        second_row = valid.iloc[1]
        p_both = best_row["p_game_hit"] * second_row["p_game_hit"]
        if p_both >= DOUBLE_THRESHOLD:
            double_pick = pick_from_row(second_row)
            click.echo(f"  DOUBLE DOWN: + {double_pick.batter_name} "
                        f"({double_pick.p_game_hit:.1%}), P(both): {p_both:.1%}")

    # Step 7: Build runner-up info
    runner_up = None
    if len(valid) >= 2:
        ru = valid.iloc[1]
        runner_up = {"batter_name": ru["batter_name"], "p_game_hit": float(ru["p_game_hit"])}

    # Step 8: Save pick
    daily = DailyPick(
        date=date,
        run_time=datetime.now(timezone.utc).isoformat(),
        pick=new_pick,
        double_down=double_pick,
        runner_up=runner_up,
        streak=streak,
        bluesky_posted=current.bluesky_posted if current else False,
        bluesky_uri=current.bluesky_uri if current else None,
    )
    save_pick(daily, picks_path)
    click.echo(f"  Saved to {picks_path / f'{date}.json'}")

    # Step 9: Post to Bluesky if appropriate
    if dry_run:
        text = format_post(
            new_pick.batter_name, new_pick.team, new_pick.pitcher_name,
            new_pick.p_game_hit, streak,
            double_pick.batter_name if double_pick else None,
            double_pick.p_game_hit if double_pick else None,
        )
        click.echo(f"  Would post:\n{text}")
        return

    if should_post_now(new_pick.game_time, daily.bluesky_posted):
        text = format_post(
            new_pick.batter_name, new_pick.team, new_pick.pitcher_name,
            new_pick.p_game_hit, streak,
            double_pick.batter_name if double_pick else None,
            double_pick.p_game_hit if double_pick else None,
        )
        try:
            uri = post_to_bluesky(text)
            daily.bluesky_posted = True
            daily.bluesky_uri = uri
            save_pick(daily, picks_path)
            click.echo(f"  Posted to Bluesky: {uri}")
        except Exception as e:
            click.echo(f"  Bluesky post failed: {e}", err=True)
    else:
        click.echo("  Not posting yet (game not within 3h, not evening run)")
```

- [ ] **Step 2: Manual smoke test with --dry-run**

```bash
cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run bts run --date 2026-03-31 --dry-run
```

Expected: Runs the pipeline, shows pick, saves JSON to `data/picks/2026-03-31.json`, shows "Would post" text.

- [ ] **Step 3: Verify pick file was created**

```bash
cat /Users/stone/projects/bts/data/picks/2026-03-31.json | python3 -m json.tool
```

Expected: Valid JSON matching the spec format with all fields populated.

- [ ] **Step 4: Run all tests to check for regressions**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/bts/cli.py
git commit -m "feat: add 'bts run' command for daily automation"
```

---

### Task 6: `bts check-results` command — result checking and streak update

**Files:**
- Modify: `src/bts/cli.py` — add `check-results` command

- [ ] **Step 1: Write the `bts check-results` command**

Add to `src/bts/cli.py`:

```python
@cli.command(name="check-results")
@click.option("--date", required=True, help="Date to check results for (YYYY-MM-DD)")
@click.option("--picks-dir", default="data/picks", type=click.Path(), help="Picks directory")
def check_results(date: str, picks_dir: str):
    """Check if yesterday's pick got a hit and update the streak.

    Designed to run via cron at 1am ET (after all games finish).
    """
    from pathlib import Path
    from bts.picks import load_pick, check_hit, update_streak, save_pick

    picks_path = Path(picks_dir)
    daily = load_pick(date, picks_path)

    if daily is None:
        click.echo(f"No pick found for {date}.")
        return

    # Check primary pick
    click.echo(f"Checking {daily.pick.batter_name} (game {daily.pick.game_pk})...")
    primary_result = check_hit(daily.pick.game_pk, daily.pick.batter_id)

    if primary_result is None:
        click.echo(f"Game {daily.pick.game_pk} not final yet. Try again later.")
        return

    results = [primary_result]

    # Check double-down if applicable
    if daily.double_down:
        click.echo(f"Checking {daily.double_down.batter_name} (game {daily.double_down.game_pk})...")
        double_result = check_hit(daily.double_down.game_pk, daily.double_down.batter_id)
        if double_result is None:
            click.echo(f"Game {daily.double_down.game_pk} not final yet. Try again later.")
            return
        results.append(double_result)

    # Update streak
    new_streak = update_streak(results, picks_path)

    # Report
    if all(results):
        hit_names = [daily.pick.batter_name]
        if daily.double_down:
            hit_names.append(daily.double_down.batter_name)
        click.echo(f"HIT! {' + '.join(hit_names)}. Streak: {new_streak}")
    else:
        miss_names = []
        if not results[0]:
            miss_names.append(daily.pick.batter_name)
        if len(results) > 1 and not results[1]:
            miss_names.append(daily.double_down.batter_name)
        click.echo(f"MISS: {', '.join(miss_names)}. Streak reset to 0.")

    # Save result to pick file for reference
    daily.streak = new_streak
    save_pick(daily, picks_path)
```

- [ ] **Step 2: Manual smoke test**

```bash
# Check a past date (if pick file exists from Task 5 smoke test)
cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run bts check-results --date 2026-03-31
```

Expected: Queries MLB API, reports hit/miss, updates `data/picks/streak.json`.

- [ ] **Step 3: Run all tests**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/bts/cli.py
git commit -m "feat: add 'bts check-results' command for streak tracking"
```

---

### Task 7: Cron setup on Mac

**Files:**
- Create: `scripts/cron-setup.sh` — helper to install/verify cron entries

- [ ] **Step 1: Create cron setup script**

```bash
# scripts/cron-setup.sh
#!/usr/bin/env bash
# BTS daily automation cron setup for Mac.
# All times are in the system timezone (ET assumed).
#
# Schedule:
#   11:00 AM ET — Early games (1-3pm starts)
#   4:00 PM ET  — Bulk games (6-8pm starts)
#   7:30 PM ET  — West coast (9-10pm starts), always posts
#   1:00 AM ET  — Check yesterday's results, update streak
#
# Usage: bash scripts/cron-setup.sh [install|show|remove]

set -euo pipefail

BTS_DIR="/Users/stone/projects/bts"
LOG_DIR="$BTS_DIR/data/picks"
UV_PREFIX="UV_CACHE_DIR=/tmp/uv-cache"
MARKER="# BTS-AUTOMATION"

CRON_LINES="$MARKER
0 11 * * * cd $BTS_DIR && $UV_PREFIX uv run bts run --date \$(date +\%Y-\%m-\%d) >> $LOG_DIR/automation.log 2>&1 $MARKER
0 16 * * * cd $BTS_DIR && $UV_PREFIX uv run bts run --date \$(date +\%Y-\%m-\%d) >> $LOG_DIR/automation.log 2>&1 $MARKER
30 19 * * * cd $BTS_DIR && $UV_PREFIX uv run bts run --date \$(date +\%Y-\%m-\%d) >> $LOG_DIR/automation.log 2>&1 $MARKER
0 1 * * * cd $BTS_DIR && $UV_PREFIX uv run bts check-results --date \$(date -v-1d +\%Y-\%m-\%d) >> $LOG_DIR/automation.log 2>&1 $MARKER"

case "${1:-show}" in
    install)
        mkdir -p "$LOG_DIR"
        # Remove old BTS entries, add new ones
        (crontab -l 2>/dev/null | grep -v "$MARKER"; echo "$CRON_LINES") | crontab -
        echo "Installed BTS cron jobs. Verify with: crontab -l"
        ;;
    show)
        echo "Current BTS cron entries:"
        crontab -l 2>/dev/null | grep "$MARKER" || echo "(none)"
        echo ""
        echo "Would install:"
        echo "$CRON_LINES"
        ;;
    remove)
        crontab -l 2>/dev/null | grep -v "$MARKER" | crontab -
        echo "Removed BTS cron jobs."
        ;;
    *)
        echo "Usage: $0 [install|show|remove]"
        exit 1
        ;;
esac
```

- [ ] **Step 2: Verify cron script shows correct entries**

```bash
cd /Users/stone/projects/bts && bash scripts/cron-setup.sh show
```

Expected: Shows the 4 cron entries (11am, 4pm, 7:30pm runs + 1am results check).

- [ ] **Step 3: Commit**

```bash
git add scripts/cron-setup.sh
git commit -m "feat: add cron setup script for daily BTS automation"
```

**Note:** Don't install cron yet — run `bts run --dry-run` manually for a few days first to validate the pipeline. Install cron when confident.

---

## Post-Plan Verification Checklist

After all tasks complete:

- [ ] `bts predict --date YYYY-MM-DD` still works (no regression)
- [ ] `bts post --dry-run` still works (no regression)
- [ ] `bts run --date YYYY-MM-DD --dry-run` produces valid pick JSON
- [ ] `bts check-results --date YYYY-MM-DD` correctly reports hit/miss
- [ ] `data/picks/streak.json` updates correctly
- [ ] All tests pass: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v`
- [ ] No credentials in code (Bluesky password from keychain only)
