# Milestone 1: Data Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data pipeline that pulls MLB game feeds from the API and transforms them into PA-level Parquet tables ready for feature engineering.

**Architecture:** CLI-driven staged pipeline. `bts data pull` downloads raw game feed JSON from the MLB Stats API v1.1 (one file per game, incremental resume). `bts data build` parses those JSONs into plate-appearance-level records stored as Parquet with nested pitch sequence arrays. Optional weather enrichment via Open-Meteo.

**Tech Stack:** Python 3.12+, uv, click, pandas, pyarrow, urllib (stdlib for API calls)

---

## File Map

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Create: project metadata, dependencies, CLI entrypoint |
| `src/bts/__init__.py` | Create: package init |
| `src/bts/cli.py` | Create: click CLI group with `data pull` and `data build` commands |
| `src/bts/data/__init__.py` | Create: data subpackage |
| `src/bts/data/schema.py` | Create: PA column definitions, constants (hit events, PA-ending events) |
| `src/bts/data/pull.py` | Create: schedule discovery, game feed downloader, weather enrichment |
| `src/bts/data/build.py` | Create: game feed parser, PA extractor, Parquet writer |
| `tests/__init__.py` | Create: test package |
| `tests/data/__init__.py` | Create: test subpackage |
| `tests/data/test_schema.py` | Create: schema constant tests |
| `tests/data/test_pull.py` | Create: schedule + downloader tests |
| `tests/data/test_build.py` | Create: parser + builder tests |
| `tests/conftest.py` | Create: shared fixtures (sample game feed JSON) |
| `.gitignore` | Create: ignore data/, .venv, __pycache__, etc. |
| `ARCHITECTURE.md` | Create: project architecture overview |

---

### Task 1: Scaffold Project

**Files:**
- Create: `pyproject.toml`
- Create: `src/bts/__init__.py`
- Create: `src/bts/cli.py`
- Create: `src/bts/data/__init__.py`
- Create: `.gitignore`
- Create: `ARCHITECTURE.md`
- Create: `tests/__init__.py`
- Create: `tests/data/__init__.py`

- [ ] **Step 1: Initialize uv project**

```bash
cd /Users/stone/projects/bts
uv init --lib --name bts --python 3.12
```

This creates a basic `pyproject.toml` and `src/bts/__init__.py`.

- [ ] **Step 2: Edit pyproject.toml**

Replace the generated `pyproject.toml` with:

```toml
[project]
name = "bts"
version = "0.1.0"
description = "Beat the Streak v2 — PA-level MLB hit prediction"
requires-python = ">=3.12"
dependencies = [
    "click>=8.1",
    "pandas>=2.2",
    "pyarrow>=15.0",
]

[project.scripts]
bts = "bts.cli:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"

[dependency-groups]
dev = [
    "pytest>=8.0",
]
```

- [ ] **Step 3: Create directory structure and .gitignore**

```bash
mkdir -p src/bts/data tests/data
touch src/bts/data/__init__.py tests/__init__.py tests/data/__init__.py
```

Create `.gitignore`:

```
data/raw/
data/processed/
data/models/
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
.pytest_cache/
```

- [ ] **Step 4: Create minimal CLI entrypoint**

`src/bts/cli.py`:

```python
import click


@click.group()
def cli():
    """Beat the Streak v2 — PA-level MLB hit prediction."""
    pass


@cli.group()
def data():
    """Data pipeline commands."""
    pass
```

- [ ] **Step 5: Create ARCHITECTURE.md**

`ARCHITECTURE.md`:

```markdown
# BTS Architecture

## Overview

Two-stage hit prediction model for MLB's Beat the Streak contest.
Stage 1 predicts P(hit) per plate appearance. Stage 2 aggregates to P(>=1 hit) per game.

## Pipeline

1. `bts data pull` — Downloads raw game feeds from MLB Stats API v1.1 to `data/raw/{season}/{gamePk}.json`
2. `bts data build` — Parses game feeds into PA-level Parquet at `data/processed/pa_{season}.parquet`
3. Feature engineering (Milestone 2+) — Computes rolling stats, pitcher archetypes, context features at training time
4. Training and evaluation (Milestone 3+) — PA-level LightGBM, game-level aggregation, walk-forward backtesting

## Key Design Decisions

- PA-level modeling (not game-level) — more training data, natural lineup position handling
- Raw pitch sequences preserved in Parquet nested arrays — EDA determines aggregations
- Features computed at training time, not baked into build — enables fast iteration
- Walk-forward validation with shift(1) — double defense against temporal leakage

## Data Flow

```
MLB Stats API → data/raw/{season}/{gamePk}.json
                         ↓ bts data build
              data/processed/pa_{season}.parquet
                         ↓ feature engineering
              Training DataFrame with all features
                         ↓ model training
              PA-level P(hit) predictions
                         ↓ aggregation
              Game-level P(>=1 hit) rankings
```

## Spec

See `docs/superpowers/specs/2026-03-29-bts-v2-design.md` for full design.
```

- [ ] **Step 6: Install dependencies and verify CLI**

```bash
cd /Users/stone/projects/bts
uv sync
uv run bts --help
```

Expected output includes `data` subcommand.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "scaffold: uv project with click CLI and directory structure"
```

---

### Task 2: Schema Module

**Files:**
- Create: `src/bts/data/schema.py`
- Create: `tests/data/test_schema.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_schema.py`:

```python
from bts.data.schema import PA_COLUMNS, HIT_EVENTS, PA_ENDING_EVENTS


def test_pa_columns_has_required_fields():
    required = [
        "game_pk", "date", "season", "batter_id", "pitcher_id",
        "lineup_position", "is_home", "hp_umpire_id", "venue_id",
        "pitch_count", "pitch_types", "pitch_calls", "pitch_px", "pitch_pz",
        "sz_top", "sz_bottom", "final_count_balls", "final_count_strikes",
        "launch_speed", "launch_angle", "event_type", "is_hit",
        "weather_temp", "weather_wind_speed", "weather_wind_dir", "roof_type",
    ]
    for col in required:
        assert col in PA_COLUMNS, f"Missing column: {col}"


def test_hit_events_are_subset_of_pa_ending():
    for event in HIT_EVENTS:
        assert event in PA_ENDING_EVENTS, f"{event} not in PA_ENDING_EVENTS"


def test_hit_events_contains_all_hit_types():
    assert "single" in HIT_EVENTS
    assert "double" in HIT_EVENTS
    assert "triple" in HIT_EVENTS
    assert "home_run" in HIT_EVENTS
    assert len(HIT_EVENTS) == 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/stone/projects/bts
uv run pytest tests/data/test_schema.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Write implementation**

`src/bts/data/schema.py`:

```python
"""PA table schema definitions and event constants."""

HIT_EVENTS = frozenset({"single", "double", "triple", "home_run"})

PA_ENDING_EVENTS = frozenset({
    "single", "double", "triple", "home_run",
    "field_out", "strikeout", "walk", "hit_by_pitch",
    "force_out", "grounded_into_double_play", "double_play",
    "field_error", "sac_fly", "sac_bunt",
    "fielders_choice", "fielders_choice_out",
    "strikeout_double_play", "catcher_interf",
    "sac_fly_double_play", "triple_play", "sac_bunt_double_play",
})

# Ordered list of columns in the PA Parquet table.
# Pitch sequence columns (pitch_types, pitch_calls, pitch_px, pitch_pz)
# are stored as nested lists in Parquet.
PA_COLUMNS = [
    "game_pk",
    "date",
    "season",
    "batter_id",
    "pitcher_id",
    "lineup_position",
    "is_home",
    "hp_umpire_id",
    "venue_id",
    "pitch_count",
    "pitch_types",
    "pitch_calls",
    "pitch_px",
    "pitch_pz",
    "sz_top",
    "sz_bottom",
    "final_count_balls",
    "final_count_strikes",
    "launch_speed",
    "launch_angle",
    "event_type",
    "is_hit",
    "weather_temp",
    "weather_wind_speed",
    "weather_wind_dir",
    "roof_type",
]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/data/test_schema.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/data/schema.py tests/data/test_schema.py
git commit -m "feat: PA schema definitions and event constants"
```

---

### Task 3: Schedule Discovery

**Files:**
- Create: `src/bts/data/pull.py`
- Create: `tests/data/test_pull.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_pull.py`:

```python
import json
from unittest.mock import patch, MagicMock
from bts.data.pull import discover_games


MOCK_SCHEDULE_RESPONSE = {
    "dates": [{
        "games": [
            {
                "gamePk": 823651,
                "officialDate": "2025-06-01",
                "status": {"detailedState": "Final"},
                "teams": {
                    "away": {"team": {"name": "Pittsburgh Pirates"}},
                    "home": {"team": {"name": "New York Mets"}},
                },
            },
            {
                "gamePk": 823652,
                "officialDate": "2025-06-01",
                "status": {"detailedState": "Scheduled"},
                "teams": {
                    "away": {"team": {"name": "Boston Red Sox"}},
                    "home": {"team": {"name": "Cincinnati Reds"}},
                },
            },
        ]
    }]
}


def _mock_urlopen(url, **kwargs):
    resp = MagicMock()
    resp.read.return_value = json.dumps(MOCK_SCHEDULE_RESPONSE).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


@patch("bts.data.pull.urlopen", side_effect=_mock_urlopen)
def test_discover_games_returns_final_games_only(mock_open):
    games = discover_games("2025-06-01", "2025-06-01")
    assert len(games) == 1
    assert games[0]["gamePk"] == 823651


@patch("bts.data.pull.urlopen", side_effect=_mock_urlopen)
def test_discover_games_includes_date(mock_open):
    games = discover_games("2025-06-01", "2025-06-01")
    assert games[0]["date"] == "2025-06-01"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/data/test_pull.py::test_discover_games_returns_final_games_only -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write implementation**

`src/bts/data/pull.py`:

```python
"""MLB Stats API data pulling: schedule discovery and game feed download."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request

API_BASE = "https://statsapi.mlb.com"


def discover_games(start_date: str, end_date: str) -> list[dict]:
    """Discover completed MLB games in a date range.

    Args:
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD

    Returns:
        List of dicts with keys: gamePk, date
    """
    games = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        url = f"{API_BASE}/api/v1/schedule?sportId=1&date={date_str}"
        resp = urlopen(url, timeout=15)
        data = json.loads(resp.read())

        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                if game["status"]["detailedState"] == "Final":
                    games.append({
                        "gamePk": game["gamePk"],
                        "date": game.get("officialDate", date_str),
                    })

        current += timedelta(days=1)

    return games
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/data/test_pull.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/data/pull.py tests/data/test_pull.py
git commit -m "feat: schedule discovery from MLB Stats API"
```

---

### Task 4: Game Feed Downloader

**Files:**
- Modify: `src/bts/data/pull.py`
- Modify: `tests/data/test_pull.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_pull.py`:

```python
from bts.data.pull import download_game_feed, pull_feeds


def test_download_game_feed_writes_json(tmp_path):
    sample_feed = {"gameData": {"game": {"pk": 123456}}, "liveData": {}}

    def _mock_urlopen_feed(url, **kwargs):
        resp = MagicMock()
        resp.read.return_value = json.dumps(sample_feed).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("bts.data.pull.urlopen", side_effect=_mock_urlopen_feed):
        path = download_game_feed(123456, tmp_path / "2025")

    assert path.exists()
    assert path.name == "123456.json"
    data = json.loads(path.read_text())
    assert data["gameData"]["game"]["pk"] == 123456


def test_download_game_feed_skips_existing(tmp_path):
    out_dir = tmp_path / "2025"
    out_dir.mkdir()
    existing = out_dir / "123456.json"
    existing.write_text('{"already": "here"}')

    # Should not call urlopen at all
    with patch("bts.data.pull.urlopen") as mock_open:
        path = download_game_feed(123456, out_dir)

    mock_open.assert_not_called()
    assert json.loads(path.read_text()) == {"already": "here"}


def test_pull_feeds_orchestrates(tmp_path):
    games = [{"gamePk": 111, "date": "2025-06-01"}, {"gamePk": 222, "date": "2025-06-01"}]
    sample_feed = {"gameData": {}, "liveData": {}}

    def _mock_urlopen_any(url, **kwargs):
        resp = MagicMock()
        resp.read.return_value = json.dumps(sample_feed).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("bts.data.pull.discover_games", return_value=games), \
         patch("bts.data.pull.urlopen", side_effect=_mock_urlopen_any):
        paths = pull_feeds("2025-06-01", "2025-06-01", tmp_path, delay=0)

    assert len(paths) == 2
    assert (tmp_path / "2025" / "111.json").exists()
    assert (tmp_path / "2025" / "222.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/data/test_pull.py::test_download_game_feed_writes_json -v
```

Expected: FAIL with `ImportError` for `download_game_feed`.

- [ ] **Step 3: Write implementation**

Add to `src/bts/data/pull.py`:

```python
def download_game_feed(game_pk: int, output_dir: Path) -> Path:
    """Download a single game feed JSON. Skips if already exists.

    Args:
        game_pk: MLB game PK identifier
        output_dir: Directory to write {game_pk}.json into

    Returns:
        Path to the JSON file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{game_pk}.json"

    if output_path.exists():
        return output_path

    url = f"{API_BASE}/api/v1.1/game/{game_pk}/feed/live"
    resp = urlopen(url, timeout=30)
    data = resp.read()

    output_path.write_bytes(data)
    return output_path


def pull_feeds(
    start_date: str,
    end_date: str,
    data_dir: Path,
    delay: float = 0.5,
) -> list[Path]:
    """Pull all game feeds for a date range.

    Args:
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        data_dir: Root data directory (files go to data_dir/{season}/{gamePk}.json)
        delay: Seconds between API requests

    Returns:
        List of paths to downloaded JSON files
    """
    games = discover_games(start_date, end_date)
    paths = []

    for i, game in enumerate(games):
        season = game["date"][:4]
        output_dir = data_dir / season
        path = download_game_feed(game["gamePk"], output_dir)
        paths.append(path)

        if delay > 0 and i < len(games) - 1:
            time.sleep(delay)

    return paths
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/data/test_pull.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/data/pull.py tests/data/test_pull.py
git commit -m "feat: game feed downloader with incremental resume"
```

---

### Task 5: `bts data pull` CLI Command

**Files:**
- Modify: `src/bts/cli.py`
- Test: manual CLI smoke test

- [ ] **Step 1: Wire up the pull command**

Replace `src/bts/cli.py`:

```python
import click
from pathlib import Path


@click.group()
def cli():
    """Beat the Streak v2 — PA-level MLB hit prediction."""
    pass


@cli.group()
def data():
    """Data pipeline commands."""
    pass


@data.command()
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--data-dir", default="data/raw", type=click.Path(), help="Output directory")
@click.option("--delay", default=0.5, type=float, help="Seconds between API requests")
def pull(start: str, end: str, data_dir: str, delay: float):
    """Pull game feeds from MLB Stats API."""
    from bts.data.pull import pull_feeds

    output = Path(data_dir)
    click.echo(f"Pulling games from {start} to {end} into {output}/")
    paths = pull_feeds(start, end, output, delay=delay)
    click.echo(f"Done. {len(paths)} game feeds downloaded.")
```

- [ ] **Step 2: Smoke test with a single day**

```bash
cd /Users/stone/projects/bts
uv run bts data pull --start 2025-06-01 --end 2025-06-01 --data-dir data/raw
```

Expected: downloads ~15 game feed JSON files to `data/raw/2025/`. Verify one exists:

```bash
ls data/raw/2025/ | head -5
python3 -c "import json; d=json.load(open('data/raw/2025/$(ls data/raw/2025/ | head -1)')); print(d['gameData']['teams']['home']['name'])"
```

- [ ] **Step 3: Commit**

```bash
git add src/bts/cli.py
git commit -m "feat: bts data pull CLI command"
```

---

### Task 6: Game Feed Parser

**Files:**
- Create: `src/bts/data/build.py`
- Create: `tests/conftest.py`
- Create: `tests/data/test_build.py`

This is the most complex task. The parser extracts plate appearances from a game feed JSON, including pitch sequences, lineup position, umpire, weather, and hit data.

- [ ] **Step 1: Create test fixture**

`tests/conftest.py`:

```python
import json
import pytest
from pathlib import Path


@pytest.fixture
def sample_game_feed():
    """Minimal game feed JSON with enough structure to test PA extraction."""
    return {
        "gameData": {
            "datetime": {"officialDate": "2025-06-15"},
            "teams": {
                "away": {"id": 134, "name": "Pittsburgh Pirates"},
                "home": {"id": 121, "name": "New York Mets"},
            },
            "venue": {
                "id": 3289,
                "name": "Citi Field",
                "fieldInfo": {"roofType": "Open"},
            },
            "weather": {
                "condition": "Sunny",
                "temp": "78",
                "wind": "9 mph, Out To CF",
            },
            "game": {"pk": 999999, "season": "2025"},
        },
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {
                        "batters": [100001],
                        "players": {
                            "ID100001": {
                                "person": {"id": 100001, "fullName": "Test Batter"},
                                "battingOrder": "300",
                            },
                        },
                    },
                    "home": {
                        "batters": [200001],
                        "players": {
                            "ID200001": {
                                "person": {"id": 200001, "fullName": "Home Batter"},
                                "battingOrder": "100",
                            },
                        },
                    },
                },
                "officials": [
                    {
                        "official": {"id": 427215, "fullName": "Test Umpire"},
                        "officialType": "Home Plate",
                    },
                ],
            },
            "plays": {
                "allPlays": [
                    {
                        "result": {
                            "eventType": "single",
                            "description": "Test Batter singles.",
                        },
                        "about": {
                            "halfInning": "top",
                            "inning": 1,
                            "hasReview": False,
                        },
                        "matchup": {
                            "batter": {"id": 100001},
                            "pitcher": {"id": 300001},
                        },
                        "count": {"balls": 1, "strikes": 2},
                        "playEvents": [
                            {
                                "isPitch": True,
                                "pitchNumber": 1,
                                "details": {
                                    "call": {"code": "B", "description": "Ball"},
                                    "type": {"code": "FF", "description": "Four-Seam Fastball"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": -0.5, "pZ": 2.8},
                                    "strikeZoneTop": 3.4,
                                    "strikeZoneBottom": 1.7,
                                },
                                "count": {"balls": 1, "strikes": 0},
                            },
                            {
                                "isPitch": True,
                                "pitchNumber": 2,
                                "details": {
                                    "call": {"code": "C", "description": "Called Strike"},
                                    "type": {"code": "SL", "description": "Slider"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": 0.3, "pZ": 2.1},
                                    "strikeZoneTop": 3.4,
                                    "strikeZoneBottom": 1.7,
                                },
                                "count": {"balls": 1, "strikes": 1},
                            },
                            {
                                "isPitch": True,
                                "pitchNumber": 3,
                                "details": {
                                    "call": {"code": "S", "description": "Swinging Strike"},
                                    "type": {"code": "CH", "description": "Changeup"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": 0.1, "pZ": 1.9},
                                    "strikeZoneTop": 3.4,
                                    "strikeZoneBottom": 1.7,
                                },
                                "count": {"balls": 1, "strikes": 2},
                            },
                            {
                                "isPitch": True,
                                "pitchNumber": 4,
                                "details": {
                                    "call": {"code": "X", "description": "In play, no out"},
                                    "type": {"code": "FF", "description": "Four-Seam Fastball"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": -0.2, "pZ": 2.5},
                                    "strikeZoneTop": 3.4,
                                    "strikeZoneBottom": 1.7,
                                },
                                "hitData": {
                                    "launchSpeed": 98.3,
                                    "launchAngle": 12.0,
                                },
                                "count": {"balls": 1, "strikes": 2},
                            },
                        ],
                    },
                    {
                        "result": {
                            "eventType": "strikeout",
                            "description": "Home Batter strikes out.",
                        },
                        "about": {
                            "halfInning": "bottom",
                            "inning": 1,
                            "hasReview": False,
                        },
                        "matchup": {
                            "batter": {"id": 200001},
                            "pitcher": {"id": 400001},
                        },
                        "count": {"balls": 0, "strikes": 3},
                        "playEvents": [
                            {
                                "isPitch": True,
                                "pitchNumber": 1,
                                "details": {
                                    "call": {"code": "S", "description": "Swinging Strike"},
                                    "type": {"code": "FF", "description": "Four-Seam Fastball"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": 0.0, "pZ": 2.5},
                                    "strikeZoneTop": 3.3,
                                    "strikeZoneBottom": 1.6,
                                },
                                "count": {"balls": 0, "strikes": 1},
                            },
                            {
                                "isPitch": True,
                                "pitchNumber": 2,
                                "details": {
                                    "call": {"code": "S", "description": "Swinging Strike"},
                                    "type": {"code": "SL", "description": "Slider"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": 0.8, "pZ": 1.8},
                                    "strikeZoneTop": 3.3,
                                    "strikeZoneBottom": 1.6,
                                },
                                "count": {"balls": 0, "strikes": 2},
                            },
                            {
                                "isPitch": True,
                                "pitchNumber": 3,
                                "details": {
                                    "call": {"code": "S", "description": "Swinging Strike"},
                                    "type": {"code": "CH", "description": "Changeup"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": 0.4, "pZ": 2.0},
                                    "strikeZoneTop": 3.3,
                                    "strikeZoneBottom": 1.6,
                                },
                                "count": {"balls": 0, "strikes": 3},
                            },
                        ],
                    },
                ],
            },
        },
    }


@pytest.fixture
def sample_feed_path(sample_game_feed, tmp_path):
    """Write sample feed to a file and return its path."""
    raw_dir = tmp_path / "raw" / "2025"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "999999.json"
    path.write_text(json.dumps(sample_game_feed))
    return path
```

- [ ] **Step 2: Write the failing tests**

`tests/data/test_build.py`:

```python
import pandas as pd
from bts.data.build import parse_game_feed
from bts.data.schema import PA_COLUMNS


def test_parse_game_feed_returns_correct_count(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    assert len(rows) == 2  # one single + one strikeout


def test_parse_game_feed_hit_fields(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    single = rows[0]
    assert single["is_hit"] == 1
    assert single["event_type"] == "single"
    assert single["batter_id"] == 100001
    assert single["pitcher_id"] == 300001


def test_parse_game_feed_non_hit_fields(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    strikeout = rows[1]
    assert strikeout["is_hit"] == 0
    assert strikeout["event_type"] == "strikeout"
    assert strikeout["batter_id"] == 200001


def test_parse_game_feed_pitch_sequences(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    single = rows[0]
    assert single["pitch_count"] == 4
    assert single["pitch_types"] == ["FF", "SL", "CH", "FF"]
    assert single["pitch_calls"] == ["B", "C", "S", "X"]
    assert len(single["pitch_px"]) == 4
    assert len(single["pitch_pz"]) == 4


def test_parse_game_feed_context(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    single = rows[0]
    assert single["game_pk"] == 999999
    assert single["date"] == "2025-06-15"
    assert single["season"] == 2025
    assert single["venue_id"] == 3289
    assert single["hp_umpire_id"] == 427215
    assert single["weather_temp"] == 78
    assert single["weather_wind_speed"] == 9
    assert single["weather_wind_dir"] == "Out To CF"
    assert single["roof_type"] == "Open"


def test_parse_game_feed_lineup_position(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    # Away batter has battingOrder "300" -> lineup position 3
    assert rows[0]["lineup_position"] == 3
    # Home batter has battingOrder "100" -> lineup position 1
    assert rows[1]["lineup_position"] == 1


def test_parse_game_feed_is_home(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    assert rows[0]["is_home"] is False  # top inning = away
    assert rows[1]["is_home"] is True   # bottom inning = home


def test_parse_game_feed_strike_zone(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    # Uses the last pitch's strike zone
    assert rows[0]["sz_top"] == 3.4
    assert rows[0]["sz_bottom"] == 1.7


def test_parse_game_feed_launch_data(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    single = rows[0]
    assert single["launch_speed"] == 98.3
    assert single["launch_angle"] == 12.0
    # Strikeout has no hitData
    strikeout = rows[1]
    assert strikeout["launch_speed"] is None
    assert strikeout["launch_angle"] is None


def test_parse_game_feed_count(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    assert rows[0]["final_count_balls"] == 1
    assert rows[0]["final_count_strikes"] == 2
    assert rows[1]["final_count_balls"] == 0
    assert rows[1]["final_count_strikes"] == 3


def test_parse_game_feed_has_all_columns(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    for col in PA_COLUMNS:
        assert col in rows[0], f"Missing column: {col}"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/data/test_build.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 4: Write implementation**

`src/bts/data/build.py`:

```python
"""Parse MLB game feed JSON into plate appearance records."""

import json
import re
from pathlib import Path

import pandas as pd

from bts.data.schema import HIT_EVENTS, PA_ENDING_EVENTS


def _parse_wind(wind_str: str) -> tuple[int | None, str | None]:
    """Parse '9 mph, Out To CF' into (9, 'Out To CF')."""
    if not wind_str:
        return None, None
    match = re.match(r"(\d+)\s*mph,?\s*(.*)", wind_str)
    if match:
        return int(match.group(1)), match.group(2).strip()
    return None, wind_str


def _get_lineup_positions(boxscore: dict) -> dict[int, int]:
    """Build a map of batter_id -> lineup_position from boxscore.

    battingOrder is stored as "100", "200", etc. Divide by 100.
    Pinch hitters and substitutes may have values like "101" -> still position 1.
    """
    positions = {}
    for side in ("away", "home"):
        players = boxscore["teams"][side].get("players", {})
        for key, player in players.items():
            batting_order = player.get("battingOrder")
            if batting_order:
                batter_id = player["person"]["id"]
                positions[batter_id] = int(batting_order) // 100
    return positions


def _get_hp_umpire_id(boxscore: dict) -> int | None:
    """Extract home plate umpire ID from officials list."""
    for official in boxscore.get("officials", []):
        if official.get("officialType") == "Home Plate":
            return official["official"]["id"]
    return None


def parse_game_feed(feed: dict) -> list[dict]:
    """Parse a game feed into a list of plate appearance dicts.

    Each dict has all columns defined in schema.PA_COLUMNS.
    Pitch sequence fields are stored as Python lists.
    """
    game_data = feed["gameData"]
    live_data = feed["liveData"]
    boxscore = live_data["boxscore"]

    # Game-level context
    game_pk = game_data["game"]["pk"]
    date = game_data["datetime"]["officialDate"]
    season = int(game_data["game"]["season"])
    venue_id = game_data["venue"]["id"]
    roof_type = game_data["venue"].get("fieldInfo", {}).get("roofType")

    weather = game_data.get("weather", {})
    weather_temp_str = weather.get("temp")
    weather_temp = int(weather_temp_str) if weather_temp_str else None
    wind_speed, wind_dir = _parse_wind(weather.get("wind", ""))

    hp_umpire_id = _get_hp_umpire_id(boxscore)
    lineup_positions = _get_lineup_positions(boxscore)

    rows = []
    for play in live_data["plays"]["allPlays"]:
        event_type = play["result"].get("eventType", "")
        if event_type not in PA_ENDING_EVENTS:
            continue

        batter_id = play["matchup"]["batter"]["id"]
        pitcher_id = play["matchup"]["pitcher"]["id"]
        is_home = play["about"]["halfInning"] == "bottom"
        count = play.get("count", {})

        # Extract pitch sequence
        pitch_types = []
        pitch_calls = []
        pitch_px = []
        pitch_pz = []
        sz_top = None
        sz_bottom = None
        launch_speed = None
        launch_angle = None

        for event in play.get("playEvents", []):
            if not event.get("isPitch"):
                continue
            details = event.get("details", {})
            pitch_data = event.get("pitchData", {})
            coords = pitch_data.get("coordinates", {})

            pitch_types.append(details.get("type", {}).get("code", "UN"))
            pitch_calls.append(details.get("call", {}).get("code", ""))
            pitch_px.append(coords.get("pX"))
            pitch_pz.append(coords.get("pZ"))

            sz_top = pitch_data.get("strikeZoneTop", sz_top)
            sz_bottom = pitch_data.get("strikeZoneBottom", sz_bottom)

            hit_data = event.get("hitData")
            if hit_data:
                launch_speed = hit_data.get("launchSpeed")
                launch_angle = hit_data.get("launchAngle")

        rows.append({
            "game_pk": game_pk,
            "date": date,
            "season": season,
            "batter_id": batter_id,
            "pitcher_id": pitcher_id,
            "lineup_position": lineup_positions.get(batter_id),
            "is_home": is_home,
            "hp_umpire_id": hp_umpire_id,
            "venue_id": venue_id,
            "pitch_count": len(pitch_types),
            "pitch_types": pitch_types,
            "pitch_calls": pitch_calls,
            "pitch_px": pitch_px,
            "pitch_pz": pitch_pz,
            "sz_top": sz_top,
            "sz_bottom": sz_bottom,
            "final_count_balls": count.get("balls", 0),
            "final_count_strikes": count.get("strikes", 0),
            "launch_speed": launch_speed,
            "launch_angle": launch_angle,
            "event_type": event_type,
            "is_hit": 1 if event_type in HIT_EVENTS else 0,
            "weather_temp": weather_temp,
            "weather_wind_speed": wind_speed,
            "weather_wind_dir": wind_dir,
            "roof_type": roof_type,
        })

    return rows
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/data/test_build.py -v
```

Expected: 11 passed.

- [ ] **Step 6: Commit**

```bash
git add src/bts/data/build.py tests/conftest.py tests/data/test_build.py
git commit -m "feat: game feed parser extracts PA records with pitch sequences"
```

---

### Task 7: Parquet Builder and `bts data build` CLI

**Files:**
- Modify: `src/bts/data/build.py`
- Modify: `src/bts/cli.py`
- Modify: `tests/data/test_build.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_build.py`:

```python
from bts.data.build import build_season


def test_build_season_creates_parquet(sample_feed_path, tmp_path):
    raw_dir = sample_feed_path.parent.parent  # tmp_path/raw
    output_path = tmp_path / "processed" / "pa_2025.parquet"

    build_season(raw_dir, output_path, season=2025)

    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert len(df) == 2
    assert df["is_hit"].sum() == 1
    assert df["game_pk"].iloc[0] == 999999


def test_build_season_preserves_pitch_lists(sample_feed_path, tmp_path):
    raw_dir = sample_feed_path.parent.parent
    output_path = tmp_path / "processed" / "pa_2025.parquet"

    build_season(raw_dir, output_path, season=2025)

    df = pd.read_parquet(output_path)
    row = df.iloc[0]
    assert isinstance(row["pitch_types"], list)
    assert row["pitch_types"] == ["FF", "SL", "CH", "FF"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/data/test_build.py::test_build_season_creates_parquet -v
```

Expected: FAIL with `ImportError` for `build_season`.

- [ ] **Step 3: Write implementation**

Add to `src/bts/data/build.py`:

```python
def build_season(raw_dir: Path, output_path: Path, season: int) -> pd.DataFrame:
    """Build PA-level Parquet from raw game feed JSONs for one season.

    Args:
        raw_dir: Root raw directory (looks in raw_dir/{season}/*.json)
        output_path: Path to write the Parquet file
        season: Year to process

    Returns:
        The built DataFrame
    """
    season_dir = raw_dir / str(season)
    if not season_dir.exists():
        raise FileNotFoundError(f"No raw data for season {season} at {season_dir}")

    all_rows = []
    json_files = sorted(season_dir.glob("*.json"))

    for json_path in json_files:
        # Skip weather sidecar files
        if json_path.stem.endswith("_weather"):
            continue
        feed = json.loads(json_path.read_text())
        rows = parse_game_feed(feed)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/data/test_build.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Wire up CLI command**

Add to `src/bts/cli.py`, after the `pull` command:

```python
@data.command()
@click.option("--seasons", required=True, help="Comma-separated seasons (e.g., 2023,2024,2025)")
@click.option("--raw-dir", default="data/raw", type=click.Path(), help="Raw data directory")
@click.option("--out-dir", default="data/processed", type=click.Path(), help="Output directory")
def build(seasons: str, raw_dir: str, out_dir: str):
    """Build PA-level Parquet from raw game feeds."""
    from bts.data.build import build_season

    raw = Path(raw_dir)
    out = Path(out_dir)

    for season_str in seasons.split(","):
        season = int(season_str.strip())
        output_path = out / f"pa_{season}.parquet"
        click.echo(f"Building {output_path} from {raw}/{season}/...")
        df = build_season(raw, output_path, season)
        click.echo(f"  {len(df)} plate appearances written.")
```

- [ ] **Step 6: Smoke test end-to-end with real data**

Requires game feeds from Task 5's smoke test:

```bash
uv run bts data build --seasons 2025 --raw-dir data/raw --out-dir data/processed
```

Then verify:

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/processed/pa_2025.parquet')
print(f'Rows: {len(df)}')
print(f'Hits: {df.is_hit.sum()} ({df.is_hit.mean():.1%})')
print(f'Columns: {list(df.columns)}')
print(df[['batter_id','pitcher_id','pitch_count','event_type','is_hit']].head(10))
"
```

Expected: ~70-80 PA rows from one day's games, ~22% hit rate, all PA_COLUMNS present.

- [ ] **Step 7: Commit**

```bash
git add src/bts/data/build.py src/bts/cli.py tests/data/test_build.py
git commit -m "feat: bts data build creates PA-level Parquet from raw feeds"
```

---

### Task 8: Weather Enrichment (Open-Meteo)

**Files:**
- Modify: `src/bts/data/pull.py`
- Modify: `src/bts/data/build.py`
- Modify: `src/bts/cli.py`
- Modify: `tests/data/test_pull.py`
- Modify: `tests/data/test_build.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_pull.py`:

```python
from bts.data.pull import enrich_weather


MOCK_OPEN_METEO_RESPONSE = {
    "hourly": {
        "time": ["2025-06-15T19:00"],
        "surface_pressure": [1013.2],
        "relative_humidity_2m": [65],
        "dewpoint_2m": [18.5],
    }
}


def test_enrich_weather_writes_sidecar(tmp_path):
    game_feed = {
        "gameData": {
            "datetime": {"officialDate": "2025-06-15", "dateTime": "2025-06-15T23:10:00Z"},
            "venue": {
                "location": {"defaultCoordinates": {"latitude": 40.757, "longitude": -73.845}},
            },
        },
    }
    raw_dir = tmp_path / "2025"
    raw_dir.mkdir()
    (raw_dir / "999999.json").write_text(json.dumps(game_feed))

    def _mock_meteo(url, **kwargs):
        resp = MagicMock()
        resp.read.return_value = json.dumps(MOCK_OPEN_METEO_RESPONSE).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("bts.data.pull.urlopen", side_effect=_mock_meteo):
        enrich_weather(raw_dir)

    weather_path = raw_dir / "999999_weather.json"
    assert weather_path.exists()
    data = json.loads(weather_path.read_text())
    assert data["surface_pressure"] == 1013.2
    assert data["relative_humidity"] == 65


def test_enrich_weather_skips_existing(tmp_path):
    raw_dir = tmp_path / "2025"
    raw_dir.mkdir()
    (raw_dir / "999999.json").write_text("{}")
    (raw_dir / "999999_weather.json").write_text('{"already": "enriched"}')

    with patch("bts.data.pull.urlopen") as mock_open:
        enrich_weather(raw_dir)

    mock_open.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/data/test_pull.py::test_enrich_weather_writes_sidecar -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write implementation**

Add to `src/bts/data/pull.py`:

```python
OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"


def enrich_weather(season_dir: Path, delay: float = 0.3) -> int:
    """Fetch atmospheric data from Open-Meteo for all games in a season directory.

    Writes {gamePk}_weather.json sidecar files with pressure, humidity, dewpoint.
    Skips games that already have weather files.

    Returns:
        Number of games enriched
    """
    count = 0
    game_files = sorted(season_dir.glob("*.json"))

    for game_path in game_files:
        if game_path.stem.endswith("_weather"):
            continue

        weather_path = season_dir / f"{game_path.stem}_weather.json"
        if weather_path.exists():
            continue

        feed = json.loads(game_path.read_text())
        game_data = feed.get("gameData", {})

        coords = (
            game_data.get("venue", {})
            .get("location", {})
            .get("defaultCoordinates", {})
        )
        lat = coords.get("latitude")
        lon = coords.get("longitude")
        date = game_data.get("datetime", {}).get("officialDate")

        if not all([lat, lon, date]):
            continue

        url = (
            f"{OPEN_METEO_BASE}"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={date}&end_date={date}"
            f"&hourly=surface_pressure,relative_humidity_2m,dewpoint_2m"
        )

        try:
            resp = urlopen(url, timeout=15)
            data = json.loads(resp.read())
            hourly = data.get("hourly", {})

            # Use the value closest to typical game time (~19:00 local).
            # For simplicity, average all hourly values for the day.
            pressures = [p for p in hourly.get("surface_pressure", []) if p is not None]
            humidities = [h for h in hourly.get("relative_humidity_2m", []) if h is not None]
            dewpoints = [d for d in hourly.get("dewpoint_2m", []) if d is not None]

            weather_data = {
                "surface_pressure": sum(pressures) / len(pressures) if pressures else None,
                "relative_humidity": sum(humidities) / len(humidities) if humidities else None,
                "dewpoint": sum(dewpoints) / len(dewpoints) if dewpoints else None,
            }

            weather_path.write_text(json.dumps(weather_data))
            count += 1

            if delay > 0:
                time.sleep(delay)

        except Exception:
            # Don't fail the whole enrichment if one game's weather lookup fails
            continue

    return count
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/data/test_pull.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Update build.py to merge weather sidecar data**

In `src/bts/data/build.py`, update `build_season` to read weather sidecars:

Replace the `for json_path in json_files:` loop in `build_season` with:

```python
    for json_path in json_files:
        # Skip weather sidecar files
        if json_path.stem.endswith("_weather"):
            continue
        feed = json.loads(json_path.read_text())
        rows = parse_game_feed(feed)

        # Merge weather sidecar if present
        weather_path = json_path.parent / f"{json_path.stem}_weather.json"
        if weather_path.exists():
            weather = json.loads(weather_path.read_text())
            for row in rows:
                row["atm_pressure"] = weather.get("surface_pressure")
                row["humidity"] = weather.get("relative_humidity")

        all_rows.extend(rows)
```

Also add `"atm_pressure"` and `"humidity"` to `schema.py`'s `PA_COLUMNS` list:

Add after `"roof_type"`:
```python
    "atm_pressure",
    "humidity",
```

And update `parse_game_feed` to include default None values for those columns:

In the `rows.append({...})` dict, add after `"roof_type": roof_type,`:
```python
            "atm_pressure": None,
            "humidity": None,
```

- [ ] **Step 6: Add weather test for build**

Append to `tests/data/test_build.py`:

```python
def test_build_season_merges_weather(sample_feed_path, tmp_path):
    # Write a weather sidecar
    weather_path = sample_feed_path.parent / "999999_weather.json"
    weather_path.write_text(json.dumps({
        "surface_pressure": 1010.5,
        "relative_humidity": 72.0,
    }))

    raw_dir = sample_feed_path.parent.parent
    output_path = tmp_path / "processed" / "pa_2025.parquet"

    build_season(raw_dir, output_path, season=2025)

    df = pd.read_parquet(output_path)
    assert df["atm_pressure"].iloc[0] == 1010.5
    assert df["humidity"].iloc[0] == 72.0
```

- [ ] **Step 7: Update schema test**

Update the `required` list in `test_pa_columns_has_required_fields` in `tests/data/test_schema.py` to include `"atm_pressure"` and `"humidity"`.

- [ ] **Step 8: Wire CLI command**

Add to `src/bts/cli.py`, in the `pull` command function, after the `pull_feeds` call:

```python
@data.command(name="enrich-weather")
@click.option("--data-dir", default="data/raw", type=click.Path(), help="Raw data directory")
@click.option("--seasons", required=True, help="Comma-separated seasons (e.g., 2023,2024,2025)")
@click.option("--delay", default=0.3, type=float, help="Seconds between API requests")
def enrich_weather_cmd(data_dir: str, seasons: str, delay: float):
    """Enrich game feeds with atmospheric data from Open-Meteo."""
    from bts.data.pull import enrich_weather

    raw = Path(data_dir)
    for season_str in seasons.split(","):
        season = int(season_str.strip())
        season_dir = raw / str(season)
        if not season_dir.exists():
            click.echo(f"Skipping {season}: no raw data at {season_dir}")
            continue
        click.echo(f"Enriching {season} weather data...")
        count = enrich_weather(season_dir, delay=delay)
        click.echo(f"  {count} games enriched.")
```

- [ ] **Step 9: Run all tests**

```bash
uv run pytest -v
```

Expected: all tests pass (16+).

- [ ] **Step 10: Commit**

```bash
git add src/bts/data/pull.py src/bts/data/build.py src/bts/data/schema.py src/bts/cli.py tests/
git commit -m "feat: Open-Meteo weather enrichment with pressure and humidity"
```

---

### Task 9: Integration Validation with Real API Data

**Files:**
- No new files — this is a validation step against real data

- [ ] **Step 1: Pull one day of real data**

If not already pulled in Task 5:

```bash
cd /Users/stone/projects/bts
uv run bts data pull --start 2025-06-15 --end 2025-06-15
```

- [ ] **Step 2: Build Parquet from that day**

```bash
uv run bts data build --seasons 2025
```

- [ ] **Step 3: Validate the output**

```bash
python3 -c "
import pandas as pd

df = pd.read_parquet('data/processed/pa_2025.parquet')
print(f'Total PAs: {len(df)}')
print(f'Unique games: {df.game_pk.nunique()}')
print(f'Hit rate: {df.is_hit.mean():.1%}')
print(f'Avg pitch count: {df.pitch_count.mean():.1f}')
print(f'Lineup positions: {sorted(df.lineup_position.dropna().unique())}')
print(f'Has umpire: {df.hp_umpire_id.notna().all()}')
print(f'Has weather: {df.weather_temp.notna().mean():.0%}')
print(f'Has launch data: {df.launch_speed.notna().mean():.0%}')
print()
print('Sample rows:')
print(df[['batter_id','lineup_position','pitch_count','event_type','is_hit','weather_temp']].head(10))
print()
print('Pitch type sample:')
print(df.iloc[0]['pitch_types'])
"
```

Expected:
- ~550-700 PAs from ~15 games
- Hit rate ~22%
- Avg pitch count ~3.5-4.0
- Lineup positions 1-9
- Umpire present for all rows
- Weather present for all rows
- Launch data present for ~65% of rows (only batted ball events)

- [ ] **Step 4: Clean up test data (don't commit to git)**

```bash
# Verify .gitignore is working
git status
```

Expected: `data/` directory not shown in untracked files.

- [ ] **Step 5: Final commit if any fixes were needed**

Only commit if code changes were required during validation:

```bash
git add -A && git commit -m "fix: adjustments from integration validation" || echo "Nothing to commit"
```

---

## Post-Plan Summary

After all 9 tasks, you have:
- `bts data pull` downloading raw game feeds with incremental resume
- `bts data build` parsing feeds into PA-level Parquet with pitch sequences
- `bts data enrich-weather` adding atmospheric data from Open-Meteo
- ~16+ tests covering schema, parsing, download, weather enrichment
- Clean project structure ready for Milestone 2 (EDA + feature engineering)

**Next:** Pull the full 2023-2025 dataset (~7,200 games, ~2-3 hours), then write the Milestone 2 plan for EDA and feature engineering.
