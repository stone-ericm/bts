# BTS Leaderboard Watcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a twice-daily scraper that captures the MLB.com BTS leaderboard plus per-user picks logs into parquet, decoupled from the production picks pipeline, with health-check integration and CLI tooling.

**Architecture:** Direct JSON API via reverse-engineered endpoints (Playwright contingency only); pydantic-validated parquet storage under `data/leaderboard/`; `pass`-managed session cookies; systemd timer on bts-hetzner; integrated into `bts.health.runner`.

**Tech Stack:** Python 3.12 + uv, click (CLI), pydantic (schemas), httpx (HTTP), pyarrow (parquet), pytest, Playwright (auth contingency only), systemd (scheduling).

**Spec:** `docs/superpowers/specs/2026-05-01-bts-leaderboard-watcher-design.md` (commit `e297c04`).

**Project conventions** (read these before starting):
- All `uv` commands: `UV_CACHE_DIR=/tmp/uv-cache uv ...`
- Tests: `tests/<package>/test_<module>.py` mirrors `src/bts/<package>/<module>.py`
- Health checks: module exposes `check(...)` returning `list[Alert]`; registered in `src/bts/health/runner.py`
- Click CLI: subcommand groups added via `cli.add_command(...)` in `src/bts/cli.py`
- Bluesky DM: `from bts.dm import send_dm`
- Memory references: `feedback_dont_estimate_time.md` (no time estimates), `feedback_commit_to_hard_work.md` (commit when direction chosen)

---

## Task 0: Branch + dependency bumps

**Files:**
- Modify: `pyproject.toml` (add `httpx`, `pydantic` if not already pinned)
- Create: branch `feature/leaderboard-watcher`

- [ ] **Step 1: Create feature branch**

```bash
cd /Users/stone/projects/bts
git fetch origin
git checkout -b feature/leaderboard-watcher origin/main
```

- [ ] **Step 2: Add deps to pyproject.toml**

Inspect current state:
```bash
grep -E "httpx|pydantic" pyproject.toml
```

If `httpx` is absent, add to the main `dependencies` array:
```toml
"httpx>=0.27",
"pydantic>=2.7",
```

(Don't add `playwright` as a hard dep — it's contingency; install ad-hoc if Phase 1 needs it.)

- [ ] **Step 3: Sync deps + verify import**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import httpx, pydantic; print(httpx.__version__, pydantic.VERSION)"
```

Expected: prints two version strings, no errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): add httpx + pydantic for leaderboard watcher"
```

---

## Phase 1 — API Discovery (Mac local)

**Goal:** Identify the JSON endpoints behind the BTS app and document them. This phase is research, not TDD. Output is a populated `endpoints.py` and a working manual scrape.

### Task 1.1: Create package skeleton

**Files:**
- Create: `src/bts/leaderboard/__init__.py`
- Create: `src/bts/leaderboard/endpoints.py` (placeholder)
- Create: `tests/leaderboard/__init__.py`

- [ ] **Step 1: Create directories**

```bash
cd /Users/stone/projects/bts
mkdir -p src/bts/leaderboard tests/leaderboard tests/leaderboard/fixtures
touch src/bts/leaderboard/__init__.py tests/leaderboard/__init__.py
```

- [ ] **Step 2: Write skeleton endpoints.py**

```python
# src/bts/leaderboard/endpoints.py
"""Discovered MLB.com BTS API endpoints.

Filled in during Phase 1 by reverse-engineering the BTS app via Chrome
DevTools or the superpowers-chrome MCP. Each constant should be a
Python-format-string template parameterized on the runtime args
(date, username, tab, page).

When you discover a new endpoint, add it here, document the response
shape in models.py, and write a fixture in tests/leaderboard/fixtures/.
"""

# TODO Phase 1 — fill in actual URLs after DevTools observation
LEADERBOARD_URL_TEMPLATE: str = ""
USER_PICKS_URL_TEMPLATE: str = ""
USER_STATS_URL_TEMPLATE: str = ""
```

(Yes, `TODO` is the one allowed exception in the codebase here — Phase 1's whole job is to fill these in.)

- [ ] **Step 3: Verify import works**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from bts.leaderboard import endpoints; print('ok')"
```

Expected: prints `ok`, no errors.

- [ ] **Step 4: Commit**

```bash
git add src/bts/leaderboard/ tests/leaderboard/
git commit -m "feat(leaderboard): package skeleton + endpoints placeholder"
```

### Task 1.2: Capture session cookies via Playwright (one-time)

**Files:**
- Create: `scripts/capture_bts_cookies.py`

- [ ] **Step 1: Install Playwright (one-time)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv pip install playwright
UV_CACHE_DIR=/tmp/uv-cache uv run playwright install chromium
```

- [ ] **Step 2: Write the capture script**

```python
# scripts/capture_bts_cookies.py
"""One-time interactive cookie capture for MLB.com BTS.

Opens Playwright Chromium, lets the user log in interactively, then
serializes session cookies to stdout (or to `pass` if PASS_STORE_KEY
env var is set).

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/capture_bts_cookies.py

Login manually in the browser window that opens. Once you can see
the leaderboard at https://www.mlb.com/apps/beat-the-streak/game,
press Enter in the terminal. Cookies are extracted and printed as JSON.
"""
import json
import os
import subprocess
import sys

from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.mlb.com/apps/beat-the-streak/game")
        print("Browser opened. Log in to MLB.com, navigate to the leaderboard, then press Enter here...", file=sys.stderr)
        input()
        cookies = context.cookies()
        cookie_json = json.dumps(cookies, indent=2)

        pass_key = os.environ.get("PASS_STORE_KEY")
        if pass_key:
            subprocess.run(["pass", "insert", "-m", pass_key], input=cookie_json.encode(), check=True)
            print(f"Saved {len(cookies)} cookies to pass:{pass_key}", file=sys.stderr)
        else:
            print(cookie_json)
        browser.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it once and capture cookies**

```bash
PASS_STORE_KEY=mlb-bts-session-cookies UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/capture_bts_cookies.py
```

A Chromium window opens. Log in to MLB.com BTS. Once you can see the leaderboard, return to the terminal and press Enter. The script saves cookies to `pass`.

Verify:
```bash
pass show mlb-bts-session-cookies | head -5
```
Expected: JSON starts with `[\n  {`

- [ ] **Step 4: Commit the script (cookies stay in `pass`, not the repo)**

```bash
git add scripts/capture_bts_cookies.py
git commit -m "feat(leaderboard): one-time Playwright cookie capture script"
```

### Task 1.3: Discover the leaderboard endpoint

**Files:**
- Create: `tests/leaderboard/fixtures/leaderboard_active_streak.json`
- Modify: `src/bts/leaderboard/endpoints.py`

- [ ] **Step 1: Open Chrome DevTools and capture leaderboard fetch**

Manual interactive step. Two ways:

**Option A (recommended): superpowers-chrome MCP**

```
Use the chrome MCP to:
  1. Navigate to https://www.mlb.com/apps/beat-the-streak/game
  2. Click the "Leaderboard" tab
  3. Click "Active Streak"
  4. Inspect Network panel for XHR/fetch requests
  5. Identify the JSON endpoint that returned the top-N data
  6. Copy the request URL, headers, and response body
```

**Option B: Manual Chrome DevTools**

Open Chrome → DevTools → Network → filter "Fetch/XHR" → click leaderboard tab → identify the JSON endpoint that returned the leaderboard rows (the one with usernames + streak counts in the response).

- [ ] **Step 2: Save the response as fixture**

Save the captured JSON response as `tests/leaderboard/fixtures/leaderboard_active_streak.json`. The fixture must contain the raw response bytes (not formatted) — that's what we'll test parsers against.

- [ ] **Step 3: Update endpoints.py with discovered URL**

```python
# src/bts/leaderboard/endpoints.py — replace TODO line with actual URL
LEADERBOARD_URL_TEMPLATE: str = "https://api.example.mlb.com/bts/leaderboard?type={tab}&page={page}"  # ACTUAL URL FROM PHASE 1
```

(Replace with the actual URL discovered in Step 1.)

- [ ] **Step 4: Hand-verify with httpx**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "
import json, httpx
from bts.leaderboard.endpoints import LEADERBOARD_URL_TEMPLATE
cookies_raw = open('/tmp/cookies.json').read() if False else __import__('subprocess').check_output(['pass', 'show', 'mlb-bts-session-cookies']).decode()
cookies = {c['name']: c['value'] for c in json.loads(cookies_raw)}
url = LEADERBOARD_URL_TEMPLATE.format(tab='active', page=1)
r = httpx.get(url, cookies=cookies, headers={'User-Agent': 'bts-leaderboard-watcher/1.0'})
print('status:', r.status_code)
print('first row sample:', json.dumps(r.json(), indent=2)[:500])
"
```

Expected: status 200; JSON contains usernames + streak counts matching what's visible in the UI.

- [ ] **Step 5: Commit**

```bash
git add tests/leaderboard/fixtures/leaderboard_active_streak.json src/bts/leaderboard/endpoints.py
git commit -m "feat(leaderboard): discover + document leaderboard endpoint"
```

### Task 1.4: Discover the user picks endpoint

**Files:**
- Create: `tests/leaderboard/fixtures/user_picks_tombrady12.json`
- Modify: `src/bts/leaderboard/endpoints.py`

- [ ] **Step 1: Capture user picks fetch**

In DevTools (or superpowers-chrome), click any leaderboard row to drill into a user. Identify the XHR that returns the user's picks log + season stats. Capture URL, headers, response body.

- [ ] **Step 2: Save fixture**

Save response as `tests/leaderboard/fixtures/user_picks_tombrady12.json`.

If the response also contains season stats (best streak / active streak / accuracy), great — note it. If those come from a separate endpoint, capture that fixture too as `tests/leaderboard/fixtures/user_stats_tombrady12.json`.

- [ ] **Step 3: Update endpoints.py**

```python
# src/bts/leaderboard/endpoints.py — append
USER_PICKS_URL_TEMPLATE: str = "https://api.example.mlb.com/bts/user/{username}/picks"  # ACTUAL URL
USER_STATS_URL_TEMPLATE: str = ""  # set if separate from picks endpoint, else leave empty
```

- [ ] **Step 4: Hand-verify**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "
import json, httpx, subprocess
from bts.leaderboard.endpoints import USER_PICKS_URL_TEMPLATE
cookies_raw = subprocess.check_output(['pass', 'show', 'mlb-bts-session-cookies']).decode()
cookies = {c['name']: c['value'] for c in json.loads(cookies_raw)}
url = USER_PICKS_URL_TEMPLATE.format(username='tombrady12')
r = httpx.get(url, cookies=cookies, headers={'User-Agent': 'bts-leaderboard-watcher/1.0'})
print('status:', r.status_code)
print('keys:', list(r.json().keys()) if r.headers.get('content-type','').startswith('application/json') else 'non-json')
"
```

Expected: 200; JSON keys include picks-log-like data.

- [ ] **Step 5: Commit**

```bash
git add tests/leaderboard/fixtures/user_picks_tombrady12.json src/bts/leaderboard/endpoints.py
git commit -m "feat(leaderboard): discover + document user picks endpoint"
```

### Task 1.5: Document Phase 1 findings in spec

**Files:**
- Modify: `docs/superpowers/specs/2026-05-01-bts-leaderboard-watcher-design.md` Section 13

- [ ] **Step 1: Update Section 13 with answers**

Open the spec. In Section 13 (Open Questions), replace each open question with its discovered answer. Example pattern:

```markdown
- ~~**Auth scheme**: cookie-only, or Bearer token, or session JWT?~~ **CLOSED 2026-05-01**: cookies-only. The session cookie `bamsessionId` (or whatever the actual name is) is sufficient; no Bearer token observed.
```

Repeat for: endpoint URLs, cookie lifetime (best estimate based on observed expiry header), pagination, picks log depth.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-01-bts-leaderboard-watcher-design.md
git commit -m "docs(leaderboard-spec): close Phase 1 open questions with discovered answers"
```

---

## Phase 2 — Storage (Mac local, TDD)

### Task 2.1: LeaderboardRow pydantic model

**Files:**
- Create: `src/bts/leaderboard/models.py`
- Create: `tests/leaderboard/test_models.py`

- [ ] **Step 1: Write failing test**

```python
# tests/leaderboard/test_models.py
"""Tests for pydantic schemas representing leaderboard data."""
from __future__ import annotations

from datetime import datetime, date

import pytest
from pydantic import ValidationError

from bts.leaderboard.models import LeaderboardRow, PickRow, SeasonStats


class TestLeaderboardRow:
    def test_valid_active_streak_row(self):
        row = LeaderboardRow(
            captured_at=datetime(2026, 5, 1, 14, 0, 0),
            tab="active_streak",
            rank=1,
            username="tombrady12",
            streak=35,
            hits_today=None,
        )
        assert row.username == "tombrady12"
        assert row.tab == "active_streak"

    def test_yesterday_tab_has_hits_today(self):
        row = LeaderboardRow(
            captured_at=datetime(2026, 5, 1, 14, 0, 0),
            tab="yesterday",
            rank=1,
            username="someone",
            streak=None,
            hits_today=2,
        )
        assert row.hits_today == 2

    def test_invalid_tab_rejected(self):
        with pytest.raises(ValidationError):
            LeaderboardRow(
                captured_at=datetime(2026, 5, 1),
                tab="not_a_real_tab",
                rank=1,
                username="x",
                streak=10,
                hits_today=None,
            )

    def test_negative_rank_rejected(self):
        with pytest.raises(ValidationError):
            LeaderboardRow(
                captured_at=datetime(2026, 5, 1),
                tab="active_streak",
                rank=0,
                username="x",
                streak=10,
                hits_today=None,
            )
```

- [ ] **Step 2: Run test (expect failure)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_models.py::TestLeaderboardRow -v
```

Expected: ImportError (`models.py` doesn't exist yet).

- [ ] **Step 3: Implement models.py**

```python
# src/bts/leaderboard/models.py
"""Pydantic schemas for the BTS leaderboard watcher data model.

These mirror the parquet column schemas documented in
docs/superpowers/specs/2026-05-01-bts-leaderboard-watcher-design.md
Section 4 (Data model). Validation happens on parsing — bad rows are
rejected before they reach storage.
"""
from __future__ import annotations

from datetime import datetime, date
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


TabName = Literal["active_streak", "all_season", "all_time", "yesterday"]
HomeAway = Literal["home", "away"]


class LeaderboardRow(BaseModel):
    model_config = ConfigDict(strict=True)

    captured_at: datetime
    tab: TabName
    rank: int = Field(gt=0)
    username: str = Field(min_length=1)
    streak: int | None = Field(default=None, ge=0)
    hits_today: int | None = Field(default=None, ge=0)


class PickRow(BaseModel):
    model_config = ConfigDict(strict=True)

    captured_at: datetime
    pick_date: date
    batter_name: str = Field(min_length=1)
    batter_team: str = Field(min_length=1)
    opponent_team: str = Field(min_length=1)
    home_or_away: HomeAway
    at_bats: int = Field(ge=0)
    hits: int = Field(ge=0)
    streak_after: int = Field(ge=0)
    batter_id: int | None = None


class SeasonStats(BaseModel):
    model_config = ConfigDict(strict=True)

    captured_at: datetime
    username: str = Field(min_length=1)
    best_streak: int = Field(ge=0)
    active_streak: int = Field(ge=0)
    pick_accuracy_pct: float = Field(ge=0.0, le=100.0)
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_models.py::TestLeaderboardRow -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/leaderboard/models.py tests/leaderboard/test_models.py
git commit -m "feat(leaderboard): LeaderboardRow pydantic model"
```

### Task 2.2: PickRow + SeasonStats validation tests

**Files:**
- Modify: `tests/leaderboard/test_models.py`

- [ ] **Step 1: Append PickRow + SeasonStats tests**

```python
# tests/leaderboard/test_models.py — append to existing file

class TestPickRow:
    def test_valid_pick(self):
        row = PickRow(
            captured_at=datetime(2026, 5, 1, 14, 0),
            pick_date=date(2026, 4, 30),
            batter_name="Juan Soto",
            batter_team="NYM",
            opponent_team="WSH",
            home_or_away="home",
            at_bats=3,
            hits=2,
            streak_after=35,
            batter_id=665742,
        )
        assert row.hits == 2
        assert row.streak_after == 35

    def test_batter_id_optional(self):
        row = PickRow(
            captured_at=datetime(2026, 5, 1),
            pick_date=date(2026, 4, 30),
            batter_name="Some Player",
            batter_team="ABC",
            opponent_team="XYZ",
            home_or_away="away",
            at_bats=3,
            hits=1,
            streak_after=5,
        )
        assert row.batter_id is None

    def test_invalid_home_away_rejected(self):
        with pytest.raises(ValidationError):
            PickRow(
                captured_at=datetime(2026, 5, 1),
                pick_date=date(2026, 4, 30),
                batter_name="X", batter_team="A", opponent_team="B",
                home_or_away="elsewhere",  # invalid
                at_bats=3, hits=1, streak_after=5,
            )


class TestSeasonStats:
    def test_valid_stats(self):
        s = SeasonStats(
            captured_at=datetime(2026, 5, 1, 14, 0),
            username="tombrady12",
            best_streak=35,
            active_streak=35,
            pick_accuracy_pct=100.0,
        )
        assert s.pick_accuracy_pct == 100.0

    def test_accuracy_out_of_bounds_rejected(self):
        with pytest.raises(ValidationError):
            SeasonStats(
                captured_at=datetime(2026, 5, 1),
                username="x",
                best_streak=10,
                active_streak=5,
                pick_accuracy_pct=101.0,  # > 100
            )
```

- [ ] **Step 2: Run all model tests (expect pass — models already cover these)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_models.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/leaderboard/test_models.py
git commit -m "test(leaderboard): PickRow + SeasonStats validation tests"
```

### Task 2.3: Storage write_leaderboard_snapshot

**Files:**
- Create: `src/bts/leaderboard/storage.py`
- Create: `tests/leaderboard/test_storage.py`

- [ ] **Step 1: Write failing test**

```python
# tests/leaderboard/test_storage.py
"""Tests for parquet I/O in the leaderboard package."""
from __future__ import annotations

from datetime import datetime, date

import pyarrow.parquet as pq
import pytest

from bts.leaderboard.models import LeaderboardRow, PickRow, SeasonStats
from bts.leaderboard.storage import (
    write_leaderboard_snapshot,
    append_user_picks,
    write_season_stats,
    read_user_picks,
)


def _row(rank: int, username: str, streak: int) -> LeaderboardRow:
    return LeaderboardRow(
        captured_at=datetime(2026, 5, 1, 14, 0, 0),
        tab="active_streak",
        rank=rank,
        username=username,
        streak=streak,
        hits_today=None,
    )


class TestWriteLeaderboardSnapshot:
    def test_writes_parquet_with_all_rows(self, tmp_path):
        rows = [_row(1, "alpha", 35), _row(2, "beta", 34)]
        out = tmp_path / "leaderboard_snapshots" / "2026-05-01.parquet"
        write_leaderboard_snapshot(out, rows)
        assert out.exists()
        table = pq.read_table(out)
        assert table.num_rows == 2
        assert "username" in table.column_names

    def test_creates_parent_dir(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "snapshots" / "2026-05-01.parquet"
        write_leaderboard_snapshot(out, [_row(1, "x", 10)])
        assert out.exists()

    def test_empty_rows_writes_empty_parquet(self, tmp_path):
        out = tmp_path / "snapshots" / "empty.parquet"
        write_leaderboard_snapshot(out, [])
        assert out.exists()
        table = pq.read_table(out)
        assert table.num_rows == 0
        # Schema is preserved even when empty so downstream readers don't crash
        assert "username" in table.column_names
```

- [ ] **Step 2: Run test (expect failure)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_storage.py::TestWriteLeaderboardSnapshot -v
```

Expected: ImportError (`storage.py` doesn't exist).

- [ ] **Step 3: Implement storage.py**

```python
# src/bts/leaderboard/storage.py
"""Parquet I/O for leaderboard data.

Writes are atomic-by-write-then-rename pattern. Reads return pyarrow
Tables for downstream pandas/polars conversion as needed.

Schema is enforced via pydantic models (see models.py) — non-conforming
rows raise on construction, before they reach storage.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bts.leaderboard.models import LeaderboardRow, PickRow, SeasonStats


_LEADERBOARD_SCHEMA = pa.schema([
    ("captured_at", pa.timestamp("ms")),
    ("tab", pa.string()),
    ("rank", pa.int32()),
    ("username", pa.string()),
    ("streak", pa.int32()),
    ("hits_today", pa.int32()),
])


def _rows_to_table(rows: list[LeaderboardRow], schema: pa.Schema) -> pa.Table:
    cols: dict[str, list] = {f.name: [] for f in schema}
    for r in rows:
        d = r.model_dump()
        for f in schema:
            cols[f.name].append(d.get(f.name))
    return pa.table(cols, schema=schema)


def write_leaderboard_snapshot(path: Path, rows: list[LeaderboardRow]) -> None:
    """Write leaderboard snapshot rows to parquet (atomic-rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _rows_to_table(rows, _LEADERBOARD_SCHEMA)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.rename(path)
```

(Stub the other functions for now — they get tested + implemented in 2.4 and 2.5.)

```python
def append_user_picks(path: Path, rows: list[PickRow]) -> None:
    raise NotImplementedError("Task 2.4")


def write_season_stats(path: Path, rows: list[SeasonStats]) -> None:
    raise NotImplementedError("Task 2.5")


def read_user_picks(path: Path):
    raise NotImplementedError("Task 2.4")
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_storage.py::TestWriteLeaderboardSnapshot -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/leaderboard/storage.py tests/leaderboard/test_storage.py
git commit -m "feat(leaderboard): write_leaderboard_snapshot with atomic rename"
```

### Task 2.4: Storage append_user_picks (with dedup-aware read)

**Files:**
- Modify: `src/bts/leaderboard/storage.py`
- Modify: `tests/leaderboard/test_storage.py`

- [ ] **Step 1: Append failing tests**

```python
# tests/leaderboard/test_storage.py — append

def _pick(username_unused: str, pick_date_str: str, batter: str, captured_iso: str) -> PickRow:
    return PickRow(
        captured_at=datetime.fromisoformat(captured_iso),
        pick_date=date.fromisoformat(pick_date_str),
        batter_name=batter,
        batter_team="NYM",
        opponent_team="WSH",
        home_or_away="home",
        at_bats=3,
        hits=2,
        streak_after=10,
    )


class TestAppendUserPicks:
    def test_first_write_creates_file(self, tmp_path):
        path = tmp_path / "user_picks" / "tombrady12.parquet"
        append_user_picks(path, [_pick("tombrady12", "2026-04-30", "Soto", "2026-05-01T14:00:00")])
        assert path.exists()
        assert pq.read_table(path).num_rows == 1

    def test_append_preserves_existing(self, tmp_path):
        path = tmp_path / "user_picks" / "tombrady12.parquet"
        append_user_picks(path, [_pick("x", "2026-04-30", "Soto", "2026-05-01T14:00:00")])
        append_user_picks(path, [_pick("x", "2026-05-01", "Vlad", "2026-05-02T14:00:00")])
        table = pq.read_table(path)
        assert table.num_rows == 2
        names = table.column("batter_name").to_pylist()
        assert "Soto" in names and "Vlad" in names

    def test_append_keeps_observations_with_different_captured_at(self, tmp_path):
        """Same pick observed twice: append-only keeps both rows for audit trail."""
        path = tmp_path / "user_picks" / "tombrady12.parquet"
        append_user_picks(path, [_pick("x", "2026-04-30", "Soto", "2026-05-01T14:00:00")])
        append_user_picks(path, [_pick("x", "2026-04-30", "Soto", "2026-05-01T20:00:00")])
        assert pq.read_table(path).num_rows == 2


class TestReadUserPicks:
    def test_returns_latest_per_pick_date(self, tmp_path):
        path = tmp_path / "user_picks" / "x.parquet"
        # Write same pick_date twice with different captured_at — newest wins on read
        append_user_picks(path, [_pick("x", "2026-04-30", "Old", "2026-05-01T08:00:00")])
        append_user_picks(path, [_pick("x", "2026-04-30", "New", "2026-05-01T20:00:00")])
        latest = read_user_picks(path, dedupe="latest_per_pick_date")
        assert latest.num_rows == 1
        assert latest.column("batter_name").to_pylist() == ["New"]

    def test_no_dedupe_returns_all_observations(self, tmp_path):
        path = tmp_path / "user_picks" / "x.parquet"
        append_user_picks(path, [_pick("x", "2026-04-30", "Old", "2026-05-01T08:00:00")])
        append_user_picks(path, [_pick("x", "2026-04-30", "New", "2026-05-01T20:00:00")])
        all_rows = read_user_picks(path, dedupe=None)
        assert all_rows.num_rows == 2
```

- [ ] **Step 2: Run test (expect fail — NotImplementedError)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_storage.py::TestAppendUserPicks tests/leaderboard/test_storage.py::TestReadUserPicks -v
```

Expected: 5 fail with NotImplementedError or AttributeError.

- [ ] **Step 3: Implement append_user_picks + read_user_picks**

```python
# src/bts/leaderboard/storage.py — replace the stubs

_USER_PICKS_SCHEMA = pa.schema([
    ("captured_at", pa.timestamp("ms")),
    ("pick_date", pa.date32()),
    ("batter_name", pa.string()),
    ("batter_team", pa.string()),
    ("opponent_team", pa.string()),
    ("home_or_away", pa.string()),
    ("at_bats", pa.int32()),
    ("hits", pa.int32()),
    ("streak_after", pa.int32()),
    ("batter_id", pa.int64()),
])


def append_user_picks(path: Path, rows: list[PickRow]) -> None:
    """Append-only writer for per-user picks log.

    Reads existing parquet (if any), concatenates new rows, writes back.
    Every observation is preserved — dedup happens on read, not write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new_table = _rows_to_table(rows, _USER_PICKS_SCHEMA)
    if path.exists():
        existing = pq.read_table(path)
        combined = pa.concat_tables([existing, new_table])
    else:
        combined = new_table
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(combined, tmp)
    tmp.rename(path)


def read_user_picks(path: Path, dedupe: str | None = None) -> pa.Table:
    """Read user picks with optional dedup.

    dedupe=None: return raw appended observations
    dedupe='latest_per_pick_date': return only newest captured_at for each pick_date
    """
    if not path.exists():
        return pa.table({}, schema=_USER_PICKS_SCHEMA)
    table = pq.read_table(path)
    if dedupe is None:
        return table
    if dedupe == "latest_per_pick_date":
        df = table.to_pandas().sort_values("captured_at").drop_duplicates(
            subset=["pick_date"], keep="last"
        )
        return pa.Table.from_pandas(df, schema=_USER_PICKS_SCHEMA)
    raise ValueError(f"unknown dedupe mode: {dedupe!r}")
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_storage.py::TestAppendUserPicks tests/leaderboard/test_storage.py::TestReadUserPicks -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/leaderboard/storage.py tests/leaderboard/test_storage.py
git commit -m "feat(leaderboard): append_user_picks + read_user_picks with dedup-on-read"
```

### Task 2.5: Storage write_season_stats

**Files:**
- Modify: `src/bts/leaderboard/storage.py`
- Modify: `tests/leaderboard/test_storage.py`

- [ ] **Step 1: Append failing tests**

```python
# tests/leaderboard/test_storage.py — append

class TestWriteSeasonStats:
    def test_writes_stats_parquet(self, tmp_path):
        out = tmp_path / "season_stats" / "2026-05-01.parquet"
        rows = [SeasonStats(
            captured_at=datetime(2026, 5, 1, 14, 0),
            username="tombrady12",
            best_streak=35,
            active_streak=35,
            pick_accuracy_pct=100.0,
        )]
        write_season_stats(out, rows)
        table = pq.read_table(out)
        assert table.num_rows == 1
        assert table.column("pick_accuracy_pct").to_pylist() == [100.0]

    def test_empty_rows_writes_header_only(self, tmp_path):
        out = tmp_path / "season_stats" / "empty.parquet"
        write_season_stats(out, [])
        table = pq.read_table(out)
        assert table.num_rows == 0
        assert "username" in table.column_names
```

- [ ] **Step 2: Run test (expect fail)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_storage.py::TestWriteSeasonStats -v
```

Expected: NotImplementedError.

- [ ] **Step 3: Implement write_season_stats**

```python
# src/bts/leaderboard/storage.py — replace the season-stats stub

_SEASON_STATS_SCHEMA = pa.schema([
    ("captured_at", pa.timestamp("ms")),
    ("username", pa.string()),
    ("best_streak", pa.int32()),
    ("active_streak", pa.int32()),
    ("pick_accuracy_pct", pa.float64()),
])


def write_season_stats(path: Path, rows: list[SeasonStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _rows_to_table(rows, _SEASON_STATS_SCHEMA)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.rename(path)
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_storage.py -v
```

Expected: all storage tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bts/leaderboard/storage.py tests/leaderboard/test_storage.py
git commit -m "feat(leaderboard): write_season_stats"
```

---

## Phase 3 — Scraper (Mac local, TDD)

### Task 3.1: auth.load_session_cookies

**Files:**
- Create: `src/bts/leaderboard/auth.py`
- Create: `tests/leaderboard/test_auth.py`

- [ ] **Step 1: Write failing test**

```python
# tests/leaderboard/test_auth.py
"""Tests for auth flow: cookie loading + session probe."""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from bts.leaderboard.auth import load_session_cookies, is_session_valid, AuthError


class TestLoadSessionCookies:
    def test_loads_cookies_from_pass(self):
        fake_cookies = json.dumps([
            {"name": "session", "value": "abc123", "domain": ".mlb.com"},
            {"name": "uid", "value": "tombrady12", "domain": ".mlb.com"},
        ])
        with patch("bts.leaderboard.auth.subprocess.check_output", return_value=fake_cookies.encode()):
            cookies = load_session_cookies()
        assert cookies == {"session": "abc123", "uid": "tombrady12"}

    def test_raises_auth_error_when_pass_fails(self):
        import subprocess
        with patch("bts.leaderboard.auth.subprocess.check_output",
                   side_effect=subprocess.CalledProcessError(1, "pass")):
            with pytest.raises(AuthError, match="cookie store"):
                load_session_cookies()
```

- [ ] **Step 2: Run test (expect fail — module not found)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_auth.py::TestLoadSessionCookies -v
```

Expected: ImportError.

- [ ] **Step 3: Implement auth.py**

```python
# src/bts/leaderboard/auth.py
"""Session cookie management for the BTS leaderboard scraper.

Cookies live in `pass:mlb-bts-session-cookies` as a JSON dump from
Playwright's context.cookies(). At scrape time we transform that into
a {name: value} dict suitable for httpx.

A cookie refresh is a manual workflow: see scripts/capture_bts_cookies.py.
"""
from __future__ import annotations

import json
import logging
import subprocess

import httpx

log = logging.getLogger(__name__)

PASS_KEY = "mlb-bts-session-cookies"


class AuthError(Exception):
    """Raised when session cookies are missing, expired, or rejected."""


def load_session_cookies() -> dict[str, str]:
    """Return name->value cookie dict from `pass`. Raises AuthError on failure."""
    try:
        raw = subprocess.check_output(["pass", "show", PASS_KEY], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise AuthError(f"could not read cookie store {PASS_KEY!r}: {e}") from e
    try:
        cookies_list = json.loads(raw.decode())
    except json.JSONDecodeError as e:
        raise AuthError(f"cookie store {PASS_KEY!r} is not valid JSON: {e}") from e
    return {c["name"]: c["value"] for c in cookies_list}


def is_session_valid(probe_url: str, cookies: dict[str, str], timeout: float = 10.0) -> bool:
    """GET `probe_url` with cookies; return True iff status 200."""
    try:
        r = httpx.get(
            probe_url,
            cookies=cookies,
            timeout=timeout,
            headers={"User-Agent": "bts-leaderboard-watcher/1.0 (research; contact: stone.ericm@gmail.com)"},
        )
    except httpx.HTTPError as e:
        log.warning(f"session probe network error: {e}")
        return False
    if r.status_code == 200:
        return True
    log.warning(f"session probe returned {r.status_code}")
    return False
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_auth.py::TestLoadSessionCookies -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/leaderboard/auth.py tests/leaderboard/test_auth.py
git commit -m "feat(leaderboard): auth.load_session_cookies + AuthError"
```

### Task 3.2: auth.is_session_valid (probe)

**Files:**
- Modify: `tests/leaderboard/test_auth.py`

- [ ] **Step 1: Append test**

```python
# tests/leaderboard/test_auth.py — append

class TestIsSessionValid:
    def test_returns_true_on_200(self):
        with patch("bts.leaderboard.auth.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert is_session_valid("http://x", {"a": "b"}) is True

    def test_returns_false_on_401(self):
        with patch("bts.leaderboard.auth.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=401)
            assert is_session_valid("http://x", {"a": "b"}) is False

    def test_returns_false_on_403(self):
        with patch("bts.leaderboard.auth.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=403)
            assert is_session_valid("http://x", {"a": "b"}) is False

    def test_returns_false_on_network_error(self):
        import httpx as _httpx
        with patch("bts.leaderboard.auth.httpx.get",
                   side_effect=_httpx.ConnectError("nope")):
            assert is_session_valid("http://x", {"a": "b"}) is False
```

- [ ] **Step 2: Run test (expect pass — already implemented in 3.1)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_auth.py -v
```

Expected: all auth tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/leaderboard/test_auth.py
git commit -m "test(leaderboard): is_session_valid probe coverage"
```

### Task 3.3: rate_limit decorator

**Files:**
- Create: `src/bts/leaderboard/ratelimit.py`
- Create: `tests/leaderboard/test_ratelimit.py`

- [ ] **Step 1: Write failing test**

```python
# tests/leaderboard/test_ratelimit.py
"""Tests for the rate-limit decorator."""
from __future__ import annotations

import time

from bts.leaderboard.ratelimit import rate_limited


class TestRateLimited:
    def test_first_call_is_immediate(self):
        @rate_limited(min_interval_s=1.0)
        def f():
            return time.monotonic()

        t0 = time.monotonic()
        f()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1, f"first call should not block; took {elapsed}s"

    def test_second_call_within_interval_is_delayed(self):
        @rate_limited(min_interval_s=0.2)
        def f():
            return None

        f()
        t1 = time.monotonic()
        f()
        elapsed = time.monotonic() - t1
        assert 0.15 <= elapsed <= 0.4, f"expected ~0.2s gap, got {elapsed}s"

    def test_third_call_after_interval_is_immediate(self):
        @rate_limited(min_interval_s=0.1)
        def f():
            return None

        f()
        time.sleep(0.2)
        t1 = time.monotonic()
        f()
        elapsed = time.monotonic() - t1
        assert elapsed < 0.05, f"call after interval should be immediate; took {elapsed}s"
```

- [ ] **Step 2: Run test (expect fail)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_ratelimit.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement ratelimit.py**

```python
# src/bts/leaderboard/ratelimit.py
"""Per-function rate-limit decorator.

Conservative posture: enforce a minimum gap between calls to the SAME
decorated callable. Not a global limiter — each decorated function has
its own cadence. Adequate for our use case (one function per endpoint).
"""
from __future__ import annotations

import functools
import time
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def rate_limited(min_interval_s: float) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator: enforce >=min_interval_s between successive calls."""
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        last_called = 0.0

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal last_called
            now = time.monotonic()
            wait = (last_called + min_interval_s) - now
            if wait > 0:
                time.sleep(wait)
            last_called = time.monotonic()
            return fn(*args, **kwargs)

        return wrapper
    return decorator
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_ratelimit.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/leaderboard/ratelimit.py tests/leaderboard/test_ratelimit.py
git commit -m "feat(leaderboard): rate_limited decorator (per-function min gap)"
```

### Task 3.4: scraper.scrape_leaderboard (parser + integration)

**Files:**
- Create: `src/bts/leaderboard/scraper.py`
- Create: `tests/leaderboard/test_scraper.py`

- [ ] **Step 1: Write failing test against the captured fixture**

```python
# tests/leaderboard/test_scraper.py
"""Scraper tests using captured HTTP fixtures."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from bts.leaderboard.scraper import (
    parse_leaderboard_response,
    scrape_leaderboard,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


class TestParseLeaderboardResponse:
    def test_parses_active_streak_fixture(self):
        body = _load_fixture("leaderboard_active_streak.json")
        rows = parse_leaderboard_response(
            body, tab="active_streak", captured_at=datetime(2026, 5, 1, 14, 0),
        )
        # The exact response shape depends on Phase 1 discovery — assertions below
        # assume rows align with the visible UI (rank 1 = highest streak)
        assert len(rows) > 0
        assert rows[0].rank == 1
        assert rows[0].streak >= rows[-1].streak  # ranking descending
        assert all(r.tab == "active_streak" for r in rows)


class TestScrapeLeaderboard:
    def test_calls_endpoint_with_cookies_and_user_agent(self):
        with patch("bts.leaderboard.scraper.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, json=lambda: _load_fixture("leaderboard_active_streak.json"),
            )
            scrape_leaderboard(tab="active_streak", cookies={"a": "b"})
            args, kwargs = mock_get.call_args
            assert kwargs["cookies"] == {"a": "b"}
            assert "bts-leaderboard-watcher" in kwargs["headers"]["User-Agent"]
```

- [ ] **Step 2: Run test (expect fail)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_scraper.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement scraper.scrape_leaderboard + parse_leaderboard_response**

The parser specifics depend on Phase 1's discovered response shape. Below is a template — adapt the field-extraction lines to whatever the actual JSON keys are.

```python
# src/bts/leaderboard/scraper.py
"""Core scraping orchestration for the BTS leaderboard watcher.

Each top-level scrape function (scrape_leaderboard, scrape_user_picks,
scrape_user_stats) takes session cookies and returns typed model rows.
HTTP failures are surfaced as exceptions; the orchestrator above catches
and decides retry vs bail.
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Literal

import httpx

from bts.leaderboard.endpoints import (
    LEADERBOARD_URL_TEMPLATE,
    USER_PICKS_URL_TEMPLATE,
    USER_STATS_URL_TEMPLATE,
)
from bts.leaderboard.models import LeaderboardRow, PickRow, SeasonStats
from bts.leaderboard.ratelimit import rate_limited

log = logging.getLogger(__name__)

USER_AGENT = "bts-leaderboard-watcher/1.0 (research; contact: stone.ericm@gmail.com)"
DEFAULT_MIN_INTERVAL_S = 2.0


TabName = Literal["active_streak", "all_season", "all_time", "yesterday"]


def parse_leaderboard_response(
    body: dict, tab: TabName, captured_at: datetime,
) -> list[LeaderboardRow]:
    """Parse a leaderboard JSON body into typed rows.

    EDIT THIS FUNCTION based on Phase 1's discovered response shape.
    Below is the placeholder structure — replace `body[...]` accesses
    with the real keys from tests/leaderboard/fixtures/leaderboard_active_streak.json.
    """
    raw_rows = body.get("rows", body.get("entries", body))  # adjust per Phase 1
    out: list[LeaderboardRow] = []
    for r in raw_rows:
        out.append(LeaderboardRow(
            captured_at=captured_at,
            tab=tab,
            rank=int(r["rank"]),
            username=str(r["username"]),  # Phase 1: confirm key name
            streak=int(r["streak"]) if "streak" in r else None,
            hits_today=int(r["hitsToday"]) if "hitsToday" in r else None,
        ))
    return out


@rate_limited(min_interval_s=DEFAULT_MIN_INTERVAL_S)
def scrape_leaderboard(tab: TabName, cookies: dict[str, str], page: int = 1) -> list[LeaderboardRow]:
    """Fetch and parse one leaderboard tab. Raises on non-200."""
    url = LEADERBOARD_URL_TEMPLATE.format(tab=tab, page=page)
    r = httpx.get(
        url, cookies=cookies, timeout=30.0,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    r.raise_for_status()
    return parse_leaderboard_response(r.json(), tab=tab, captured_at=datetime.utcnow())
```

- [ ] **Step 4: Run test (expect pass — adjust parser if fixture shape differs)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_scraper.py -v
```

If the parser fails because Phase 1's actual response shape differs from the template, **edit `parse_leaderboard_response` to match the real keys**. Re-run until passing.

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/leaderboard/scraper.py tests/leaderboard/test_scraper.py
git commit -m "feat(leaderboard): scrape_leaderboard + parse_leaderboard_response"
```

### Task 3.5: scraper.scrape_user_picks + scrape_user_stats

**Files:**
- Modify: `src/bts/leaderboard/scraper.py`
- Modify: `tests/leaderboard/test_scraper.py`

- [ ] **Step 1: Append failing tests**

```python
# tests/leaderboard/test_scraper.py — append

from bts.leaderboard.scraper import (
    parse_user_picks_response,
    parse_user_stats_response,
    scrape_user_picks,
    scrape_user_stats,
)


class TestParseUserPicksResponse:
    def test_parses_tombrady12_fixture(self):
        body = _load_fixture("user_picks_tombrady12.json")
        rows = parse_user_picks_response(
            body, captured_at=datetime(2026, 5, 1, 14, 0),
        )
        assert len(rows) >= 5  # at least 5 picks visible per screenshot
        # First row should be most recent: Soto on 2026-04-30 with streak 35
        first = sorted(rows, key=lambda r: r.pick_date, reverse=True)[0]
        assert first.batter_name == "Juan Soto"
        assert first.streak_after == 35


class TestParseUserStatsResponse:
    def test_parses_tombrady12_stats(self):
        body = _load_fixture("user_picks_tombrady12.json")  # may be same endpoint
        stats = parse_user_stats_response(
            body, username="tombrady12", captured_at=datetime(2026, 5, 1, 14, 0),
        )
        assert stats.best_streak == 35
        assert stats.active_streak == 35
        assert stats.pick_accuracy_pct == 100.0
```

- [ ] **Step 2: Run test (expect fail — functions undefined)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_scraper.py::TestParseUserPicksResponse tests/leaderboard/test_scraper.py::TestParseUserStatsResponse -v
```

Expected: ImportError.

- [ ] **Step 3: Implement parsers + scrape functions**

```python
# src/bts/leaderboard/scraper.py — append

def parse_user_picks_response(body: dict, captured_at: datetime) -> list[PickRow]:
    """Parse user picks JSON into PickRow list. EDIT per Phase 1 shape."""
    raw_picks = body.get("picks", body.get("picksLog", []))
    out: list[PickRow] = []
    for p in raw_picks:
        out.append(PickRow(
            captured_at=captured_at,
            pick_date=date.fromisoformat(p["pickDate"]),  # confirm key name
            batter_name=str(p["batterName"]),
            batter_team=str(p["batterTeam"]),
            opponent_team=str(p["opponentTeam"]),
            home_or_away="home" if p["isHome"] else "away",
            at_bats=int(p["atBats"]),
            hits=int(p["hits"]),
            streak_after=int(p["streakAfter"]),
            batter_id=int(p["batterId"]) if "batterId" in p else None,
        ))
    return out


def parse_user_stats_response(body: dict, username: str, captured_at: datetime) -> SeasonStats:
    """Parse season-stats fields. EDIT per Phase 1 shape."""
    s = body.get("seasonStats", body)
    return SeasonStats(
        captured_at=captured_at,
        username=username,
        best_streak=int(s["bestStreak"]),
        active_streak=int(s["activeStreak"]),
        pick_accuracy_pct=float(s["pickAccuracyPct"]),
    )


@rate_limited(min_interval_s=DEFAULT_MIN_INTERVAL_S)
def scrape_user_picks(username: str, cookies: dict[str, str]) -> list[PickRow]:
    url = USER_PICKS_URL_TEMPLATE.format(username=username)
    r = httpx.get(url, cookies=cookies, timeout=30.0,
                  headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    r.raise_for_status()
    return parse_user_picks_response(r.json(), captured_at=datetime.utcnow())


@rate_limited(min_interval_s=DEFAULT_MIN_INTERVAL_S)
def scrape_user_stats(username: str, cookies: dict[str, str]) -> SeasonStats:
    url = (USER_STATS_URL_TEMPLATE or USER_PICKS_URL_TEMPLATE).format(username=username)
    r = httpx.get(url, cookies=cookies, timeout=30.0,
                  headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    r.raise_for_status()
    return parse_user_stats_response(r.json(), username=username, captured_at=datetime.utcnow())
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_scraper.py -v
```

Expected: all scraper tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bts/leaderboard/scraper.py tests/leaderboard/test_scraper.py
git commit -m "feat(leaderboard): scrape_user_picks + scrape_user_stats"
```

### Task 3.6: scraper.run() — full daily orchestration

**Files:**
- Modify: `src/bts/leaderboard/scraper.py`
- Modify: `tests/leaderboard/test_scraper.py`

- [ ] **Step 1: Append failing test**

```python
# tests/leaderboard/test_scraper.py — append

from bts.leaderboard.scraper import run as scraper_run


class TestRun:
    def test_writes_all_4_tabs_and_picks_for_top_n(self, tmp_path):
        leaderboard_fixture = _load_fixture("leaderboard_active_streak.json")
        picks_fixture = _load_fixture("user_picks_tombrady12.json")

        with patch("bts.leaderboard.scraper.httpx.get") as mock_get:
            def _resp(*a, **kw):
                url = a[0] if a else kw.get("url", "")
                if "leaderboard" in url:
                    return MagicMock(status_code=200, json=lambda: leaderboard_fixture, raise_for_status=lambda: None)
                return MagicMock(status_code=200, json=lambda: picks_fixture, raise_for_status=lambda: None)
            mock_get.side_effect = _resp

            scraper_run(
                cookies={"a": "b"},
                output_dir=tmp_path / "leaderboard",
                top_n=2,
                tabs=("active_streak",),  # one tab keeps test fast
            )

        # Snapshot file written
        snaps = list((tmp_path / "leaderboard" / "leaderboard_snapshots").glob("*.parquet"))
        assert len(snaps) == 1
        # Per-user picks file written for top 2 users
        user_files = list((tmp_path / "leaderboard" / "user_picks").glob("*.parquet"))
        assert len(user_files) == 2
```

- [ ] **Step 2: Run test (expect fail)**

Expected: AttributeError on `scraper_run`.

- [ ] **Step 3: Implement run()**

```python
# src/bts/leaderboard/scraper.py — append

from pathlib import Path

from bts.leaderboard.storage import (
    write_leaderboard_snapshot,
    append_user_picks,
    write_season_stats,
)


def run(
    cookies: dict[str, str],
    output_dir: Path,
    top_n: int = 100,
    tabs: tuple[TabName, ...] = ("active_streak", "all_season", "all_time", "yesterday"),
) -> None:
    """Full daily scrape: 4 leaderboards + per-user picks for top_n users.

    Failures during per-user iteration are logged but do not abort the run —
    we preserve the partial dataset for the slot we got and let the next
    scheduled run resume.
    """
    captured_date = date.today().isoformat()
    snapshot_path = output_dir / "leaderboard_snapshots" / f"{captured_date}.parquet"
    stats_path = output_dir / "season_stats" / f"{captured_date}.parquet"

    all_rows: list[LeaderboardRow] = []
    tracked_usernames: set[str] = set()

    for tab in tabs:
        try:
            rows = scrape_leaderboard(tab=tab, cookies=cookies)
        except httpx.HTTPError as e:
            log.exception(f"failed to scrape leaderboard tab {tab}: {e}")
            continue
        all_rows.extend(rows[:top_n])
        for r in rows[:top_n]:
            tracked_usernames.add(r.username)

    write_leaderboard_snapshot(snapshot_path, all_rows)
    log.info(f"wrote {len(all_rows)} leaderboard rows to {snapshot_path}")

    season_rows: list[SeasonStats] = []
    for username in sorted(tracked_usernames):
        try:
            picks = scrape_user_picks(username, cookies=cookies)
            stats = scrape_user_stats(username, cookies=cookies)
        except httpx.HTTPError as e:
            log.warning(f"skipping user {username}: {e}")
            continue
        user_path = output_dir / "user_picks" / f"{username}.parquet"
        append_user_picks(user_path, picks)
        season_rows.append(stats)

    write_season_stats(stats_path, season_rows)
    log.info(f"wrote {len(season_rows)} season-stats rows to {stats_path}")
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_scraper.py::TestRun -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/leaderboard/scraper.py tests/leaderboard/test_scraper.py
git commit -m "feat(leaderboard): scraper.run() orchestrates 4 tabs + top-N picks"
```

### Task 3.7: CLI wiring

**Files:**
- Create: `src/bts/leaderboard/cli.py`
- Modify: `src/bts/cli.py`
- Create: `tests/leaderboard/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/leaderboard/test_cli.py
"""Smoke tests for the leaderboard CLI."""
from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from bts.leaderboard.cli import leaderboard


class TestLeaderboardCLI:
    def test_scrape_invokes_run(self, tmp_path):
        runner = CliRunner()
        with patch("bts.leaderboard.cli.scraper_run") as mock_run, \
             patch("bts.leaderboard.cli.load_session_cookies", return_value={"a": "b"}):
            result = runner.invoke(leaderboard, [
                "scrape", "--output-dir", str(tmp_path), "--top-n", "10",
            ])
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()

    def test_status_when_no_data(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(leaderboard, ["status", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "no successful scrape yet" in result.output.lower()

    def test_scrape_handles_auth_error(self, tmp_path):
        from bts.leaderboard.auth import AuthError
        runner = CliRunner()
        with patch("bts.leaderboard.cli.load_session_cookies",
                   side_effect=AuthError("expired")):
            result = runner.invoke(leaderboard, ["scrape", "--output-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "auth" in result.output.lower()
```

- [ ] **Step 2: Run test (expect fail)**

Expected: ImportError.

- [ ] **Step 3: Implement cli.py**

```python
# src/bts/leaderboard/cli.py
"""CLI for the BTS leaderboard watcher.

  bts leaderboard scrape    — run the daily scrape
  bts leaderboard status    — last successful scrape, lag, errors
  bts leaderboard backfill  — re-walk historical visible picks for a user
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import click

from bts.leaderboard.auth import AuthError, load_session_cookies
from bts.leaderboard.scraper import run as scraper_run

DEFAULT_OUTPUT_DIR = Path("data/leaderboard")


@click.group()
def leaderboard():
    """BTS leaderboard watcher commands."""
    pass


@leaderboard.command()
@click.option("--output-dir", type=click.Path(), default=str(DEFAULT_OUTPUT_DIR))
@click.option("--top-n", type=int, default=100)
def scrape(output_dir: str, top_n: int):
    """Run a full daily scrape: 4 leaderboards + per-user picks for top-N."""
    try:
        cookies = load_session_cookies()
    except AuthError as e:
        click.echo(f"auth error: {e}", err=True)
        sys.exit(2)
    scraper_run(cookies=cookies, output_dir=Path(output_dir), top_n=top_n)
    click.echo(f"scrape complete: {datetime.utcnow().isoformat()}Z")


@leaderboard.command()
@click.option("--output-dir", type=click.Path(), default=str(DEFAULT_OUTPUT_DIR))
def status(output_dir: str):
    """Report last successful scrape time + lag."""
    od = Path(output_dir)
    snaps = sorted((od / "leaderboard_snapshots").glob("*.parquet")) if od.exists() else []
    if not snaps:
        click.echo("no successful scrape yet")
        return
    latest = snaps[-1]
    click.echo(f"last scrape: {latest.name}")
    click.echo(f"size: {latest.stat().st_size} bytes")


@leaderboard.command()
@click.argument("username")
@click.option("--output-dir", type=click.Path(), default=str(DEFAULT_OUTPUT_DIR))
def backfill(username: str, output_dir: str):
    """Re-fetch a single user's full visible picks log."""
    from bts.leaderboard.scraper import scrape_user_picks, scrape_user_stats
    from bts.leaderboard.storage import append_user_picks
    try:
        cookies = load_session_cookies()
    except AuthError as e:
        click.echo(f"auth error: {e}", err=True)
        sys.exit(2)
    picks = scrape_user_picks(username, cookies=cookies)
    user_path = Path(output_dir) / "user_picks" / f"{username}.parquet"
    append_user_picks(user_path, picks)
    click.echo(f"backfilled {len(picks)} picks for {username}")
```

- [ ] **Step 4: Wire into top-level CLI**

```python
# src/bts/cli.py — add near the other cli.add_command(...) calls (around line 14-15)
from bts.leaderboard.cli import leaderboard
cli.add_command(leaderboard)
```

- [ ] **Step 5: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_cli.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Smoke test CLI**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts leaderboard --help
UV_CACHE_DIR=/tmp/uv-cache uv run bts leaderboard status --output-dir /tmp/empty
```

Expected: help text appears; status reports "no successful scrape yet".

- [ ] **Step 7: Commit**

```bash
git add src/bts/leaderboard/cli.py src/bts/cli.py tests/leaderboard/test_cli.py
git commit -m "feat(leaderboard): CLI scrape/status/backfill commands"
```

### Task 3.8: End-to-end live smoke test (Mac)

**Files:** none (operational verification)

- [ ] **Step 1: Run full live scrape on Mac**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts leaderboard scrape --output-dir /tmp/leaderboard-smoke --top-n 5
```

Expected: completes in < 60 s. No tracebacks.

- [ ] **Step 2: Verify outputs**

```bash
ls -lah /tmp/leaderboard-smoke/leaderboard_snapshots/ /tmp/leaderboard-smoke/user_picks/ /tmp/leaderboard-smoke/season_stats/
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "
import pyarrow.parquet as pq
from pathlib import Path
for snap in sorted(Path('/tmp/leaderboard-smoke/leaderboard_snapshots').glob('*.parquet')):
    t = pq.read_table(snap)
    print(snap.name, t.num_rows, 'rows')
    print(t.to_pandas().head().to_string())
"
```

Expected: a leaderboard_snapshots parquet exists with 5+ rows containing real usernames + streak counts; user_picks/ contains 5 per-user files.

- [ ] **Step 3: If outputs match the live UI, mark Phase 3 complete and continue to Phase 4. If outputs differ from UI, fix parsers and re-run.**

No commit needed if all is well. If parser changes were needed:

```bash
git add src/bts/leaderboard/scraper.py
git commit -m "fix(leaderboard): parser corrections from live smoke test"
```

---

## Phase 4 — Production deploy (bts-hetzner)

### Task 4.1: leaderboard_freshness health check

**Files:**
- Create: `src/bts/health/leaderboard_freshness.py`
- Create: `tests/health/test_leaderboard_freshness.py`

- [ ] **Step 1: Write failing test**

```python
# tests/health/test_leaderboard_freshness.py
"""Tests for the leaderboard_freshness health check."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bts.health.leaderboard_freshness import check, SOURCE


def _write_snapshot(dir_path: Path, hours_ago: float):
    dir_path.mkdir(parents=True, exist_ok=True)
    p = dir_path / "2026-05-01.parquet"
    p.write_bytes(b"PAR1placeholder")
    # Set mtime to hours_ago hours in the past
    import os, time
    mtime = time.time() - (hours_ago * 3600)
    os.utime(p, (mtime, mtime))


class TestLeaderboardFreshness:
    def test_no_alert_when_recent(self, tmp_path):
        snaps = tmp_path / "leaderboard_snapshots"
        _write_snapshot(snaps, hours_ago=2)
        assert check(tmp_path) == []

    def test_warn_when_12h_to_36h(self, tmp_path):
        snaps = tmp_path / "leaderboard_snapshots"
        _write_snapshot(snaps, hours_ago=20)
        alerts = check(tmp_path)
        assert len(alerts) == 1
        assert alerts[0].level == "WARN"
        assert alerts[0].source == SOURCE

    def test_critical_when_more_than_36h(self, tmp_path):
        snaps = tmp_path / "leaderboard_snapshots"
        _write_snapshot(snaps, hours_ago=40)
        alerts = check(tmp_path)
        assert alerts[0].level == "CRITICAL"

    def test_warn_when_no_snapshots_at_all(self, tmp_path):
        # Empty leaderboard dir = no successful scrapes ever
        (tmp_path / "leaderboard_snapshots").mkdir()
        alerts = check(tmp_path)
        assert alerts[0].level == "WARN"

    def test_no_alert_when_dir_missing_entirely(self, tmp_path):
        # If leaderboard_snapshots doesn't exist at all, watcher isn't deployed yet
        # — silent (don't alarm before first deploy)
        assert check(tmp_path) == []
```

- [ ] **Step 2: Run test (expect fail)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/health/test_leaderboard_freshness.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement leaderboard_freshness.py**

```python
# src/bts/health/leaderboard_freshness.py
"""Tier 2 health check: detect a stale or absent leaderboard scrape.

Watches `data/leaderboard/leaderboard_snapshots/` mtimes. Fires WARN
when the last successful scrape is between 12 h and 36 h old, CRITICAL
beyond 36 h.

The 36 h threshold is intentional: with twice-daily scrapes (10:00 ET
and 01:00 ET), a healthy gap is at most ~9 h. A 36 h gap means we
missed both the morning and the late-night slot — almost certainly an
auth-cookie expiry or persistent network issue.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "leaderboard_freshness"

DEFAULT_THRESHOLDS = {
    "warn_hours": 12.0,
    "critical_hours": 36.0,
}


def check(leaderboard_dir: Path, thresholds: dict | None = None) -> list[Alert]:
    """Returns INFO/WARN/CRITICAL when leaderboard scrape is stale."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    snaps_dir = leaderboard_dir / "leaderboard_snapshots"
    # Watcher not yet deployed → silent (don't alarm pre-launch)
    if not snaps_dir.exists():
        return []
    snaps = sorted(snaps_dir.glob("*.parquet"))
    if not snaps:
        return [Alert(
            level="WARN", source=SOURCE,
            message="leaderboard_snapshots directory empty — no successful scrapes recorded",
        )]
    latest = max(snaps, key=lambda p: p.stat().st_mtime)
    age_h = (datetime.now().timestamp() - latest.stat().st_mtime) / 3600
    if age_h >= t["critical_hours"]:
        return [Alert(
            level="CRITICAL", source=SOURCE,
            message=(f"leaderboard scrape stale by {age_h:.1f}h "
                     f"(latest: {latest.name}). Auth cookies likely expired — "
                     f"refresh via scripts/capture_bts_cookies.py on Mac."),
        )]
    if age_h >= t["warn_hours"]:
        return [Alert(
            level="WARN", source=SOURCE,
            message=f"leaderboard scrape lagging: latest {age_h:.1f}h ago ({latest.name})",
        )]
    return []
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/health/test_leaderboard_freshness.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/health/leaderboard_freshness.py tests/health/test_leaderboard_freshness.py
git commit -m "feat(health): leaderboard_freshness tier-2 alert"
```

### Task 4.2: Wire health check into runner

**Files:**
- Modify: `src/bts/health/runner.py`
- Modify: `tests/health/test_runner.py`

- [ ] **Step 1: Update runner.py imports + signature**

In `src/bts/health/runner.py`:

```python
# Add to the imports block:
from bts.health import (
    blend_training,
    calibration,
    disk_fill,
    leaderboard_freshness,  # NEW
    memory_growth,
    pitcher_sparsity,
    pooled_training,
    post_failure,
    predicted_vs_realized,
    projected_lineup,
    realized_calibration,
    restart_spike,
    same_team_corr,
    streak_validation,
)
```

Add `leaderboard_dir: Path | None = None` to `run_all_checks()` signature, then add:

```python
    # Leaderboard scrape freshness — only check if leaderboard_dir is configured
    if leaderboard_dir is not None:
        alerts.extend(_safe_run("leaderboard_freshness", lambda: leaderboard_freshness.check(
            leaderboard_dir, thresholds=overrides.get("leaderboard_freshness"),
        )))
```

(Add this block alongside the other tier-2 checks.)

- [ ] **Step 2: Update existing test_runner.py to pass new arg as None (default)**

The new param is optional with default None — existing callers and tests don't change.

- [ ] **Step 3: Run all health tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/health/ -v
```

Expected: all health tests pass (including the new leaderboard_freshness ones).

- [ ] **Step 4: Wire scheduler to actually pass `leaderboard_dir`**

Find the `run_all_checks(...)` call site in `src/bts/scheduler.py` (one call). Add the new kwarg:

```bash
grep -n "run_all_checks(" src/bts/scheduler.py
```

In that call, add `leaderboard_dir=Path("data/leaderboard")` to the keyword arguments.

- [ ] **Step 5: Re-run scheduler tests if any exist**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -k scheduler -v
```

Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/bts/health/runner.py src/bts/scheduler.py tests/health/
git commit -m "feat(health): wire leaderboard_freshness into runner + scheduler"
```

### Task 4.3: Bluesky DM hook for auth failures

**Files:**
- Modify: `src/bts/leaderboard/cli.py`
- Modify: `tests/leaderboard/test_cli.py`

- [ ] **Step 1: Append failing test**

```python
# tests/leaderboard/test_cli.py — append

class TestAuthErrorTriggersDM:
    def test_auth_error_calls_send_dm(self, tmp_path):
        from bts.leaderboard.auth import AuthError
        runner = CliRunner()
        with patch("bts.leaderboard.cli.load_session_cookies",
                   side_effect=AuthError("expired")), \
             patch("bts.leaderboard.cli.send_dm") as mock_dm:
            result = runner.invoke(leaderboard, [
                "scrape", "--output-dir", str(tmp_path),
                "--dm-recipient", "stoneericm.bsky.social",
            ])
        assert result.exit_code != 0
        mock_dm.assert_called_once()
        args, _ = mock_dm.call_args
        assert "stoneericm.bsky.social" in args[0]
        assert "leaderboard" in args[1].lower()
        assert "cookie" in args[1].lower() or "auth" in args[1].lower()
```

- [ ] **Step 2: Run test (expect fail)**

Expected: `send_dm` not imported in cli.

- [ ] **Step 3: Add DM hook to scrape command**

```python
# src/bts/leaderboard/cli.py — modify scrape command

from bts.dm import send_dm

@leaderboard.command()
@click.option("--output-dir", type=click.Path(), default=str(DEFAULT_OUTPUT_DIR))
@click.option("--top-n", type=int, default=100)
@click.option("--dm-recipient", default=None,
              help="Bluesky handle for auth-failure notifications")
def scrape(output_dir: str, top_n: int, dm_recipient: str | None):
    """Run a full daily scrape."""
    try:
        cookies = load_session_cookies()
    except AuthError as e:
        msg = f"BTS leaderboard scrape: auth/cookie error — refresh via capture_bts_cookies.py on Mac. ({e})"
        click.echo(msg, err=True)
        if dm_recipient:
            try:
                send_dm(dm_recipient, msg)
            except Exception as dm_err:
                click.echo(f"(DM also failed: {dm_err})", err=True)
        sys.exit(2)
    scraper_run(cookies=cookies, output_dir=Path(output_dir), top_n=top_n)
    click.echo(f"scrape complete: {datetime.utcnow().isoformat()}Z")
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_cli.py -v
```

Expected: all CLI tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bts/leaderboard/cli.py tests/leaderboard/test_cli.py
git commit -m "feat(leaderboard): Bluesky DM on auth failure (--dm-recipient flag)"
```

### Task 4.4: systemd unit + timer files

**Files:**
- Create: `deploy/systemd/bts-leaderboard.service`
- Create: `deploy/systemd/bts-leaderboard.timer`

- [ ] **Step 1: Create deploy/systemd directory**

```bash
mkdir -p /Users/stone/projects/bts/deploy/systemd
```

- [ ] **Step 2: Write the .service file**

```ini
# deploy/systemd/bts-leaderboard.service
[Unit]
Description=BTS Leaderboard Scraper (oneshot)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/bts/projects/bts
Environment=UV_CACHE_DIR=/tmp/uv-cache
Environment=PATH=/home/bts/.local/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=/home/bts/projects/bts/.env
ExecStart=/home/bts/.local/bin/uv run bts leaderboard scrape --output-dir /home/bts/projects/bts/data/leaderboard --top-n 100 --dm-recipient stoneericm.bsky.social

[Install]
WantedBy=default.target
```

- [ ] **Step 3: Write the .timer file**

```ini
# deploy/systemd/bts-leaderboard.timer
[Unit]
Description=BTS Leaderboard Scraper twice-daily timer

[Timer]
# 10:00 ET (14:00 UTC) — morning intentions before most pick-locks
OnCalendar=*-*-* 14:00:00 UTC
# 01:00 ET (05:00 UTC) — late-night, post-game-resolution
OnCalendar=*-*-* 05:00:00 UTC
RandomizedDelaySec=300
Persistent=true
Unit=bts-leaderboard.service

[Install]
WantedBy=timers.target
```

- [ ] **Step 4: Commit**

```bash
git add deploy/systemd/bts-leaderboard.service deploy/systemd/bts-leaderboard.timer
git commit -m "feat(leaderboard): systemd unit + twice-daily timer"
```

### Task 4.5: Install scripts on bts-hetzner

**Files:**
- Modify: `scripts/cron-setup-hetzner.sh`

- [ ] **Step 1: Inspect existing script**

```bash
cat /Users/stone/projects/bts/scripts/cron-setup-hetzner.sh | head -40
```

Note its install/show/uninstall pattern.

- [ ] **Step 2: Add a leaderboard-systemd installer**

Append a new function to `scripts/cron-setup-hetzner.sh` that handles installing the systemd unit + timer. Pattern (adapt to existing script style):

```bash
# scripts/cron-setup-hetzner.sh — append

install_leaderboard_systemd() {
    local target="$HOME/.config/systemd/user"
    mkdir -p "$target"
    cp deploy/systemd/bts-leaderboard.service "$target/"
    cp deploy/systemd/bts-leaderboard.timer "$target/"
    systemctl --user daemon-reload
    systemctl --user enable --now bts-leaderboard.timer
    echo "installed bts-leaderboard.{service,timer}; enabled timer"
    systemctl --user list-timers bts-leaderboard.timer
}
```

(Wire it into the script's main argument-dispatch alongside `install` / `show`.)

- [ ] **Step 3: Commit**

```bash
git add scripts/cron-setup-hetzner.sh
git commit -m "chore(deploy): add install_leaderboard_systemd to cron-setup script"
```

### Task 4.6: Sync cookies to bts-hetzner pass store

**Files:** none (operational)

- [ ] **Step 1: Verify pass works on bts-hetzner**

```bash
ssh bts@bts-hetzner 'pass --version' 2>&1 | head -2
```

If pass isn't installed, install it: `ssh bts@bts-hetzner sudo apt-get install -y pass`.

- [ ] **Step 2: Push cookies into the bts-hetzner pass store**

The simplest reliable path: re-run `capture_bts_cookies.py` on Mac with `PASS_STORE_KEY=mlb-bts-session-cookies`, then transfer the encrypted blob to bts-hetzner via the existing pass-sync mechanism (or scp the gpg-encrypted file from `~/.password-store/`).

```bash
# Option: copy the encrypted file directly
scp "$HOME/.password-store/mlb-bts-session-cookies.gpg" bts@bts-hetzner:~/.password-store/
# Verify on remote
ssh bts@bts-hetzner 'pass show mlb-bts-session-cookies | head -3'
```

Expected: shows JSON cookie array start.

- [ ] **Step 3: No commit (operational only)**

### Task 4.7: Push to deploy + verify production scrape

**Files:** none (deploy + verify)

- [ ] **Step 1: Merge feature branch to main**

```bash
cd /Users/stone/projects/bts
git checkout main
git merge --no-ff feature/leaderboard-watcher
git push origin main
```

- [ ] **Step 2: Push to deploy branch (triggers canary workflow)**

```bash
git push origin main:deploy
gh run watch $(gh run list --branch deploy --limit 1 --json databaseId -q '.[0].databaseId') --exit-status
```

Expected: deploy workflow completes successfully (~50s).

- [ ] **Step 3: Install systemd timer on bts-hetzner**

```bash
ssh bts@bts-hetzner 'cd ~/projects/bts && bash scripts/cron-setup-hetzner.sh install_leaderboard_systemd'
```

Expected: shows the timer enabled with NEXT and LEFT columns.

- [ ] **Step 4: Manually trigger a test scrape**

```bash
ssh bts@bts-hetzner 'systemctl --user start bts-leaderboard.service'
sleep 30
ssh bts@bts-hetzner 'systemctl --user status bts-leaderboard.service --no-pager | head -20'
ssh bts@bts-hetzner 'ls -lah ~/projects/bts/data/leaderboard/leaderboard_snapshots/ ~/projects/bts/data/leaderboard/user_picks/ ~/projects/bts/data/leaderboard/season_stats/'
```

Expected: status shows last run succeeded (Active: inactive (dead), code=0). Parquets exist with non-zero sizes.

- [ ] **Step 5: Verify health check sees fresh data**

```bash
ssh bts@bts-hetzner 'cd ~/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run python -c "
from pathlib import Path
from bts.health.leaderboard_freshness import check
print(check(Path(\"data/leaderboard\")))
"'
```

Expected: empty list `[]` (no alert, scrape is fresh).

- [ ] **Step 6: 24h soak — verify both timer slots fire**

After ~24 h, check:

```bash
ssh bts@bts-hetzner 'journalctl --user -u bts-leaderboard.service --since "24h ago" --no-pager | grep -E "Started|complete|error" | head -20'
ssh bts@bts-hetzner 'ls ~/projects/bts/data/leaderboard/leaderboard_snapshots/'
```

Expected: 2+ successful runs in journal; ≥1 snapshot file written.

- [ ] **Step 7: Mark Phase 4 complete**

No commit needed if all green. Plan complete through deploy.

---

## Phase 5 — Backfill + analysis (parallel, ongoing)

Phase 5 begins ONLY after Phase 4's 24h soak passes. It's intentionally smaller per task because the data is now flowing and we can iterate quickly.

### Task 5.1: Backfill loop for all current top-N users

**Files:**
- Create: `scripts/backfill_leaderboard_users.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
# scripts/backfill_leaderboard_users.py
"""One-shot backfill of full visible picks log for every user currently in any top-100.

Run on bts-hetzner once after first scrape lands. Subsequent daily scrapes
maintain the data; this just ensures we capture the full visible streak
history of users who appeared on day 1.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/backfill_leaderboard_users.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import pyarrow.parquet as pq

from bts.leaderboard.auth import load_session_cookies
from bts.leaderboard.scraper import scrape_user_picks
from bts.leaderboard.storage import append_user_picks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LEADERBOARD_DIR = Path("data/leaderboard")


def main():
    snaps = sorted((LEADERBOARD_DIR / "leaderboard_snapshots").glob("*.parquet"))
    if not snaps:
        log.error("no snapshots — run a scrape first")
        return
    table = pq.read_table(snaps[-1])
    usernames = sorted(set(table.column("username").to_pylist()))
    log.info(f"backfilling {len(usernames)} users")

    cookies = load_session_cookies()
    for u in usernames:
        try:
            picks = scrape_user_picks(u, cookies=cookies)
            append_user_picks(LEADERBOARD_DIR / "user_picks" / f"{u}.parquet", picks)
            log.info(f"  {u}: {len(picks)} picks")
        except Exception as e:
            log.warning(f"  {u}: failed — {e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on bts-hetzner**

```bash
ssh bts@bts-hetzner 'cd ~/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/backfill_leaderboard_users.py 2>&1 | tail -10'
```

Expected: ~100 users backfilled in ~3-4 minutes.

- [ ] **Step 3: Commit**

```bash
git add scripts/backfill_leaderboard_users.py
git commit -m "feat(leaderboard): backfill script for full visible picks of current top-N"
```

### Task 5.2: analysis.consensus_pick

**Files:**
- Create: `src/bts/leaderboard/analysis.py`
- Create: `tests/leaderboard/test_analysis.py`

- [ ] **Step 1: Write failing test**

```python
# tests/leaderboard/test_analysis.py
"""Tests for analysis: consensus pick + percentile rank."""
from __future__ import annotations

from datetime import datetime, date
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bts.leaderboard.models import PickRow
from bts.leaderboard.storage import append_user_picks
from bts.leaderboard.analysis import consensus_pick


def _pick(user_unused: str, batter: str, pick_date_iso: str) -> PickRow:
    return PickRow(
        captured_at=datetime(2026, 5, 1, 14, 0),
        pick_date=date.fromisoformat(pick_date_iso),
        batter_name=batter,
        batter_team="X", opponent_team="Y", home_or_away="home",
        at_bats=3, hits=1, streak_after=10,
    )


class TestConsensusPick:
    def test_returns_modal_batter_when_majority(self, tmp_path):
        leaderboard = tmp_path / "leaderboard"
        for u in ["alpha", "beta", "gamma"]:
            append_user_picks(
                leaderboard / "user_picks" / f"{u}.parquet",
                [_pick(u, "Soto", "2026-05-01")],
            )
        append_user_picks(
            leaderboard / "user_picks" / "delta.parquet",
            [_pick("delta", "Vlad", "2026-05-01")],
        )
        result = consensus_pick(leaderboard, pick_date=date(2026, 5, 1))
        assert result["consensus_batter"] == "Soto"
        assert result["consensus_share"] == 0.75

    def test_returns_none_when_no_picks(self, tmp_path):
        result = consensus_pick(tmp_path / "leaderboard", pick_date=date(2026, 5, 1))
        assert result is None
```

- [ ] **Step 2: Run test (expect fail)**

Expected: ImportError.

- [ ] **Step 3: Implement analysis.consensus_pick**

```python
# src/bts/leaderboard/analysis.py
"""Analysis layer: consensus pick + percentile rank from leaderboard data.

These are read-only queries against the parquet store. Produces inputs
for the dashboard view at port 3003.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from bts.leaderboard.storage import read_user_picks


def consensus_pick(leaderboard_dir: Path, pick_date: date) -> dict | None:
    """Return modal batter pick across all tracked users for `pick_date`.

    Returns dict with keys: consensus_batter, consensus_share, n_users.
    Returns None if no picks for that date are recorded.
    """
    pick_dir = leaderboard_dir / "user_picks"
    if not pick_dir.exists():
        return None
    user_files = list(pick_dir.glob("*.parquet"))
    if not user_files:
        return None

    batters: list[str] = []
    for f in user_files:
        table = read_user_picks(f, dedupe="latest_per_pick_date")
        df = table.to_pandas()
        match = df[df["pick_date"] == pick_date]
        if not match.empty:
            batters.append(match.iloc[0]["batter_name"])

    if not batters:
        return None
    counts = Counter(batters)
    top, top_n = counts.most_common(1)[0]
    return {
        "consensus_batter": top,
        "consensus_share": top_n / len(batters),
        "n_users": len(batters),
    }
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_analysis.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/leaderboard/analysis.py tests/leaderboard/test_analysis.py
git commit -m "feat(leaderboard): analysis.consensus_pick"
```

### Task 5.3: analysis.percentile_rank

**Files:**
- Modify: `src/bts/leaderboard/analysis.py`
- Modify: `tests/leaderboard/test_analysis.py`

- [ ] **Step 1: Append failing test**

```python
# tests/leaderboard/test_analysis.py — append

from bts.leaderboard.models import LeaderboardRow
from bts.leaderboard.storage import write_leaderboard_snapshot
from bts.leaderboard.analysis import percentile_rank


class TestPercentileRank:
    def test_our_streak_ranks_among_active_users(self, tmp_path):
        leaderboard = tmp_path / "leaderboard"
        rows = [
            LeaderboardRow(captured_at=datetime(2026, 5, 1, 14, 0), tab="active_streak",
                           rank=i + 1, username=f"u{i}", streak=35 - i, hits_today=None)
            for i in range(10)
        ]
        write_leaderboard_snapshot(
            leaderboard / "leaderboard_snapshots" / "2026-05-01.parquet", rows,
        )
        # Our streak of 30 → 5 of 10 users have higher → 50th percentile
        rank = percentile_rank(leaderboard, our_streak=30)
        assert 0.4 <= rank["pct"] <= 0.6
```

- [ ] **Step 2: Run test (expect fail)**

Expected: AttributeError.

- [ ] **Step 3: Implement percentile_rank**

```python
# src/bts/leaderboard/analysis.py — append

def percentile_rank(leaderboard_dir: Path, our_streak: int) -> dict:
    """Compute our percentile rank in the active_streak leaderboard.

    Returns {pct, n_above, n_total}. Higher pct = better rank.
    """
    snaps = sorted((leaderboard_dir / "leaderboard_snapshots").glob("*.parquet"))
    if not snaps:
        return {"pct": None, "n_above": 0, "n_total": 0}
    df = pq.read_table(snaps[-1]).to_pandas()
    active = df[df["tab"] == "active_streak"]
    if active.empty:
        return {"pct": None, "n_above": 0, "n_total": 0}
    n_above = (active["streak"] > our_streak).sum()
    n_total = len(active)
    return {
        "pct": 1 - (n_above / n_total) if n_total > 0 else None,
        "n_above": int(n_above),
        "n_total": int(n_total),
    }
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/leaderboard/test_analysis.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/leaderboard/analysis.py tests/leaderboard/test_analysis.py
git commit -m "feat(leaderboard): analysis.percentile_rank"
```

### Task 5.4: Dashboard adapter (read-only view at /leaderboard)

**Files:**
- Modify: `src/bts/web.py` (or whatever serves port 3003 — verify via `ssh bts@bts-hetzner 'systemctl --user cat bts-dashboard.service'`)

- [ ] **Step 1: Verify dashboard module location**

```bash
grep -n "uvicorn\|flask\|fastapi\|app = " src/bts/web/*.py 2>&1 | head
ls src/bts/web/ 2>&1
```

Identify the app entrypoint.

- [ ] **Step 2: Add /leaderboard endpoint**

(Pattern depends on framework — adapt the snippet below for actual framework. Below assumes FastAPI; if it's Flask, use `@app.route('/leaderboard')`.)

```python
# src/bts/web/__init__.py (or correct module) — add a new endpoint
from datetime import date
from pathlib import Path

from bts.leaderboard.analysis import consensus_pick, percentile_rank


@app.get("/leaderboard")
def leaderboard_view():
    base = Path("data/leaderboard")
    today = date.today()
    consensus = consensus_pick(base, today)
    return {
        "today": today.isoformat(),
        "consensus": consensus,
        # percentile rank requires knowing our streak — pass current
        # production streak via the same source the existing dashboard reads
        # (e.g., streak.json). Replace the placeholder below.
        "our_streak_placeholder": "TODO Phase 5.5 — wire to streak.json",
    }
```

- [ ] **Step 3: Smoke test locally**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m bts.web &
sleep 2
curl -s http://localhost:3003/leaderboard | head -10
kill %1
```

Expected: JSON response with today's consensus pick.

- [ ] **Step 4: Deploy + verify on bts-hetzner**

```bash
git add src/bts/web/
git commit -m "feat(leaderboard): /leaderboard dashboard endpoint"
git push origin main:deploy
gh run watch $(gh run list --branch deploy --limit 1 --json databaseId -q '.[0].databaseId') --exit-status
ssh bts@bts-hetzner 'curl -s http://localhost:3003/leaderboard | head -5'
```

Expected: deploy succeeds; endpoint returns JSON.

### Task 5.5: Wire our streak into the percentile-rank view

**Files:**
- Modify: `src/bts/web/__init__.py` (or correct module)

- [ ] **Step 1: Find where existing dashboard reads our current streak**

```bash
grep -rln "streak\.json\|active_streak\|current_streak" src/bts/web/ 2>&1 | head -5
```

- [ ] **Step 2: Replace placeholder with real read**

Update the `/leaderboard` endpoint to read our active streak from the existing source and feed it into `percentile_rank()`.

- [ ] **Step 3: Smoke test, deploy, verify**

(Same pattern as 5.4.)

- [ ] **Step 4: Commit**

```bash
git add src/bts/web/
git commit -m "feat(leaderboard): wire our active streak into percentile-rank view"
```

### Task 5.6: 7-day data-accumulation gate

**Files:** none (operational milestone)

- [ ] **Step 1: After 7 days of data, run a sanity audit**

```bash
ssh bts@bts-hetzner 'UV_CACHE_DIR=/tmp/uv-cache uv run python -c "
import pyarrow.parquet as pq
from pathlib import Path
base = Path(\"data/leaderboard\")
snaps = sorted((base / \"leaderboard_snapshots\").glob(\"*.parquet\"))
users = list((base / \"user_picks\").glob(\"*.parquet\"))
print(f\"snapshots: {len(snaps)}, users tracked: {len(users)}\")
print(f\"first: {snaps[0].name if snaps else \"-\"}\")
print(f\"last: {snaps[-1].name if snaps else \"-\"}\")
"'
```

Expected: ≥7 snapshots, ≥100 user files, span covers a week.

- [ ] **Step 2: If health check is green and dashboard renders, declare Phase 5 done.**

Plan is complete.

---

## Self-review notes

**Spec coverage**: every section of the design spec has at least one task implementing it (Phases 1–5 map 1:1; Section 12 failure modes covered across Tasks 4.1, 4.3, and the test fixtures in 3.4–3.6).

**Type consistency**: `LeaderboardRow`, `PickRow`, `SeasonStats` named identically across all tasks. `scrape_leaderboard` / `scrape_user_picks` / `scrape_user_stats` / `run` consistent. `check()` signature matches existing health-check pattern.

**Placeholder scan**: one acknowledged TODO in `endpoints.py` (Task 1.1) — that's literally Phase 1's deliverable, not a plan failure. One TODO in Task 5.4 placeholder is replaced by Task 5.5. No other TBD/TODO/"add appropriate error handling" patterns.

**Worktree**: this plan is being executed on `feature/leaderboard-watcher` branch (Task 0). Final merge to main + push to deploy happens in Task 4.7.
