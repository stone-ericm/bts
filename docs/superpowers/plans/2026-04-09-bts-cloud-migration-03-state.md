# BTS Cloud Migration — Plan 03: State Management

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the state management CLI commands that enable the cloud migration's recovery story: `bts state export` (one-time migration snapshot), `bts state regenerate` (rebuild from Bluesky + MLB API + initial snapshot for disaster recovery), and `bts state verify` (drift check that runs periodically to detect bit-rot).

**Architecture:** Export command walks `data/picks/` and refuses to run if any pick is unresolved. Regenerate command parses the committed initial snapshot for pre-cutoff data and parses Bluesky post history for post-cutoff data. Verify command runs regeneration to a temp directory and diffs against current live state.

**Tech Stack:** Python 3.12, atproto SDK (already a dep via `posting.py`), Click, stdlib. No new external deps.

**Dependencies on other plans:** None. Uses existing `bts.posting` and `bts.picks` modules.

**Parent spec:** `docs/superpowers/specs/2026-04-09-bts-cloud-migration-design.md` (§ State persistence, § Initial state snapshot, § Risks: regenerate bit-rot)

---

## File Structure

- Create `src/bts/state/__init__.py` — empty package marker
- Create `src/bts/state/export.py` — initial snapshot export, ~120 lines
- Create `src/bts/state/regenerate.py` — Bluesky + MLB regeneration, ~400 lines
- Create `src/bts/state/verify.py` — drift check, ~100 lines
- Modify `src/bts/cli.py` — register three new commands under `bts state`
- Create `tests/test_state_export.py` — ~150 lines
- Create `tests/test_state_regenerate.py` — ~250 lines
- Create `tests/test_state_verify.py` — ~80 lines

---

### Task 1: Create `src/bts/state` package and export skeleton

**Files:**
- Create: `src/bts/state/__init__.py`
- Create: `src/bts/state/export.py`
- Create: `tests/test_state_export.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_state_export.py`:

```python
"""Tests for bts state export."""
import json
from pathlib import Path

import pytest

from bts.state.export import export_initial_state, UnresolvedPickError


def _write_pick(picks_dir: Path, date: str, result: str | None, double_down: bool = False):
    pick = {
        "date": date,
        "run_time": f"{date}T12:00:00+00:00",
        "pick": {
            "batter_name": "Test Batter",
            "batter_id": 100,
            "team": "NYY",
            "pitcher_name": "Test Pitcher",
            "pitcher_id": 200,
            "game_pk": 12345,
            "game_time": f"{date}T19:05:00-04:00",
            "p_game_hit": 0.85,
            "p_hit_pa": 0.31,
            "projected_lineup": False,
        },
        "double_down": {
            "batter_name": "Other Batter",
            "batter_id": 101,
            "team": "BOS",
            "pitcher_name": "Other Pitcher",
            "pitcher_id": 201,
            "game_pk": 12346,
            "game_time": f"{date}T19:10:00-04:00",
            "p_game_hit": 0.80,
            "p_hit_pa": 0.28,
            "projected_lineup": False,
        } if double_down else None,
        "runner_up": None,
        "bluesky_posted": True,
        "bluesky_uri": f"at://did:test/app.bsky.feed.post/{date}",
        "result": result,
    }
    (picks_dir / f"{date}.json").write_text(json.dumps(pick))


def test_export_refuses_when_unresolved(tmp_path: Path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pick(picks_dir, "2026-04-01", "hit")
    _write_pick(picks_dir, "2026-04-02", None)  # Unresolved
    _write_pick(picks_dir, "2026-04-03", "hit")

    (picks_dir / "streak.json").write_text('{"current": 2, "saver_available": true}')

    out_path = tmp_path / "initial-state.json"
    with pytest.raises(UnresolvedPickError, match="2026-04-02"):
        export_initial_state(picks_dir=picks_dir, output_path=out_path)


def test_export_succeeds_when_all_resolved(tmp_path: Path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pick(picks_dir, "2026-04-01", "hit")
    _write_pick(picks_dir, "2026-04-02", "miss")
    _write_pick(picks_dir, "2026-04-03", "hit")
    (picks_dir / "streak.json").write_text('{"current": 1, "saver_available": true}')

    out_path = tmp_path / "initial-state.json"
    export_initial_state(picks_dir=picks_dir, output_path=out_path)

    exported = json.loads(out_path.read_text())
    assert exported["version"] == 1
    assert exported["cutoff_date"] == "2026-04-03"
    assert exported["streak_at_cutoff"] == 1
    assert exported["saver_available"] is True
    assert len(exported["historical_picks"]) == 3
    assert {p["date"] for p in exported["historical_picks"]} == {"2026-04-01", "2026-04-02", "2026-04-03"}


def test_export_excludes_non_date_files(tmp_path: Path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pick(picks_dir, "2026-04-01", "hit")
    (picks_dir / "streak.json").write_text('{"current": 1, "saver_available": false}')
    (picks_dir / "notes.txt").write_text("not a pick")
    (picks_dir / "orchestrator.log").write_text("log data")

    out_path = tmp_path / "initial-state.json"
    export_initial_state(picks_dir=picks_dir, output_path=out_path)

    exported = json.loads(out_path.read_text())
    # Only the one pick, not notes.txt or orchestrator.log
    assert len(exported["historical_picks"]) == 1


def test_export_includes_bluesky_uri(tmp_path: Path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pick(picks_dir, "2026-04-01", "hit")
    (picks_dir / "streak.json").write_text('{"current": 1, "saver_available": false}')

    out_path = tmp_path / "initial-state.json"
    export_initial_state(picks_dir=picks_dir, output_path=out_path)

    exported = json.loads(out_path.read_text())
    assert exported["historical_picks"][0]["bluesky_uri"] == "at://did:test/app.bsky.feed.post/2026-04-01"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_state_export.py -v
```

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'bts.state'`.

- [ ] **Step 3: Create the package and export module**

Create `src/bts/state/__init__.py`:

```python
"""State management: export initial snapshot, regenerate from sources, verify drift."""
```

Create `src/bts/state/export.py`:

```python
"""Export current BTS state to a committable snapshot file.

Used once, at the moment of cloud migration cutover, to freeze the
pre-migration history into a git-tracked file. After export, the
regenerate command uses this file as the source of truth for dates
before the cutoff and uses Bluesky + MLB API for dates after.
"""
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

EXPORT_VERSION = 1
DATE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


class UnresolvedPickError(RuntimeError):
    """Raised when export is attempted with unresolved picks present."""


def export_initial_state(picks_dir: Path, output_path: Path) -> dict:
    """Export the full BTS state as a committable snapshot.

    Enforces the invariant that no pick may be in an unresolved state.
    Raises UnresolvedPickError with the list of offending files if any
    pick has `result is None`.

    Returns the exported dict (also written to output_path).
    """
    pick_files = _collect_pick_files(picks_dir)

    unresolved = []
    historical: list[dict] = []
    for pf in pick_files:
        data = json.loads(pf.read_text())
        if data.get("result") is None:
            unresolved.append(pf.name)
            continue
        historical.append(_pick_to_historical(data))

    if unresolved:
        raise UnresolvedPickError(
            f"Refusing to export: {len(unresolved)} pick(s) still unresolved: "
            f"{', '.join(sorted(unresolved)[:5])}"
            f"{'...' if len(unresolved) > 5 else ''}. "
            f"Wait for results to finalize and try again."
        )

    streak_file = picks_dir / "streak.json"
    if not streak_file.exists():
        raise RuntimeError(
            f"streak.json not found in {picks_dir}. "
            f"Export requires a streak file to determine the starting point."
        )
    streak_data = json.loads(streak_file.read_text())

    historical.sort(key=lambda p: p["date"])
    cutoff_date = historical[-1]["date"] if historical else "none"

    snapshot = {
        "version": EXPORT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "cutoff_date": cutoff_date,
        "streak_at_cutoff": streak_data.get("current", 0),
        "saver_available": streak_data.get("saver_available", True),
        "historical_picks": historical,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2))
    return snapshot


def _collect_pick_files(picks_dir: Path) -> list[Path]:
    """Return only files whose name matches YYYY-MM-DD.json."""
    return [
        p for p in picks_dir.iterdir()
        if p.is_file() and DATE_PATTERN.match(p.name)
    ]


def _pick_to_historical(pick_data: dict) -> dict:
    """Project a full pick file into its committable historical form."""
    return {
        "date": pick_data["date"],
        "pick": _project_pick(pick_data.get("pick")),
        "double_down": _project_pick(pick_data.get("double_down")) if pick_data.get("double_down") else None,
        "result": pick_data.get("result"),
        "bluesky_posted": pick_data.get("bluesky_posted", False),
        "bluesky_uri": pick_data.get("bluesky_uri"),
    }


def _project_pick(pick: dict | None) -> dict | None:
    """Extract the minimum fields to reconstruct a pick."""
    if pick is None:
        return None
    return {
        "batter_name": pick["batter_name"],
        "batter_id": pick["batter_id"],
        "team": pick["team"],
        "pitcher_name": pick["pitcher_name"],
        "pitcher_id": pick["pitcher_id"],
        "game_pk": pick["game_pk"],
        "game_time": pick["game_time"],
        "p_game_hit": pick["p_game_hit"],
    }
```

- [ ] **Step 4: Run tests to verify**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_state_export.py -v
```

Expected: All four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/state/__init__.py src/bts/state/export.py tests/test_state_export.py
git commit -m "feat(state): add export_initial_state with resolved-state guard"
```

---

### Task 2: CLI command `bts state export`

**Files:**
- Modify: `src/bts/cli.py`

- [ ] **Step 1: Add the state group and export command to cli.py**

In `src/bts/cli.py`, find the existing command groups (`data`, `simulate`, etc.) and add:

```python
@cli.group()
def state():
    """State management: export / regenerate / verify BTS state."""


@state.command(name="export")
@click.option("--picks-dir", default="data/picks", type=click.Path(exists=True))
@click.option("--to", "output_path", default="data/state/initial-state.json", type=click.Path())
def state_export(picks_dir, output_path):
    """Export current state to a committable snapshot file.

    Refuses to run if any pick in picks-dir is unresolved. Used at
    the moment of cloud migration cutover to freeze pre-migration history.
    """
    from pathlib import Path
    from bts.state.export import export_initial_state, UnresolvedPickError

    try:
        snapshot = export_initial_state(
            picks_dir=Path(picks_dir),
            output_path=Path(output_path),
        )
    except UnresolvedPickError as e:
        click.echo(str(e), err=True)
        raise SystemExit(2)

    click.echo(
        f"Exported {len(snapshot['historical_picks'])} picks to {output_path}\n"
        f"  cutoff_date: {snapshot['cutoff_date']}\n"
        f"  streak_at_cutoff: {snapshot['streak_at_cutoff']}\n"
        f"  saver_available: {snapshot['saver_available']}"
    )
```

- [ ] **Step 2: Smoke test**

On Pi5 where actual state lives:

```bash
ssh stonehengee@pi5.local 'cd ~/projects/bts && git pull && UV_CACHE_DIR=/tmp/uv-cache uv run bts state export --to /tmp/test-export.json'
```

Expected: either exports successfully (if all picks are resolved) or fails with "Refusing to export: N pick(s) still unresolved". Do not commit `/tmp/test-export.json` — this is just a smoke test.

- [ ] **Step 3: Commit**

```bash
git add src/bts/cli.py
git commit -m "feat(state): add 'bts state export' CLI command"
```

---

### Task 3: Regenerate module — fetch Bluesky post history

**Files:**
- Create: `src/bts/state/regenerate.py`
- Create: `tests/test_state_regenerate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_state_regenerate.py`:

```python
"""Tests for bts state regenerate."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from bts.state.regenerate import (
    fetch_bluesky_posts,
    parse_pick_from_post,
    ParsedPost,
)


def _make_post(text: str, uri: str, created_at: str, is_reply: bool = False):
    """Build a fake atproto post response."""
    post = MagicMock()
    post.post.uri = uri
    post.post.record.text = text
    post.post.record.created_at = created_at
    if is_reply:
        post.post.record.reply = MagicMock()
    else:
        post.post.record.reply = None
    return post


def test_fetch_bluesky_posts_returns_posts_in_order():
    fake_posts = [
        _make_post("pick 3", "at://post3", "2026-04-03T12:00:00Z"),
        _make_post("pick 2", "at://post2", "2026-04-02T12:00:00Z"),
        _make_post("pick 1", "at://post1", "2026-04-01T12:00:00Z"),
    ]
    fake_response = MagicMock()
    fake_response.feed = fake_posts
    fake_response.cursor = None

    with patch("bts.state.regenerate._bluesky_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.get_author_feed.return_value = fake_response
        mock_client_factory.return_value = mock_client

        posts = fetch_bluesky_posts(handle="test.bsky.social", from_date="2026-04-01")

    # Should be sorted chronologically
    assert len(posts) == 3
    assert posts[0].uri == "at://post1"
    assert posts[-1].uri == "at://post3"


def test_parse_pick_post_extracts_single_pick():
    text = "Today's BTS pick: Nico Hoerner (CHC) vs RHP Test Pitcher — 78.3% 🎯\n\nStreak: 2"
    parsed = parse_pick_from_post(text)
    assert parsed is not None
    assert parsed.batter_name == "Nico Hoerner"
    assert parsed.team == "CHC"
    assert parsed.is_double_down is False
    assert parsed.double_down_batter is None


def test_parse_pick_post_extracts_double_down():
    text = (
        "Today's BTS pick: Jose Altuve (HOU) vs RHP Pitcher A — 82.0% 🎯\n"
        "Double down: Kyle Tucker (HOU) vs Pitcher B — 80.0%\n\n"
        "Streak: 5"
    )
    parsed = parse_pick_from_post(text)
    assert parsed is not None
    assert parsed.batter_name == "Jose Altuve"
    assert parsed.is_double_down is True
    assert parsed.double_down_batter == "Kyle Tucker"


def test_parse_skip_post():
    text = "Today's BTS pick: SKIP — top prob 76.5%, below 80% threshold. Streak holds at 3."
    parsed = parse_pick_from_post(text)
    assert parsed is not None
    assert parsed.is_skip is True


def test_parse_unrecognized_post_returns_none():
    text = "random promotional content, not a pick"
    parsed = parse_pick_from_post(text)
    assert parsed is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_state_regenerate.py -v
```

Expected: Tests FAIL with `ModuleNotFoundError: No module named 'bts.state.regenerate'`.

- [ ] **Step 3: Write the regenerate module (fetch + parse only for now)**

Create `src/bts/state/regenerate.py`:

```python
"""Reconstruct BTS state from Bluesky post history + MLB API.

Used for disaster recovery: if the Fly machine loses its volume entirely,
or during migration between providers, this command rebuilds the full
state from authoritative external sources. Pre-cutoff state comes from
the committed initial-state.json snapshot.

The heart of this module is a post parser that must handle the human-readable
format produced by src/bts/posting.py's format_post() function. If that
format changes, this parser must be updated in lockstep.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional


@dataclass
class ParsedPost:
    """A Bluesky post parsed into structured pick fields."""
    uri: str
    created_at: str
    text: str
    is_reply: bool
    # For pick posts
    is_skip: bool = False
    batter_name: Optional[str] = None
    team: Optional[str] = None
    is_double_down: bool = False
    double_down_batter: Optional[str] = None
    double_down_team: Optional[str] = None
    streak_at_time: Optional[int] = None
    # For result reply posts
    is_result: bool = False
    result: Optional[str] = None  # "hit" | "miss" | "skip"
    streak_after: Optional[int] = None


def _bluesky_client():
    """Lazy import wrapper so tests can mock it."""
    from atproto import Client
    return Client()


def fetch_bluesky_posts(
    handle: str,
    from_date: str,
    limit: int = 5000,
) -> list[ParsedPost]:
    """Fetch the author's post history via atproto get_author_feed.

    Returns posts in chronological order (oldest first). Filters to
    posts created on or after from_date.
    """
    client = _bluesky_client()
    all_posts: list = []
    cursor = None

    while True:
        response = client.get_author_feed(
            actor=handle,
            limit=100,
            cursor=cursor,
        )
        feed = response.feed
        if not feed:
            break
        all_posts.extend(feed)
        cursor = response.cursor
        if not cursor or len(all_posts) >= limit:
            break

    # Filter by date + parse
    parsed: list[ParsedPost] = []
    for entry in all_posts:
        record = entry.post.record
        created_at = record.created_at
        if created_at < f"{from_date}T00:00:00":
            continue
        is_reply = getattr(record, "reply", None) is not None
        parsed.append(ParsedPost(
            uri=entry.post.uri,
            created_at=created_at,
            text=record.text,
            is_reply=is_reply,
        ))

    parsed.sort(key=lambda p: p.created_at)
    return parsed


# Regex patterns for post format (see src/bts/posting.py format_post)
_PICK_RE = re.compile(
    r"Today's BTS pick:\s*(?P<name>[^(]+?)\s*\((?P<team>[A-Z]{2,3})\)",
    re.MULTILINE,
)
_DOUBLE_RE = re.compile(
    r"Double down:\s*(?P<name>[^(]+?)\s*\((?P<team>[A-Z]{2,3})\)",
    re.MULTILINE,
)
_SKIP_RE = re.compile(r"SKIP", re.IGNORECASE)
_STREAK_RE = re.compile(r"Streak[:\s]+(?P<streak>\d+)", re.IGNORECASE)


def parse_pick_from_post(text: str) -> Optional[ParsedPost]:
    """Parse a pick post's text into structured fields.

    Returns None if the post doesn't look like a BTS pick. Returns a
    ParsedPost with is_skip=True if it's a skip announcement.
    """
    if _SKIP_RE.search(text) and "Today's BTS pick" in text:
        return ParsedPost(
            uri="", created_at="", text=text, is_reply=False,
            is_skip=True,
        )

    match = _PICK_RE.search(text)
    if not match:
        return None

    parsed = ParsedPost(
        uri="", created_at="", text=text, is_reply=False,
        batter_name=match.group("name").strip(),
        team=match.group("team").strip(),
    )

    double_match = _DOUBLE_RE.search(text)
    if double_match:
        parsed.is_double_down = True
        parsed.double_down_batter = double_match.group("name").strip()
        parsed.double_down_team = double_match.group("team").strip()

    streak_match = _STREAK_RE.search(text)
    if streak_match:
        parsed.streak_at_time = int(streak_match.group("streak"))

    return parsed
```

- [ ] **Step 4: Run tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_state_regenerate.py -v
```

Expected: All five tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/state/regenerate.py tests/test_state_regenerate.py
git commit -m "feat(state): add Bluesky post fetcher + pick parser for regenerate"
```

---

### Task 4: Regenerate — result reply parsing + streak reconstruction

**Files:**
- Modify: `src/bts/state/regenerate.py`
- Modify: `tests/test_state_regenerate.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_state_regenerate.py`:

```python
from bts.state.regenerate import (
    parse_result_from_reply,
    reconstruct_pick_timeline,
    Timeline,
    HistoricalPickRecord,
)


def test_parse_result_reply_hit():
    parsed = parse_result_from_reply("✅ HIT — Streak: 3")
    assert parsed.is_result is True
    assert parsed.result == "hit"
    assert parsed.streak_after == 3


def test_parse_result_reply_miss():
    parsed = parse_result_from_reply("❌ MISS — Streak reset to 0")
    assert parsed.is_result is True
    assert parsed.result == "miss"
    assert parsed.streak_after == 0


def test_parse_result_reply_not_a_result():
    parsed = parse_result_from_reply("Random reply text")
    assert parsed.is_result is False


def test_reconstruct_timeline_alternates_picks_and_results():
    posts = [
        ParsedPost(uri="at://p1", created_at="2026-04-01T12:00:00Z",
                   text="Today's BTS pick: A (NYY)", is_reply=False,
                   batter_name="A", team="NYY"),
        ParsedPost(uri="at://r1", created_at="2026-04-01T23:00:00Z",
                   text="✅ HIT — Streak: 1", is_reply=True,
                   is_result=True, result="hit", streak_after=1),
        ParsedPost(uri="at://p2", created_at="2026-04-02T12:00:00Z",
                   text="Today's BTS pick: B (BOS)", is_reply=False,
                   batter_name="B", team="BOS"),
        ParsedPost(uri="at://r2", created_at="2026-04-02T23:00:00Z",
                   text="❌ MISS — Streak reset to 0", is_reply=True,
                   is_result=True, result="miss", streak_after=0),
    ]

    timeline = reconstruct_pick_timeline(posts)
    assert len(timeline.pick_records) == 2
    assert timeline.pick_records[0].date == "2026-04-01"
    assert timeline.pick_records[0].batter_name == "A"
    assert timeline.pick_records[0].result == "hit"
    assert timeline.pick_records[0].bluesky_uri == "at://p1"
    assert timeline.final_streak == 0
    assert timeline.pick_records[1].result == "miss"


def test_reconstruct_timeline_handles_unresolved_last_day():
    posts = [
        ParsedPost(uri="at://p1", created_at="2026-04-01T12:00:00Z",
                   text="Today's BTS pick: A (NYY)", is_reply=False,
                   batter_name="A", team="NYY"),
        ParsedPost(uri="at://r1", created_at="2026-04-01T23:00:00Z",
                   text="✅ HIT — Streak: 1", is_reply=True,
                   is_result=True, result="hit", streak_after=1),
        ParsedPost(uri="at://p2", created_at="2026-04-02T12:00:00Z",
                   text="Today's BTS pick: B (BOS)", is_reply=False,
                   batter_name="B", team="BOS"),
        # No reply for p2 — still in progress or regeneration runs mid-day
    ]
    timeline = reconstruct_pick_timeline(posts)
    assert len(timeline.pick_records) == 2
    assert timeline.pick_records[1].result is None
    assert timeline.final_streak == 1  # Last known resolved streak
```

- [ ] **Step 2: Run to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_state_regenerate.py -v
```

Expected: New tests FAIL with missing names.

- [ ] **Step 3: Write the implementation**

Append to `src/bts/state/regenerate.py`:

```python
_HIT_RE = re.compile(r"✅|HIT", re.IGNORECASE)
_MISS_RE = re.compile(r"❌|MISS", re.IGNORECASE)


def parse_result_from_reply(text: str) -> ParsedPost:
    """Parse a Bluesky reply post as a BTS result announcement.

    Returns a ParsedPost with is_result=False if the text doesn't look
    like a BTS result reply.
    """
    is_hit = bool(_HIT_RE.search(text))
    is_miss = bool(_MISS_RE.search(text))
    if not (is_hit or is_miss):
        return ParsedPost(uri="", created_at="", text=text, is_reply=True, is_result=False)

    result = "hit" if is_hit and not is_miss else "miss"

    # Extract streak number
    streak_after = None
    match = re.search(r"[Ss]treak[:\s]+(\d+)", text)
    if match:
        streak_after = int(match.group(1))
    elif "reset to 0" in text.lower():
        streak_after = 0

    return ParsedPost(
        uri="", created_at="", text=text, is_reply=True,
        is_result=True,
        result=result,
        streak_after=streak_after,
    )


@dataclass
class HistoricalPickRecord:
    """A pick + its result as reconstructed from Bluesky."""
    date: str
    batter_name: str
    team: str
    is_double_down: bool
    double_down_batter: Optional[str]
    double_down_team: Optional[str]
    bluesky_uri: str
    result: Optional[str]
    streak_after: Optional[int]


@dataclass
class Timeline:
    """Full reconstructed timeline from Bluesky."""
    pick_records: list[HistoricalPickRecord] = field(default_factory=list)
    final_streak: int = 0
    saver_available_at_end: bool = True


def _date_from_created_at(created_at: str) -> str:
    """Extract YYYY-MM-DD from an ISO timestamp."""
    return created_at[:10]


def reconstruct_pick_timeline(posts: list[ParsedPost]) -> Timeline:
    """Walk posts chronologically to reconstruct pick history + streak.

    Posts must be sorted oldest-first. Pairs pick posts with their result
    replies based on date proximity (result for day D is the first reply
    seen after the pick for day D with an is_result=True payload).
    """
    # Parse reply text where we haven't already
    for p in posts:
        if p.is_reply and not p.is_result:
            parsed = parse_result_from_reply(p.text)
            p.is_result = parsed.is_result
            p.result = parsed.result
            p.streak_after = parsed.streak_after

    records_by_date: dict[str, HistoricalPickRecord] = {}
    timeline_order: list[str] = []

    for post in posts:
        date = _date_from_created_at(post.created_at)
        if not post.is_reply:
            # Pick post
            if post.is_skip or post.batter_name is None:
                continue
            if date not in records_by_date:
                records_by_date[date] = HistoricalPickRecord(
                    date=date,
                    batter_name=post.batter_name,
                    team=post.team or "",
                    is_double_down=post.is_double_down,
                    double_down_batter=post.double_down_batter,
                    double_down_team=post.double_down_team,
                    bluesky_uri=post.uri,
                    result=None,
                    streak_after=None,
                )
                timeline_order.append(date)
        elif post.is_result:
            # Reply post carrying a result; attribute to most recent unresolved record
            for date_key in reversed(timeline_order):
                record = records_by_date[date_key]
                if record.result is None:
                    record.result = post.result
                    record.streak_after = post.streak_after
                    break

    pick_records = [records_by_date[d] for d in timeline_order]

    # Last known resolved streak is the final_streak
    final_streak = 0
    for r in reversed(pick_records):
        if r.streak_after is not None:
            final_streak = r.streak_after
            break

    return Timeline(
        pick_records=pick_records,
        final_streak=final_streak,
        saver_available_at_end=True,  # Conservative default; Task 5 refines
    )
```

- [ ] **Step 4: Run tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_state_regenerate.py -v
```

Expected: All nine tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/state/regenerate.py tests/test_state_regenerate.py
git commit -m "feat(state): add result reply parsing + timeline reconstruction"
```

---

### Task 5: Saver availability reconstruction

**Files:**
- Modify: `src/bts/state/regenerate.py`
- Modify: `tests/test_state_regenerate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state_regenerate.py`:

```python
def test_saver_consumed_on_first_miss_at_streak_10():
    """Saver fires on first miss at streak 10-15 per MDP rules."""
    posts = [
        # Build up to streak 10
        *[
            ParsedPost(
                uri=f"at://p{i}", created_at=f"2026-04-{i:02d}T12:00:00Z",
                text=f"pick {i}", is_reply=False,
                batter_name=f"B{i}", team="NYY",
            )
            for i in range(1, 11)
        ],
        *[
            ParsedPost(
                uri=f"at://r{i}", created_at=f"2026-04-{i:02d}T23:00:00Z",
                text=f"hit {i}", is_reply=True,
                is_result=True, result="hit", streak_after=i,
            )
            for i in range(1, 11)
        ],
    ]
    # Sort by timestamp
    posts.sort(key=lambda p: p.created_at)

    # Day 11 = miss at streak 10 — saver should fire, streak stays at 10
    posts += [
        ParsedPost(uri="at://p11", created_at="2026-04-11T12:00:00Z",
                   text="pick 11", is_reply=False,
                   batter_name="B11", team="NYY"),
        ParsedPost(uri="at://r11", created_at="2026-04-11T23:00:00Z",
                   text="❌ MISS — Streak: 10 (saver used)", is_reply=True,
                   is_result=True, result="miss", streak_after=10),
    ]

    timeline = reconstruct_pick_timeline(posts)
    # Saver consumed — no longer available
    assert timeline.saver_available_at_end is False
    assert timeline.final_streak == 10
```

- [ ] **Step 2: Run to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_state_regenerate.py::test_saver_consumed_on_first_miss_at_streak_10 -v
```

Expected: FAIL — current implementation returns `saver_available_at_end=True` always.

- [ ] **Step 3: Update reconstruct_pick_timeline to track saver**

Modify the tail of `reconstruct_pick_timeline` in `src/bts/state/regenerate.py` to compute saver state:

Replace:
```python
    return Timeline(
        pick_records=pick_records,
        final_streak=final_streak,
        saver_available_at_end=True,  # Conservative default; Task 5 refines
    )
```

with:

```python
    # Saver: available until first miss at streak 10-15
    saver_available = True
    for r in pick_records:
        if r.result != "miss":
            continue
        # Determine streak just before this miss. Walk backward to find the
        # previous streak_after (or 0 for the first pick).
        idx = pick_records.index(r)
        streak_before = 0
        for prior in reversed(pick_records[:idx]):
            if prior.streak_after is not None:
                streak_before = prior.streak_after
                break
        # Saver consumed if streak_before in [10, 15] — MDP saver phase
        if 10 <= streak_before <= 15:
            saver_available = False
            break

    return Timeline(
        pick_records=pick_records,
        final_streak=final_streak,
        saver_available_at_end=saver_available,
    )
```

- [ ] **Step 4: Run tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_state_regenerate.py -v
```

Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/state/regenerate.py tests/test_state_regenerate.py
git commit -m "feat(state): reconstruct saver availability in timeline"
```

---

### Task 6: Regenerate command that composes initial snapshot + Bluesky

**Files:**
- Modify: `src/bts/state/regenerate.py`
- Modify: `tests/test_state_regenerate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state_regenerate.py`:

```python
def test_regenerate_composes_snapshot_and_bluesky(tmp_path: Path):
    """Regeneration should produce state files using snapshot + Bluesky data."""
    # Initial snapshot file
    snapshot = {
        "version": 1,
        "exported_at": "2026-04-01T00:00:00Z",
        "cutoff_date": "2026-03-31",
        "streak_at_cutoff": 5,
        "saver_available": True,
        "historical_picks": [
            {
                "date": "2026-03-31",
                "pick": {
                    "batter_name": "Aaron Judge",
                    "batter_id": 100, "team": "NYY",
                    "pitcher_name": "X", "pitcher_id": 200,
                    "game_pk": 111, "game_time": "2026-03-31T19:05:00-04:00",
                    "p_game_hit": 0.85,
                },
                "double_down": None,
                "result": "hit",
                "bluesky_posted": True,
                "bluesky_uri": "at://old/post",
            },
        ],
    }
    snapshot_path = tmp_path / "initial-state.json"
    snapshot_path.write_text(json.dumps(snapshot))

    # Bluesky timeline from 2026-04-01 onward
    timeline = Timeline(
        pick_records=[
            HistoricalPickRecord(
                date="2026-04-01",
                batter_name="Nico Hoerner",
                team="CHC",
                is_double_down=False,
                double_down_batter=None,
                double_down_team=None,
                bluesky_uri="at://new/post",
                result="hit",
                streak_after=6,
            ),
        ],
        final_streak=6,
        saver_available_at_end=True,
    )

    from bts.state.regenerate import compose_state_from_snapshot_and_timeline

    out_dir = tmp_path / "regenerated"
    compose_state_from_snapshot_and_timeline(
        snapshot_path=snapshot_path,
        timeline=timeline,
        out_picks_dir=out_dir,
    )

    # Pre-cutoff pick file
    old_pick = json.loads((out_dir / "2026-03-31.json").read_text())
    assert old_pick["pick"]["batter_name"] == "Aaron Judge"
    assert old_pick["result"] == "hit"
    assert old_pick["bluesky_uri"] == "at://old/post"

    # Post-cutoff pick file
    new_pick = json.loads((out_dir / "2026-04-01.json").read_text())
    assert new_pick["pick"]["batter_name"] == "Nico Hoerner"
    assert new_pick["result"] == "hit"
    assert new_pick["bluesky_uri"] == "at://new/post"

    # Streak file
    streak = json.loads((out_dir / "streak.json").read_text())
    assert streak["current"] == 6
    assert streak["saver_available"] is True
```

- [ ] **Step 2: Run to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_state_regenerate.py::test_regenerate_composes_snapshot_and_bluesky -v
```

Expected: FAIL with `ImportError: cannot import name 'compose_state_from_snapshot_and_timeline'`.

- [ ] **Step 3: Write the implementation**

Append to `src/bts/state/regenerate.py`:

```python
import json as _json
from pathlib import Path as _Path


def compose_state_from_snapshot_and_timeline(
    snapshot_path: Path,
    timeline: Timeline,
    out_picks_dir: Path,
) -> None:
    """Write pick files + streak.json from a snapshot + Bluesky timeline.

    Pre-cutoff records come from the committed initial snapshot; post-cutoff
    records come from the Bluesky timeline. Writes pick files to
    out_picks_dir/{date}.json in the format the scheduler expects.
    """
    out_picks_dir.mkdir(parents=True, exist_ok=True)

    snapshot = _json.loads(snapshot_path.read_text())

    # Write historical picks from snapshot
    for hist in snapshot.get("historical_picks", []):
        pick_file = _hist_to_pick_file(hist)
        out_path = out_picks_dir / f"{hist['date']}.json"
        out_path.write_text(_json.dumps(pick_file, indent=2))

    # Write picks from Bluesky timeline (post-cutoff)
    cutoff = snapshot.get("cutoff_date", "0000-00-00")
    for record in timeline.pick_records:
        if record.date <= cutoff:
            continue  # Snapshot already wrote this
        pick_file = _record_to_pick_file(record)
        out_path = out_picks_dir / f"{record.date}.json"
        out_path.write_text(_json.dumps(pick_file, indent=2))

    # Streak file: prefer the timeline's final_streak, fall back to snapshot
    streak_data = {
        "current": (timeline.final_streak
                    if timeline.pick_records
                    else snapshot.get("streak_at_cutoff", 0)),
        "saver_available": (timeline.saver_available_at_end
                            if timeline.pick_records
                            else snapshot.get("saver_available", True)),
    }
    (out_picks_dir / "streak.json").write_text(_json.dumps(streak_data, indent=2))


def _hist_to_pick_file(hist: dict) -> dict:
    """Convert a snapshot historical record back to a pick file shape."""
    return {
        "date": hist["date"],
        "run_time": hist.get("run_time", f"{hist['date']}T12:00:00+00:00"),
        "pick": hist["pick"],
        "double_down": hist.get("double_down"),
        "runner_up": None,
        "bluesky_posted": hist.get("bluesky_posted", True),
        "bluesky_uri": hist.get("bluesky_uri"),
        "result": hist.get("result"),
    }


def _record_to_pick_file(record: HistoricalPickRecord) -> dict:
    """Convert a regenerated HistoricalPickRecord to a pick file shape.

    Note: some fields (batter_id, pitcher info, p_game_hit) cannot be
    recovered from Bluesky alone and are left as None. A follow-up
    pass could use the MLB API to backfill batter_id from name+team+date.
    """
    return {
        "date": record.date,
        "run_time": f"{record.date}T12:00:00+00:00",
        "pick": {
            "batter_name": record.batter_name,
            "batter_id": None,
            "team": record.team,
            "pitcher_name": None,
            "pitcher_id": None,
            "game_pk": None,
            "game_time": None,
            "p_game_hit": None,
            "p_hit_pa": None,
            "projected_lineup": False,
        },
        "double_down": {
            "batter_name": record.double_down_batter,
            "batter_id": None,
            "team": record.double_down_team,
            "pitcher_name": None, "pitcher_id": None,
            "game_pk": None, "game_time": None,
            "p_game_hit": None, "p_hit_pa": None,
            "projected_lineup": False,
        } if record.is_double_down else None,
        "runner_up": None,
        "bluesky_posted": True,
        "bluesky_uri": record.bluesky_uri,
        "result": record.result,
    }
```

- [ ] **Step 4: Run tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_state_regenerate.py -v
```

Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/state/regenerate.py tests/test_state_regenerate.py
git commit -m "feat(state): compose regenerated state from snapshot + timeline"
```

---

### Task 7: Top-level regenerate function + CLI

**Files:**
- Modify: `src/bts/state/regenerate.py`
- Modify: `src/bts/cli.py`

- [ ] **Step 1: Add the regenerate() entry point to regenerate.py**

Append to `src/bts/state/regenerate.py`:

```python
def regenerate(
    snapshot_path: Path,
    bluesky_handle: str,
    out_picks_dir: Path,
) -> dict:
    """Full regeneration: fetch Bluesky, compose with snapshot, write pick files.

    Returns a summary dict with counts of regenerated picks and the
    final streak.
    """
    snapshot = _json.loads(snapshot_path.read_text())
    cutoff = snapshot.get("cutoff_date", "0000-00-00")

    posts = fetch_bluesky_posts(handle=bluesky_handle, from_date=cutoff)

    # Parse pick posts (non-reply) through parse_pick_from_post
    parsed_posts: list[ParsedPost] = []
    for p in posts:
        if p.is_reply:
            parsed_posts.append(p)
            continue
        pick_parse = parse_pick_from_post(p.text)
        if pick_parse is None:
            continue
        pick_parse.uri = p.uri
        pick_parse.created_at = p.created_at
        pick_parse.is_reply = False
        parsed_posts.append(pick_parse)

    timeline = reconstruct_pick_timeline(parsed_posts)
    compose_state_from_snapshot_and_timeline(
        snapshot_path=snapshot_path,
        timeline=timeline,
        out_picks_dir=out_picks_dir,
    )

    return {
        "snapshot_cutoff": cutoff,
        "snapshot_picks": len(snapshot.get("historical_picks", [])),
        "bluesky_picks": len(timeline.pick_records),
        "final_streak": timeline.final_streak,
        "saver_available": timeline.saver_available_at_end,
    }
```

- [ ] **Step 2: Add CLI command**

In `src/bts/cli.py` under the `state` group:

```python
@state.command(name="regenerate")
@click.option("--snapshot", default="data/state/initial-state.json",
              type=click.Path(exists=True))
@click.option("--handle", default="beatthestreakbot.bsky.social")
@click.option("--out-picks-dir", default="data/picks", type=click.Path())
def state_regenerate(snapshot, handle, out_picks_dir):
    """Rebuild BTS state from committed snapshot + Bluesky post history.

    Used for disaster recovery when the Fly volume is lost or during
    migration between providers. Post-cutoff data comes from Bluesky;
    pre-cutoff data comes from the committed initial snapshot.
    """
    from pathlib import Path
    from bts.state.regenerate import regenerate

    summary = regenerate(
        snapshot_path=Path(snapshot),
        bluesky_handle=handle,
        out_picks_dir=Path(out_picks_dir),
    )
    click.echo("Regeneration complete:")
    for k, v in summary.items():
        click.echo(f"  {k}: {v}")
```

- [ ] **Step 3: Commit**

```bash
git add src/bts/state/regenerate.py src/bts/cli.py
git commit -m "feat(state): add 'bts state regenerate' top-level command"
```

---

### Task 8: Verify command for drift detection

**Files:**
- Create: `src/bts/state/verify.py`
- Create: `tests/test_state_verify.py`
- Modify: `src/bts/cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_state_verify.py`:

```python
"""Tests for bts state verify."""
import json
from pathlib import Path

import pytest

from bts.state.verify import diff_pick_files, DriftReport


def test_diff_identical_dirs_returns_no_drift(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    pick = {"date": "2026-04-01", "pick": {"batter_name": "X"}, "result": "hit"}
    (a / "2026-04-01.json").write_text(json.dumps(pick))
    (b / "2026-04-01.json").write_text(json.dumps(pick))
    (a / "streak.json").write_text('{"current": 1, "saver_available": true}')
    (b / "streak.json").write_text('{"current": 1, "saver_available": true}')

    report = diff_pick_files(a, b)
    assert report.is_clean


def test_diff_streak_mismatch_reported(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "streak.json").write_text('{"current": 5, "saver_available": true}')
    (b / "streak.json").write_text('{"current": 3, "saver_available": true}')

    report = diff_pick_files(a, b)
    assert not report.is_clean
    assert any("streak" in issue for issue in report.issues)


def test_diff_missing_pick_file_reported(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "2026-04-01.json").write_text('{"date": "2026-04-01"}')
    (a / "streak.json").write_text('{"current": 1, "saver_available": true}')
    (b / "streak.json").write_text('{"current": 1, "saver_available": true}')

    report = diff_pick_files(a, b)
    assert not report.is_clean
    assert any("2026-04-01" in issue for issue in report.issues)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_state_verify.py -v
```

Expected: Tests FAIL with `ModuleNotFoundError: No module named 'bts.state.verify'`.

- [ ] **Step 3: Write the implementation**

Create `src/bts/state/verify.py`:

```python
"""State drift detection: compare live state to what regeneration produces.

Runs periodically (weekly cron on the Fly machine) to catch bit-rot in
the regeneration logic before it matters in a real recovery event. Also
catches silent corruption in the live state files.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

DATE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


@dataclass
class DriftReport:
    """Result of comparing two state directories."""
    issues: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.issues


def diff_pick_files(live_dir: Path, regenerated_dir: Path) -> DriftReport:
    """Compare live state to regenerated state. Returns a DriftReport.

    Compares pick files (by date), result field, batter_name, and streak.
    Does not compare fields that cannot be recovered from Bluesky alone
    (batter_id, p_game_hit, etc.) because those are expected to differ.
    """
    report = DriftReport()

    # Collect pick dates from both
    live_picks = {p.stem: p for p in live_dir.glob("*.json")
                  if DATE_PATTERN.match(p.name)}
    regen_picks = {p.stem: p for p in regenerated_dir.glob("*.json")
                   if DATE_PATTERN.match(p.name)}

    # Dates in live but missing in regen
    for date in sorted(set(live_picks) - set(regen_picks)):
        report.issues.append(f"pick {date} exists in live but not in regenerated")
    for date in sorted(set(regen_picks) - set(live_picks)):
        report.issues.append(f"pick {date} exists in regenerated but not in live")

    # Common dates — compare recoverable fields
    for date in sorted(set(live_picks) & set(regen_picks)):
        live = json.loads(live_picks[date].read_text())
        regen = json.loads(regen_picks[date].read_text())
        if live.get("result") != regen.get("result"):
            report.issues.append(
                f"pick {date} result mismatch: "
                f"live={live.get('result')}, regen={regen.get('result')}"
            )
        if live.get("pick", {}).get("batter_name") != regen.get("pick", {}).get("batter_name"):
            report.issues.append(
                f"pick {date} batter name mismatch: "
                f"live={live.get('pick', {}).get('batter_name')}, "
                f"regen={regen.get('pick', {}).get('batter_name')}"
            )

    # Streak
    try:
        live_streak = json.loads((live_dir / "streak.json").read_text())
        regen_streak = json.loads((regenerated_dir / "streak.json").read_text())
        if live_streak.get("current") != regen_streak.get("current"):
            report.issues.append(
                f"streak mismatch: live={live_streak.get('current')}, "
                f"regen={regen_streak.get('current')}"
            )
    except FileNotFoundError as e:
        report.issues.append(f"streak file missing: {e}")

    return report
```

- [ ] **Step 4: Run tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_state_verify.py -v
```

Expected: All three tests PASS.

- [ ] **Step 5: Register CLI command**

Add to `src/bts/cli.py` under the `state` group:

```python
@state.command(name="verify")
@click.option("--live-dir", default="data/picks", type=click.Path(exists=True))
@click.option("--snapshot", default="data/state/initial-state.json",
              type=click.Path(exists=True))
@click.option("--handle", default="beatthestreakbot.bsky.social")
def state_verify(live_dir, snapshot, handle):
    """Regenerate state to a temp dir and diff against live state.

    Run periodically as a drift check. Exits 0 if clean, 1 if drift found.
    """
    import tempfile
    from pathlib import Path
    from bts.state.regenerate import regenerate
    from bts.state.verify import diff_pick_files

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "picks"
        summary = regenerate(
            snapshot_path=Path(snapshot),
            bluesky_handle=handle,
            out_picks_dir=tmp_path,
        )
        report = diff_pick_files(Path(live_dir), tmp_path)

    if report.is_clean:
        click.echo(f"Drift check CLEAN. {summary['snapshot_picks']} snapshot + {summary['bluesky_picks']} Bluesky picks.")
        return

    click.echo("Drift detected:", err=True)
    for issue in report.issues:
        click.echo(f"  - {issue}", err=True)
    raise SystemExit(1)
```

- [ ] **Step 6: Commit**

```bash
git add src/bts/state/verify.py src/bts/cli.py tests/test_state_verify.py
git commit -m "feat(state): add 'bts state verify' drift detection"
```

---

## Completion criteria for Plan 03

- [ ] All tests pass: `uv run pytest tests/test_state_export.py tests/test_state_regenerate.py tests/test_state_verify.py -v`
- [ ] `bts state export` works on real Pi5 state (produces a valid initial-state.json or refuses with unresolved picks error)
- [ ] `bts state regenerate` works against the live Bluesky bot account and produces pick files
- [ ] `bts state verify` runs end-to-end and either reports clean or enumerates specific drift

**Next plan:** `04-scheduler.md` — Scheduler refactors for heartbeat, config extraction, and Bluesky password consolidation. Can be executed in parallel with this one.
