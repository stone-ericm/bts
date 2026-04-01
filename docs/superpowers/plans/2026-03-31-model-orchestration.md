# Model Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pi5 orchestrates daily BTS predictions across Mac -> Alienware -> cloud VPS cascade, with strategy logic extracted into a shared module.

**Architecture:** Workers run `bts predict-json` (date in, JSON out via SSH). Pi5 receives predictions, applies pick strategy, saves picks, posts to Bluesky, DMs on failure. LightGBM is an optional dependency so Pi5 can run pick logic without model deps.

**Tech Stack:** Python 3.12, Click, pandas, subprocess (SSH), tomllib (stdlib), urllib (Bluesky API)

---

### Task 1: Make LightGBM an Optional Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Move LightGBM and scikit-learn to optional dependencies**

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

[project.optional-dependencies]
model = [
    "lightgbm>=4.6.0",
    "scikit-learn>=1.8.0",
]
```

- [ ] **Step 2: Verify tests still pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model && UV_CACHE_DIR=/tmp/uv-cache uv run pytest -v`

Expected: All existing tests pass (they mock run_pipeline, never import lightgbm directly).

- [ ] **Step 3: Commit**

```bash
cd /Users/stone/projects/bts
git add pyproject.toml uv.lock
git commit -m "refactor: make LightGBM an optional dependency

Move lightgbm and scikit-learn to [project.optional-dependencies.model].
Pi5 can now install the package without model deps for orchestration.
Workers install with: uv sync --extra model"
```

---

### Task 2: Extract Strategy Module

**Files:**
- Create: `src/bts/strategy.py`
- Create: `tests/test_strategy.py`

- [ ] **Step 1: Write failing tests for select_pick**

```python
# tests/test_strategy.py
"""Tests for BTS pick strategy (densest bucket + override)."""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from bts.picks import Pick, DailyPick, save_pick


def _predictions(rows):
    """Build a predictions DataFrame from simplified row dicts."""
    defaults = {
        "batter_id": 100001,
        "team": "NYM",
        "lineup": 1,
        "pitcher_name": "Test Pitcher",
        "pitcher_id": 200001,
        "game_pk": 778899,
        "game_time": "2026-04-01T23:10:00Z",  # 7:10pm ET — prime window
        "p_hit_pa": 0.30,
        "flags": "",
    }
    full_rows = []
    for i, r in enumerate(rows):
        row = {**defaults, **r}
        row.setdefault("batter_name", f"Batter {i+1}")
        row.setdefault("p_game_hit", 0.75 - i * 0.02)
        full_rows.append(row)
    return pd.DataFrame(full_rows)


class TestSelectPick:
    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    def test_basic_pick(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Jacob Wilson", "p_game_hit": 0.763},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result is not None
        assert not result.locked
        assert result.daily.pick.batter_name == "Jacob Wilson"
        assert result.daily.pick.p_game_hit == 0.763

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    def test_double_down_when_threshold_met(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Wilson", "p_game_hit": 0.82, "game_pk": 778899},
            {"batter_name": "Mangum", "p_game_hit": 0.81, "game_pk": 778900},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result.daily.double_down is not None
        assert result.daily.double_down.batter_name == "Mangum"

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    def test_no_double_down_below_threshold(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Wilson", "p_game_hit": 0.75, "game_pk": 778899},
            {"batter_name": "Mangum", "p_game_hit": 0.70, "game_pk": 778900},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result.daily.double_down is None

    @patch("bts.strategy.get_game_statuses", return_value={778899: "F"})
    def test_locked_when_game_started(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        existing = DailyPick(
            date="2026-04-01",
            run_time="2026-04-01T15:00:00+00:00",
            pick=Pick(
                batter_name="Wilson", batter_id=100001, team="ATH",
                lineup_position=1, pitcher_name="Suarez", pitcher_id=200001,
                p_game_hit=0.76, flags=[], projected_lineup=False,
                game_pk=778899, game_time="2026-04-01T23:10:00Z",
            ),
            double_down=None, runner_up=None,
        )
        save_pick(existing, tmp_path)

        preds = _predictions([{"batter_name": "Wilson", "game_pk": 778899}])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result.locked
        assert result.daily.pick.batter_name == "Wilson"

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P"})
    def test_locked_when_already_posted(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        existing = DailyPick(
            date="2026-04-01",
            run_time="2026-04-01T15:00:00+00:00",
            pick=Pick(
                batter_name="Wilson", batter_id=100001, team="ATH",
                lineup_position=1, pitcher_name="Suarez", pitcher_id=200001,
                p_game_hit=0.76, flags=[], projected_lineup=False,
                game_pk=778899, game_time="2026-04-01T23:10:00Z",
            ),
            double_down=None, runner_up=None,
            bluesky_posted=True, bluesky_uri="at://did:plc:test/post/123",
        )
        save_pick(existing, tmp_path)

        preds = _predictions([{"batter_name": "Wilson", "game_pk": 778899}])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result.locked

    @patch("bts.strategy.get_game_statuses", return_value={778899: "F"})
    def test_all_games_started_no_prior_pick(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        preds = _predictions([{"game_pk": 778899}])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result is None

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    def test_runner_up_populated(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Wilson", "p_game_hit": 0.76, "game_pk": 778899},
            {"batter_name": "Mangum", "p_game_hit": 0.72, "game_pk": 778900},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result.daily.runner_up is not None
        assert result.daily.runner_up["batter_name"] == "Mangum"

    @patch("bts.strategy.get_game_statuses", return_value={
        778899: "P", 778900: "P", 778901: "P", 778902: "P",
    })
    def test_override_from_non_densest_window(self, mock_statuses, tmp_path):
        """A non-densest pick above 78% should override the densest window."""
        from bts.strategy import select_pick

        preds = _predictions([
            # Early game (before 4pm ET = before 20:00 UTC) with high probability
            {"batter_name": "Early Star", "p_game_hit": 0.85,
             "game_pk": 778899, "game_time": "2026-04-01T17:10:00Z"},  # 1:10pm ET
            # Three prime games (densest window)
            {"batter_name": "Prime 1", "p_game_hit": 0.74,
             "game_pk": 778900, "game_time": "2026-04-01T23:10:00Z"},
            {"batter_name": "Prime 2", "p_game_hit": 0.72,
             "game_pk": 778901, "game_time": "2026-04-01T23:40:00Z"},
            {"batter_name": "Prime 3", "p_game_hit": 0.70,
             "game_pk": 778902, "game_time": "2026-04-02T00:10:00Z"},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        # Early Star overrides because 85% > 78% threshold
        assert result.daily.pick.batter_name == "Early Star"

    @patch("bts.strategy.get_game_statuses", return_value={
        778899: "P", 778900: "P", 778901: "P", 778902: "P",
    })
    def test_no_override_below_threshold(self, mock_statuses, tmp_path):
        """A non-densest pick below 78% should NOT override."""
        from bts.strategy import select_pick

        preds = _predictions([
            # Early game below threshold
            {"batter_name": "Early OK", "p_game_hit": 0.77,
             "game_pk": 778899, "game_time": "2026-04-01T17:10:00Z"},
            # Three prime games (densest)
            {"batter_name": "Prime 1", "p_game_hit": 0.74,
             "game_pk": 778900, "game_time": "2026-04-01T23:10:00Z"},
            {"batter_name": "Prime 2", "p_game_hit": 0.72,
             "game_pk": 778901, "game_time": "2026-04-01T23:40:00Z"},
            {"batter_name": "Prime 3", "p_game_hit": 0.70,
             "game_pk": 778902, "game_time": "2026-04-02T00:10:00Z"},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        # Falls back to densest window — Prime 1
        assert result.daily.pick.batter_name == "Prime 1"

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P"})
    def test_empty_predictions(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        preds = pd.DataFrame()
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_strategy.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'bts.strategy'`

- [ ] **Step 3: Implement strategy.py**

```python
# src/bts/strategy.py
"""Pick strategy: densest bucket + asymmetric override.

Extracted from cli.py so both `bts run` (local) and the Pi5 orchestrator
share the same decision logic.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from bts.picks import (
    DailyPick, Pick, pick_from_row, load_pick, get_game_statuses,
)

OVERRIDE_THRESHOLD = 0.78
DOUBLE_THRESHOLD = 0.65


@dataclass
class PickResult:
    """Result of pick strategy.

    daily: the selected DailyPick (new or existing locked)
    locked: True if pick was already locked (game started or posted)
    """
    daily: DailyPick
    locked: bool = False


def _classify_et_hour(game_time_utc: str) -> int:
    """Convert UTC game time to ET hour (UTC - 4)."""
    try:
        utc = datetime.fromisoformat(str(game_time_utc).replace("Z", "+00:00"))
        return (utc - timedelta(hours=4)).hour
    except Exception:
        return 18  # default to prime


def _apply_densest_bucket(valid: pd.DataFrame) -> pd.DataFrame:
    """Apply densest bucket + override strategy.

    Returns the filtered DataFrame to pick from.
    """
    if "game_time" not in valid.columns:
        return valid

    valid = valid.copy()
    valid["_et_hour"] = valid["game_time"].apply(_classify_et_hour)

    early = valid[valid["_et_hour"] < 16]
    prime = valid[(valid["_et_hour"] >= 16) & (valid["_et_hour"] < 20)]
    west = valid[valid["_et_hour"] >= 20]

    buckets = {"early": early, "prime": prime, "west": west}
    densest_name = max(buckets, key=lambda k: len(buckets[k]))

    top_overall = valid.iloc[0]
    top_hour = top_overall["_et_hour"]
    top_window = "early" if top_hour < 16 else ("prime" if top_hour < 20 else "west")

    if top_window == densest_name:
        return buckets[densest_name]
    elif top_overall["p_game_hit"] > OVERRIDE_THRESHOLD:
        return valid  # override — top pick from non-densest beats threshold
    else:
        return buckets[densest_name]


def select_pick(
    predictions: pd.DataFrame,
    date: str,
    picks_dir: Path,
) -> PickResult | None:
    """Apply densest bucket + override strategy to predictions.

    Returns PickResult with the selected DailyPick, or None if there's
    nothing to pick (no games, all started, empty predictions).
    """
    if predictions.empty:
        return None

    current = load_pick(date, picks_dir)
    statuses = get_game_statuses(date)

    # Check if current pick is locked
    if current and (
        statuses.get(current.pick.game_pk) != "P" or current.bluesky_posted
    ):
        return PickResult(daily=current, locked=True)

    # Filter to games not yet started
    not_started = predictions["game_pk"].map(lambda pk: statuses.get(pk) == "P")
    available = predictions[not_started]

    if available.empty:
        if current:
            return PickResult(daily=current, locked=True)
        return None

    # Filter to valid predictions
    valid = available[available["p_game_hit"].notna()]
    if valid.empty:
        return None

    # Apply densest bucket + override
    valid = _apply_densest_bucket(valid)

    best_row = valid.iloc[0]
    new_pick = pick_from_row(best_row)

    # Double-down check
    double_pick = None
    if len(valid) >= 2:
        second_row = valid.iloc[1]
        p_both = best_row["p_game_hit"] * second_row["p_game_hit"]
        if p_both >= DOUBLE_THRESHOLD:
            double_pick = pick_from_row(second_row)

    # Runner-up
    runner_up = None
    if len(valid) >= 2:
        ru = valid.iloc[1]
        runner_up = {"batter_name": ru["batter_name"], "p_game_hit": float(ru["p_game_hit"])}

    daily = DailyPick(
        date=date,
        run_time=datetime.now(timezone.utc).isoformat(),
        pick=new_pick,
        double_down=double_pick,
        runner_up=runner_up,
        bluesky_posted=False,
        bluesky_uri=None,
    )

    return PickResult(daily=daily, locked=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_strategy.py -v`

Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/stone/projects/bts
git add src/bts/strategy.py tests/test_strategy.py
git commit -m "feat: extract pick strategy into shared module

Moves densest bucket + override logic from cli.py into strategy.py.
select_pick() takes predictions DataFrame + date + picks_dir, returns
PickResult with the DailyPick and locked status. Both bts run (local)
and the Pi5 orchestrator will use this same function."
```

---

### Task 3: Refactor `bts run` to Use Strategy Module

**Files:**
- Modify: `src/bts/cli.py:73-310`

- [ ] **Step 1: Rewrite the `run` command to use `select_pick`**

Replace the `run` function in `cli.py` (everything from `@cli.command()` for `run` through the end of the function) with:

```python
@cli.command()
@click.option("--date", required=True, help="Date to predict (YYYY-MM-DD)")
@click.option("--data-dir", default="data/processed", type=click.Path(), help="Processed data directory")
@click.option("--picks-dir", default="data/picks", type=click.Path(), help="Picks output directory")
@click.option("--models-dir", default="data/models", type=click.Path(), help="Cached models directory")
@click.option("--top", default=10, type=int, help="Number of ranked picks to show")
@click.option("--dry-run", is_flag=True, help="Print rankings only — don't save pick or post to Bluesky")
def run(date: str, data_dir: str, picks_dir: str, models_dir: str, top: int, dry_run: bool):
    """Run daily BTS automation: predict, save pick, post to Bluesky.

    Designed to run via cron at 11am, 4pm, and 7:30pm ET.
    Uses densest-bucket strategy with 78% override threshold.
    Use --dry-run to preview rankings without saving or posting.
    """
    import pandas as pd
    from datetime import datetime, timezone
    from bts.model.predict import run_pipeline, save_blend, load_blend
    from bts.picks import save_pick, load_streak
    from bts.posting import format_post, post_to_bluesky, should_post_now
    from bts.strategy import select_pick

    picks_path = Path(picks_dir)
    models_path = Path(models_dir)

    # Step 1: Run prediction pipeline (with model caching)
    click.echo(f"[{datetime.now(timezone.utc).strftime('%H:%M UTC')}] Running predictions for {date}...")
    cache_path = models_path / f"blend_{date}.pkl"
    cached_blend = None
    if cache_path.exists():
        click.echo(f"  Loading cached model from {cache_path}")
        cached_blend = load_blend(cache_path)

    try:
        predictions = run_pipeline(
            date, data_dir,
            cached_blend=cached_blend,
            save_blend_path=cache_path if not cached_blend else None,
        )
    except RuntimeError as e:
        click.echo(f"ERROR: {e}", err=True)
        return
    except Exception as e:
        click.echo(f"ERROR: Pipeline failed — {e}", err=True)
        return

    if predictions.empty:
        click.echo("No games found for this date.")
        return

    # Print ranked picks
    click.echo(f"\n{'='*80}")
    click.echo(f"BTS PICKS — {date}")
    click.echo(f"{'='*80}")
    click.echo(f"{'#':<4} {'Batter':<22} {'Team':<5} {'Pos':>3} {'vs Pitcher':<22} {'P(PA)':>6} {'P(Game)':>7}  {'Flags'}")
    click.echo(f"{'-'*80}")
    shown = 0
    for _, row in predictions.iterrows():
        if shown >= top:
            break
        if pd.isna(row.get("p_game_hit")):
            continue
        p_pa = row.get("p_hit_pa", row.get("p_game_hit", 0))
        click.echo(
            f"{shown+1:<4} {row['batter_name']:<22} {row['team']:<5} "
            f"{int(row.get('lineup', 0)):>3} {row['pitcher_name']:<22} "
            f"{p_pa:>5.1%} {row['p_game_hit']:>6.1%}  {row.get('flags', '')}"
        )
        shown += 1

    if dry_run:
        click.echo("\n  (--dry-run: not saving or posting)")
        return

    # Step 2: Apply strategy
    result = select_pick(predictions, date, picks_path)

    if result is None:
        click.echo("No valid picks available.")
        return

    if result.locked:
        reason = "already posted" if result.daily.bluesky_posted else "game started"
        click.echo(f"Pick locked: {result.daily.pick.batter_name} ({reason})")
        # Catch-up posting if needed
        if not result.daily.bluesky_posted:
            streak = load_streak(picks_path)
            text = format_post(
                result.daily.pick.batter_name, result.daily.pick.team,
                result.daily.pick.pitcher_name, result.daily.pick.p_game_hit, streak,
                result.daily.double_down.batter_name if result.daily.double_down else None,
                result.daily.double_down.p_game_hit if result.daily.double_down else None,
            )
            try:
                uri = post_to_bluesky(text)
                result.daily.bluesky_posted = True
                result.daily.bluesky_uri = uri
                save_pick(result.daily, picks_path)
                click.echo(f"  Posted to Bluesky (catch-up): {uri}")
            except Exception as e:
                click.echo(f"  Bluesky catch-up post failed: {e}", err=True)
        return

    # New or updated pick
    daily = result.daily
    click.echo(f"Pick: {daily.pick.batter_name} ({daily.pick.p_game_hit:.1%}) "
               f"vs {daily.pick.pitcher_name}")
    if daily.double_down:
        p_both = daily.pick.p_game_hit * daily.double_down.p_game_hit
        click.echo(f"  DOUBLE DOWN: + {daily.double_down.batter_name} "
                    f"({daily.double_down.p_game_hit:.1%}), P(both): {p_both:.1%}")

    save_pick(daily, picks_path)
    click.echo(f"  Saved to {picks_path / f'{date}.json'}")

    # Post to Bluesky if appropriate
    streak = load_streak(picks_path)
    if should_post_now(daily.pick.game_time, daily.bluesky_posted):
        text = format_post(
            daily.pick.batter_name, daily.pick.team, daily.pick.pitcher_name,
            daily.pick.p_game_hit, streak,
            daily.double_down.batter_name if daily.double_down else None,
            daily.double_down.p_game_hit if daily.double_down else None,
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

- [ ] **Step 2: Verify existing integration tests still pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli_integration.py -v`

Expected: All 7 integration tests PASS. The mocks for `get_game_statuses` and `run_pipeline` still apply because `select_pick` calls the same `bts.picks.get_game_statuses` function.

Note: If tests fail because the mock path changed (strategy imports get_game_statuses), add a mock for `bts.strategy.get_game_statuses` alongside the existing `bts.picks.get_game_statuses` mock.

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest -v`

Expected: All tests PASS (strategy tests + integration tests + all others).

- [ ] **Step 4: Commit**

```bash
cd /Users/stone/projects/bts
git add src/bts/cli.py
git commit -m "refactor: bts run uses strategy.select_pick

Replaces inline pick logic in cli.py with call to strategy.select_pick().
Run command is now: predict -> select_pick -> save -> post.
Behavior is identical — same densest bucket, override, double-down logic."
```

---

### Task 4: Add `bts predict-json` Command

**Files:**
- Modify: `src/bts/cli.py` (add new command)
- Create: `tests/test_predict_json.py`

- [ ] **Step 1: Write failing test for predict-json**

```python
# tests/test_predict_json.py
"""Tests for bts predict-json CLI command."""

import json
import pytest
from unittest.mock import patch
from click.testing import CliRunner

from bts.cli import cli


def _mock_predictions():
    """Build a mock predictions DataFrame."""
    import pandas as pd
    return pd.DataFrame([
        {
            "batter_name": "Jacob Wilson",
            "batter_id": 700363,
            "team": "ATH",
            "lineup": 1,
            "pitcher_name": "Jose Suarez",
            "pitcher_id": 660761,
            "p_game_hit": 0.763,
            "p_hit_pa": 0.312,
            "flags": "",
            "game_pk": 778899,
            "game_time": "2026-04-01T23:10:00Z",
        },
        {
            "batter_name": "Jake Mangum",
            "batter_id": 700100,
            "team": "NYM",
            "lineup": 2,
            "pitcher_name": "Logan Webb",
            "pitcher_id": 657277,
            "p_game_hit": 0.726,
            "p_hit_pa": 0.295,
            "flags": "PROJECTED",
            "game_pk": 778900,
            "game_time": "2026-04-01T23:10:00Z",
        },
    ])


class TestPredictJson:
    @patch("bts.model.predict.run_pipeline")
    def test_outputs_valid_json(self, mock_pipeline):
        mock_pipeline.return_value = _mock_predictions()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "predict-json", "--date", "2026-04-01",
            "--data-dir", "data/processed",
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2
        assert data[0]["batter_name"] == "Jacob Wilson"
        assert data[0]["p_game_hit"] == 0.763

    @patch("bts.model.predict.run_pipeline")
    def test_includes_all_required_fields(self, mock_pipeline):
        mock_pipeline.return_value = _mock_predictions()

        runner = CliRunner()
        result = runner.invoke(cli, ["predict-json", "--date", "2026-04-01"])
        data = json.loads(result.output)

        required = ["batter_name", "batter_id", "team", "lineup",
                     "pitcher_name", "pitcher_id", "game_pk",
                     "game_time", "p_game_hit", "flags"]
        for field in required:
            assert field in data[0], f"Missing field: {field}"

    @patch("bts.model.predict.run_pipeline")
    def test_empty_predictions(self, mock_pipeline):
        import pandas as pd
        mock_pipeline.return_value = pd.DataFrame()

        runner = CliRunner()
        result = runner.invoke(cli, ["predict-json", "--date", "2026-04-01"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []

    @patch("bts.model.predict.run_pipeline")
    def test_error_exits_nonzero(self, mock_pipeline):
        mock_pipeline.side_effect = RuntimeError("No data")

        runner = CliRunner()
        result = runner.invoke(cli, ["predict-json", "--date", "2026-04-01"])

        assert result.exit_code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_predict_json.py -v`

Expected: FAIL with `No such command 'predict-json'`

- [ ] **Step 3: Implement predict-json command**

Add to `src/bts/cli.py`, after the `run` command:

```python
@cli.command(name="predict-json")
@click.option("--date", required=True, help="Date to predict (YYYY-MM-DD)")
@click.option("--data-dir", default="data/processed", type=click.Path(), help="Processed data directory")
@click.option("--models-dir", default="data/models", type=click.Path(), help="Cached models directory")
def predict_json(date: str, data_dir: str, models_dir: str):
    """Run predictions and output JSON to stdout.

    Worker command for remote orchestration. Outputs a JSON array of
    ranked predictions. All log messages go to stderr.
    """
    import json as _json
    import sys
    from datetime import datetime, timezone
    from bts.model.predict import run_pipeline, save_blend, load_blend

    models_path = Path(models_dir)

    click.echo(
        f"[{datetime.now(timezone.utc).strftime('%H:%M UTC')}] "
        f"Running predictions for {date}...",
        err=True,
    )

    cache_path = models_path / f"blend_{date}.pkl"
    cached_blend = None
    if cache_path.exists():
        click.echo(f"  Loading cached model from {cache_path}", err=True)
        cached_blend = load_blend(cache_path)

    try:
        predictions = run_pipeline(
            date, data_dir,
            cached_blend=cached_blend,
            save_blend_path=cache_path if not cached_blend else None,
        )
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    if predictions.empty:
        click.echo("[]")
        return

    # Select columns needed by the orchestrator
    columns = [
        "batter_name", "batter_id", "team", "lineup",
        "pitcher_name", "pitcher_id", "game_pk", "game_time",
        "p_hit_pa", "p_game_hit", "flags",
    ]
    output_cols = [c for c in columns if c in predictions.columns]
    output = predictions[output_cols].to_dict(orient="records")

    # Clean up NaN/None for JSON serialization
    for row in output:
        for k, v in row.items():
            if isinstance(v, float) and (v != v):  # NaN check
                row[k] = None
            elif hasattr(v, 'item'):  # numpy scalar
                row[k] = v.item()

    click.echo(_json.dumps(output, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_predict_json.py -v`

Expected: All 4 tests PASS.

- [ ] **Step 5: Manual smoke test**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run bts predict-json --help`

Expected: Shows help text for predict-json command.

- [ ] **Step 6: Commit**

```bash
cd /Users/stone/projects/bts
git add src/bts/cli.py tests/test_predict_json.py
git commit -m "feat: add bts predict-json for remote orchestration

New CLI command outputs ranked predictions as JSON to stdout.
All log messages go to stderr. Exits non-zero on pipeline errors.
This is the worker interface — Pi5 calls it via SSH on compute machines."
```

---

### Task 5: Bluesky DM Notifications

**Files:**
- Create: `src/bts/dm.py`
- Create: `tests/test_dm.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dm.py
"""Tests for Bluesky DM notifications."""

import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

from bts.dm import send_dm, get_dm_password


class TestGetDmPassword:
    @patch("subprocess.run")
    def test_keychain_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="test-password\n")
        assert get_dm_password() == "test-password"

    @patch("subprocess.run")
    def test_env_fallback(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        with patch.dict("os.environ", {"BTS_BLUESKY_DM_PASSWORD": "env-pw"}):
            assert get_dm_password() == "env-pw"

    @patch("subprocess.run")
    def test_raises_when_not_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="DM password not found"):
                get_dm_password()


def _mock_urlopen_responses(responses):
    """Create a side_effect that returns successive mock responses."""
    mocks = []
    for resp in responses:
        m = MagicMock()
        m.read.return_value = json.dumps(resp).encode()
        mocks.append(m)
    return mocks


class TestSendDm:
    @patch("bts.dm.retry_urlopen")
    @patch("bts.dm.get_dm_password", return_value="test-password")
    def test_sends_dm_successfully(self, mock_pw, mock_urlopen):
        mock_urlopen.side_effect = _mock_urlopen_responses([
            # createSession
            {"accessJwt": "jwt-token", "did": "did:plc:bot"},
            # resolveHandle
            {"did": "did:plc:recipient"},
            # getConvoForMembers
            {"convo": {"id": "convo-123"}},
            # sendMessage
            {"id": "msg-456", "sentAt": "2026-04-01T12:00:00Z"},
        ])

        result = send_dm("stonehengee.bsky.social", "Test message")
        assert result == "msg-456"
        assert mock_urlopen.call_count == 4

    @patch("bts.dm.retry_urlopen")
    @patch("bts.dm.get_dm_password", return_value="test-password")
    def test_auth_failure_raises(self, mock_pw, mock_urlopen):
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            url="", code=401, msg="Unauthorized",
            hdrs=None, fp=BytesIO(b"bad auth"),
        )

        with pytest.raises(RuntimeError, match="DM auth failed"):
            send_dm("stonehengee.bsky.social", "Test")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_dm.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'bts.dm'`

- [ ] **Step 3: Implement dm.py**

```python
# src/bts/dm.py
"""Bluesky DM notifications for BTS orchestrator failures."""

import json
import os
import subprocess
from urllib.error import HTTPError
from urllib.request import Request

from bts.util import retry_urlopen

BSKY_HOST = "https://bsky.social/xrpc"
CHAT_PROXY = "did:web:api.bsky.chat#bsky_chat"
BOT_HANDLE = "beatthestreakbot.bsky.social"


def get_dm_password() -> str:
    """Get Bluesky DM app password from keychain or environment.

    Uses the DM-scoped password (separate from posting password).
    """
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", "claude-cli",
             "-s", "bluesky-bts-app-password-dm", "-w"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    password = os.environ.get("BTS_BLUESKY_DM_PASSWORD")
    if password:
        return password

    raise RuntimeError(
        "DM password not found. Set BTS_BLUESKY_DM_PASSWORD or add "
        "bluesky-bts-app-password-dm to keychain."
    )


def send_dm(recipient_handle: str, text: str) -> str:
    """Send a Bluesky DM to recipient_handle. Returns message ID.

    Steps: authenticate -> resolve handle -> get/create convo -> send.
    """
    password = get_dm_password()

    # Step 1: Authenticate
    auth_data = json.dumps({
        "identifier": BOT_HANDLE,
        "password": password,
    }).encode()
    req = Request(
        f"{BSKY_HOST}/com.atproto.server.createSession",
        data=auth_data,
        headers={"Content-Type": "application/json"},
    )
    try:
        session = json.loads(retry_urlopen(req, timeout=15).read())
    except HTTPError as e:
        if e.code == 401:
            raise RuntimeError("DM auth failed — check bluesky-bts-app-password-dm") from e
        raise RuntimeError(f"DM auth error (HTTP {e.code})") from e

    jwt = session["accessJwt"]

    # Step 2: Resolve recipient handle to DID
    req = Request(
        f"{BSKY_HOST}/com.atproto.identity.resolveHandle?handle={recipient_handle}",
        headers={"Authorization": f"Bearer {jwt}"},
    )
    target_did = json.loads(retry_urlopen(req, timeout=15).read())["did"]

    # Step 3: Get or create conversation
    req = Request(
        f"{BSKY_HOST}/chat.bsky.convo.getConvoForMembers?members={target_did}",
        headers={
            "Authorization": f"Bearer {jwt}",
            "atproto-proxy": CHAT_PROXY,
        },
    )
    convo_id = json.loads(retry_urlopen(req, timeout=15).read())["convo"]["id"]

    # Step 4: Send message
    msg_data = json.dumps({
        "convoId": convo_id,
        "message": {"text": text},
    }).encode()
    req = Request(
        f"{BSKY_HOST}/chat.bsky.convo.sendMessage",
        data=msg_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt}",
            "atproto-proxy": CHAT_PROXY,
        },
    )
    result = json.loads(retry_urlopen(req, timeout=15).read())
    return result["id"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_dm.py -v`

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/stone/projects/bts
git add src/bts/dm.py tests/test_dm.py
git commit -m "feat: Bluesky DM notifications for orchestrator failures

New dm.py module sends DMs via the AT Protocol chat API.
Uses separate app password (bluesky-bts-app-password-dm) with DM scope.
The orchestrator DMs @stonehengee when all compute tiers fail."
```

---

### Task 6: Orchestrator

**Files:**
- Create: `src/bts/orchestrator.py`
- Create: `tests/test_orchestrator.py`
- Modify: `src/bts/cli.py` (add `orchestrate` command)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orchestrator.py
"""Tests for Pi5 orchestrator cascade logic."""

import json
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from bts.orchestrator import ssh_predict, run_cascade, load_config


SAMPLE_PREDICTIONS = json.dumps([
    {
        "batter_name": "Jacob Wilson",
        "batter_id": 700363,
        "team": "ATH",
        "lineup": 1,
        "pitcher_name": "Jose Suarez",
        "pitcher_id": 660761,
        "game_pk": 778899,
        "game_time": "2026-04-01T23:10:00Z",
        "p_hit_pa": 0.312,
        "p_game_hit": 0.763,
        "flags": "",
    },
])


class TestSshPredict:
    @patch("subprocess.run")
    def test_success_returns_dataframe(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=SAMPLE_PREDICTIONS, stderr="Running..."
        )
        df = ssh_predict("mac", "/path/to/bts", "2026-04-01", timeout_sec=300)

        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["batter_name"] == "Jacob Wilson"

    @patch("subprocess.run")
    def test_ssh_failure_returns_none(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=300)
        df = ssh_predict("mac", "/path/to/bts", "2026-04-01", timeout_sec=300)
        assert df is None

    @patch("subprocess.run")
    def test_nonzero_exit_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="ERROR: No data"
        )
        df = ssh_predict("mac", "/path/to/bts", "2026-04-01", timeout_sec=300)
        assert df is None

    @patch("subprocess.run")
    def test_invalid_json_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="not json", stderr=""
        )
        df = ssh_predict("mac", "/path/to/bts", "2026-04-01", timeout_sec=300)
        assert df is None


class TestRunCascade:
    @patch("bts.orchestrator.ssh_predict")
    def test_first_tier_succeeds(self, mock_ssh):
        import pandas as pd
        mock_ssh.return_value = pd.DataFrame(json.loads(SAMPLE_PREDICTIONS))

        tiers = [
            {"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5},
        ]
        df, tier_name = run_cascade(tiers, "2026-04-01")

        assert tier_name == "mac"
        assert len(df) == 1

    @patch("bts.orchestrator.ssh_predict")
    def test_falls_through_to_second_tier(self, mock_ssh):
        import pandas as pd

        def side_effect(host, bts_dir, date, timeout_sec):
            if host == "mac":
                return None  # Mac failed
            return pd.DataFrame(json.loads(SAMPLE_PREDICTIONS))

        mock_ssh.side_effect = side_effect

        tiers = [
            {"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5},
            {"name": "alienware", "ssh_host": "alienware", "bts_dir": "/bts", "timeout_min": 10},
        ]
        df, tier_name = run_cascade(tiers, "2026-04-01")

        assert tier_name == "alienware"
        assert len(df) == 1

    @patch("bts.orchestrator.ssh_predict")
    def test_all_tiers_fail_returns_none(self, mock_ssh):
        mock_ssh.return_value = None

        tiers = [
            {"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5},
            {"name": "alienware", "ssh_host": "alienware", "bts_dir": "/bts", "timeout_min": 10},
        ]
        df, tier_name = run_cascade(tiers, "2026-04-01")

        assert df is None
        assert tier_name is None


class TestLoadConfig:
    def test_loads_toml(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[orchestrator]
picks_dir = "/home/bts/data/picks"

[bluesky]
dm_recipient = "stonehengee.bsky.social"

[[tiers]]
name = "mac"
ssh_host = "macbook-pro.local"
bts_dir = "/Users/stone/projects/bts"
timeout_min = 5

[[tiers]]
name = "alienware"
ssh_host = "alienware"
bts_dir = "/c/Users/stone/projects/bts"
timeout_min = 10
""")
        config = load_config(config_path)

        assert len(config["tiers"]) == 2
        assert config["tiers"][0]["name"] == "mac"
        assert config["orchestrator"]["picks_dir"] == "/home/bts/data/picks"
        assert config["bluesky"]["dm_recipient"] == "stonehengee.bsky.social"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'bts.orchestrator'`

- [ ] **Step 3: Implement orchestrator.py**

```python
# src/bts/orchestrator.py
"""Pi5 orchestrator: cascade model runs across compute machines via SSH."""

import json
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_config(path: Path) -> dict:
    """Load orchestrator config from TOML file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def ssh_predict(
    ssh_host: str,
    bts_dir: str,
    date: str,
    timeout_sec: int = 300,
) -> pd.DataFrame | None:
    """Run bts predict-json on a remote machine via SSH.

    Returns predictions DataFrame on success, None on any failure.
    """
    cmd = (
        f"cd {bts_dir} && "
        f"UV_CACHE_DIR=/tmp/uv-cache uv run bts predict-json --date {date}"
    )
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
             ssh_host, cmd],
            capture_output=True, text=True, timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        print(f"  [{ssh_host}] Timeout after {timeout_sec}s", file=sys.stderr)
        return None
    except OSError as e:
        print(f"  [{ssh_host}] SSH error: {e}", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"  [{ssh_host}] Exit code {result.returncode}", file=sys.stderr)
        if result.stderr:
            # Print last 5 lines of stderr for context
            lines = result.stderr.strip().split("\n")
            for line in lines[-5:]:
                print(f"    {line}", file=sys.stderr)
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  [{ssh_host}] Invalid JSON output", file=sys.stderr)
        return None

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def run_cascade(
    tiers: list[dict],
    date: str,
) -> tuple[pd.DataFrame | None, str | None]:
    """Try each tier in order until one succeeds.

    Returns (predictions_df, tier_name) or (None, None) if all fail.
    """
    for tier in tiers:
        name = tier["name"]
        print(f"Trying {name}...", file=sys.stderr)
        df = ssh_predict(
            tier["ssh_host"],
            tier["bts_dir"],
            date,
            timeout_sec=tier["timeout_min"] * 60,
        )
        if df is not None:
            print(f"  [{name}] Success — {len(df)} predictions", file=sys.stderr)
            return df, name

    return None, None


def orchestrate(config_path: Path, date: str) -> bool:
    """Run the full orchestration: cascade -> strategy -> save -> post.

    Returns True if a pick was made, False otherwise.
    """
    from bts.dm import send_dm
    from bts.picks import save_pick, load_streak
    from bts.posting import format_post, post_to_bluesky, should_post_now
    from bts.strategy import select_pick

    config = load_config(config_path)
    picks_dir = Path(config["orchestrator"]["picks_dir"])
    dm_recipient = config["bluesky"]["dm_recipient"]

    # Run cascade
    predictions, tier_name = run_cascade(config["tiers"], date)

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

    # Apply strategy
    result = select_pick(predictions, date, picks_dir)

    if result is None:
        print("No valid picks available.", file=sys.stderr)
        return False

    if result.locked:
        print(f"Pick locked: {result.daily.pick.batter_name}", file=sys.stderr)
        # Catch-up posting
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

    # New pick
    daily = result.daily
    save_pick(daily, picks_dir)
    print(
        f"Pick ({tier_name}): {daily.pick.batter_name} "
        f"({daily.pick.p_game_hit:.1%})",
        file=sys.stderr,
    )

    # Post to Bluesky
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

- [ ] **Step 4: Add `bts orchestrate` CLI command**

Add to `src/bts/cli.py`:

```python
@cli.command()
@click.option("--date", required=True, help="Date to predict (YYYY-MM-DD)")
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True), help="Orchestrator config TOML file")
def orchestrate(date: str, config_path: str):
    """Orchestrate predictions across compute tiers (Pi5 command).

    Cascades through SSH tiers (Mac -> Alienware -> Cloud), applies
    pick strategy, saves pick, posts to Bluesky. DMs on total failure.
    """
    from bts.orchestrator import orchestrate as _orchestrate

    success = _orchestrate(Path(config_path), date)
    if not success:
        raise SystemExit(1)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py -v`

Expected: All 8 tests PASS.

- [ ] **Step 6: Run full test suite**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest -v`

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
cd /Users/stone/projects/bts
git add src/bts/orchestrator.py tests/test_orchestrator.py src/bts/cli.py
git commit -m "feat: Pi5 orchestrator with SSH cascade

Cascades bts predict-json across Mac -> Alienware -> Cloud via SSH.
On success, applies strategy, saves pick, posts to Bluesky.
On total failure, DMs @stonehengee. Config via TOML file.
New CLI: bts orchestrate --date X --config path/to/config.toml"
```

---

### Task 7: Config Template and Cron Setup

**Files:**
- Create: `config/orchestrator.example.toml`
- Create: `scripts/cron-setup-pi5.sh`

- [ ] **Step 1: Create example config**

```toml
# config/orchestrator.example.toml
# Copy to ~/.bts-orchestrator.toml on Pi5 and adjust paths.

[orchestrator]
picks_dir = "/home/stonehengee/projects/bts/data/picks"

[bluesky]
dm_recipient = "stonehengee.bsky.social"

[[tiers]]
name = "mac"
ssh_host = "macbook-pro.local"
bts_dir = "/Users/stone/projects/bts"
timeout_min = 5

[[tiers]]
name = "alienware"
ssh_host = "alienware"
bts_dir = "/c/Users/stone/projects/bts"
timeout_min = 10

# Uncomment when cloud VPS is set up:
# [[tiers]]
# name = "cloud"
# ssh_host = "bts-cloud"
# bts_dir = "/home/bts/projects/bts"
# timeout_min = 15
```

- [ ] **Step 2: Create Pi5 cron setup script**

```bash
#!/usr/bin/env bash
# BTS orchestrator cron setup for Pi5.
# All times are in system timezone (ET assumed).
#
# Schedule:
#   11:00 AM ET — Early game check
#   4:00 PM ET  — Prime time run
#   7:30 PM ET  — West coast run
#   1:00 AM ET  — Check yesterday's results
#
# Usage: bash scripts/cron-setup-pi5.sh [install|show|remove]

set -euo pipefail

BTS_DIR="$HOME/projects/bts"
CONFIG="$HOME/.bts-orchestrator.toml"
LOG_DIR="$BTS_DIR/data/picks"
UV_PREFIX="UV_CACHE_DIR=/tmp/uv-cache"
MARKER="# BTS-ORCHESTRATOR"

# Cross-platform yesterday date
if date -v-1d >/dev/null 2>&1; then
    YESTERDAY_CMD='$(date -v-1d +\%Y-\%m-\%d)'  # macOS
else
    YESTERDAY_CMD='$(date -d "yesterday" +\%Y-\%m-\%d)'  # Linux
fi

CRON_LINES="$MARKER
0 11 * * * cd $BTS_DIR && $UV_PREFIX uv run bts orchestrate --date \$(date +\%Y-\%m-\%d) --config $CONFIG >> $LOG_DIR/orchestrator.log 2>&1 $MARKER
0 16 * * * cd $BTS_DIR && $UV_PREFIX uv run bts orchestrate --date \$(date +\%Y-\%m-\%d) --config $CONFIG >> $LOG_DIR/orchestrator.log 2>&1 $MARKER
30 19 * * * cd $BTS_DIR && $UV_PREFIX uv run bts orchestrate --date \$(date +\%Y-\%m-\%d) --config $CONFIG >> $LOG_DIR/orchestrator.log 2>&1 $MARKER
0 1 * * * cd $BTS_DIR && $UV_PREFIX uv run bts check-results --date $YESTERDAY_CMD >> $LOG_DIR/orchestrator.log 2>&1 $MARKER"

case "${1:-show}" in
    install)
        if [ ! -f "$CONFIG" ]; then
            echo "ERROR: Config not found at $CONFIG"
            echo "Copy config/orchestrator.example.toml to $CONFIG first."
            exit 1
        fi
        mkdir -p "$LOG_DIR"
        (crontab -l 2>/dev/null | grep -v "$MARKER"; echo "$CRON_LINES") | crontab -
        echo "Installed BTS orchestrator cron jobs. Verify with: crontab -l"
        ;;
    show)
        echo "Current BTS orchestrator cron entries:"
        crontab -l 2>/dev/null | grep "$MARKER" || echo "(none)"
        echo ""
        echo "Would install:"
        echo "$CRON_LINES"
        ;;
    remove)
        crontab -l 2>/dev/null | grep -v "$MARKER" | crontab -
        echo "Removed BTS orchestrator cron jobs."
        ;;
    *)
        echo "Usage: $0 [install|show|remove]"
        exit 1
        ;;
esac
```

- [ ] **Step 3: Commit**

```bash
cd /Users/stone/projects/bts
chmod +x scripts/cron-setup-pi5.sh
git add config/orchestrator.example.toml scripts/cron-setup-pi5.sh
git commit -m "feat: orchestrator config template and Pi5 cron setup

Example TOML config with Mac + Alienware tiers (cloud commented out).
Pi5 cron script runs bts orchestrate 3x daily + check-results at 1am."
```

---

## Post-Implementation Verification

After all tasks are complete:

- [ ] Run full test suite: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -v`
- [ ] Verify `bts run` still works locally: `UV_CACHE_DIR=/tmp/uv-cache uv run bts run --date 2026-03-31 --dry-run`
- [ ] Verify `bts predict-json` outputs clean JSON: `UV_CACHE_DIR=/tmp/uv-cache uv run bts predict-json --date 2026-03-31 2>/dev/null | python -m json.tool | head -20`
- [ ] Verify `bts orchestrate --help` shows expected options
- [ ] Test SSH predict from Mac to itself: `ssh localhost "cd ~/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run bts predict-json --date 2026-03-31" 2>/dev/null | python -m json.tool | head -5`

## Deployment Steps (manual, after code is merged)

1. **Pi5**: Clone repo, `uv sync`, copy config, install cron, store Bluesky passwords
2. **Alienware**: Clone repo, `uv sync --extra model`, rsync parquets from Mac
3. **Cloud VPS**: Research provider, provision, clone repo, stage data
4. **Verify**: Pi5 -> Mac SSH predict works, Pi5 -> Alienware SSH predict works
5. **Go live**: Install Pi5 cron, remove Mac cron
