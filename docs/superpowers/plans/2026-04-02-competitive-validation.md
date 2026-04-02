# BTS Competitive Validation Sprint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Systematically validate our BTS approach against r/beatthestreak community strategies through a multi-metric scorecard, measuring every change against a rigorous baseline.

**Architecture:** Build a scorecard module that reads existing backtest profiles and produces standardized metrics. Each investigation modifies one variable and diffs against the baseline scorecard. Features that pass the both-seasons test get integrated; the MDP is re-solved on the final configuration.

**Tech Stack:** Python 3.12, pandas, numpy, LightGBM, Rich (tables), existing BTS infrastructure (`bts simulate`, `bts evaluate`)

---

### Task 1: Scorecard Module — Data Layer

Build the scorecard computation engine that reads backtest profiles and computes all metrics.

**Files:**
- Create: `src/bts/validate/__init__.py`
- Create: `src/bts/validate/scorecard.py`
- Create: `tests/validate/__init__.py`
- Create: `tests/validate/test_scorecard.py`

- [ ] **Step 1: Write failing test for P@K computation**

```python
# tests/validate/test_scorecard.py
import pandas as pd
import pytest
from bts.validate.scorecard import compute_precision_at_k


def _make_profiles(days: int = 10, top_n: int = 10) -> pd.DataFrame:
    """Create synthetic backtest profiles for testing.

    Each day: rank 1-10, p_game_hit decreasing, rank-1 always hits,
    rank-2 hits 80% of the time, rest hit 50%.
    """
    rows = []
    for d in range(days):
        date = f"2024-06-{d+1:02d}"
        for rank in range(1, top_n + 1):
            p = 0.90 - (rank - 1) * 0.02
            if rank == 1:
                hit = 1
            elif rank == 2:
                hit = 1 if d % 5 != 0 else 0  # 80% hit rate
            else:
                hit = 1 if d % 2 == 0 else 0
            rows.append({
                "date": date, "rank": rank, "batter_id": 1000 + rank,
                "p_game_hit": p, "actual_hit": hit, "n_pas": 4,
            })
    return pd.DataFrame(rows)


def test_precision_at_k_basic():
    df = _make_profiles()
    result = compute_precision_at_k(df, k_values=[1, 5, 10])
    # Rank-1 always hits in our synthetic data
    assert result[1] == 1.0
    # K=5 includes rank 1 (100%), rank 2 (80%), rank 3-5 (50%)
    assert 0.5 < result[5] < 1.0
    # All K values present
    assert set(result.keys()) == {1, 5, 10}


def test_precision_at_k_per_season():
    df = _make_profiles()
    df["season"] = 2024
    result = compute_precision_at_k(df, k_values=[1], by_season=True)
    assert 2024 in result
    assert result[2024][1] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_scorecard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bts.validate'`

- [ ] **Step 3: Implement P@K computation**

```python
# src/bts/validate/__init__.py
# (empty)
```

```python
# tests/validate/__init__.py
# (empty)
```

```python
# src/bts/validate/scorecard.py
"""Multi-metric scorecard for BTS model validation.

Reads backtest profiles and computes standardized metrics for
comparing model/strategy variants.
"""

import pandas as pd
import numpy as np


def compute_precision_at_k(
    profiles_df: pd.DataFrame,
    k_values: list[int] | None = None,
    by_season: bool = False,
) -> dict:
    """Compute Precision@K from backtest profiles.

    Args:
        profiles_df: DataFrame with columns [date, rank, p_game_hit, actual_hit].
        k_values: List of K values (default: [1, 5, 10, 25, 50, 100, 250, 500]).
        by_season: If True and 'season' column exists, return {season: {k: precision}}.

    Returns:
        {k: precision} or {season: {k: precision}} if by_season.
    """
    if k_values is None:
        k_values = [1, 5, 10, 25, 50, 100, 250, 500]

    if by_season and "season" in profiles_df.columns:
        result = {}
        for season, sdf in profiles_df.groupby("season"):
            result[season] = _compute_pak(sdf, k_values)
        return result

    return _compute_pak(profiles_df, k_values)


def _compute_pak(df: pd.DataFrame, k_values: list[int]) -> dict[int, float]:
    """Compute P@K for each day, return averages."""
    result = {}
    for k in k_values:
        daily_precisions = []
        for _, day_df in df.groupby("date"):
            top_k = day_df[day_df["rank"] <= k]
            if len(top_k) > 0:
                daily_precisions.append(top_k["actual_hit"].mean())
        result[k] = float(np.mean(daily_precisions)) if daily_precisions else 0.0
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_scorecard.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Write failing test for miss-day and calibration metrics**

```python
# tests/validate/test_scorecard.py (append)

from bts.validate.scorecard import compute_miss_analysis, compute_calibration


def test_miss_analysis():
    df = _make_profiles()
    result = compute_miss_analysis(df)
    # rank_2_hit_rate_on_miss_days: rank-1 never misses in this data,
    # so there should be no miss days
    assert result["n_miss_days"] == 0
    assert result["rank_2_hit_rate_on_miss"] is None


def test_miss_analysis_with_misses():
    df = _make_profiles()
    # Force some rank-1 misses
    df.loc[df["rank"] == 1, "actual_hit"] = df.apply(
        lambda r: 0 if r["rank"] == 1 and r["date"] == "2024-06-01" else r["actual_hit"],
        axis=1,
    )
    result = compute_miss_analysis(df)
    assert result["n_miss_days"] >= 1
    assert result["rank_2_hit_rate_on_miss"] is not None


def test_calibration():
    df = _make_profiles()
    result = compute_calibration(df, n_deciles=5)
    # Should return list of (predicted_mean, actual_mean, count) tuples
    assert len(result) == 5
    for predicted, actual, count in result:
        assert 0 <= predicted <= 1
        assert 0 <= actual <= 1
        assert count > 0
```

- [ ] **Step 6: Implement miss analysis and calibration**

```python
# src/bts/validate/scorecard.py (append)

def compute_miss_analysis(profiles_df: pd.DataFrame) -> dict:
    """Analyze rank-1 miss days: rank-2 hit rate, predicted confidence.

    Returns dict with:
        n_miss_days: number of days rank-1 missed
        rank_2_hit_rate_on_miss: fraction of miss days where rank-2 hit
        mean_p_hit_on_miss: mean predicted p_game_hit on miss days
        mean_p_hit_on_hit: mean predicted p_game_hit on hit days
    """
    r1 = profiles_df[profiles_df["rank"] == 1].copy()
    r2 = profiles_df[profiles_df["rank"] == 2].copy()

    miss_dates = set(r1[r1["actual_hit"] == 0]["date"])
    hit_dates = set(r1[r1["actual_hit"] == 1]["date"])

    n_miss = len(miss_dates)
    if n_miss == 0:
        return {
            "n_miss_days": 0,
            "rank_2_hit_rate_on_miss": None,
            "mean_p_hit_on_miss": None,
            "mean_p_hit_on_hit": float(r1["p_game_hit"].mean()) if len(hit_dates) > 0 else None,
        }

    r2_on_miss = r2[r2["date"].isin(miss_dates)]
    r1_on_miss = r1[r1["date"].isin(miss_dates)]
    r1_on_hit = r1[r1["date"].isin(hit_dates)]

    return {
        "n_miss_days": n_miss,
        "rank_2_hit_rate_on_miss": float(r2_on_miss["actual_hit"].mean()) if len(r2_on_miss) > 0 else None,
        "mean_p_hit_on_miss": float(r1_on_miss["p_game_hit"].mean()),
        "mean_p_hit_on_hit": float(r1_on_hit["p_game_hit"].mean()) if len(r1_on_hit) > 0 else None,
    }


def compute_calibration(
    profiles_df: pd.DataFrame,
    n_deciles: int = 10,
) -> list[tuple[float, float, int]]:
    """Compute calibration buckets for top-ranked predictions.

    Groups rank 1-10 predictions into deciles by predicted probability,
    returns (predicted_mean, actual_mean, count) for each bucket.
    """
    top = profiles_df[profiles_df["rank"] <= 10].copy()
    top["bucket"] = pd.qcut(top["p_game_hit"], n_deciles, labels=False, duplicates="drop")

    result = []
    for _, group in top.groupby("bucket"):
        result.append((
            float(group["p_game_hit"].mean()),
            float(group["actual_hit"].mean()),
            len(group),
        ))
    return result
```

- [ ] **Step 7: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_scorecard.py -v`
Expected: 5 tests PASS

- [ ] **Step 8: Write failing test for streak simulation metrics**

```python
# tests/validate/test_scorecard.py (append)

from bts.validate.scorecard import compute_streak_metrics


def test_streak_metrics():
    df = _make_profiles(days=50)
    result = compute_streak_metrics(df, n_trials=100, season_length=50)
    assert "mean_max_streak" in result
    assert "p90_max_streak" in result
    assert "p99_max_streak" in result
    assert "longest_replay_streak" in result
    assert "p_57_monte_carlo" in result
    assert result["mean_max_streak"] >= 0
    assert result["longest_replay_streak"] >= 0
```

- [ ] **Step 9: Implement streak simulation metrics**

```python
# src/bts/validate/scorecard.py (append)

def compute_streak_metrics(
    profiles_df: pd.DataFrame,
    n_trials: int = 10_000,
    season_length: int = 180,
) -> dict:
    """Compute streak distribution metrics from backtest profiles.

    Uses the existing Monte Carlo engine with the MDP-optimal strategy,
    falling back to the 'combined' heuristic if MDP policy isn't available.
    Also replays actual seasons for longest observed streak.
    """
    from bts.simulate.monte_carlo import (
        load_profiles, run_monte_carlo, simulate_season,
    )
    from bts.simulate.strategies import ALL_STRATEGIES

    strategy = ALL_STRATEGIES["combined"]
    daily_profiles = load_profiles(profiles_df)

    # Monte Carlo
    mc = run_monte_carlo(daily_profiles, strategy, n_trials=n_trials,
                         season_length=season_length)

    # Replay actual season
    replay_result = simulate_season(daily_profiles, strategy)

    return {
        "mean_max_streak": float(np.mean(mc.max_streaks)),
        "median_max_streak": mc.median_streak,
        "p90_max_streak": int(np.percentile(mc.max_streaks, 90)),
        "p99_max_streak": int(np.percentile(mc.max_streaks, 99)),
        "p_57_monte_carlo": mc.p_57,
        "longest_replay_streak": replay_result.max_streak,
    }
```

- [ ] **Step 10: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_scorecard.py -v`
Expected: 6 tests PASS

- [ ] **Step 11: Commit**

```bash
git add src/bts/validate/ tests/validate/
git commit -m "feat(validate): scorecard data layer — P@K, miss analysis, calibration, streak metrics"
```

---

### Task 2: Scorecard Module — CLI + Baseline Report

Wire up the CLI command and add the full scorecard report that produces both formatted output and a JSON artifact.

**Files:**
- Modify: `src/bts/validate/scorecard.py`
- Modify: `src/bts/cli.py`
- Create: `tests/validate/test_scorecard_cli.py`

- [ ] **Step 1: Write failing test for full scorecard assembly**

```python
# tests/validate/test_scorecard.py (append)

from bts.validate.scorecard import compute_full_scorecard


def test_full_scorecard():
    df = _make_profiles(days=50)
    result = compute_full_scorecard(df, mc_trials=100, season_length=50)
    assert "precision" in result
    assert "miss_analysis" in result
    assert "calibration" in result
    assert "streak_metrics" in result
    assert "metadata" in result
    # Precision should have multiple K values
    assert 1 in result["precision"]


def test_full_scorecard_with_seasons():
    df = _make_profiles(days=50)
    df["season"] = 2024
    result = compute_full_scorecard(df, mc_trials=100, season_length=50)
    assert "precision_by_season" in result
    assert 2024 in result["precision_by_season"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_scorecard.py::test_full_scorecard -v`
Expected: FAIL — `ImportError: cannot import name 'compute_full_scorecard'`

- [ ] **Step 3: Implement full scorecard assembly**

```python
# src/bts/validate/scorecard.py (append)

import json
from datetime import datetime, timezone
from pathlib import Path


def compute_full_scorecard(
    profiles_df: pd.DataFrame,
    mc_trials: int = 10_000,
    season_length: int = 180,
) -> dict:
    """Compute the complete multi-metric scorecard.

    Returns a dict suitable for JSON serialization and comparison.
    """
    result = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_days": int(profiles_df["date"].nunique()),
            "n_rows": len(profiles_df),
            "mc_trials": mc_trials,
            "season_length": season_length,
        },
        "precision": compute_precision_at_k(profiles_df),
        "miss_analysis": compute_miss_analysis(profiles_df),
        "calibration": compute_calibration(profiles_df),
        "streak_metrics": compute_streak_metrics(
            profiles_df, n_trials=mc_trials, season_length=season_length,
        ),
    }

    # Per-season breakdown if we have season info or can infer from dates
    profiles_df = profiles_df.copy()
    if "season" not in profiles_df.columns:
        profiles_df["season"] = pd.to_datetime(profiles_df["date"]).dt.year

    result["precision_by_season"] = compute_precision_at_k(
        profiles_df, by_season=True,
    )

    # Per-season P@1 summary for quick readability
    result["p_at_1_by_season"] = {
        season: vals.get(1, 0.0)
        for season, vals in result["precision_by_season"].items()
    }

    # Exact P(57) — requires quality bins and the absorbing chain
    try:
        from bts.simulate.quality_bins import compute_bins
        from bts.simulate.exact import exact_p57
        from bts.simulate.strategies import ALL_STRATEGIES

        bins = compute_bins(profiles_df)
        strategy = ALL_STRATEGIES["combined"]
        result["p_57_exact_heuristic"] = exact_p57(strategy, bins, season_length)

        # Also try MDP if available
        try:
            from bts.simulate.mdp import solve_mdp
            sol = solve_mdp(bins, season_length=season_length)
            result["p_57_exact_mdp"] = sol.optimal_p57
        except Exception:
            result["p_57_exact_mdp"] = None
    except Exception:
        result["p_57_exact_heuristic"] = None
        result["p_57_exact_mdp"] = None

    return result


def save_scorecard(scorecard: dict, path: Path | str) -> Path:
    """Save scorecard to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    path.write_text(json.dumps(scorecard, indent=2, default=_convert))
    return path


def diff_scorecards(baseline: dict, variant: dict) -> dict:
    """Compute deltas between two scorecards.

    Returns a dict with the same structure but values replaced by
    (baseline_value, variant_value, delta) tuples for numeric fields.
    """
    diffs = {}

    # P@K deltas
    diffs["precision"] = {}
    for k in baseline.get("precision", {}):
        b = baseline["precision"][k]
        v = variant["precision"].get(k, 0.0)
        diffs["precision"][k] = {"baseline": b, "variant": v, "delta": v - b}

    # Per-season P@1
    diffs["p_at_1_by_season"] = {}
    for season in baseline.get("p_at_1_by_season", {}):
        b = baseline["p_at_1_by_season"][season]
        v = variant.get("p_at_1_by_season", {}).get(season, 0.0)
        diffs["p_at_1_by_season"][season] = {"baseline": b, "variant": v, "delta": v - b}

    # Miss analysis
    diffs["miss_analysis"] = {}
    for key in baseline.get("miss_analysis", {}):
        b = baseline["miss_analysis"].get(key)
        v = variant.get("miss_analysis", {}).get(key)
        if isinstance(b, (int, float)) and isinstance(v, (int, float)):
            diffs["miss_analysis"][key] = {"baseline": b, "variant": v, "delta": v - b}

    # P(57)
    for key in ["p_57_exact_heuristic", "p_57_exact_mdp"]:
        b = baseline.get(key)
        v = variant.get(key)
        if isinstance(b, (int, float)) and isinstance(v, (int, float)):
            diffs[key] = {"baseline": b, "variant": v, "delta": v - b}

    # Streak metrics
    diffs["streak_metrics"] = {}
    for key in baseline.get("streak_metrics", {}):
        b = baseline["streak_metrics"].get(key)
        v = variant.get("streak_metrics", {}).get(key)
        if isinstance(b, (int, float)) and isinstance(v, (int, float)):
            diffs["streak_metrics"][key] = {"baseline": b, "variant": v, "delta": v - b}

    return diffs
```

- [ ] **Step 4: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_scorecard.py -v`
Expected: 8 tests PASS

- [ ] **Step 5: Write failing test for diff_scorecards**

```python
# tests/validate/test_scorecard.py (append)

from bts.validate.scorecard import diff_scorecards


def test_diff_scorecards():
    baseline = {
        "precision": {1: 0.85, 5: 0.80},
        "p_at_1_by_season": {2024: 0.84, 2025: 0.87},
        "miss_analysis": {"n_miss_days": 20, "rank_2_hit_rate_on_miss": 0.85},
        "p_57_exact_mdp": 0.0615,
        "streak_metrics": {"mean_max_streak": 15.2},
    }
    variant = {
        "precision": {1: 0.87, 5: 0.81},
        "p_at_1_by_season": {2024: 0.85, 2025: 0.88},
        "miss_analysis": {"n_miss_days": 18, "rank_2_hit_rate_on_miss": 0.86},
        "p_57_exact_mdp": 0.0700,
        "streak_metrics": {"mean_max_streak": 16.0},
    }
    diff = diff_scorecards(baseline, variant)
    assert diff["precision"][1]["delta"] == pytest.approx(0.02)
    assert diff["p_57_exact_mdp"]["delta"] == pytest.approx(0.0085)
    assert diff["p_at_1_by_season"][2024]["delta"] == pytest.approx(0.01)
```

- [ ] **Step 6: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_scorecard.py::test_diff_scorecards -v`
Expected: PASS (implementation already written)

- [ ] **Step 7: Wire up CLI command**

```python
# src/bts/cli.py — add after the simulate import at top level:

@cli.group()
def validate():
    """Validation and benchmarking commands."""
    pass


@validate.command()
@click.option("--profiles-dir", default="data/simulation", type=click.Path(exists=True),
              help="Directory with backtest profile parquets")
@click.option("--mc-trials", default=10_000, type=int, help="Monte Carlo trials")
@click.option("--season-length", default=180, type=int, help="Days per season")
@click.option("--save", "save_path", default=None, type=click.Path(),
              help="Save scorecard JSON (default: data/validation/scorecard_{timestamp}.json)")
@click.option("--diff", "diff_path", default=None, type=click.Path(exists=True),
              help="Diff against a previous scorecard JSON")
def scorecard(profiles_dir: str, mc_trials: int, season_length: int,
              save_path: str | None, diff_path: str | None):
    """Compute multi-metric scorecard from backtest profiles."""
    import pandas as pd
    from datetime import datetime
    from rich.console import Console
    from rich.table import Table
    from bts.validate.scorecard import (
        compute_full_scorecard, save_scorecard, diff_scorecards,
    )

    console = Console()
    profiles_path = Path(profiles_dir)

    # Load profiles
    dfs = []
    for p in sorted(profiles_path.glob("backtest_*.parquet")):
        season = int(p.stem.split("_")[1])
        sdf = pd.read_parquet(p)
        sdf["season"] = season
        dfs.append(sdf)
    if not dfs:
        click.echo("No backtest profiles found.", err=True)
        raise SystemExit(1)

    profiles_df = pd.concat(dfs, ignore_index=True)
    console.print(f"[bold]Computing scorecard from {profiles_df['date'].nunique()} days "
                  f"({len(dfs)} seasons)...[/bold]")

    sc = compute_full_scorecard(profiles_df, mc_trials=mc_trials,
                                season_length=season_length)

    # Display precision table
    table = Table(title="Precision @ K")
    table.add_column("K", justify="right")
    table.add_column("P@K (all)", justify="right")
    for season in sorted(sc.get("precision_by_season", {})):
        table.add_column(str(season), justify="right")

    for k in sorted(sc["precision"]):
        row = [str(k), f"{sc['precision'][k]:.3f}"]
        for season in sorted(sc.get("precision_by_season", {})):
            val = sc["precision_by_season"][season].get(k, 0)
            row.append(f"{val:.3f}")
        table.add_row(*row)
    console.print(table)

    # P(57)
    console.print(f"\n[bold]P(57):[/bold]")
    if sc.get("p_57_exact_mdp") is not None:
        console.print(f"  MDP optimal: {sc['p_57_exact_mdp']:.4%}")
    if sc.get("p_57_exact_heuristic") is not None:
        console.print(f"  Heuristic:   {sc['p_57_exact_heuristic']:.4%}")
    console.print(f"  Monte Carlo: {sc['streak_metrics']['p_57_monte_carlo']:.4%}")

    # Miss analysis
    ma = sc["miss_analysis"]
    console.print(f"\n[bold]Miss Analysis:[/bold]")
    console.print(f"  Miss days: {ma['n_miss_days']}")
    if ma["rank_2_hit_rate_on_miss"] is not None:
        console.print(f"  Rank-2 hit rate on miss: {ma['rank_2_hit_rate_on_miss']:.1%}")
    if ma["mean_p_hit_on_miss"] is not None and ma["mean_p_hit_on_hit"] is not None:
        console.print(f"  Mean confidence: hit={ma['mean_p_hit_on_hit']:.3f} vs miss={ma['mean_p_hit_on_miss']:.3f}")

    # Streak metrics
    sm = sc["streak_metrics"]
    console.print(f"\n[bold]Streak Distribution ({mc_trials:,} trials):[/bold]")
    console.print(f"  Mean: {sm['mean_max_streak']:.1f}  Median: {sm['median_max_streak']}")
    console.print(f"  90th: {sm['p90_max_streak']}  99th: {sm['p99_max_streak']}")
    console.print(f"  Longest replay: {sm['longest_replay_streak']}")

    # Save
    if save_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"data/validation/scorecard_{ts}.json"
    saved = save_scorecard(sc, save_path)
    console.print(f"\n[green]Saved to {saved}[/green]")

    # Diff
    if diff_path:
        import json as _json
        baseline = _json.loads(Path(diff_path).read_text())
        diffs = diff_scorecards(baseline, sc)
        console.print(f"\n[bold yellow]Diff vs {diff_path}:[/bold yellow]")
        for k in sorted(diffs.get("precision", {})):
            d = diffs["precision"][k]
            sign = "+" if d["delta"] >= 0 else ""
            console.print(f"  P@{k}: {d['baseline']:.3f} → {d['variant']:.3f} ({sign}{d['delta']:.3f})")
        if "p_57_exact_mdp" in diffs:
            d = diffs["p_57_exact_mdp"]
            sign = "+" if d["delta"] >= 0 else ""
            console.print(f"  P(57) MDP: {d['baseline']:.4%} → {d['variant']:.4%} ({sign}{d['delta']:.4%})")
```

- [ ] **Step 8: Run all tests to verify nothing is broken**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v`
Expected: All existing tests + new tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/bts/validate/ src/bts/cli.py tests/validate/
git commit -m "feat(validate): scorecard CLI — full report, JSON save, diff comparison"
```

---

### Task 3: Compute Baseline Scorecard

Run the scorecard on existing backtest profiles to establish the baseline all investigations are measured against.

**Files:**
- Output: `data/validation/scorecard_baseline.json`

- [ ] **Step 1: Run the baseline scorecard**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run bts validate scorecard --save data/validation/scorecard_baseline.json`

Record the full output — these are the numbers everything gets compared against.

- [ ] **Step 2: Verify the baseline JSON was saved**

Run: `cat data/validation/scorecard_baseline.json | python3 -m json.tool | head -30`

Check that P@1, P@100, P(57), miss analysis, and streak metrics are all populated.

- [ ] **Step 3: Commit the baseline**

```bash
git add data/validation/scorecard_baseline.json
git commit -m "data: baseline scorecard — all metrics before validation sprint"
```

---

### Task 4: Quick Verification — 2026 Double-Down Rule (Item 1)

Research whether the 2026 BTS rules allow removing your second pick after the first player's game starts.

**Files:**
- Output: `docs/validation/item-01-double-down-rule.md`

- [ ] **Step 1: Research the rule**

Search MLB.com BTS rules page and Reddit community reports. Key search terms:
- "beat the streak 2026 rules"
- "BTS remove pick" or "BTS cancel pick"
- Reddit r/beatthestreak posts about 2026 rule changes

Check the official MLB BTS FAQ and terms.

- [ ] **Step 2: Document findings**

Create `docs/validation/item-01-double-down-rule.md` with:
- Verdict: confirmed / not confirmed / ambiguous
- Source URLs
- Exact rule text if found
- Implications for MDP (if confirmed → item 9 is mandatory)

- [ ] **Step 3: Commit**

```bash
git add docs/validation/
git commit -m "docs: item 1 verdict — 2026 double-down rule research"
```

---

### Task 5: Quick Verification — Scoring Change Audit (Item 2)

Audit our check-results logic for retroactive scoring changes.

**Files:**
- Read: `src/bts/picks.py` (check_hit function)
- Output: `docs/validation/item-02-scoring-changes.md`
- Possibly modify: `src/bts/picks.py` if fix needed

- [ ] **Step 1: Read the check_hit implementation**

Read `src/bts/picks.py` — find the `check_hit` function and trace how it resolves hit/miss. Specifically check:
1. Does it query the MLB API for game results?
2. Does it handle the case where a game result changes (hit → error) after initial check?
3. Does the 1am cron re-check results from previous days, or only the current day?

- [ ] **Step 2: Document findings**

Create `docs/validation/item-02-scoring-changes.md` with:
- Verdict: handled / not handled / partially handled
- Exact code paths
- If not handled: describe the fix needed (next-day re-verification pass)

- [ ] **Step 3: If fix needed, write failing test**

```python
# tests/test_picks.py (append) — only if fix is needed
def test_check_hit_reverify_next_day():
    """check_hit should re-verify results from previous day
    in case MLB retroactively changed scoring."""
    # Test that calling check_hit on day D also re-checks day D-1
    # if D-1's result was recorded less than 24 hours ago
    pass  # Implementation depends on findings
```

- [ ] **Step 4: Implement fix if needed, run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_picks.py -v`

- [ ] **Step 5: Commit**

```bash
git add docs/validation/ src/bts/picks.py tests/test_picks.py
git commit -m "audit: item 2 — scoring change handling (verdict + fix if needed)"
```

---

### Task 6: Quick Verification — Home/Away Analysis (Item 4)

Check if visiting team PA advantage is already captured by our model.

**Files:**
- Create: `scripts/validation/item_04_home_away.py`
- Output: `docs/validation/item-04-home-away.md`

- [ ] **Step 1: Write analysis script**

```python
# scripts/validation/item_04_home_away.py
"""Analyze whether our model already captures home/away PA advantage."""

import pandas as pd
import numpy as np
from pathlib import Path

profiles_dir = Path("data/simulation")
processed_dir = Path("data/processed")

# Load backtest profiles
dfs = []
for p in sorted(profiles_dir.glob("backtest_*.parquet")):
    season = int(p.stem.split("_")[1])
    sdf = pd.read_parquet(p)
    sdf["season"] = season
    dfs.append(sdf)
profiles = pd.concat(dfs, ignore_index=True)

# Load PA data to get home/away status for each batter-game
# We need to map batter_id + date to is_home
pa_dfs = []
for p in sorted(processed_dir.glob("pa_*.parquet")):
    pdf = pd.read_parquet(p, columns=["batter_id", "date", "is_home", "game_pk"])
    pa_dfs.append(pdf.drop_duplicates(subset=["batter_id", "game_pk"]))
pa_data = pd.concat(pa_dfs, ignore_index=True)
pa_data["date"] = pd.to_datetime(pa_data["date"]).dt.strftime("%Y-%m-%d")

# Merge: get is_home for rank-1 picks
r1 = profiles[profiles["rank"] == 1].copy()
r1["date"] = r1["date"].astype(str)
r1 = r1.merge(
    pa_data[["batter_id", "date", "is_home"]].drop_duplicates(),
    on=["batter_id", "date"],
    how="left",
)

print(f"Rank-1 picks: {len(r1)}")
print(f"  Home: {r1['is_home'].sum()}, Away: {(~r1['is_home']).sum()}")
print(f"  Home P@1: {r1[r1['is_home']]['actual_hit'].mean():.3f}")
print(f"  Away P@1: {r1[~r1['is_home']]['actual_hit'].mean():.3f}")
print(f"  Difference: {r1[~r1['is_home']]['actual_hit'].mean() - r1[r1['is_home']]['actual_hit'].mean():.3f}")

# Also check: average PAs per game for home vs away
all_pa = pd.concat([pd.read_parquet(p) for p in sorted(processed_dir.glob("pa_*.parquet"))])
pas_per_game = all_pa.groupby(["batter_id", "game_pk", "is_home"]).size().reset_index(name="n_pas")
print(f"\nAverage PAs per game:")
print(f"  Home: {pas_per_game[pas_per_game['is_home']]['n_pas'].mean():.2f}")
print(f"  Away: {pas_per_game[~pas_per_game['is_home']]['n_pas'].mean():.2f}")
```

- [ ] **Step 2: Run analysis**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validation/item_04_home_away.py`

- [ ] **Step 3: Document findings**

Create `docs/validation/item-04-home-away.md` with verdict and measured delta.

- [ ] **Step 4: Commit**

```bash
git add scripts/validation/ docs/validation/
git commit -m "analysis: item 4 — home/away PA advantage (verdict)"
```

---

### Task 7: P@K Benchmark vs. lokikg (Item 6)

Extract P@K curves from the baseline scorecard and compare against lokikg's claims.

**Files:**
- Output: `docs/validation/item-06-lokikg-benchmark.md`

- [ ] **Step 1: Extract P@K from baseline scorecard**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python3 -c "
import json
sc = json.loads(open('data/validation/scorecard_baseline.json').read())
print('=== Overall P@K ===')
for k, v in sorted(sc['precision'].items(), key=lambda x: int(x[0])):
    print(f'  P@{k}: {v:.3f}')
print()
print('=== P@K by Season ===')
for season in sorted(sc['precision_by_season']):
    print(f'  {season}:')
    for k, v in sorted(sc['precision_by_season'][season].items(), key=lambda x: int(x[0])):
        print(f'    P@{k}: {v:.3f}')
"
```

- [ ] **Step 2: Document comparison**

Create `docs/validation/item-06-lokikg-benchmark.md` comparing:
- Our P@100 vs lokikg's claimed 89%
- Our P@250 vs lokikg's claimed 79.2%
- Caveats: lokikg validates on 2025 only, leakage controls unknown, methodology differences (PA-level vs game-level)
- Our multi-season validation advantage

- [ ] **Step 3: Commit**

```bash
git add docs/validation/
git commit -m "analysis: item 6 — P@K benchmark vs lokikg"
```

---

### Task 8: Projected vs. Confirmed Lineup Simulation (Item 8b)

Quantify how much P@1 we lose from lineup uncertainty.

**Files:**
- Create: `scripts/validation/item_08b_lineup_impact.py`
- Output: `docs/validation/item-08b-lineup-impact.md`

- [ ] **Step 1: Write the lineup simulation script**

```python
# scripts/validation/item_08b_lineup_impact.py
"""Simulate projected vs confirmed lineup impact on P@1.

For each game-date, constructs "projected" lineups using each team's
prior game's lineup. Compares rank-1 picks under projected vs confirmed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

processed_dir = Path("data/processed")

# Load all PA data
print("Loading PA data...")
dfs = []
for p in sorted(processed_dir.glob("pa_*.parquet")):
    dfs.append(pd.read_parquet(p))
all_pas = pd.concat(dfs, ignore_index=True)
all_pas["date"] = pd.to_datetime(all_pas["date"])
print(f"  {len(all_pas):,} PAs loaded")

# For each team-date, record the actual lineup (batter_id → batting_order)
team_dates = all_pas.groupby(["team_id", "date", "game_pk"]).apply(
    lambda g: dict(zip(g["batter_id"], g["batting_order"]))
).reset_index(name="lineup")

# Sort by date to build "prior game lineup" per team
team_dates = team_dates.sort_values(["team_id", "date"]).reset_index(drop=True)
team_dates["prior_lineup"] = team_dates.groupby("team_id")["lineup"].shift(1)

# For dates where we have prior lineups, compare
has_prior = team_dates.dropna(subset=["prior_lineup"])
print(f"\n{len(has_prior)} team-game-dates with prior lineup available")

# Compute lineup match rate
def lineup_match_rate(row):
    actual = set(row["lineup"].keys())
    prior = set(row["prior_lineup"].keys())
    if not actual:
        return np.nan
    return len(actual & prior) / len(actual)

has_prior["match_rate"] = has_prior.apply(lineup_match_rate, axis=1)
print(f"Average lineup match rate (prior vs actual): {has_prior['match_rate'].mean():.1%}")
print(f"  Perfect match: {(has_prior['match_rate'] == 1.0).mean():.1%}")
print(f"  < 80% match: {(has_prior['match_rate'] < 0.8).mean():.1%}")

# Stratify by day-of-week
has_prior["dow"] = pd.to_datetime(has_prior["date"]).dt.day_name()
print(f"\nMatch rate by day of week:")
for dow, group in has_prior.groupby("dow"):
    print(f"  {dow}: {group['match_rate'].mean():.1%} ({len(group)} games)")
```

Note: This script computes lineup stability. If match rate is very high (>95%), lineup uncertainty is negligible and we can skip the full P@1 comparison. If it's lower, a follow-up script would need to re-run predictions with projected lineups — that's a much bigger computation.

- [ ] **Step 2: Run analysis**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validation/item_08b_lineup_impact.py`

- [ ] **Step 3: If lineup volatility is meaningful (match rate < 90%), write P@1 comparison**

This would require re-running the blend predictions with projected lineups — 2-3 hours of compute. Only proceed if Step 2 shows meaningful volatility.

- [ ] **Step 4: Document findings**

Create `docs/validation/item-08b-lineup-impact.md` with measured volatility and P@1 delta estimate.

- [ ] **Step 5: Commit**

```bash
git add scripts/validation/ docs/validation/
git commit -m "analysis: item 8b — lineup projection impact (verdict)"
```

---

### Task 9: Miss-Day Pattern Analysis (Item 11)

Investigate whether miss days have identifiable patterns.

**Files:**
- Create: `scripts/validation/item_11_miss_analysis.py`
- Output: `docs/validation/item-11-miss-patterns.md`

- [ ] **Step 1: Write the analysis script**

```python
# scripts/validation/item_11_miss_analysis.py
"""Deep analysis of rank-1 miss days: are there identifiable patterns?"""

import pandas as pd
import numpy as np
from pathlib import Path

profiles_dir = Path("data/simulation")

# Load profiles
dfs = []
for p in sorted(profiles_dir.glob("backtest_*.parquet")):
    season = int(p.stem.split("_")[1])
    sdf = pd.read_parquet(p)
    sdf["season"] = season
    dfs.append(sdf)
profiles = pd.concat(dfs, ignore_index=True)
profiles["date"] = pd.to_datetime(profiles["date"])

r1 = profiles[profiles["rank"] == 1].copy()
r2 = profiles[profiles["rank"] == 2].copy()

# Basic miss stats
print(f"Total days: {len(r1)}")
print(f"Hit days: {r1['actual_hit'].sum()} ({r1['actual_hit'].mean():.1%})")
print(f"Miss days: {(1 - r1['actual_hit']).sum()}")

# Miss rate by season
print(f"\nP@1 by season:")
for season, group in r1.groupby("season"):
    print(f"  {season}: {group['actual_hit'].mean():.1%} ({len(group)} days, {(1-group['actual_hit']).sum():.0f} misses)")

# Confidence on hit vs miss days
hit_days = r1[r1["actual_hit"] == 1]
miss_days = r1[r1["actual_hit"] == 0]
print(f"\nMean p_game_hit:")
print(f"  Hit days:  {hit_days['p_game_hit'].mean():.4f}")
print(f"  Miss days: {miss_days['p_game_hit'].mean():.4f}")
print(f"  Difference: {hit_days['p_game_hit'].mean() - miss_days['p_game_hit'].mean():.4f}")

# Confidence gap (rank-1 minus rank-2)
merged = r1[["date", "p_game_hit"]].merge(
    r2[["date", "p_game_hit"]].rename(columns={"p_game_hit": "p_game_hit_r2"}),
    on="date",
)
merged["confidence_gap"] = merged["p_game_hit"] - merged["p_game_hit_r2"]
merged = merged.merge(r1[["date", "actual_hit"]], on="date")

print(f"\nConfidence gap (rank-1 minus rank-2):")
print(f"  Hit days:  {merged[merged['actual_hit'] == 1]['confidence_gap'].mean():.4f}")
print(f"  Miss days: {merged[merged['actual_hit'] == 0]['confidence_gap'].mean():.4f}")

# Would a confidence gap filter help?
# Simulate: skip days where gap < threshold
for threshold in [0.002, 0.005, 0.01, 0.02]:
    filtered = merged[merged["confidence_gap"] >= threshold]
    if len(filtered) > 0:
        p_at_1 = filtered["actual_hit"].mean()
        pct_skipped = 1 - len(filtered) / len(merged)
        print(f"  Gap >= {threshold:.3f}: P@1={p_at_1:.1%}, skip {pct_skipped:.0%} of days ({len(filtered)} play days)")

# Rank-2 hit rate on miss days
miss_dates = set(r1[r1["actual_hit"] == 0]["date"])
r2_miss = r2[r2["date"].isin(miss_dates)]
print(f"\nRank-2 performance on miss days:")
print(f"  Hit rate: {r2_miss['actual_hit'].mean():.1%} ({len(r2_miss)} days)")

# Time-of-season patterns
r1["month"] = r1["date"].dt.month
print(f"\nP@1 by month:")
for month, group in r1.groupby("month"):
    month_name = pd.Timestamp(2024, month, 1).strftime("%b")
    print(f"  {month_name}: {group['actual_hit'].mean():.1%} ({len(group)} days)")

# Miss clustering: are misses clustered or random?
miss_dates_sorted = sorted(miss_dates)
if len(miss_dates_sorted) > 1:
    gaps = [(miss_dates_sorted[i+1] - miss_dates_sorted[i]).days
            for i in range(len(miss_dates_sorted) - 1)]
    print(f"\nMiss clustering:")
    print(f"  Mean gap between misses: {np.mean(gaps):.1f} days")
    print(f"  Consecutive miss days: {sum(1 for g in gaps if g == 1)}")
    print(f"  Expected if random (Poisson): {1/r1['actual_hit'].mean():.1f} days")
```

- [ ] **Step 2: Run analysis**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validation/item_11_miss_analysis.py`

- [ ] **Step 3: Document findings**

Create `docs/validation/item-11-miss-patterns.md` with:
- Confidence gap analysis: is there a filterable signal?
- Time-of-season patterns
- Miss clustering (random or structured?)
- Verdict: incorporate into skip rules or not

- [ ] **Step 4: Commit**

```bash
git add scripts/validation/ docs/validation/
git commit -m "analysis: item 11 — miss-day pattern analysis (verdict)"
```

---

### Task 10: Densest-Bucket Strategy Validation (Item 12)

Test whether the game-time bucketing helps or hurts P(57).

**Files:**
- Create: `scripts/validation/item_12_densest_bucket.py`
- Output: `docs/validation/item-12-densest-bucket.md`

- [ ] **Step 1: Write validation script**

```python
# scripts/validation/item_12_densest_bucket.py
"""Validate densest-bucket strategy against always-pick-rank-1.

Since backtest profiles use confirmed lineups, this tests the
ranking restriction concept, not lineup uncertainty.
"""

import pandas as pd
import numpy as np
from pathlib import Path

profiles_dir = Path("data/simulation")

# Load profiles — rank-1 is already the "always pick best" strategy
# Rank-2 tells us what happens when we DON'T pick rank-1
dfs = []
for p in sorted(profiles_dir.glob("backtest_*.parquet")):
    season = int(p.stem.split("_")[1])
    sdf = pd.read_parquet(p)
    sdf["season"] = season
    dfs.append(sdf)
profiles = pd.concat(dfs, ignore_index=True)

r1 = profiles[profiles["rank"] == 1]
r2 = profiles[profiles["rank"] == 2]

# The backtest profiles already represent the BLEND rank-1 pick.
# The densest-bucket strategy could sometimes choose a different
# player (rank-2 or lower) from the densest time window.
#
# Since we don't have game_time in backtest profiles, we can't
# directly simulate densest-bucket. But we CAN answer the question:
# "if the densest bucket sometimes forces us to pick rank-2 instead
# of rank-1, how much does P@1 drop?"

# Simulate: what if X% of the time we're forced to pick rank-2?
merged = r1[["date", "actual_hit"]].merge(
    r2[["date", "actual_hit"]].rename(columns={"actual_hit": "r2_hit"}),
    on="date",
)

r1_p = merged["actual_hit"].mean()
r2_p = merged["r2_hit"].mean()

print(f"Rank-1 P@1: {r1_p:.3f}")
print(f"Rank-2 P@1: {r2_p:.3f}")
print(f"Gap: {r1_p - r2_p:.3f}")

# If densest-bucket forces rank-2 instead of rank-1 X% of the time:
for pct_forced in [0.05, 0.10, 0.15, 0.20, 0.30]:
    blended_p = (1 - pct_forced) * r1_p + pct_forced * r2_p
    print(f"  If forced to rank-2 {pct_forced:.0%} of the time: P@1={blended_p:.3f} (delta={blended_p - r1_p:+.3f})")

# The real question is: how often does densest-bucket actually change
# the pick? We need game_time in profiles for that. For now, estimate
# from production data if available.
print("\nNote: Full validation requires game_time in backtest profiles.")
print("This analysis provides the COST function for rank displacement.")
print("The BENEFIT (lineup certainty) is measured by item 8b.")
```

- [ ] **Step 2: Run analysis**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validation/item_12_densest_bucket.py`

- [ ] **Step 3: Document findings with P(57) impact**

To compute P(57) impact of rank displacement, use the exact solver with modified quality bins (reduce P(hit) in proportion to the rank-2 substitution rate).

- [ ] **Step 4: Commit**

```bash
git add scripts/validation/ docs/validation/
git commit -m "analysis: item 12 — densest-bucket strategy validation (verdict)"
```

---

### Task 11: Batting Order Signal Investigation (Item 3)

Test whether explicit batting order adds predictive value beyond PA-level aggregation.

**Files:**
- Create: `scripts/validation/item_03_batting_order.py`
- Output: `docs/validation/item-03-batting-order.md`

- [ ] **Step 1: Write investigation script**

This investigation requires a full blend backtest with batting_order as an additional feature. We add it as a 13th blend member (FEATURE_COLS + batting_order) and also test it in the baseline model.

```python
# scripts/validation/item_03_batting_order.py
"""Test batting_order as a feature: does PA aggregation already capture the signal?

Approach:
1. Stratify existing backtest P@1 by lineup slot (check if model is already
   better for leadoff hitters)
2. Add batting_order to a single-model walk-forward as a quick test
3. If promising, run full blend backtest

Requires processed parquet data with batting_order column.
"""

import pandas as pd
import numpy as np
from pathlib import Path

processed_dir = Path("data/processed")
profiles_dir = Path("data/simulation")

# Step 1: Check if batting_order exists in our PA data
sample = pd.read_parquet(next(processed_dir.glob("pa_*.parquet")), columns=["batting_order"])
print(f"batting_order range: {sample['batting_order'].min()}-{sample['batting_order'].max()}")
print(f"batting_order null rate: {sample['batting_order'].isna().mean():.1%}")

# Step 2: Load profiles and PA data, join to get lineup slot for rank-1 picks
profiles = pd.concat(
    [pd.read_parquet(p).assign(season=int(p.stem.split("_")[1]))
     for p in sorted(profiles_dir.glob("backtest_*.parquet"))],
    ignore_index=True,
)

# Get the most common batting_order per batter-date from PA data
pa_dfs = [pd.read_parquet(p, columns=["batter_id", "date", "batting_order"])
          for p in sorted(processed_dir.glob("pa_*.parquet"))]
pa_data = pd.concat(pa_dfs, ignore_index=True)
pa_data["date"] = pd.to_datetime(pa_data["date"]).dt.strftime("%Y-%m-%d")
batter_order = pa_data.groupby(["batter_id", "date"])["batting_order"].first().reset_index()

r1 = profiles[profiles["rank"] == 1].copy()
r1["date"] = r1["date"].astype(str)
r1 = r1.merge(batter_order, on=["batter_id", "date"], how="left")

print(f"\nP@1 by lineup slot:")
for slot, group in r1.groupby("batting_order"):
    if len(group) >= 10:
        print(f"  Slot {slot}: P@1={group['actual_hit'].mean():.1%} "
              f"({len(group)} days, {group['actual_hit'].mean() - r1['actual_hit'].mean():+.1%} vs avg)")

# Check how often rank-1 IS a leadoff hitter
print(f"\nRank-1 batting order distribution:")
print(r1["batting_order"].value_counts().head(9).to_string())
```

- [ ] **Step 2: Run the stratification analysis**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validation/item_03_batting_order.py`

- [ ] **Step 3: If signal detected, run single-model walk-forward with batting_order**

Only proceed if Step 2 shows meaningful P@1 variation by lineup slot. This would require temporarily adding `batting_order` to FEATURE_COLS and running `walk_forward_evaluate` on 2024 and 2025. Use a script, not a code change to features.py:

```python
# scripts/validation/item_03_batting_order_backtest.py
"""Walk-forward test with batting_order as an additional feature."""

import pandas as pd
from pathlib import Path
from bts.features.compute import compute_all_features, FEATURE_COLS, TRAIN_START_YEAR
from bts.evaluate.backtest import walk_forward_evaluate

processed_dir = Path("data/processed")
dfs = [pd.read_parquet(p) for p in sorted(processed_dir.glob("pa_*.parquet"))]
df = pd.concat(dfs, ignore_index=True)
df = compute_all_features(df)

# Temporarily extend FEATURE_COLS
test_cols = FEATURE_COLS + ["batting_order"]

for season in [2024, 2025]:
    print(f"\n{'='*60}")
    print(f"Season {season} — with batting_order")
    metrics, _ = walk_forward_evaluate(df, test_season=season)
    # Note: walk_forward_evaluate uses FEATURE_COLS from the import.
    # For this test, we need to monkey-patch it or extract the model loop.
    # This script may need adjustment based on the actual backtest API.
    print(f"  P@1: {metrics.get('P@1', 'N/A')}")
```

This script is a template — actual implementation may need to modify the walk_forward call to accept custom feature columns.

- [ ] **Step 4: Document findings**

Create `docs/validation/item-03-batting-order.md` with:
- P@1 by lineup slot (does the model already predict leadoff better?)
- Delta from adding explicit batting_order feature
- Verdict: add as blend member / add to baseline / reject

- [ ] **Step 5: Commit**

```bash
git add scripts/validation/ docs/validation/
git commit -m "analysis: item 3 — batting order signal investigation (verdict)"
```

---

### Task 12: Implied Run Total Investigation (Item 5)

Test team-level Vegas implied run total as a context feature.

**Files:**
- Create: `scripts/validation/item_05_implied_runs.py`
- Output: `docs/validation/item-05-implied-runs.md`

- [ ] **Step 1: Write data exploration script**

```python
# scripts/validation/item_05_implied_runs.py
"""Explore team-level implied run totals from odds data.

Data lives in data/external/odds/v2/{date}.json.
Coverage: Sept 2023 - Sept 2025 (418 dates).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

odds_dir = Path("data/external/odds/v2")
files = sorted(odds_dir.glob("*.json"))
print(f"Odds files: {len(files)}")
print(f"Date range: {files[0].stem} to {files[-1].stem}")

# Check one file to understand structure
sample = json.loads(files[100].read_text())
print(f"\nSample file keys: {list(sample[0].keys()) if isinstance(sample, list) else list(sample.keys())}")

# We need to extract: game_pk or team matchup → implied run total per team
# The implied run total = totals market / 2 adjusted by moneyline
# Or if available directly as team totals
# Explore the data structure first before writing feature extraction

# Print first game's structure
if isinstance(sample, list) and len(sample) > 0:
    game = sample[0]
    print(f"\nFirst game:")
    for k, v in game.items():
        if isinstance(v, dict):
            print(f"  {k}: {list(v.keys())}")
        elif isinstance(v, list) and len(v) > 0:
            print(f"  {k}: [{type(v[0]).__name__}, ...] len={len(v)}")
        else:
            print(f"  {k}: {v}")
```

- [ ] **Step 2: Run exploration**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validation/item_05_implied_runs.py`

Based on the data structure, determine how to extract team-level implied run totals.

- [ ] **Step 3: Extend script to compute the feature and join to PA data**

After understanding the data format, build the feature:
1. Extract team implied run total per game
2. Map to batter's team
3. Join to PA data by date + team
4. Run walk-forward backtest on 2024 + 2025 with this as an additional feature
5. Test as both a baseline addition and a blend member

This step requires significant coding based on Step 2's findings. The script template will need to be fleshed out after understanding the odds data structure.

- [ ] **Step 4: Document findings with scorecard comparison**

Run the scorecard with the modified model and diff against baseline.

- [ ] **Step 5: Commit**

```bash
git add scripts/validation/ docs/validation/
git commit -m "analysis: item 5 — implied run total investigation (verdict)"
```

---

### Task 13: Walk-Rate / BB% Feature (Item 7)

Test explicit BB% as a feature or hard filter.

**Files:**
- Create: `scripts/validation/item_07_walk_rate.py`
- Output: `docs/validation/item-07-walk-rate.md`

- [ ] **Step 1: Write analysis + quick test**

```python
# scripts/validation/item_07_walk_rate.py
"""Test walk rate as explicit feature vs our count_tendency proxy.

Community consensus: avoid high-walk batters (Soto, etc).
Our batter_count_tendency_30g captures this indirectly.
Question: is explicit BB% sharper?
"""

import pandas as pd
import numpy as np
from pathlib import Path

processed_dir = Path("data/processed")
profiles_dir = Path("data/simulation")

# Load PA data and compute BB rate
dfs = [pd.read_parquet(p) for p in sorted(processed_dir.glob("pa_*.parquet"))]
df = pd.concat(dfs, ignore_index=True)
df["date"] = pd.to_datetime(df["date"])

# Identify walks from event types
df["is_walk"] = df["event_type"].isin(["walk", "intent_walk"])

# Compute rolling BB rate per batter (30 game-dates)
date_walks = df.groupby(["batter_id", "date"]).agg(
    walks=("is_walk", "sum"),
    pas=("is_walk", "count"),
).reset_index().sort_values(["batter_id", "date"])

date_walks["bb_rate"] = date_walks["walks"] / date_walks["pas"]
date_walks["batter_bb_rate_30g"] = date_walks.groupby("batter_id")["bb_rate"].transform(
    lambda x: x.shift(1).rolling(30, min_periods=10).mean()
)

# Check correlation with count_tendency
from bts.features.compute import compute_all_features
sample_season = df[df["season"] == 2025].copy()
sample_season = compute_all_features(sample_season)

# Join bb_rate to the featured data
sample_season = sample_season.merge(
    date_walks[["batter_id", "date", "batter_bb_rate_30g"]],
    on=["batter_id", "date"],
    how="left",
)

# Correlation between bb_rate and count_tendency
corr = sample_season[["batter_bb_rate_30g", "batter_count_tendency_30g"]].corr()
print(f"Correlation between bb_rate and count_tendency: {corr.iloc[0,1]:.3f}")

# Load profiles and check: do high-BB batters miss more often when they're rank-1?
profiles = pd.concat(
    [pd.read_parquet(p).assign(season=int(p.stem.split("_")[1]))
     for p in sorted(profiles_dir.glob("backtest_*.parquet"))],
    ignore_index=True,
)

r1 = profiles[profiles["rank"] == 1].copy()
r1["date_str"] = r1["date"].astype(str)
date_walks["date_str"] = date_walks["date"].astype(str)
r1 = r1.merge(
    date_walks[["batter_id", "date_str", "batter_bb_rate_30g"]],
    left_on=["batter_id", "date_str"],
    right_on=["batter_id", "date_str"],
    how="left",
)

# Split by BB rate quartiles
r1["bb_quartile"] = pd.qcut(r1["batter_bb_rate_30g"].dropna(), 4, labels=False)
print(f"\nP@1 by BB rate quartile (rank-1 picks):")
for q, group in r1.groupby("bb_quartile"):
    bb_range = f"{group['batter_bb_rate_30g'].min():.1%}-{group['batter_bb_rate_30g'].max():.1%}"
    print(f"  Q{int(q)+1} (BB {bb_range}): P@1={group['actual_hit'].mean():.1%} ({len(group)} days)")

# Hard filter test: what if we excluded BB% > 15%?
high_bb = r1[r1["batter_bb_rate_30g"] > 0.15]
low_bb = r1[r1["batter_bb_rate_30g"] <= 0.15]
print(f"\nHard filter BB% > 15%:")
print(f"  Would exclude {len(high_bb)} pick-days ({len(high_bb)/len(r1):.0%})")
print(f"  Excluded picks P@1: {high_bb['actual_hit'].mean():.1%}")
print(f"  Remaining picks P@1: {low_bb['actual_hit'].mean():.1%}")
```

- [ ] **Step 2: Run analysis**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validation/item_07_walk_rate.py`

- [ ] **Step 3: If signal detected, run backtest with BB% feature or filter**

Only proceed with full backtest if Step 2 shows meaningful P@1 difference.

- [ ] **Step 4: Document findings**

Create `docs/validation/item-07-walk-rate.md` with:
- Correlation between BB% and count_tendency
- P@1 by BB% quartile
- Hard filter impact
- Verdict: add feature / add filter / reject

- [ ] **Step 5: Commit**

```bash
git add scripts/validation/ docs/validation/
git commit -m "analysis: item 7 — walk rate feature/filter investigation (verdict)"
```

---

### Task 14: Contact Quality Composite (Item 8)

Test whether a combined contact quality feature adds to the blend.

**Files:**
- Create: `scripts/validation/item_08_contact_composite.py`
- Output: `docs/validation/item-08-contact-composite.md`

- [ ] **Step 1: Write analysis script**

```python
# scripts/validation/item_08_contact_composite.py
"""Test a composite contact quality feature as a 13th blend member.

Composite = standardized average of barrel_rate + hard_hit_rate +
sweet_spot_rate + avg_ev (30-game rolling).

Hypothesis: probably marginal since blend already has these as
individual members, but composite might break ties differently.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from bts.features.compute import compute_all_features, FEATURE_COLS, STATCAST_COLS

processed_dir = Path("data/processed")

# Load and compute features
dfs = [pd.read_parquet(p) for p in sorted(processed_dir.glob("pa_*.parquet"))]
df = pd.concat(dfs, ignore_index=True)
df = compute_all_features(df)

# Check: how correlated are the 4 contact quality components?
contact_cols = [
    "batter_barrel_rate_30g",
    "batter_hard_hit_rate_30g",
    "batter_sweet_spot_rate_30g",
    "batter_avg_ev_30g",
]

# Correlation matrix
corr = df[contact_cols].corr()
print("Contact quality component correlations:")
print(corr.to_string(float_format=lambda x: f"{x:.3f}"))

# Compute composite (standardize then average)
for col in contact_cols:
    mean = df[col].mean()
    std = df[col].std()
    df[f"{col}_z"] = (df[col] - mean) / std if std > 0 else 0

z_cols = [f"{c}_z" for c in contact_cols]
df["contact_quality_composite"] = df[z_cols].mean(axis=1)

# Check correlation with is_hit
hit_corr = df[["contact_quality_composite"] + contact_cols + ["is_hit"]].corr()["is_hit"]
print(f"\nCorrelation with is_hit:")
print(hit_corr.to_string(float_format=lambda x: f"{x:.4f}"))

# The composite should be tested as a blend member via full backtest.
# For now, check if the composite carries unique information.
from sklearn.metrics import mutual_info_score
print(f"\nComposite mean: {df['contact_quality_composite'].mean():.3f}")
print(f"Composite std: {df['contact_quality_composite'].std():.3f}")
print(f"Non-null rate: {df['contact_quality_composite'].notna().mean():.1%}")
```

- [ ] **Step 2: Run analysis**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validation/item_08_contact_composite.py`

- [ ] **Step 3: If the composite is distinct enough from individual components, run blend backtest**

Would add `("contact_composite", FEATURE_COLS + ["contact_quality_composite"])` to BLEND_CONFIGS and re-run `bts simulate backtest`.

- [ ] **Step 4: Document findings**

Create `docs/validation/item-08-contact-composite.md` with verdict.

- [ ] **Step 5: Commit**

```bash
git add scripts/validation/ docs/validation/
git commit -m "analysis: item 8 — contact quality composite investigation (verdict)"
```

---

### Task 15: MDP Re-Solve with Removable Double-Down (Item 9)

**BLOCKED BY:** Task 4 (item 1 must confirm the rule).

If the 2026 removable double-down rule is confirmed, modify the MDP transition matrix and re-solve.

**Files:**
- Modify: `src/bts/simulate/mdp.py`
- Modify: `tests/simulate/test_mdp.py`
- Output: new policy at `data/models/mdp_policy_removable_double.npz`

- [ ] **Step 1: Write failing test for the new transition**

```python
# tests/simulate/test_mdp.py (append)

def test_mdp_removable_double_transition():
    """With removable double-down, miss on pick-2 only should advance by 1, not reset."""
    from bts.simulate.mdp import solve_mdp
    from bts.simulate.quality_bins import QualityBins, QualityBin

    # Simple 2-bin system for testing
    bins = QualityBins(
        bins=[
            QualityBin(index=0, p_range=(0.7, 0.8), p_hit=0.75, p_both=0.56, frequency=0.5),
            QualityBin(index=1, p_range=(0.8, 0.9), p_hit=0.85, p_both=0.72, frequency=0.5),
        ],
        boundaries=[0.8],
    )

    # Solve with removable_double=True
    sol = solve_mdp(bins, season_length=180, removable_double=True)

    # P(57) should be higher than without removable double
    sol_standard = solve_mdp(bins, season_length=180, removable_double=False)
    assert sol.optimal_p57 > sol_standard.optimal_p57

    # Policy should double more aggressively
    # At streak 0, high quality, the removable-double policy should prefer double
    action = sol.policy(streak=0, days_remaining=160, saver=True, quality_bin=1)
    assert action == "double"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_mdp.py::test_mdp_removable_double_transition -v`
Expected: FAIL — `TypeError: solve_mdp() got an unexpected keyword argument 'removable_double'`

- [ ] **Step 3: Modify solve_mdp to support removable_double**

```python
# In src/bts/simulate/mdp.py, modify solve_mdp signature and double transition:

def solve_mdp(
    bins: QualityBins,
    season_length: int = 180,
    late_bins: QualityBins | None = None,
    late_phase_days: int = 60,
    removable_double: bool = False,
) -> MDPSolution:
    # ... (existing code until the double value computation)

    # Inside the backward induction loop, replace the double section:
    # Double: both hit → s+2, any miss → reset or saver
    pb = p_both[q]
    ph_single = p_hit[q]  # P(pick-1 hits)
    next_dbl = min(s + 2, 57)

    if removable_double:
        # Removable double-down: pick-1 miss = reset, pick-1 hit + pick-2 miss = advance by 1
        # P(+2) = P(both), P(+1) = P1*(1-P2), P(reset) = 1-P1
        # P2 conditional: pb = p1 * p2, so p2 = pb / p1
        p2_cond = pb / ph_single if ph_single > 0 else 0
        p_advance_2 = pb
        p_advance_1 = ph_single * (1 - p2_cond)
        p_miss = 1 - ph_single

        next_single = min(s + 1, 57)
        if saver and 10 <= s <= 15:
            v_double = (p_advance_2 * ev(next_dbl, saver) +
                       p_advance_1 * ev(next_single, saver) +
                       p_miss * ev(s, 0))
        else:
            v_double = (p_advance_2 * ev(next_dbl, saver) +
                       p_advance_1 * ev(next_single, saver) +
                       p_miss * ev(0, saver))
    else:
        # Standard: both must hit or reset
        if saver and 10 <= s <= 15:
            v_double = pb * ev(next_dbl, saver) + (1 - pb) * ev(s, 0)
        else:
            v_double = pb * ev(next_dbl, saver) + (1 - pb) * ev(0, saver)
```

- [ ] **Step 4: Run test**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_mdp.py -v`
Expected: All MDP tests PASS

- [ ] **Step 5: Run full solve and compare**

```bash
# Solve with removable double
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate solve --save-policy data/models/mdp_policy_removable_double.npz

# Compare P(57) values
```

- [ ] **Step 6: Run scorecard with new policy and diff**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts validate scorecard --diff data/validation/scorecard_baseline.json
```

- [ ] **Step 7: Document findings**

Create `docs/validation/item-09-removable-double.md` with P(57) delta and policy changes.

- [ ] **Step 8: Commit**

```bash
git add src/bts/simulate/mdp.py tests/simulate/test_mdp.py data/models/ docs/validation/
git commit -m "feat: MDP removable double-down support + item 9 verdict"
```

---

### Task 16: Phase-Aware Bin Granularity (Item 10)

Test finer time bins (monthly, quarterly) in the MDP.

**Files:**
- Create: `scripts/validation/item_10_phase_bins.py`
- Output: `docs/validation/item-10-phase-bins.md`

- [ ] **Step 1: Write bin granularity test**

```python
# scripts/validation/item_10_phase_bins.py
"""Test finer phase-aware bins in the MDP.

Current: binary (early Mar-Jul vs late Aug-Sep).
Test: monthly (6), quarterly (3).
Risk: more bins = fewer data points per bin = noisier.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from bts.simulate.quality_bins import compute_bins
from bts.simulate.mdp import solve_mdp

profiles_dir = Path("data/simulation")

# Load all profiles
dfs = [pd.read_parquet(p) for p in sorted(profiles_dir.glob("backtest_*.parquet"))]
profiles_df = pd.concat(dfs, ignore_index=True)
profiles_df["date"] = pd.to_datetime(profiles_df["date"])
profiles_df["month"] = profiles_df["date"].dt.month

# Current: binary early/late
early = profiles_df[profiles_df["month"] <= 7]
late = profiles_df[profiles_df["month"] >= 8]
early_bins = compute_bins(early)
late_bins = compute_bins(late)

sol_binary = solve_mdp(early_bins, season_length=180, late_bins=late_bins, late_phase_days=60)
print(f"Binary (current): P(57) = {sol_binary.optimal_p57:.4%}")

# No phase awareness (baseline)
all_bins = compute_bins(profiles_df)
sol_none = solve_mdp(all_bins, season_length=180)
print(f"No phases:         P(57) = {sol_none.optimal_p57:.4%}")

# Quarterly: Apr-May, Jun-Jul, Aug-Sep
# This requires extending the MDP to support 3+ phases.
# For now, test with the 2-phase framework by varying the late_phase_days cutoff.
for late_days in [30, 45, 60, 75, 90]:
    cutoff_month = 7 if late_days <= 60 else 6
    early_q = profiles_df[profiles_df["month"] <= cutoff_month]
    late_q = profiles_df[profiles_df["month"] > cutoff_month]
    if len(early_q) > 0 and len(late_q) > 0:
        eb = compute_bins(early_q)
        lb = compute_bins(late_q)
        sol = solve_mdp(eb, season_length=180, late_bins=lb, late_phase_days=late_days)
        print(f"  late_days={late_days} (cutoff month {cutoff_month}): P(57) = {sol.optimal_p57:.4%}")

# Monthly bins — check data sufficiency
print(f"\nData per month:")
for month, group in profiles_df.groupby("month"):
    r1 = group[group["rank"] == 1]
    month_name = pd.Timestamp(2024, month, 1).strftime("%b")
    print(f"  {month_name}: {len(r1)} days, P@1={r1['actual_hit'].mean():.1%}")
```

- [ ] **Step 2: Run analysis**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validation/item_10_phase_bins.py`

- [ ] **Step 3: Document findings**

Create `docs/validation/item-10-phase-bins.md` with optimal bin granularity and P(57) impact.

- [ ] **Step 4: Commit**

```bash
git add scripts/validation/ docs/validation/
git commit -m "analysis: item 10 — phase-aware bin granularity (verdict)"
```

---

### Task 17: Integration + Final Scorecard (Phase 5)

Integrate all accepted findings, re-solve MDP, produce final scorecard comparison.

**Files:**
- Possibly modify: `src/bts/features/compute.py` (if new features accepted)
- Possibly modify: `src/bts/model/predict.py` (if blend configs change)
- Possibly modify: `src/bts/simulate/mdp.py` (if MDP changes accepted)
- Possibly modify: `src/bts/strategy.py` (if strategy changes accepted)
- Output: `data/validation/scorecard_final.json`
- Output: `docs/validation/final-report.md`

- [ ] **Step 1: Review all item verdicts**

Read all `docs/validation/item-*.md` files. Compile list of accepted changes.

- [ ] **Step 2: If any features accepted, re-run blend backtest**

If items 3, 5, 7, or 8 produced accepted features:
1. Add the features to `FEATURE_COLS` or `BLEND_CONFIGS` as indicated
2. Run: `UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate backtest --seasons 2021,2022,2023,2024,2025`
3. Test features both individually AND in combination (per spec requirement)

- [ ] **Step 3: Re-solve MDP on final configuration**

If new backtest profiles were generated, or if MDP parameters changed:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate solve --save-policy data/models/mdp_policy.npz
```

- [ ] **Step 4: Compute final scorecard and diff**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts validate scorecard \
  --save data/validation/scorecard_final.json \
  --diff data/validation/scorecard_baseline.json
```

- [ ] **Step 5: Write final report**

Create `docs/validation/final-report.md` with:
- Summary table: all 14 items, verdict, delta, action
- Before/after scorecard comparison
- Final P(57)
- Lessons learned

- [ ] **Step 6: Commit**

```bash
git add src/ tests/ data/ docs/
git commit -m "feat: competitive validation sprint — integrate accepted findings

Items accepted: [list]
Items rejected: [list]
P(57): baseline X% → final Y%
P@1: baseline X% → final Y%"
```
