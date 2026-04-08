"""Phase 6: Scheduler timing simulation.

Simulates lineup confirmation dynamics over historical data to measure
how often the scheduler would lock a suboptimal pick by locking too early.

Uses the existing backtest profiles as the "oracle" ranking, then simulates
which lineups are confirmed at each check time.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase6_scheduler_timing.py
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

ET = ZoneInfo("America/New_York")
TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]

# Variants to test
VARIANTS = [
    {"name": "baseline", "early_lock_gap": 0.03, "min_confirmed_pct": 0.0},
    {"name": "min_50pct", "early_lock_gap": 0.03, "min_confirmed_pct": 0.50},
    {"name": "min_75pct", "early_lock_gap": 0.03, "min_confirmed_pct": 0.75},
    {"name": "gap_5pct", "early_lock_gap": 0.05, "min_confirmed_pct": 0.0},
    {"name": "gap5_min50", "early_lock_gap": 0.05, "min_confirmed_pct": 0.50},
]

N_TRIALS = 100


def build_game_time_lookup(raw_dir: str = "data/raw") -> dict[int, str]:
    """Build game_pk → game_time (UTC ISO) from raw feeds."""
    lookup = {}
    raw = Path(raw_dir)
    for season_dir in sorted(raw.iterdir()):
        if not season_dir.is_dir():
            continue
        for f in season_dir.glob("*.json"):
            try:
                d = json.loads(f.read_text())
                pk = int(f.stem)
                gt = d.get("gameData", {}).get("datetime", {}).get("dateTime")
                if gt:
                    lookup[pk] = gt
            except Exception:
                continue
    print(f"Built game_time lookup: {len(lookup)} games", file=sys.stderr)
    return lookup


def enrich_profiles_with_game_info(
    profiles: pd.DataFrame,
    pa_data: pd.DataFrame,
    gt_lookup: dict,
) -> pd.DataFrame:
    """Add game_pk and game_time to profiles by joining to PA data."""
    # For each (date, batter_id), find their game_pk from PA data
    pa_games = pa_data.groupby(["date", "batter_id"]).agg(
        game_pk=("game_pk", "first"),  # take first game (handles doubleheaders)
    ).reset_index()
    pa_games["date"] = pd.to_datetime(pa_games["date"]).dt.date

    profiles = profiles.copy()
    profiles["_date_key"] = pd.to_datetime(profiles["date"]).dt.date

    profiles = profiles.merge(
        pa_games.rename(columns={"date": "_date_key"}),
        on=["_date_key", "batter_id"],
        how="left",
    )

    # Add game_time from lookup
    profiles["game_time"] = profiles["game_pk"].map(gt_lookup)
    profiles.drop(columns=["_date_key"], inplace=True)

    coverage = profiles["game_pk"].notna().sum()
    print(f"Profile enrichment: {coverage}/{len(profiles)} rows have game_pk", file=sys.stderr)

    return profiles


def game_time_to_et(gt_str: str) -> datetime:
    """Convert UTC ISO game time to ET datetime."""
    try:
        utc = datetime.fromisoformat(str(gt_str).replace("Z", "+00:00"))
        return utc.astimezone(ET)
    except Exception:
        return None


def simulate_confirmation_times(
    game_pks: list[int],
    game_times_et: dict[int, datetime],
    rng: np.random.Generator,
) -> dict[int, datetime]:
    """Simulate when each game's lineup would be confirmed.

    Samples from N(game_time - 60min, 15min), clipped to
    [game_time - 120min, game_time - 15min].
    """
    confirmations = {}
    for pk in game_pks:
        gt = game_times_et.get(pk)
        if gt is None:
            continue
        # Sample offset in minutes before game time
        offset = rng.normal(60, 15)
        offset = np.clip(offset, 15, 120)
        confirmations[pk] = gt - timedelta(minutes=float(offset))
    return confirmations


def compute_check_times(game_times_et: dict[int, datetime], offset_min: int = 45) -> list[datetime]:
    """Compute scheduler check times (game_time - offset), sorted."""
    checks = set()
    for gt in game_times_et.values():
        if gt:
            checks.add(gt - timedelta(minutes=offset_min))
    return sorted(checks)


def simulate_day(
    day_profiles: pd.DataFrame,
    game_times_et: dict[int, datetime],
    confirmation_times: dict[int, datetime],
    early_lock_gap: float = 0.03,
    min_confirmed_pct: float = 0.0,
) -> dict:
    """Simulate one day's scheduler lock flow.

    Returns dict with: locked_batter_id, locked_game_pk, oracle_batter_id,
    lock_check_num, n_confirmed_at_lock, n_total_games, matched_oracle.
    """
    if len(day_profiles) == 0:
        return None

    # Oracle: rank-1 from full backtest (all lineups confirmed)
    oracle = day_profiles[day_profiles["rank"] == 1].iloc[0]
    oracle_bid = oracle["batter_id"]

    # All game_pks on this day
    all_game_pks = set(day_profiles["game_pk"].dropna().astype(int).unique())
    n_total = len(all_game_pks)
    if n_total == 0:
        return None

    # Check times
    check_times = compute_check_times(game_times_et)
    if not check_times:
        return None

    # Simulate check-by-check
    for check_num, check_time in enumerate(check_times):
        # Which games have confirmed lineups at this time?
        confirmed_pks = {
            pk for pk, ct in confirmation_times.items()
            if ct <= check_time
        }
        n_confirmed = len(confirmed_pks & all_game_pks)

        # Min confirmed percentage guardrail
        if min_confirmed_pct > 0 and n_confirmed / n_total < min_confirmed_pct:
            continue

        # Find top confirmed pick and best projected pick
        confirmed_batters = day_profiles[day_profiles["game_pk"].isin(confirmed_pks)]
        projected_batters = day_profiles[~day_profiles["game_pk"].isin(confirmed_pks)]

        if len(confirmed_batters) == 0:
            continue

        top_confirmed = confirmed_batters.iloc[0]  # already ranked by p_game_hit

        # should_lock: top pick confirmed AND (all confirmed OR gap >= threshold)
        if len(projected_batters) == 0:
            # All confirmed — lock
            pass
        else:
            best_projected = projected_batters.iloc[0]
            gap = top_confirmed["p_game_hit"] - best_projected["p_game_hit"]
            if gap < early_lock_gap:
                continue  # gap too small, wait

        # Lock!
        return {
            "locked_batter_id": int(top_confirmed["batter_id"]),
            "locked_game_pk": int(top_confirmed["game_pk"]),
            "locked_p": float(top_confirmed["p_game_hit"]),
            "locked_hit": int(top_confirmed["actual_hit"]),
            "oracle_batter_id": int(oracle_bid),
            "oracle_p": float(oracle["p_game_hit"]),
            "oracle_hit": int(oracle["actual_hit"]),
            "matched_oracle": int(top_confirmed["batter_id"]) == int(oracle_bid),
            "lock_check_num": check_num,
            "n_confirmed_at_lock": n_confirmed,
            "n_total_games": n_total,
        }

    # Fallback: never locked via gap check. Post rank-1 of whatever is confirmed last.
    # In practice this means the last check happened and we force-post.
    all_confirmed = day_profiles[day_profiles["game_pk"].isin(confirmation_times.keys())]
    if len(all_confirmed) > 0:
        top = all_confirmed.iloc[0]
        return {
            "locked_batter_id": int(top["batter_id"]),
            "locked_game_pk": int(top["game_pk"]),
            "locked_p": float(top["p_game_hit"]),
            "locked_hit": int(top["actual_hit"]),
            "oracle_batter_id": int(oracle_bid),
            "oracle_p": float(oracle["p_game_hit"]),
            "oracle_hit": int(oracle["actual_hit"]),
            "matched_oracle": int(top["batter_id"]) == int(oracle_bid),
            "lock_check_num": -1,  # fallback
            "n_confirmed_at_lock": len(confirmation_times),
            "n_total_games": n_total,
        }

    return None


def run_simulation(profiles: pd.DataFrame, variant: dict, n_trials: int) -> dict:
    """Run the timing simulation across all days for one variant."""
    dates = sorted(profiles["date"].unique())

    # Precompute game_times_et per day
    day_game_times = {}
    for date in dates:
        day_df = profiles[profiles["date"] == date]
        game_pks = day_df["game_pk"].dropna().astype(int).unique()
        gt_et = {}
        for pk in game_pks:
            gt_str = day_df[day_df["game_pk"] == pk]["game_time"].iloc[0]
            et = game_time_to_et(gt_str)
            if et:
                gt_et[int(pk)] = et
        day_game_times[date] = gt_et

    trial_results = []
    for trial in range(n_trials):
        rng = np.random.default_rng(seed=trial)
        trial_matches = 0
        trial_total = 0
        trial_locked_hits = 0
        trial_oracle_hits = 0

        for date in dates:
            day_df = profiles[profiles["date"] == date].copy()
            gt_et = day_game_times[date]
            if not gt_et:
                continue

            game_pks = list(gt_et.keys())
            conf_times = simulate_confirmation_times(game_pks, gt_et, rng)

            result = simulate_day(
                day_df, gt_et, conf_times,
                early_lock_gap=variant["early_lock_gap"],
                min_confirmed_pct=variant["min_confirmed_pct"],
            )

            if result:
                trial_total += 1
                trial_matches += result["matched_oracle"]
                trial_locked_hits += result["locked_hit"]
                trial_oracle_hits += result["oracle_hit"]

        if trial_total > 0:
            trial_results.append({
                "accuracy": trial_matches / trial_total,
                "locked_p1": trial_locked_hits / trial_total,
                "oracle_p1": trial_oracle_hits / trial_total,
                "n_days": trial_total,
            })

    if not trial_results:
        return {"accuracy": 0, "locked_p1": 0, "oracle_p1": 0}

    return {
        "accuracy_median": float(np.median([r["accuracy"] for r in trial_results])),
        "accuracy_p5": float(np.percentile([r["accuracy"] for r in trial_results], 5)),
        "accuracy_p95": float(np.percentile([r["accuracy"] for r in trial_results], 95)),
        "locked_p1_median": float(np.median([r["locked_p1"] for r in trial_results])),
        "oracle_p1_median": float(np.median([r["oracle_p1"] for r in trial_results])),
        "p1_penalty_median": float(np.median(
            [r["oracle_p1"] - r["locked_p1"] for r in trial_results]
        )),
        "n_days": trial_results[0]["n_days"],
    }


def main():
    # Load profiles
    print("Loading backtest profiles...", file=sys.stderr)
    profiles_dfs = []
    sim_dir = Path("data/simulation")
    for season in TEST_SEASONS:
        path = sim_dir / f"backtest_{season}.parquet"
        if path.exists():
            profiles_dfs.append(pd.read_parquet(path))
    profiles = pd.concat(profiles_dfs, ignore_index=True)
    profiles["date"] = pd.to_datetime(profiles["date"])

    # Load PA data for game_pk join
    print("Loading PA data for game_pk mapping...", file=sys.stderr)
    proc = Path("data/processed")
    pa_dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
    pa_data = pd.concat(pa_dfs, ignore_index=True)

    # Game time lookup
    print("Building game time lookup...", file=sys.stderr)
    gt_lookup = build_game_time_lookup()

    # Enrich profiles
    profiles = enrich_profiles_with_game_info(profiles, pa_data, gt_lookup)

    # Drop rows without game info
    valid = profiles["game_pk"].notna() & profiles["game_time"].notna()
    print(f"Valid profiles: {valid.sum()}/{len(profiles)}", file=sys.stderr)
    profiles = profiles[valid].copy()
    profiles["game_pk"] = profiles["game_pk"].astype(int)

    # Sort by date + rank for consistent ordering
    profiles = profiles.sort_values(["date", "rank"]).reset_index(drop=True)

    # Run simulation for each variant
    print(f"\nRunning {len(VARIANTS)} variants × {N_TRIALS} trials...\n", file=sys.stderr)

    results = {}
    for variant in VARIANTS:
        print(f"  {variant['name']}...", file=sys.stderr, end="", flush=True)
        r = run_simulation(profiles, variant, N_TRIALS)
        results[variant["name"]] = r
        print(f" accuracy={r['accuracy_median']:.3f} "
              f"[{r['accuracy_p5']:.3f}-{r['accuracy_p95']:.3f}], "
              f"P@1 penalty={r['p1_penalty_median']:+.4f}", file=sys.stderr)

    # Print results table
    print(f"\n{'='*90}")
    print(f"Phase 6: Scheduler Timing Simulation ({N_TRIALS} trials per variant)")
    print(f"{'='*90}")
    print(f"{'Variant':<15} | {'Accuracy':>10} | {'95% CI':>15} | {'Locked P@1':>11} | "
          f"{'Oracle P@1':>11} | {'Penalty':>8}")
    print("-" * 90)
    for name, r in results.items():
        print(f"{name:<15} | {r['accuracy_median']:>10.3f} | "
              f"[{r['accuracy_p5']:.3f}-{r['accuracy_p95']:.3f}] | "
              f"{r['locked_p1_median']:>11.4f} | {r['oracle_p1_median']:>11.4f} | "
              f"{r['p1_penalty_median']:>+8.4f}")
    print("=" * 90)
    print(f"\nAccuracy = fraction of days where locked pick matches oracle (all-confirmed) pick")
    print(f"Penalty = oracle P@1 - locked P@1 (positive = oracle is better)")

    # Decision
    baseline_acc = results["baseline"]["accuracy_median"]
    if baseline_acc > 0.95:
        print(f"\nBaseline accuracy {baseline_acc:.1%} > 95% — no guardrail needed.")
    elif baseline_acc < 0.90:
        best = min(results.items(), key=lambda x: x[1]["p1_penalty_median"])
        print(f"\nBaseline accuracy {baseline_acc:.1%} < 90% — consider guardrail: {best[0]}")
    else:
        print(f"\nBaseline accuracy {baseline_acc:.1%} — borderline. Check P(57) impact.")


if __name__ == "__main__":
    main()
