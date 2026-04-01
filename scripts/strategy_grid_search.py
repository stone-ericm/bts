"""Grid search over streak-aware strategy parameters + hybrid configs.

Run from project root:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/strategy_grid_search.py
"""

import sys
import time
from itertools import product
from pathlib import Path

from rich.console import Console
from rich.table import Table

from bts.simulate.strategies import Strategy
from bts.simulate.monte_carlo import load_all_profiles, run_monte_carlo

PROFILES_DIR = Path("data/simulation")
N_TRIALS = 10_000
SEED = 42


def make_streak_config(
    early_double: float | None = 0.55,
    saver_double: float | None = 0.60,
    mid_skip: float | None = 0.78,
    mid_double: float | None = 0.65,
    lock_skip: float | None = 0.80,
    lock_double: float | None = None,
    sprint_skip: float | None = 0.78,
    sprint_double: float | None = 0.60,
) -> tuple:
    return (
        (9, None, early_double),
        (15, None, saver_double),
        (30, mid_skip, mid_double),
        (45, lock_skip, lock_double),
        (56, sprint_skip, sprint_double),
    )


def run_search(profiles, strategies: dict[str, Strategy]) -> list[tuple]:
    """Run Monte Carlo for each strategy, return sorted results."""
    results = []
    for name, strategy in strategies.items():
        t0 = time.time()
        result = run_monte_carlo(profiles, strategy, n_trials=N_TRIALS, seed=SEED)
        elapsed = time.time() - t0
        results.append((name, result, elapsed))
    results.sort(key=lambda x: -x[1].p_57)
    return results


def print_results(results, title, console, top_n=20):
    table = Table(title=title)
    table.add_column("Strategy")
    table.add_column("P(57)", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("P(30+)", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("95th", justify="right")
    table.add_column("Play Days", justify="right")

    for name, result, _ in results[:top_n]:
        table.add_row(
            name,
            f"{result.p_57:.2%}",
            f"[{result.ci_95_lower:.2%}, {result.ci_95_upper:.2%}]",
            f"{result.p_30:.1%}",
            str(result.median_streak),
            str(result.p95_streak),
            f"{result.mean_play_days:.0f}",
        )
    console.print(table)


def main():
    console = Console()
    console.print("Loading profiles...", style="bold")
    profiles = load_all_profiles(PROFILES_DIR)
    console.print(f"Loaded {len(profiles)} daily profiles\n")

    # =========================================================
    # SECTION 1: Hybrid configs — combine skip + dynamic doubling
    # =========================================================
    console.print("[bold]Section 1: Hybrid Configs[/bold]")
    hybrids = {}

    # Current best for reference
    hybrids["streak-aware (baseline)"] = Strategy(
        name="streak-aware",
        streak_config=make_streak_config(),
    )

    # Global skip + streak-aware doubling
    for skip in [0.76, 0.78, 0.80, 0.82]:
        hybrids[f"global-skip-{skip}+sa-double"] = Strategy(
            name=f"gs{skip}",
            streak_config=(
                (9, skip, 0.55),
                (15, skip, 0.60),
                (30, skip, 0.65),
                (45, skip, None),
                (56, skip, 0.60),
            ),
        )

    # Aggressive skip early, conservative late
    hybrids["skip-gradient (none→0.82)"] = Strategy(
        name="skip-grad",
        streak_config=(
            (9, None, 0.55),
            (15, None, 0.60),
            (30, 0.78, 0.65),
            (45, 0.82, None),
            (56, 0.80, 0.60),
        ),
    )

    # No skip at all but dynamic doubling
    hybrids["no-skip+dynamic-double"] = Strategy(
        name="no-skip-dd",
        streak_config=(
            (9, None, 0.55),
            (15, None, 0.60),
            (30, None, 0.65),
            (45, None, None),
            (56, None, 0.60),
        ),
    )

    # Max aggression early, max conservation late
    hybrids["polarized"] = Strategy(
        name="polarized",
        streak_config=(
            (9, None, 0.45),
            (15, None, 0.55),
            (30, 0.78, 0.65),
            (45, 0.82, None),
            (56, 0.78, 0.55),
        ),
    )

    # Never double past 20
    hybrids["double-early-only"] = Strategy(
        name="double-early",
        streak_config=(
            (9, None, 0.55),
            (15, None, 0.60),
            (20, None, 0.65),
            (45, 0.78, None),
            (56, 0.78, None),
        ),
    )

    # Always double when possible (no lockdown)
    hybrids["always-double-sa"] = Strategy(
        name="always-dbl",
        streak_config=(
            (9, None, 0.55),
            (15, None, 0.60),
            (30, 0.78, 0.60),
            (45, 0.80, 0.60),
            (56, 0.78, 0.55),
        ),
    )

    results = run_search(profiles, hybrids)
    print_results(results, "Hybrid Configs (10K trials)", console)

    best_hybrid_name = results[0][0]
    console.print(f"\nBest hybrid: [bold green]{best_hybrid_name}[/bold green] "
                  f"({results[0][1].p_57:.2%})\n")

    # =========================================================
    # SECTION 2: Grid search on early + lockdown + sprint
    # =========================================================
    console.print("[bold]Section 2: Grid Search — Key Parameters[/bold]")
    grid = {}

    early_doubles = [0.45, 0.50, 0.55, 0.60]
    lock_configs = [(0.78, None), (0.80, None), (0.78, 0.60), (0.80, 0.65)]
    sprint_doubles = [None, 0.55, 0.60, 0.65]

    for ed, (ls, ld), sd in product(early_doubles, lock_configs, sprint_doubles):
        name = f"e{ed:.2f}_l{ls}{'+d' + str(ld) if ld else ''}_s{sd or 'none'}"
        grid[name] = Strategy(
            name=name,
            streak_config=(
                (9, None, ed),
                (15, None, 0.60),
                (30, 0.78, 0.65),
                (45, ls, ld),
                (56, 0.78, sd),
            ),
        )

    console.print(f"Testing {len(grid)} configurations...")
    results = run_search(profiles, grid)
    print_results(results, f"Grid Search — Top 15 of {len(grid)} (10K trials)", console, top_n=15)

    best_grid_name = results[0][0]
    best_grid = results[0][1]
    console.print(f"\nBest grid config: [bold green]{best_grid_name}[/bold green] "
                  f"({best_grid.p_57:.2%})\n")

    # =========================================================
    # SECTION 3: Fine-tune around the best grid config
    # =========================================================
    console.print("[bold]Section 3: Fine-tune around best[/bold]")

    # Parse the best config to understand what won
    # Then test variations around it
    fine = {}

    # Vary mid-skip threshold
    for ms in [None, 0.76, 0.77, 0.78, 0.79, 0.80]:
        fine[f"mid-skip-{ms}"] = Strategy(
            name=f"ms{ms}",
            streak_config=(
                (9, None, 0.55),
                (15, None, 0.60),
                (30, ms, 0.65),
                (45, 0.80, None),
                (56, 0.78, 0.60),
            ),
        )

    # Vary saver-zone doubling
    for sd in [0.50, 0.55, 0.60, 0.65]:
        fine[f"saver-double-{sd}"] = Strategy(
            name=f"sd{sd}",
            streak_config=(
                (9, None, 0.55),
                (15, None, sd),
                (30, 0.78, 0.65),
                (45, 0.80, None),
                (56, 0.78, 0.60),
            ),
        )

    # Different phase boundaries
    fine["boundary-12-25-40"] = Strategy(
        name="b12-25-40",
        streak_config=(
            (12, None, 0.55),
            (15, None, 0.60),
            (25, 0.78, 0.65),
            (40, 0.80, None),
            (56, 0.78, 0.60),
        ),
    )

    fine["boundary-8-15-35"] = Strategy(
        name="b8-15-35",
        streak_config=(
            (8, None, 0.55),
            (15, None, 0.60),
            (35, 0.78, 0.65),
            (45, 0.80, None),
            (56, 0.78, 0.60),
        ),
    )

    # More phases — add a "coast" phase at 46-50 and sprint 51-56
    fine["6-phase"] = Strategy(
        name="6-phase",
        streak_config=(
            (9, None, 0.55),
            (15, None, 0.60),
            (30, 0.78, 0.65),
            (45, 0.80, None),
            (50, 0.78, None),
            (56, 0.78, 0.60),
        ),
    )

    # Nuclear option: very tight skip everywhere past 20
    fine["ultra-conservative-late"] = Strategy(
        name="ultra-cons",
        streak_config=(
            (9, None, 0.55),
            (15, None, 0.60),
            (20, 0.78, 0.65),
            (45, 0.82, None),
            (56, 0.80, 0.55),
        ),
    )

    results = run_search(profiles, fine)
    print_results(results, "Fine-tuning (10K trials)", console)

    console.print(f"\nBest fine-tuned: [bold green]{results[0][0]}[/bold green] "
                  f"({results[0][1].p_57:.2%})")


if __name__ == "__main__":
    main()
