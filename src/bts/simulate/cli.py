# src/bts/simulate/cli.py
"""CLI commands for BTS strategy simulation."""

import json
import sys
from pathlib import Path

import click

from bts.simulate.strategies import ALL_STRATEGIES


@click.group()
def simulate():
    """Strategy simulation and backtesting."""
    pass


@simulate.command()
@click.option("--seasons", default="2021,2022,2023,2024,2025",
              help="Comma-separated seasons to backtest")
@click.option("--data-dir", default="data/processed", type=click.Path(),
              help="Processed parquet directory")
@click.option("--output-dir", default="data/simulation", type=click.Path(),
              help="Output directory for profile parquets")
@click.option("--retrain-every", default=7, type=int,
              help="Retrain blend models every N days")
def backtest(seasons: str, data_dir: str, output_dir: str, retrain_every: int):
    """Run blend walk-forward backtest and save daily profiles."""
    from bts.simulate.backtest_blend import run_backtest

    season_list = [int(s.strip()) for s in seasons.split(",")]
    click.echo(f"Running blend backtest for seasons: {season_list}")
    run_backtest(data_dir, output_dir, season_list, retrain_every)
    click.echo("Done.")


@simulate.command(name="run")
@click.option("--profiles-dir", default="data/simulation", type=click.Path(exists=True),
              help="Directory with backtest profile parquets")
@click.option("--trials", default=10_000, type=int,
              help="Number of Monte Carlo trials per strategy")
@click.option("--season-length", default=180, type=int,
              help="Days per simulated season")
@click.option("--strategy", "strategy_name", default=None,
              help="Run only this strategy (default: all)")
@click.option("--replay-only", is_flag=True,
              help="Only replay actual seasons, no Monte Carlo")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--save-json", default=None, type=click.Path(),
              help="Save raw results to JSON")
def run_sim(profiles_dir: str, trials: int, season_length: int,
            strategy_name: str | None, replay_only: bool, seed: int,
            save_json: str | None):
    """Run Monte Carlo strategy simulation."""
    from rich.console import Console
    from rich.table import Table
    from bts.simulate.monte_carlo import (
        load_all_profiles, load_season_profiles, run_monte_carlo, run_replay,
    )

    profiles_path = Path(profiles_dir)
    console = Console()

    strategies = ALL_STRATEGIES
    if strategy_name:
        if strategy_name not in ALL_STRATEGIES:
            click.echo(f"Unknown strategy: {strategy_name}. "
                       f"Options: {', '.join(ALL_STRATEGIES.keys())}", err=True)
            raise SystemExit(1)
        strategies = {strategy_name: ALL_STRATEGIES[strategy_name]}

    if replay_only:
        season_data = load_season_profiles(profiles_path)
        if not season_data:
            click.echo("No profile parquets found.", err=True)
            raise SystemExit(1)

        table = Table(title=f"Replay Results ({len(season_data)} seasons)")
        table.add_column("Strategy")
        for s in sorted(season_data.keys()):
            table.add_column(str(s), justify="right")
        table.add_column("Best", justify="right")

        for name, strategy in strategies.items():
            results = run_replay(season_data, strategy)
            streaks = [results[s].max_streak for s in sorted(results.keys())]
            row = [name] + [str(s) for s in streaks] + [str(max(streaks))]
            table.add_row(*row)

        console.print(table)
        return

    # Monte Carlo mode
    profiles = load_all_profiles(profiles_path)
    if not profiles:
        click.echo("No profile parquets found.", err=True)
        raise SystemExit(1)

    table = Table(title=f"Strategy Comparison ({trials:,} seasons, {len(profiles)} daily profiles)")
    table.add_column("Strategy")
    table.add_column("P(57)", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("P(30+)", justify="right")
    table.add_column("P(20+)", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("95th", justify="right")
    table.add_column("Play Days", justify="right")

    json_results = {}

    for name, strategy in strategies.items():
        result = run_monte_carlo(profiles, strategy, n_trials=trials,
                                  season_length=season_length, seed=seed)
        table.add_row(
            name,
            f"{result.p_57:.2%}",
            f"[{result.ci_95_lower:.2%}, {result.ci_95_upper:.2%}]",
            f"{result.p_30:.1%}",
            f"{result.p_20:.1%}",
            str(result.median_streak),
            str(result.p95_streak),
            f"{result.mean_play_days:.0f}",
        )
        json_results[name] = {
            "p_57": result.p_57, "p_30": result.p_30, "p_20": result.p_20,
            "median": result.median_streak, "p95": result.p95_streak,
            "mean_play_days": result.mean_play_days,
            "ci_95": [result.ci_95_lower, result.ci_95_upper],
        }

    console.print(table)

    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(save_json).write_text(json.dumps(json_results, indent=2))
        click.echo(f"Results saved to {save_json}")


@simulate.command()
@click.option("--profiles-dir", default="data/simulation", type=click.Path(exists=True),
              help="Directory with backtest profile parquets")
@click.option("--season-length", default=180, type=int, help="Days per season")
def solve(profiles_dir: str, season_length: int):
    """Solve MDP for optimal strategy — exact P(57) and policy extraction."""
    from rich.console import Console
    from bts.simulate.quality_bins import compute_bins
    from bts.simulate.exact import exact_p57
    from bts.simulate.mdp import solve_mdp

    import pandas as pd

    console = Console()
    profiles_path = Path(profiles_dir)

    # Load profiles and compute bins
    dfs = [pd.read_parquet(p) for p in sorted(profiles_path.glob("backtest_*.parquet"))]
    if not dfs:
        click.echo("No profile parquets found.", err=True)
        raise SystemExit(1)
    profiles_df = pd.concat(dfs, ignore_index=True)
    bins = compute_bins(profiles_df)

    console.print(f"[bold]Quality Bins ({len(bins.bins)} bins, "
                  f"{profiles_df['date'].nunique()} days)[/bold]")
    for b in bins.bins:
        console.print(f"  Q{b.index+1} [{b.p_range[0]:.3f}-{b.p_range[1]:.3f}]: "
                      f"P(hit)={b.p_hit:.1%}  P(both)={b.p_both:.1%}  freq={b.frequency:.1%}")

    # Solve MDP
    console.print(f"\n[bold]Solving MDP ({season_length} days)...[/bold]")
    sol = solve_mdp(bins, season_length=season_length)
    console.print(f"  [green]Optimal P(57) = {sol.optimal_p57:.4%}[/green]")

    # Compare with heuristic
    heuristic = ALL_STRATEGIES.get("combined", ALL_STRATEGIES["streak-aware"])
    p_heuristic = exact_p57(heuristic, bins, season_length=season_length)
    console.print(f"  Heuristic P(57) = {p_heuristic:.4%}")
    gap = sol.optimal_p57 - p_heuristic
    if gap > 0.0001:
        console.print(f"  [yellow]Gap: +{gap:.4%} — room for improvement[/yellow]")
    else:
        console.print(f"  [green]Gap: {gap:.4%} — heuristic is near-optimal[/green]")

    # Policy summary
    console.print(f"\n[bold]Policy Summary:[/bold]")
    console.print(sol.extract_thresholds())


@simulate.command()
@click.option("--profiles-dir", default="data/simulation", type=click.Path(exists=True),
              help="Directory with backtest profile parquets")
@click.option("--strategy", "strategy_name", default="combined",
              help="Strategy to evaluate (default: combined)")
@click.option("--season-length", default=180, type=int, help="Days per season")
def exact(profiles_dir: str, strategy_name: str, season_length: int):
    """Compute exact P(57) for a named strategy via absorbing chain."""
    from rich.console import Console
    from bts.simulate.quality_bins import compute_bins
    from bts.simulate.exact import exact_p57

    import pandas as pd

    console = Console()
    profiles_path = Path(profiles_dir)

    if strategy_name not in ALL_STRATEGIES:
        click.echo(f"Unknown strategy: {strategy_name}. "
                   f"Options: {', '.join(ALL_STRATEGIES.keys())}", err=True)
        raise SystemExit(1)

    dfs = [pd.read_parquet(p) for p in sorted(profiles_path.glob("backtest_*.parquet"))]
    if not dfs:
        click.echo("No profile parquets found.", err=True)
        raise SystemExit(1)
    profiles_df = pd.concat(dfs, ignore_index=True)
    bins = compute_bins(profiles_df)

    strategy = ALL_STRATEGIES[strategy_name]
    p = exact_p57(strategy, bins, season_length=season_length)
    console.print(f"{strategy_name}: P(57) = {p:.4%}  (exact, {season_length}-day season)")
