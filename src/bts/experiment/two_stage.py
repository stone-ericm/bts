"""Two-stage screening: run n1 seeds, kill or promote based on Bayesian posterior.

Decision rule: beta-binomial conjugate on "win rate" (fraction of seeds where
the experiment's avg P@1 delta > 0). Uniform Beta(1,1) prior.

After n1 seeds with w wins: posterior is Beta(1+w, 1+n1-w).
  P(win_rate > 0.5) < 0.15                 -> kill
  P(win_rate > 0.5) > 0.85 AND
      mean_delta > +0.001                  -> fast_track_ship_candidate
  otherwise                                -> promote
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scipy.stats import beta


@dataclass
class StageOneResult:
    name: str
    seeds_run: int
    wins: int
    mean_delta: float


@dataclass
class StageDecision:
    name: str
    action: Literal["kill", "promote", "fast_track_ship_candidate"]
    posterior_p_gt_half: float
    reason: str


def decide_after_stage_one(
    result: StageOneResult,
    kill_threshold: float = 0.15,
    fast_track_threshold: float = 0.85,
    fast_track_mean_delta_min: float = 0.001,
) -> StageDecision:
    a = 1 + result.wins
    b = 1 + (result.seeds_run - result.wins)
    p_gt_half = float(1 - beta.cdf(0.5, a, b))

    if p_gt_half < kill_threshold:
        return StageDecision(
            name=result.name, action="kill",
            posterior_p_gt_half=p_gt_half,
            reason=f"P(rate>0.5)={p_gt_half:.3f} < {kill_threshold}",
        )
    if p_gt_half > fast_track_threshold and result.mean_delta > fast_track_mean_delta_min:
        return StageDecision(
            name=result.name, action="fast_track_ship_candidate",
            posterior_p_gt_half=p_gt_half,
            reason=(f"P(rate>0.5)={p_gt_half:.3f} > {fast_track_threshold} + "
                    f"mean_delta={result.mean_delta:.4f} > {fast_track_mean_delta_min}"),
        )
    return StageDecision(
        name=result.name, action="promote",
        posterior_p_gt_half=p_gt_half,
        reason=f"inconclusive: P(rate>0.5)={p_gt_half:.3f}",
    )


def aggregate_stage_one_results(
    seed_dirs: list[Path],
    experiment_names: list[str],
) -> dict[str, StageOneResult]:
    """Given stage-one seed output dirs (each containing phase1/<exp>/diff.json),
    compute StageOneResult per experiment_name.

    Wins = number of seeds where diff["precision"]["1"]["delta"] > 0
    mean_delta = mean of diff["precision"]["1"]["delta"] across seeds
    """
    import json

    out: dict[str, StageOneResult] = {}
    for exp in experiment_names:
        deltas = []
        for seed_dir in seed_dirs:
            diff_path = seed_dir / "phase1" / exp / "diff.json"
            if not diff_path.exists():
                continue
            diff = json.loads(diff_path.read_text())
            d = diff.get("precision", {}).get("1", {}).get("delta")
            if d is not None:
                deltas.append(float(d))
        if not deltas:
            continue
        wins = sum(1 for d in deltas if d > 0)
        mean = sum(deltas) / len(deltas)
        out[exp] = StageOneResult(name=exp, seeds_run=len(deltas), wins=wins, mean_delta=mean)
    return out
