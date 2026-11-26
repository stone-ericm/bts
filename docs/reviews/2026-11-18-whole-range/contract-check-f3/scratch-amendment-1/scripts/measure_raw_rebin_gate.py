#!/usr/bin/env python3
"""Measure Gate B raw-distribution MDP re-bin support.

This is an offline diagnostic. It does not write or swap policy artifacts.

Gate B asks whether re-binning and re-solving the MDP on the current raw
production probability distribution has enough evidence to justify a future
candidate policy evaluation. This script runs the light point measurement and
support checks; heavier bootstrap or multi-season harness work should run
off-host, not on the live production machine.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Sequence

import numpy as np
import pandas as pd

from bts.simulate.exact import exact_p57_policy_table
from bts.simulate.mdp import load_policy, solve_mdp
from bts.simulate.quality_bins import QualityBin, QualityBins


DEFAULT_N_BINS = (2, 3, 4, 5)
DEFAULT_MIN_N = 200
DEFAULT_MIN_PER_BIN = 30
DEFAULT_SEASON_LENGTH = 180


@dataclass(frozen=True)
class PairRow:
    date: str
    p1: float
    p2: float
    y1: int
    y2: int


def _day_hit_lookup(pa_parquet: Path) -> dict[tuple[int, date], int]:
    pa = pd.read_parquet(pa_parquet)
    pa = pa.copy()
    pa["date"] = pd.to_datetime(pa["date"]).dt.date
    daily = pa.groupby(["batter_id", "date"], as_index=False)["is_hit"].max()
    return {
        (int(row.batter_id), row.date): int(row.is_hit)
        for row in daily.itertuples(index=False)
    }


def load_resolved_pair_rows(
    picks_dir: Path,
    pa_parquet: Path,
    today: date,
) -> list[PairRow]:
    """Load resolved production primary/DD pairs joined to actual day hits."""
    lookup = _day_hit_lookup(pa_parquet)
    rows: list[PairRow] = []
    for path in sorted(picks_dir.glob("*.json")):
        if "." in path.stem:
            continue
        try:
            body = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        try:
            pick_date = date.fromisoformat(body.get("date") or path.stem)
        except (TypeError, ValueError):
            continue
        if pick_date > today or body.get("result") not in ("hit", "miss"):
            continue

        primary = body.get("pick") or {}
        dd = body.get("double_down") or {}
        p1 = primary.get("p_game_hit")
        p2 = dd.get("p_game_hit")
        b1 = primary.get("batter_id")
        b2 = dd.get("batter_id")
        if any(value is None for value in (p1, p2, b1, b2)):
            continue
        y1 = lookup.get((int(b1), pick_date))
        y2 = lookup.get((int(b2), pick_date))
        if y1 is None or y2 is None:
            continue
        rows.append(PairRow(
            date=pick_date.isoformat(),
            p1=float(p1),
            p2=float(p2),
            y1=int(y1),
            y2=int(y2),
        ))
    return rows


def _classify(p_game_hit: float, boundaries: Sequence[float]) -> int:
    q = 0
    for boundary in boundaries:
        if p_game_hit >= boundary:
            q += 1
    return q


def _representative_p(bin_: QualityBin) -> float:
    lo, hi = bin_.p_range
    if np.isneginf(lo):
        return float(hi)
    if np.isposinf(hi):
        return float(lo)
    return float((lo + hi) / 2.0)


def quality_bins_from_pair_rows(rows: Sequence[PairRow], n_bins: int) -> QualityBins:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if not rows:
        raise ValueError("cannot compute bins from zero rows")

    p = np.asarray([row.p1 for row in rows], dtype=float)
    boundaries = [float(np.quantile(p, i / n_bins)) for i in range(1, n_bins)]
    assignments = np.digitize(p, boundaries)

    bins: list[QualityBin] = []
    for i in range(n_bins):
        cell = [row for row, q in zip(rows, assignments) if q == i]
        if not cell:
            lower = float("-inf") if i == 0 else boundaries[i - 1]
            upper = float("inf") if i == len(boundaries) else boundaries[i]
            bins.append(QualityBin(i, (lower, upper), 0.0, 0.0, 0.0))
            continue
        bins.append(QualityBin(
            index=i,
            p_range=(min(row.p1 for row in cell), max(row.p1 for row in cell)),
            p_hit=mean(row.y1 for row in cell),
            p_both=mean(1 if row.y1 and row.y2 else 0 for row in cell),
            frequency=len(cell) / len(rows),
        ))
    return QualityBins(bins=bins, boundaries=boundaries)


def project_policy_to_candidate_bins(
    policy_table: np.ndarray,
    policy_boundaries: Sequence[float],
    candidate_bins: QualityBins,
) -> tuple[np.ndarray, list[int]]:
    """Project a saved policy onto candidate bins using bin representative p.

    This is exact when each candidate bin lies wholly inside one saved-policy
    boundary interval. It is an approximation for bins that cross an old
    policy boundary; the output reports the mapping for review.
    """
    out = np.empty(
        (
            policy_table.shape[0],
            policy_table.shape[1],
            policy_table.shape[2],
            len(candidate_bins.bins),
        ),
        dtype=policy_table.dtype,
    )
    mapping: list[int] = []
    for bin_ in candidate_bins.bins:
        old_q = _classify(_representative_p(bin_), policy_boundaries)
        mapping.append(old_q)
        out[:, :, :, bin_.index] = policy_table[:, :, :, old_q]
    return out, mapping


def row_summary(rows: Sequence[PairRow]) -> dict:
    if not rows:
        return {
            "n": 0,
            "date_min": None,
            "date_max": None,
            "p1_min": None,
            "p1_max": None,
            "p1_mean": None,
            "p1_hit_rate": None,
            "p_both_independent_mean": None,
            "both_hit_rate": None,
        }
    return {
        "n": len(rows),
        "date_min": min(row.date for row in rows),
        "date_max": max(row.date for row in rows),
        "p1_min": min(row.p1 for row in rows),
        "p1_max": max(row.p1 for row in rows),
        "p1_mean": mean(row.p1 for row in rows),
        "p1_hit_rate": mean(row.y1 for row in rows),
        "p_both_independent_mean": mean(row.p1 * row.p2 for row in rows),
        "both_hit_rate": mean(1 if row.y1 and row.y2 else 0 for row in rows),
    }


def _bin_to_dict(bin_: QualityBin, total_n: int) -> dict:
    return {
        "index": int(bin_.index),
        "n": int(round(bin_.frequency * total_n)),
        "p_range": [float(bin_.p_range[0]), float(bin_.p_range[1])],
        "p_hit": float(bin_.p_hit),
        "p_both": float(bin_.p_both),
        "frequency": float(bin_.frequency),
    }


def evaluate_rebin_candidates(
    rows: Sequence[PairRow],
    *,
    policy_path: Path,
    n_bins_values: Sequence[int] = DEFAULT_N_BINS,
    min_n: int = DEFAULT_MIN_N,
    min_per_bin: int = DEFAULT_MIN_PER_BIN,
    season_length: int = DEFAULT_SEASON_LENGTH,
) -> dict:
    policy_table, policy_boundaries, policy_length = load_policy(policy_path)
    evaluations = []

    for n_bins in n_bins_values:
        candidate_bins = quality_bins_from_pair_rows(rows, n_bins)
        projected_policy, old_q_mapping = project_policy_to_candidate_bins(
            policy_table,
            policy_boundaries,
            candidate_bins,
        )
        candidate_solution = solve_mdp(candidate_bins, season_length=season_length)
        baseline_p57 = exact_p57_policy_table(
            projected_policy,
            candidate_bins,
            season_length=season_length,
        )
        candidate_p57 = exact_p57_policy_table(
            candidate_solution.policy_table,
            candidate_bins,
            season_length=season_length,
        )
        bin_counts = [int(round(bin_.frequency * len(rows))) for bin_ in candidate_bins.bins]
        evaluations.append({
            "n_bins": int(n_bins),
            "boundaries": [float(x) for x in candidate_bins.boundaries],
            "projected_policy_old_q_mapping": old_q_mapping,
            "min_bin_n": min(bin_counts) if bin_counts else 0,
            "bins": [_bin_to_dict(bin_, len(rows)) for bin_ in candidate_bins.bins],
            "projected_baseline_p57": float(baseline_p57),
            "candidate_optimal_p57": float(candidate_p57),
            "gap": float(candidate_p57 - baseline_p57),
        })

    if len(rows) < min_n:
        decision = "INSUFFICIENT_SUPPORT"
        reason = f"n={len(rows)} below min_n={min_n}"
    elif any(item["min_bin_n"] < min_per_bin for item in evaluations):
        decision = "INSUFFICIENT_SUPPORT"
        reason = f"at least one evaluated binning has min_bin_n below {min_per_bin}"
    elif max(item["gap"] for item in evaluations) <= 0:
        decision = "NO_POINT_IMPROVEMENT"
        reason = "no raw re-bin candidate improves point P(57)"
    else:
        decision = "POINT_SIGNAL_REQUIRES_BACKTEST"
        reason = "point P(57) improved; run off-host multi-season policy-file harness"

    return {
        "policy_path": str(policy_path),
        "policy_boundaries": [float(x) for x in policy_boundaries],
        "policy_season_length": int(policy_length),
        "season_length": int(season_length),
        "min_n": int(min_n),
        "min_per_bin": int(min_per_bin),
        "decision": decision,
        "reason": reason,
        "evaluations": evaluations,
    }


def profile_distribution_summary(profiles_dir: Path) -> dict | None:
    files = sorted(profiles_dir.glob("backtest_*.parquet"))
    if not files:
        return None
    frames = [pd.read_parquet(path, columns=["date", "rank", "p_game_hit"]) for path in files]
    df = pd.concat(frames, ignore_index=True)
    rank1 = df[df["rank"] == 1]["p_game_hit"].astype(float)
    if rank1.empty:
        return None
    return {
        "profiles_dir": str(profiles_dir),
        "files": [path.name for path in files],
        "n_rank1": int(rank1.shape[0]),
        "p_min": float(rank1.min()),
        "p_max": float(rank1.max()),
        "p_mean": float(rank1.mean()),
        "p_median": float(rank1.median()),
        "quantiles": {
            str(q): float(rank1.quantile(q))
            for q in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
        },
    }


def distribution_mismatch_flags(rows: Sequence[PairRow], profile_summary: dict | None) -> dict:
    if not rows or profile_summary is None:
        return {
            "available": False,
            "current_max_below_profile_q20": None,
            "current_median_below_profile_q20": None,
        }
    current = row_summary(rows)
    profile_q20 = profile_summary["quantiles"]["0.2"]
    return {
        "available": True,
        "current_p1_min": current["p1_min"],
        "current_p1_max": current["p1_max"],
        "current_p1_mean": current["p1_mean"],
        "current_p1_median": float(np.median([row.p1 for row in rows])),
        "profile_q20": profile_q20,
        "profile_median": profile_summary["p_median"],
        "current_max_below_profile_q20": bool(current["p1_max"] < profile_q20),
        "current_median_below_profile_q20": bool(
            float(np.median([row.p1 for row in rows])) < profile_q20
        ),
    }


def run_measurement(
    *,
    picks_dir: Path,
    pa_parquet: Path,
    today: date,
    policy_path: Path,
    profiles_dir: Path | None = None,
    n_bins_values: Sequence[int] = DEFAULT_N_BINS,
    min_n: int = DEFAULT_MIN_N,
    min_per_bin: int = DEFAULT_MIN_PER_BIN,
    season_length: int = DEFAULT_SEASON_LENGTH,
) -> dict:
    rows = load_resolved_pair_rows(picks_dir, pa_parquet, today)
    profile_summary = profile_distribution_summary(profiles_dir) if profiles_dir else None
    return {
        "schema_version": "raw_rebin_gate_measure_v1",
        "artifact_role": "gate_b_raw_rebin_measure",
        "production_deploy_claim": False,
        "heavy_compute": False,
        "date": today.isoformat(),
        "inputs": {
            "picks_dir": str(picks_dir),
            "pa_parquet": str(pa_parquet),
            "policy_path": str(policy_path),
            "profiles_dir": str(profiles_dir) if profiles_dir else None,
            "n_bins_values": [int(x) for x in n_bins_values],
        },
        "row_summary": row_summary(rows),
        "profile_distribution": profile_summary,
        "distribution_mismatch": distribution_mismatch_flags(rows, profile_summary),
        "gate_b": evaluate_rebin_candidates(
            rows,
            policy_path=policy_path,
            n_bins_values=n_bins_values,
            min_n=min_n,
            min_per_bin=min_per_bin,
            season_length=season_length,
        ) if rows else {
            "decision": "INSUFFICIENT_SUPPORT",
            "reason": "no resolved primary/DD pair rows",
        },
        "methodology": {
            "current_data_scope": (
                "resolved production days with both primary and double-down slots "
                "joined to actual day-hit outcomes"
            ),
            "baseline_projection": (
                "saved production policy table projected onto candidate raw bins "
                "by representative primary p; exact when candidate bins do not "
                "cross saved policy boundaries"
            ),
            "candidate": "same reachability MDP re-solved on current raw bins",
            "not_full_gate": (
                "this is a support and point-measurement screen; a deployable "
                "Gate B claim still needs an off-host multi-season policy-file "
                "P(57) backtest on a verified matching probability distribution"
            ),
        },
    }


def _parse_n_bins(value: str) -> tuple[int, ...]:
    out = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not out:
        raise argparse.ArgumentTypeError("expected comma-separated n_bins values")
    if any(x < 1 for x in out):
        raise argparse.ArgumentTypeError("n_bins values must be >= 1")
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picks-dir", type=Path, default=Path("data/picks"))
    parser.add_argument("--pa-parquet", type=Path, default=None)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--policy-path", type=Path, default=Path("data/models/mdp_policy.npz"))
    parser.add_argument("--profiles-dir", type=Path, default=Path("data/simulation"))
    parser.add_argument("--n-bins", type=_parse_n_bins, default=DEFAULT_N_BINS)
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    parser.add_argument("--min-per-bin", type=int, default=DEFAULT_MIN_PER_BIN)
    parser.add_argument("--season-length", type=int, default=DEFAULT_SEASON_LENGTH)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    today = date.fromisoformat(args.date)
    pa_parquet = args.pa_parquet or Path(f"data/processed/pa_{today.year}.parquet")
    if not pa_parquet.exists():
        raise SystemExit(f"missing PA parquet: {pa_parquet}")
    if not args.policy_path.exists():
        raise SystemExit(f"missing policy artifact: {args.policy_path}")
    output = args.output or Path(f"data/validation/raw_rebin_gate_{today.isoformat()}.json")
    result = run_measurement(
        picks_dir=args.picks_dir,
        pa_parquet=pa_parquet,
        today=today,
        policy_path=args.policy_path,
        profiles_dir=args.profiles_dir if args.profiles_dir.exists() else None,
        n_bins_values=args.n_bins,
        min_n=args.min_n,
        min_per_bin=args.min_per_bin,
        season_length=args.season_length,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    rows = result["row_summary"]
    gate = result["gate_b"]
    print(
        f"Loaded {rows['n']} resolved primary/DD pair rows "
        f"({rows['date_min']}..{rows['date_max']})"
    )
    print(f"Gate B decision={gate['decision']} reason={gate['reason']}")
    for item in gate.get("evaluations", []):
        print(
            f"  n_bins={item['n_bins']} min_bin_n={item['min_bin_n']} "
            f"gap={item['gap']:.10f}"
        )
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
