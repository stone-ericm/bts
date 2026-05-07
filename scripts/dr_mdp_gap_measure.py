"""Finite-candidate DR-MDP gap screen for the BTS reachability solver.

This script is deliberately non-production. It measures whether plausible
ambiguity around the empirical quality-bin manifold is large enough to justify
scoping a production robust solver.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.mdp import solve_mdp
from bts.simulate.pooled_policy import load_seed_tagged_profiles, seed_from_path
from bts.simulate.quality_bins import QualityBin, QualityBins


ACTION_SKIP = 0
ACTION_SINGLE = 1
ACTION_DOUBLE = 2


@dataclass(frozen=True)
class BinObservationStats:
    """Empirical sufficient statistics for one quality bin."""

    index: int
    n: int
    n_hit: int
    n_both: int
    p_range: tuple[float, float]
    p_hit: float
    p_both: float
    frequency: float


@dataclass(frozen=True)
class AmbiguityConstruction:
    """Finite ambiguity candidates consumed by the robust MDP screen."""

    name: str
    hit_candidates: list[list[tuple[float, float]]]
    frequency_candidates: list[np.ndarray]
    metadata: dict


@dataclass(frozen=True)
class RobustMDPResult:
    """Result of a finite-candidate rectangular robust reachability solve."""

    robust_p57: float
    value_table: np.ndarray
    policy_table: np.ndarray
    initial_frequency_candidate_index: int
    initial_frequency_candidate: np.ndarray


def solve_robust_mdp(
    bins: QualityBins,
    hit_candidates: list[list[tuple[float, float]]],
    frequency_candidates: list[np.ndarray],
    *,
    season_length: int = 153,
) -> RobustMDPResult:
    """Solve a rectangular finite-candidate robust MDP.

    The point-estimate MDP uses one pair ``(p_hit[q], p_both[q])`` per current
    bin and one next-day frequency vector. This robust screen replaces those
    singletons with finite candidate sets and evaluates each action against the
    worst candidate. It is exact for the provided finite candidate grid, not a
    continuous ambiguity-set optimizer.
    """
    n_bins = len(bins.bins)
    hit_grid = _validate_hit_candidates(hit_candidates, n_bins)
    freq_grid = _validate_frequency_candidates(frequency_candidates, n_bins)

    n_streaks = 58
    n_days = int(season_length) + 1
    n_saver = 2

    value = np.zeros((n_streaks, n_days, n_saver, n_bins))
    policy = np.zeros((n_streaks, n_days, n_saver, n_bins), dtype=np.int8)
    value[57, :, :, :] = 1.0

    for d in range(1, n_days):
        ev_by_freq = np.stack([
            np.tensordot(value[:, d - 1, :, :], freq, axes=([2], [0]))
            for freq in freq_grid
        ])

        for s in range(57):
            next_hit = min(s + 1, 57)
            next_double = min(s + 2, 57)

            for saver in range(n_saver):
                miss_state = s if saver and 10 <= s <= 15 else 0
                miss_saver = 0 if saver and 10 <= s <= 15 else saver

                for q in range(n_bins):
                    v_skip = float(np.min(ev_by_freq[:, s, saver]))

                    v_single = min(
                        float(np.min(
                            ph * ev_by_freq[:, next_hit, saver]
                            + (1.0 - ph) * ev_by_freq[:, miss_state, miss_saver]
                        ))
                        for ph, _pb in hit_grid[q]
                    )

                    v_double = min(
                        float(np.min(
                            pb * ev_by_freq[:, next_double, saver]
                            + (1.0 - pb) * ev_by_freq[:, miss_state, miss_saver]
                        ))
                        for _ph, pb in hit_grid[q]
                    )

                    values = (v_skip, v_single, v_double)
                    best_action = int(np.argmax(values))
                    value[s, d, saver, q] = values[best_action]
                    policy[s, d, saver, q] = best_action

    initial_values = np.array([
        float(np.dot(freq, value[0, season_length, 1, :]))
        for freq in freq_grid
    ])
    initial_idx = int(np.argmin(initial_values))

    return RobustMDPResult(
        robust_p57=float(initial_values[initial_idx]),
        value_table=value,
        policy_table=policy,
        initial_frequency_candidate_index=initial_idx,
        initial_frequency_candidate=freq_grid[initial_idx],
    )


def pair_frame_from_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported profile schemas to one row per top-1/top-2 pair."""
    direct_cols = {"top1_p", "top1_hit", "top2_hit"}
    ranked_cols = {"date", "rank", "p_game_hit", "actual_hit"}

    if direct_cols.issubset(profiles.columns):
        passthrough = [c for c in ("season", "date", "seed") if c in profiles.columns]
        return profiles[passthrough].assign(
            p_game_hit=profiles["top1_p"].astype(float),
            actual_hit=profiles["top1_hit"].astype(bool),
            top2_hit=profiles["top2_hit"].astype(bool),
        )

    if ranked_cols.issubset(profiles.columns):
        key_cols = [c for c in ("season", "date", "seed") if c in profiles.columns]
        if "date" not in key_cols:
            key_cols.append("date")

        rank1 = profiles[profiles["rank"] == 1].copy()
        rank2 = profiles[profiles["rank"] == 2].copy()
        merged = rank1[key_cols + ["p_game_hit", "actual_hit"]].merge(
            rank2[key_cols + ["actual_hit"]].rename(columns={"actual_hit": "top2_hit"}),
            on=key_cols,
            validate="one_to_one",
        )
        merged["p_game_hit"] = merged["p_game_hit"].astype(float)
        merged["actual_hit"] = merged["actual_hit"].astype(bool)
        merged["top2_hit"] = merged["top2_hit"].astype(bool)
        return merged

    raise ValueError(
        "profiles must use either direct columns "
        "(top1_p, top1_hit, top2_hit) or ranked columns "
        "(date, rank, p_game_hit, actual_hit)"
    )


def quality_bins_from_pairs(pairs: pd.DataFrame, *, n_bins: int = 5) -> tuple[QualityBins, list[BinObservationStats]]:
    """Compute fixed-count quality bins and empirical observation stats."""
    if pairs.empty:
        raise ValueError("cannot compute quality bins from empty profile pairs")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    work = pairs.copy()
    quantiles = [i / n_bins for i in range(1, n_bins)]
    boundaries = [float(work["p_game_hit"].quantile(q)) for q in quantiles]
    work["bin"] = np.digitize(work["p_game_hit"], boundaries)

    bins: list[QualityBin] = []
    stats: list[BinObservationStats] = []
    total = len(work)

    for i in range(n_bins):
        group = work[work["bin"] == i]
        stat = _stats_for_group(i, group, total, boundaries)
        stats.append(stat)
        bins.append(QualityBin(
            index=i,
            p_range=stat.p_range,
            p_hit=stat.p_hit,
            p_both=stat.p_both,
            frequency=stat.frequency,
        ))

    return QualityBins(bins=bins, boundaries=boundaries), stats


def build_wilson_simplex_construction(
    stats: list[BinObservationStats],
    *,
    z: float = 1.96,
) -> AmbiguityConstruction:
    """Build Wilson hit-rate candidates plus simplex-respecting freq shifts."""
    hit_candidates: list[list[tuple[float, float]]] = []
    counts = np.array([s.n for s in stats], dtype=float)
    total = float(np.sum(counts))

    for stat in stats:
        hit_lo, hit_hi = wilson_interval(stat.n_hit, stat.n, z=z)
        both_lo, both_hi = wilson_interval(stat.n_both, stat.n, z=z)
        hit_candidates.append(_candidate_pairs_from_bounds(
            point_hit=stat.p_hit,
            point_both=stat.p_both,
            hit_low=hit_lo,
            hit_high=hit_hi,
            both_low=both_lo,
            both_high=both_hi,
        ))

    base_freq = counts / total
    freq_low = np.array([wilson_interval(int(c), int(total), z=z)[0] for c in counts])
    freq_high = np.array([wilson_interval(int(c), int(total), z=z)[1] for c in counts])

    return AmbiguityConstruction(
        name="wilson_simplex",
        hit_candidates=hit_candidates,
        frequency_candidates=build_simplex_frequency_candidates(base_freq, freq_low, freq_high),
        metadata={
            "hit_interval": "per-bin Wilson intervals for p_hit and p_both",
            "frequency_interval": "Wilson cell bounds projected to simplex candidates",
            "z": float(z),
        },
    )


def build_bootstrap_construction(
    pairs: pd.DataFrame,
    boundaries: list[float],
    stats: list[BinObservationStats],
    *,
    n_bootstrap: int = 250,
    alpha: float = 0.05,
    seed: int = 42,
) -> AmbiguityConstruction:
    """Build paired-day bootstrap hit candidates plus multinomial frequency shifts."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")

    rng = np.random.default_rng(seed)
    n_bins = len(stats)
    p_hit_samples = np.zeros((n_bootstrap, n_bins))
    p_both_samples = np.zeros((n_bootstrap, n_bins))
    freq_samples = np.zeros((n_bootstrap, n_bins))

    for b in range(n_bootstrap):
        sample = _resample_pairs_by_day(pairs, rng)
        sample_stats = stats_from_fixed_boundaries(sample, boundaries)
        p_hit_samples[b, :] = [s.p_hit for s in sample_stats]
        p_both_samples[b, :] = [s.p_both for s in sample_stats]
        freq_samples[b, :] = [s.frequency for s in sample_stats]

    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0
    hit_low = np.quantile(p_hit_samples, q_low, axis=0)
    hit_high = np.quantile(p_hit_samples, q_high, axis=0)
    both_low = np.quantile(p_both_samples, q_low, axis=0)
    both_high = np.quantile(p_both_samples, q_high, axis=0)
    freq_low = np.quantile(freq_samples, q_low, axis=0)
    freq_high = np.quantile(freq_samples, q_high, axis=0)

    hit_candidates = [
        _candidate_pairs_from_bounds(
            point_hit=stat.p_hit,
            point_both=stat.p_both,
            hit_low=float(hit_low[i]),
            hit_high=float(hit_high[i]),
            both_low=float(both_low[i]),
            both_high=float(both_high[i]),
        )
        for i, stat in enumerate(stats)
    ]
    base_freq = np.array([s.frequency for s in stats], dtype=float)

    return AmbiguityConstruction(
        name="paired_day_bootstrap_multinomial",
        hit_candidates=hit_candidates,
        frequency_candidates=build_simplex_frequency_candidates(base_freq, freq_low, freq_high),
        metadata={
            "hit_interval": "paired-day bootstrap quantiles for p_hit and p_both",
            "frequency_interval": "bootstrap frequency quantiles projected to simplex candidates",
            "n_bootstrap": int(n_bootstrap),
            "alpha": float(alpha),
            "seed": int(seed),
        },
    )


def stats_from_fixed_boundaries(pairs: pd.DataFrame, boundaries: list[float]) -> list[BinObservationStats]:
    """Compute bin stats with fixed boundaries, retaining empty bins."""
    work = pairs.copy()
    work["bin"] = np.digitize(work["p_game_hit"], boundaries)
    total = len(work)
    return [
        _stats_for_group(i, work[work["bin"] == i], total, boundaries)
        for i in range(len(boundaries) + 1)
    ]


def build_simplex_frequency_candidates(
    base_freq: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> list[np.ndarray]:
    """Turn marginal cell bounds into deterministic simplex candidates."""
    base = _normalize_frequency(np.asarray(base_freq, dtype=float))
    lower = np.clip(np.asarray(lower_bounds, dtype=float), 0.0, 1.0)
    upper = np.clip(np.asarray(upper_bounds, dtype=float), 0.0, 1.0)
    if lower.shape != base.shape or upper.shape != base.shape:
        raise ValueError("frequency bounds must have the same shape as base_freq")

    candidates = [base]
    for idx in range(len(base)):
        candidates.append(_shift_frequency_cell(base, idx, float(lower[idx])))
        candidates.append(_shift_frequency_cell(base, idx, float(upper[idx])))

    return _dedupe_frequency_candidates(candidates)


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 1.0

    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * np.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))) / denom
    return max(0.0, float(center - margin)), min(1.0, float(center + margin))


def measure_gap(
    profiles: pd.DataFrame,
    *,
    season_length: int = 153,
    n_bins: int = 5,
    ci_half_width: float | None = None,
    z: float = 1.96,
    n_bootstrap: int = 250,
    seed: int = 42,
    source_profiles: list[str] | None = None,
) -> dict:
    """Run point-vs-robust measurement and return a JSON-ready dict."""
    pairs = pair_frame_from_profiles(profiles)
    bins, stats = quality_bins_from_pairs(pairs, n_bins=n_bins)
    point_solution = solve_mdp(bins, season_length=season_length)

    constructions = [build_wilson_simplex_construction(stats, z=z)]
    if n_bootstrap > 0:
        constructions.append(build_bootstrap_construction(
            pairs,
            bins.boundaries,
            stats,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ))

    construction_results = []
    for construction in constructions:
        robust = solve_robust_mdp(
            bins,
            construction.hit_candidates,
            construction.frequency_candidates,
            season_length=season_length,
        )
        delta = float(point_solution.optimal_p57 - robust.robust_p57)
        construction_results.append({
            "name": construction.name,
            "robust_p57": robust.robust_p57,
            "delta_p57": delta,
            "policy_disagreement_rate": policy_disagreement_rate(
                point_solution.policy_table,
                robust.policy_table,
            ),
            "n_hit_candidates_by_bin": [len(c) for c in construction.hit_candidates],
            "n_frequency_candidates": len(construction.frequency_candidates),
            "initial_frequency_candidate_index": robust.initial_frequency_candidate_index,
            "initial_frequency_candidate": robust.initial_frequency_candidate.tolist(),
            "exceeds_ci_half_width": (
                None if ci_half_width is None else bool(delta > ci_half_width)
            ),
            "metadata": construction.metadata,
        })

    max_delta = max(r["delta_p57"] for r in construction_results)
    return {
        "schema_version": 1,
        "method": "finite_candidate_rectangular_dr_mdp_screen",
        "source_profiles": source_profiles or [],
        "n_profile_rows": int(len(profiles)),
        "n_pair_rows": int(len(pairs)),
        "season_length": int(season_length),
        "n_bins": int(n_bins),
        "point_p57": float(point_solution.optimal_p57),
        "max_delta_p57": float(max_delta),
        "ci_half_width": None if ci_half_width is None else float(ci_half_width),
        "max_delta_exceeds_ci_half_width": (
            None if ci_half_width is None else bool(max_delta > ci_half_width)
        ),
        "bin_stats": [_bin_stats_to_json(stat) for stat in stats],
        "constructions": construction_results,
        "notes": [
            "Robust values are exact for the finite candidate grid only.",
            "This is a measurement screen; it does not alter production MDP or strategy code.",
        ],
    }


def policy_disagreement_rate(point_policy: np.ndarray, robust_policy: np.ndarray) -> float:
    """Share of nonterminal policy-table entries that differ."""
    streaks = min(point_policy.shape[0], robust_policy.shape[0], 57)
    days = min(point_policy.shape[1], robust_policy.shape[1])
    point_slice = point_policy[:streaks, 1:days, :, :]
    robust_slice = robust_policy[:streaks, 1:days, :, :]
    if point_slice.size == 0:
        return 0.0
    return float(np.mean(point_slice != robust_slice))


def load_profiles(
    paths_or_globs: list[str],
    *,
    derive_seed_from_path: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Load profile parquet files from one or more paths/globs.

    ``derive_seed_from_path`` is intended for pooled raw surfaces whose parquet
    payloads omit seed metadata but live under ``seedN``/``simulation_seedN``
    directories. When enabled, any embedded seed column must agree with the
    path-derived seed.
    """
    paths: list[Path] = []
    for item in paths_or_globs:
        matches = sorted(glob.glob(item))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            path = Path(item)
            if path.exists():
                paths.append(path)

    unique_paths = sorted({p.resolve() for p in paths})
    if not unique_paths:
        raise FileNotFoundError(f"no profile parquet files matched: {paths_or_globs}")

    path_seeds = {path: seed_from_path(path) for path in unique_paths}
    has_path_seed = any(seed is not None for seed in path_seeds.values())
    if derive_seed_from_path or has_path_seed:
        missing_seed_paths = [str(path) for path, seed in path_seeds.items() if seed is None]
        if missing_seed_paths:
            raise ValueError(
                "cannot mix seed-tagged and untagged profile paths; "
                f"missing seed marker in: {missing_seed_paths}"
            )
        frames = [load_seed_tagged_profiles(unique_paths)]
    else:
        frames = [pd.read_parquet(path) for path in unique_paths]
    return pd.concat(frames, ignore_index=True), [str(path) for path in unique_paths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles-glob",
        action="append",
        default=None,
        help="Profile parquet path/glob. May be repeated. Default: data/simulation/backtest_*.parquet",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    parser.add_argument("--season-length", type=int, default=153)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--ci-half-width", type=float, default=None)
    parser.add_argument("--z", type=float, default=1.96, help="Wilson z value for interval construction.")
    parser.add_argument(
        "--derive-seed-from-path",
        action="store_true",
        help=(
            "Populate/validate a seed column by parsing seedN from each profile path. "
            "Use for pooled raw surfaces whose parquet payloads omit seed metadata."
        ),
    )
    parser.add_argument(
        "--n-bootstrap-candidates",
        type=int,
        default=250,
        help="Paired-day bootstrap replicates for the bootstrap construction; set 0 to skip.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = parser.parse_args(argv)

    profile_globs = args.profiles_glob or ["data/simulation/backtest_*.parquet"]
    profiles, loaded_paths = load_profiles(
        profile_globs,
        derive_seed_from_path=args.derive_seed_from_path,
    )
    result = measure_gap(
        profiles,
        season_length=args.season_length,
        n_bins=args.n_bins,
        ci_half_width=args.ci_half_width,
        z=args.z,
        n_bootstrap=args.n_bootstrap_candidates,
        seed=args.seed,
        source_profiles=loaded_paths,
    )
    result["profile_loader"] = {
        "derive_seed_from_path": bool(args.derive_seed_from_path),
        "path_seed_marker_count": int(sum(
            seed_from_path(path) is not None
            for path in loaded_paths
        )),
        "seed_column_present": "seed" in profiles.columns,
        "n_seeds": (
            int(profiles["seed"].nunique())
            if "seed" in profiles.columns else None
        ),
    }

    text = json.dumps(result, allow_nan=False, indent=2 if args.pretty else None, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(_summary_line(result), file=sys.stderr)
    else:
        print(text)
    return 0


def _summary_line(result: dict) -> str:
    construction_parts = [
        f"{item['name']}={item['delta_p57']:.6f}"
        for item in result["constructions"]
    ]
    if result["ci_half_width"] is None:
        gate = "ci_half_width=unset"
    else:
        verdict = "EXCEEDS" if result["max_delta_exceeds_ci_half_width"] else "within"
        gate = f"ci_half_width={result['ci_half_width']:.6f} {verdict}"
    return (
        f"DR-MDP gap screen: point_p57={result['point_p57']:.6f} "
        f"max_delta={result['max_delta_p57']:.6f} "
        f"{gate} ({', '.join(construction_parts)})"
    )


def _validate_hit_candidates(
    hit_candidates: list[list[tuple[float, float]]],
    n_bins: int,
) -> list[list[tuple[float, float]]]:
    if len(hit_candidates) != n_bins:
        raise ValueError(f"expected hit candidates for {n_bins} bins, got {len(hit_candidates)}")

    validated: list[list[tuple[float, float]]] = []
    for q, candidates in enumerate(hit_candidates):
        if not candidates:
            raise ValueError(f"bin {q} has no hit candidates")
        q_candidates = []
        for ph, pb in candidates:
            ph_f = float(ph)
            pb_f = float(pb)
            if not (0.0 <= ph_f <= 1.0 and 0.0 <= pb_f <= 1.0):
                raise ValueError(f"bin {q} hit candidates must be probabilities")
            if pb_f > ph_f + 1e-12:
                raise ValueError(f"bin {q} p_both candidate {pb_f} exceeds p_hit {ph_f}")
            q_candidates.append((ph_f, pb_f))
        validated.append(q_candidates)
    return validated


def _validate_frequency_candidates(
    frequency_candidates: list[np.ndarray],
    n_bins: int,
) -> list[np.ndarray]:
    if not frequency_candidates:
        raise ValueError("at least one frequency candidate is required")

    validated = []
    for candidate in frequency_candidates:
        freq = _normalize_frequency(np.asarray(candidate, dtype=float))
        if freq.shape != (n_bins,):
            raise ValueError(f"frequency candidate shape {freq.shape} does not match n_bins={n_bins}")
        validated.append(freq)
    return validated


def _normalize_frequency(freq: np.ndarray) -> np.ndarray:
    if np.any(freq < -1e-12):
        raise ValueError("frequency candidates must be nonnegative")
    clipped = np.clip(freq.astype(float), 0.0, None)
    total = float(np.sum(clipped))
    if total <= 0.0:
        raise ValueError("frequency candidate must have positive mass")
    return clipped / total


def _shift_frequency_cell(base: np.ndarray, idx: int, target: float) -> np.ndarray:
    target = float(np.clip(target, 0.0, 1.0))
    out = np.zeros_like(base, dtype=float)
    out[idx] = target
    other = [i for i in range(len(base)) if i != idx]
    remaining = 1.0 - target
    other_mass = float(np.sum(base[other]))
    if other_mass > 0:
        out[other] = base[other] / other_mass * remaining
    elif other:
        out[other] = remaining / len(other)
    return _normalize_frequency(out)


def _dedupe_frequency_candidates(candidates: list[np.ndarray]) -> list[np.ndarray]:
    seen: set[tuple[float, ...]] = set()
    out = []
    for candidate in candidates:
        freq = _normalize_frequency(candidate)
        key = tuple(np.round(freq, 12))
        if key not in seen:
            seen.add(key)
            out.append(freq)
    return out


def _candidate_pairs_from_bounds(
    *,
    point_hit: float,
    point_both: float,
    hit_low: float,
    hit_high: float,
    both_low: float,
    both_high: float,
) -> list[tuple[float, float]]:
    raw = [
        (point_hit, point_both),
        (hit_low, min(both_low, hit_low)),
        (hit_high, min(both_high, hit_high)),
    ]
    seen: set[tuple[float, float]] = set()
    candidates = []
    for ph, pb in raw:
        ph_f = float(np.clip(ph, 0.0, 1.0))
        pb_f = float(np.clip(min(pb, ph_f), 0.0, 1.0))
        key = (round(ph_f, 12), round(pb_f, 12))
        if key not in seen:
            seen.add(key)
            candidates.append((ph_f, pb_f))
    return candidates


def _stats_for_group(
    index: int,
    group: pd.DataFrame,
    total: int,
    boundaries: list[float],
) -> BinObservationStats:
    if group.empty:
        lower = float("-inf") if index == 0 else float(boundaries[index - 1])
        upper = float("inf") if index == len(boundaries) else float(boundaries[index])
        return BinObservationStats(
            index=index,
            n=0,
            n_hit=0,
            n_both=0,
            p_range=(lower, upper),
            p_hit=0.0,
            p_both=0.0,
            frequency=0.0,
        )

    n = int(len(group))
    n_hit = int(group["actual_hit"].sum())
    both = group["actual_hit"].astype(bool) & group["top2_hit"].astype(bool)
    n_both = int(both.sum())
    return BinObservationStats(
        index=index,
        n=n,
        n_hit=n_hit,
        n_both=n_both,
        p_range=(float(group["p_game_hit"].min()), float(group["p_game_hit"].max())),
        p_hit=float(n_hit / n),
        p_both=float(n_both / n),
        frequency=float(n / total),
    )


def _resample_pairs_by_day(pairs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    day_cols = [c for c in ("season", "date") if c in pairs.columns]
    if not day_cols:
        return pairs.iloc[rng.integers(len(pairs), size=len(pairs))].reset_index(drop=True)

    grouped = [group for _key, group in pairs.groupby(day_cols, sort=True)]
    draws = rng.integers(len(grouped), size=len(grouped))
    return pd.concat([grouped[i] for i in draws], ignore_index=True)


def _bin_stats_to_json(stat: BinObservationStats) -> dict:
    return {
        "index": stat.index,
        "n": stat.n,
        "n_hit": stat.n_hit,
        "n_both": stat.n_both,
        "p_range": [_finite_or_none(v) for v in stat.p_range],
        "p_hit": stat.p_hit,
        "p_both": stat.p_both,
        "frequency": stat.frequency,
    }


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


if __name__ == "__main__":
    raise SystemExit(main())
