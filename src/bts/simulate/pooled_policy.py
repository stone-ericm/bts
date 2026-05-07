"""Pooled-seed MDP policy builder (Option 7).

See memory/project_bts_2026_04_15_audit_state.md for context.

A single-seed MDP policy inherits per-bin empirical noise from a 369-day
profile (~74 days per quintile, SE ~4-5pp on p_hit). That noise compounds
through backward induction and produces MDP P(57) estimates with ±2.1pp
cross-seed standard deviation — seed=42 scored 6.22%, the 16-seed mean
was 3.50%.

The fix is to pool profile parquets across many seeds before computing
quality bins. Bins get N×more data, per-bin SE drops by sqrt(N), and the
resulting policy is robust to any single seed's luck.

Public API:
    parse_seed_from_path      — extract seed id from 'seedN' in any path segment
    load_pooled_profiles      — load + tag-with-seed across per-seed dirs
    load_seed_tagged_profiles — load + tag arbitrary parquet paths with path seeds
    compute_pooled_bins       — quality bins with seed-aware rank-1/rank-2 merge
    split_by_phase_pooled     — phase split preserving the seed column
    build_pooled_policy       — full pipeline: profiles → bins → MDPSolution
    evaluate_mdp_policy       — forward-evaluate a fixed policy on holdout bins
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.mdp import MDPSolution, solve_mdp
from bts.simulate.quality_bins import QualityBin, QualityBins


SEED_RE = re.compile(r"seed(\d+)")
RANKED_PROFILE_REQUIRED_COLUMNS = frozenset({"date", "rank", "p_game_hit", "actual_hit"})


def seed_from_path(path: Path | str) -> int | None:
    """Extract integer seed from any path segment matching 'seed(\\d+)'; return None if absent."""
    path = Path(path)
    for part in path.parts:
        m = SEED_RE.search(part)
        if m:
            return int(m.group(1))
    return None


def parse_seed_from_path(path: Path | str) -> int:
    """Extract integer seed from any path segment matching 'seed(\\d+)'."""
    seed = seed_from_path(path)
    if seed is not None:
        return seed
    raise ValueError(f"Cannot extract seed from path: {path}")


def load_pooled_profiles(seed_dirs: list[Path | str]) -> pd.DataFrame:
    """Load per-seed profile parquets and tag each row with its seed id.

    Each directory must contain backtest_{season}.parquet files produced by
    a distinct BTS_LGBM_RANDOM_STATE seed (e.g. scripts/rebuild_policy.py or
    bts simulate backtest). A 'seed' column is added to every row so that
    compute_pooled_bins can pair rank-1 and rank-2 within a single
    (seed, date) — never across seeds.

    If season is not already in the parquet, it's parsed from the filename.
    """
    if not seed_dirs:
        raise ValueError("no seed_dirs provided")
    dfs: list[pd.DataFrame] = []
    for seed_dir in seed_dirs:
        seed_dir = Path(seed_dir)
        seed = parse_seed_from_path(seed_dir)
        parquets = sorted(seed_dir.glob("backtest_*.parquet"))
        if not parquets:
            raise ValueError(f"no backtest_*.parquet in {seed_dir}")
        for parquet in parquets:
            df = pd.read_parquet(parquet)
            if "season" not in df.columns:
                m = re.search(r"backtest_(\d{4})", parquet.stem)
                if m:
                    df["season"] = int(m.group(1))
            df["seed"] = seed
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_seed_tagged_profiles(paths: list[Path | str]) -> pd.DataFrame:
    """Load arbitrary profile parquet paths and tag/validate seed identity.

    This is the path-list equivalent of ``load_pooled_profiles``. It is used by
    measurement scripts that accept globbed parquet files rather than seed
    directories. Every path must contain a ``seedN`` segment, and any embedded
    seed column must agree with that path-derived seed.
    """
    if not paths:
        raise ValueError("no profile paths provided")

    frames: list[pd.DataFrame] = []
    for item in paths:
        path = Path(item)
        seed = parse_seed_from_path(path)
        frame = pd.read_parquet(path)
        _validate_ranked_profile_schema(frame, path)
        frame = _with_validated_seed(frame, seed, path)
        frames.append(frame)

    profiles = pd.concat(frames, ignore_index=True)
    _validate_rank_pair_integrity(profiles)
    return profiles


def _validate_ranked_profile_schema(frame: pd.DataFrame, path: Path) -> None:
    missing = sorted(RANKED_PROFILE_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required ranked-profile columns: {missing}")


def _with_validated_seed(frame: pd.DataFrame, seed: int, path: Path) -> pd.DataFrame:
    out = frame.copy()
    if "seed" in out.columns:
        embedded = pd.to_numeric(out["seed"], errors="raise")
        embedded_non_null = embedded.dropna()
        if not np.all(np.equal(np.mod(embedded_non_null, 1), 0)):
            raise ValueError(f"seed column in {path} must contain integer seed ids")
        embedded_seeds = set(int(s) for s in embedded_non_null.unique())
        if embedded_seeds and embedded_seeds != {seed}:
            raise ValueError(
                f"seed column in {path} does not match path-derived seed {seed}: "
                f"{sorted(embedded_seeds)}"
            )
    out["seed"] = seed
    return out


def _validate_rank_pair_integrity(profiles: pd.DataFrame) -> None:
    needed = profiles[profiles["rank"].isin([1, 2])]
    duplicate_counts = needed.groupby(["seed", "date", "rank"], dropna=False).size()
    duplicates = duplicate_counts[duplicate_counts > 1]
    if not duplicates.empty:
        first = duplicates.index[0]
        raise ValueError(f"duplicate rank row for seed/date/rank: {first}")

    all_seed_dates = pd.MultiIndex.from_frame(
        profiles[["seed", "date"]].drop_duplicates()
    )
    rank_counts = (
        needed.drop_duplicates(["seed", "date", "rank"])
        .groupby(["seed", "date"], dropna=False)["rank"]
        .nunique()
        .reindex(all_seed_dates, fill_value=0)
    )
    missing = rank_counts[rank_counts < 2]
    if not missing.empty:
        first = missing.index[0]
        raise ValueError(f"missing rank-1/rank-2 pair for seed/date: {first}")


def compute_pooled_bins(profiles_df: pd.DataFrame, n_bins: int = 5) -> QualityBins:
    """Quality bins from pooled-seed profiles.

    Mirrors bts.simulate.quality_bins.compute_bins but merges rank-1 and
    rank-2 within each (seed, date) pair. Without the seed key, a merge on
    date alone would cartesian-join across seeds and inflate p_both with
    rank-1 from seed A paired with rank-2 from seed B.

    After the merge, quintile boundaries are computed over the FULL pooled
    distribution — this is the whole point of pooling.
    """
    if "seed" not in profiles_df.columns:
        raise ValueError("profiles_df must have a 'seed' column; use load_pooled_profiles")

    r1 = profiles_df[profiles_df["rank"] == 1].copy()
    r2 = profiles_df[profiles_df["rank"] == 2].copy()
    merged = r1[["date", "seed", "p_game_hit", "actual_hit"]].merge(
        r2[["date", "seed", "actual_hit"]].rename(columns={"actual_hit": "top2_hit"}),
        on=["date", "seed"],
    )

    quantiles = [i / n_bins for i in range(1, n_bins)]
    boundaries = [float(merged["p_game_hit"].quantile(q)) for q in quantiles]
    merged["bin"] = np.digitize(merged["p_game_hit"], boundaries)

    bins: list[QualityBin] = []
    for i in range(n_bins):
        group = merged[merged["bin"] == i]
        if len(group) == 0:
            continue
        bins.append(QualityBin(
            index=i,
            p_range=(float(group["p_game_hit"].min()), float(group["p_game_hit"].max())),
            p_hit=float(group["actual_hit"].mean()),
            p_both=float((group["actual_hit"] & group["top2_hit"]).mean()),
            frequency=len(group) / len(merged),
        ))

    return QualityBins(bins=bins, boundaries=boundaries)


def split_by_phase_pooled(
    profiles_df: pd.DataFrame,
    late_phase_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split pooled profiles into early-phase and late-phase subsets.

    Mirrors scripts/mdp_policy_sweep.py::split_by_phase but preserves the
    seed column so downstream compute_pooled_bins still pairs ranks within
    (seed, date). Phase assignment is season-global and seed-independent:
    the LAST late_phase_days calendar dates of each season are late-phase.
    """
    if late_phase_days <= 0:
        return profiles_df, profiles_df.iloc[0:0]
    if "season" not in profiles_df.columns:
        raise ValueError("profiles_df must have a 'season' column")

    early_rows, late_rows = [], []
    for _season, group in profiles_df.groupby("season"):
        dates = sorted(pd.Series(group["date"]).unique())
        if len(dates) <= late_phase_days:
            late_rows.append(group)
            continue
        cutoff = dates[-late_phase_days]
        late_rows.append(group[group["date"] >= cutoff])
        early_rows.append(group[group["date"] < cutoff])
    early = pd.concat(early_rows, ignore_index=True) if early_rows else profiles_df.iloc[0:0]
    late = pd.concat(late_rows, ignore_index=True) if late_rows else profiles_df.iloc[0:0]
    return early, late


def evaluate_mdp_policy(
    policy_table: np.ndarray,
    early_bins: QualityBins,
    season_length: int = 180,
    late_bins: QualityBins | None = None,
    late_phase_days: int = 30,
) -> float:
    """Forward-evaluate a fixed MDP policy against a holdout bin distribution.

    Unlike `solve_mdp` (which chooses the optimal action at each state via
    argmax during backward induction), this function takes a FIXED
    `policy_table` and computes the value achieved by following it on a
    holdout bin distribution. It's the honest A/B primitive: "what P(57)
    does policy X achieve on unseen bins Y?"

    Bin-index convention: matches `bts.simulate.mdp.solve_mdp`. Bin k in
    the holdout's quintile distribution is treated as the same quality
    tier as bin k in the policy's quintile distribution. This is the
    solve_mdp convention: quintile position carries the semantic
    meaning, not absolute confidence. The holdout must have the same
    n_bins as the policy (enforced by assertion below).

    Returns E[P(reach 57) | start at streak=0, days=season_length, saver=1],
    where the expectation is taken over the HOLDOUT's early-phase bin
    frequencies — matching how `solve_mdp` computes `optimal_p57`.
    """
    n_bins_policy = policy_table.shape[3]
    n_bins_holdout = len(early_bins.bins)
    if n_bins_holdout != n_bins_policy:
        raise ValueError(
            f"policy has {n_bins_policy} bins but early_bins has {n_bins_holdout}; "
            "solve_mdp convention requires matching n_bins"
        )
    if late_bins is not None and len(late_bins.bins) != n_bins_policy:
        raise ValueError(
            f"policy has {n_bins_policy} bins but late_bins has {len(late_bins.bins)}"
        )

    def _freq_hit_both(qb: QualityBins) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array([b.frequency for b in qb.bins]),
            np.array([b.p_hit for b in qb.bins]),
            np.array([b.p_both for b in qb.bins]),
        )

    freq_early, phit_early, pboth_early = _freq_hit_both(early_bins)
    if late_bins is not None:
        freq_late, phit_late, pboth_late = _freq_hit_both(late_bins)
    else:
        freq_late, phit_late, pboth_late = freq_early, phit_early, pboth_early

    n_streaks = 58
    n_days = season_length + 1
    n_bins = n_bins_policy
    V = np.zeros((n_streaks, n_days, 2, n_bins))
    V[57, :, :, :] = 1.0

    for d in range(1, n_days):
        d_policy = min(d, policy_table.shape[1] - 1)
        is_late_now = (late_bins is not None) and (d <= late_phase_days)
        phit_now = phit_late if is_late_now else phit_early
        pboth_now = pboth_late if is_late_now else pboth_early

        next_is_late = (late_bins is not None) and ((d - 1) <= late_phase_days)
        next_freq = freq_late if next_is_late else freq_early

        for s in range(57):
            for saver in range(2):
                # Pre-compute the dot products once per (s, d, saver) —
                # they're independent of q and reused for every bin.
                ev_stay = float(np.dot(next_freq, V[s, d - 1, saver, :]))
                next_hit_s = min(s + 1, 57)
                next_dbl_s = min(s + 2, 57)
                ev_hit = float(np.dot(next_freq, V[next_hit_s, d - 1, saver, :]))
                ev_dbl = float(np.dot(next_freq, V[next_dbl_s, d - 1, saver, :]))
                ev_reset = float(np.dot(next_freq, V[0, d - 1, saver, :]))
                ev_hold_saver_off = float(np.dot(next_freq, V[s, d - 1, 0, :]))

                saver_active = bool(saver) and (10 <= s <= 15)

                for q in range(n_bins):
                    action = policy_table[s, d_policy, saver, q]
                    if action == 0:  # skip
                        V[s, d, saver, q] = ev_stay
                    elif action == 1:  # single
                        ph = phit_now[q]
                        if saver_active:
                            V[s, d, saver, q] = ph * ev_hit + (1 - ph) * ev_hold_saver_off
                        else:
                            V[s, d, saver, q] = ph * ev_hit + (1 - ph) * ev_reset
                    else:  # double
                        pb = pboth_now[q]
                        if saver_active:
                            V[s, d, saver, q] = pb * ev_dbl + (1 - pb) * ev_hold_saver_off
                        else:
                            V[s, d, saver, q] = pb * ev_dbl + (1 - pb) * ev_reset

    # Match solve_mdp's terminal computation: expected value under the
    # LAST iteration's freq (which is freq_early when season_length > late_phase_days).
    freq_terminal = freq_late if (late_bins is not None and season_length <= late_phase_days) else freq_early
    return float(np.dot(freq_terminal, V[0, season_length, 1, :]))


def build_pooled_policy(
    profiles_df: pd.DataFrame,
    season_length: int = 180,
    late_phase_days: int = 30,
    n_bins: int = 5,
) -> MDPSolution:
    """Pool profiles → split by phase → compute bins → solve MDP.

    Returns an MDPSolution whose .save() writes a policy_table compatible
    with bts.simulate.mdp.load_policy / lookup_action.
    """
    if late_phase_days <= 0:
        early_bins = compute_pooled_bins(profiles_df, n_bins=n_bins)
        return solve_mdp(early_bins, season_length=season_length)

    early_df, late_df = split_by_phase_pooled(profiles_df, late_phase_days)
    if len(late_df) == 0:
        early_bins = compute_pooled_bins(profiles_df, n_bins=n_bins)
        return solve_mdp(early_bins, season_length=season_length)

    early_bins = compute_pooled_bins(early_df, n_bins=n_bins)
    late_bins = compute_pooled_bins(late_df, n_bins=n_bins)
    return solve_mdp(
        early_bins,
        season_length=season_length,
        late_bins=late_bins,
        late_phase_days=late_phase_days,
    )
