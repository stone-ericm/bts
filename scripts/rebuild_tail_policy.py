#!/usr/bin/env python
"""Build data/models/mdp_tail_policy.npz — the exact E[season-best] tail policy.

Why (2026-09-03): the reach-57 table skips forever once streak + 2*days < 57.
The tail artifact replaces that regime. It is SEPARATE from mdp_policy.npz so
generic reach-57 writers (MDPSolution.save, `bts simulate solve`,
scripts/rebuild_policy.py) can never erase it, and it is hard-bound to the
sha256 of the base policy it pairs with (a base rebuild => rebuild this too).

Inputs: the 24-seed estimated-PA profiles (data/hetzner_results/mdp_estpa_run —
the serving-realistic basis; see CLAUDE.md "PROFILE BASIS"), late phase only
(the last 30 dates of each season: every tail state has <= 28 days left), and
PRODUCTION-SHAPED doubles: the second pick is the first lower-ranked candidate
in a DIFFERENT game (strategy.select_pick's rule), not rank 2 (Codex r1 P1).

Default is ONE quality bin (owner's call after the preview; Codex r2 P2: the
~150 real late dates cannot support quintiles — a variance-reduction trade, not
an exact quality-aware objective). `--n-bins 2` builds the top-20%-vs-rest
alternative for comparison.

Usage:
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/rebuild_tail_policy.py            # write
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/rebuild_tail_policy.py --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.mdp import DEFAULT_POLICY_PATH
from bts.simulate.pooled_policy import load_pooled_profiles, split_by_phase_pooled
from bts.simulate.tail_policy import (
    DEFAULT_TAIL_POLICY_PATH, MAX_TAIL_DAYS, OBJECTIVE_TAIL, TARGET, TailPolicy,
    load_tail_policy, save_tail_policy, sha256_file, solve_emax_season_best,
)

DEFAULT_PROFILES_ROOT = Path("data/hetzner_results/mdp_estpa_run")
LATE_PHASE_DAYS = 30


def production_shaped_days(profiles: pd.DataFrame) -> pd.DataFrame:
    """One row per (seed, date): rank-1 outcome + the executable different-game
    double's outcome. Days with no different-game candidate count as no-double
    (both=False), mirroring the live clamp double -> single."""
    rows = []
    cols = ["seed", "date", "rank", "game_pk", "p_game_hit", "actual_hit"]
    for (seed, date), g in profiles[cols].sort_values(["seed", "date", "rank"]).groupby(
            ["seed", "date"], sort=False):
        r1 = g.iloc[0]
        if int(r1["rank"]) != 1:
            continue
        others = g[(g["rank"] > 1) & (g["game_pk"] != r1["game_pk"])]
        dd = others.iloc[0] if len(others) else None
        rows.append({
            "seed": int(seed), "date": str(date), "p": float(r1["p_game_hit"]),
            "hit": bool(r1["actual_hit"]),
            "both": bool(r1["actual_hit"]) and dd is not None and bool(dd["actual_hit"]),
            "has_dd": dd is not None,
            "dd_rank": (int(dd["rank"]) if dd is not None else None),
        })
    return pd.DataFrame(rows)


def bin_rates(days: pd.DataFrame, n_bins: int):
    if n_bins == 1:
        boundaries: list[float] = []
        days = days.assign(bin=0)
    elif n_bins == 2:
        boundaries = [float(days["p"].quantile(0.8))]        # top-20% vs rest
        days = days.assign(bin=np.digitize(days["p"], boundaries))
    else:
        qs = [i / n_bins for i in range(1, n_bins)]
        boundaries = [float(days["p"].quantile(q)) for q in qs]
        days = days.assign(bin=np.digitize(days["p"], boundaries))
    g = days.groupby("bin")
    freq = (g.size() / len(days)).reindex(range(n_bins), fill_value=0.0).values
    p_hit = g["hit"].mean().reindex(range(n_bins), fill_value=0.0).values
    p_both = g["both"].mean().reindex(range(n_bins), fill_value=0.0).values
    counts = g.size().reindex(range(n_bins), fill_value=0).values
    return boundaries, freq, p_hit, p_both, counts


def preview(policy: np.ndarray, m: int = 18) -> str:
    A = np.array(["-", "S", "D"])
    lines = [f"  policy at best={m}, saver off (D double, S single, - skip); cols = streak 0..{m}"]
    lines.append("   d  " + " ".join(f"{s:>2}" for s in range(0, m + 1)))
    for d in (24, 20, 16, 12, 10, 9, 8, 6, 4, 3, 2, 1):
        lines.append(f"  {d:>2}  " + " ".join(f" {A[policy[s, m, d, 0, 0]]}" for s in range(0, m + 1)))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles-root", type=Path, default=DEFAULT_PROFILES_ROOT)
    ap.add_argument("--base-policy", type=Path, default=DEFAULT_POLICY_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_TAIL_POLICY_PATH)
    ap.add_argument("--n-bins", type=int, default=1, choices=(1, 2, 3, 4, 5))
    ap.add_argument("--late-phase-days", type=int, default=LATE_PHASE_DAYS)
    ap.add_argument("--dry-run", action="store_true", help="solve + preview, write nothing")
    args = ap.parse_args(argv)

    seed_dirs = sorted(args.profiles_root.glob("*/simulation_seed*"))
    if not seed_dirs:
        ap.error(f"no seed dirs under {args.profiles_root}")
    parquets = sorted(p for d in seed_dirs for p in d.glob("backtest_*.parquet"))
    profiles = load_pooled_profiles(seed_dirs)
    _early, late = split_by_phase_pooled(profiles, args.late_phase_days)
    days = production_shaped_days(late)
    boundaries, freq, p_hit, p_both, counts = bin_rates(days, args.n_bins)

    print(f"profiles: {len(profiles)} rows, {profiles['seed'].nunique()} seeds, "
          f"seasons {sorted(profiles['season'].unique().tolist())}")
    print(f"late phase ({args.late_phase_days} dates/season): {len(days)} seed-days, "
          f"{days['date'].nunique()} distinct dates, no-double days {int((~days['has_dd']).sum())}, "
          f"dd_rank counts {days['dd_rank'].value_counts().sort_index().to_dict()}")
    for q in range(args.n_bins):
        print(f"  bin {q}: n={int(counts[q])} freq={freq[q]:.4f} p_hit={p_hit[q]:.6f} p_both={p_both[q]:.6f}")
    if boundaries:
        print(f"  boundaries: {boundaries}")

    sol = solve_emax_season_best(freq, p_hit, p_both, target=TARGET, max_days=MAX_TAIL_DAYS)
    print(preview(sol.policy))
    print(f"  (0, best=18, d=24) -> {['skip', 'single', 'double'][sol.policy[0, 18, 24, 0, 0]]}")

    base_sha = sha256_file(args.base_policy)
    manifest = {
        "profiles_root": str(args.profiles_root),
        "seed_dirs": len(seed_dirs),
        "parquets": len(parquets),
        "parquets_sha256": hashlib.sha256("".join(sha256_file(p) for p in parquets).encode()).hexdigest(),
        "rows": int(len(profiles)),
        "seasons": sorted(int(s) for s in profiles["season"].unique()),
        "late_phase_days": int(args.late_phase_days),
        "late_seed_days": int(len(days)),
        "late_distinct_dates": int(days["date"].nunique()),
        "pairing": "first lower-ranked candidate in a different game (production rule)",
        "n_bins": int(args.n_bins),
        "bin_counts": [int(c) for c in counts],
        "hits": int(days["hit"].sum()),
        "both": int(days["both"].sum()),
        "base_policy_path": str(args.base_policy),
    }
    tp = TailPolicy(
        objective=OBJECTIVE_TAIL, policy_table=sol.policy, boundaries=boundaries,
        bin_freq=freq.tolist(), bin_p_hit=p_hit.tolist(), bin_p_both=p_both.tolist(),
        target=TARGET, max_days=MAX_TAIL_DAYS, base_policy_sha256=base_sha, manifest=manifest,
        built_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        solver="bts.simulate.tail_policy.solve_emax_season_best",
    )
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    save_tail_policy(tp, args.out)
    loaded = load_tail_policy(args.out, expected_base_sha=base_sha)   # the same contract production uses
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes) sha256={loaded.sha256}")
    print(f"paired with base policy {base_sha}")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
