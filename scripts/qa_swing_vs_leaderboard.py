#!/usr/bin/env python3
"""QA: bronze per-pitch aggregates vs the Savant leaderboard (tolerance protocol).

Never exact equality (Savant applies its own tracking/competitive filters and
denominators). Pass criteria (spec 2026-06-12): after matching season +
player-type + qualifying minimum, Spearman >= 0.98 on n_swings and whiff
rate; median absolute percent error <= 2% (p95 <= 5%); mean miss-distance
median |diff| <= 0.5 inches.

Usage: uv run python scripts/qa_swing_vs_leaderboard.py --season 2025 \
           --bronze data/processed --player-type pitcher
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bts.features.swing import daily_swing_aggregates  # noqa: E402

LB_URL = ("https://baseballsavant.mlb.com/leaderboard/bat-tracking/"
          "swing-timing-miss-distance?season%5B%5D={season}&type={ptype}&csv=true")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--bronze", type=Path, default=Path("data/processed"))
    ap.add_argument("--player-type", choices=["batter", "pitcher"], default="pitcher")
    args = ap.parse_args()

    bronze = pd.read_parquet(args.bronze / f"swing_{args.season}.parquet")
    daily = daily_swing_aggregates(bronze, entity=args.player_type)
    ours = daily.groupby(args.player_type).agg(
        n_swings=("n_swings", "sum"),
        n_whiffs=("n_whiffs", "sum"),
        n_whiffs_tracked=("n_whiffs_tracked", "sum"),
        miss_sum=("miss_sum", "sum"),
    )
    ours["whiff_rate"] = ours["n_whiffs"] / ours["n_swings"]
    ours["mean_miss"] = ours["miss_sum"] / ours["n_whiffs_tracked"]

    url = LB_URL.format(season=args.season, ptype=args.player_type)
    with urllib.request.urlopen(url, timeout=30) as r:
        lb = pd.read_csv(StringIO(r.read().decode("utf-8-sig")))
    lb = lb.set_index("id")

    joined = ours.join(lb[["n_swings", "whiff_rate", "miss_distance"]],
                       how="inner", rsuffix="_lb")
    # qualifying minimum: leaderboard's displayed population only
    print(f"joined {len(joined)} {args.player_type}s (leaderboard population)")
    if len(joined) < 50:
        print("FAIL: joined population implausibly small — id mismatch?")
        sys.exit(1)

    checks = []
    sp_swings = joined["n_swings"].corr(joined["n_swings_lb"], method="spearman")
    checks.append(("spearman n_swings >= 0.98", sp_swings, sp_swings >= 0.98))
    sp_wr = joined["whiff_rate"].corr(joined["whiff_rate_lb"], method="spearman")
    checks.append(("spearman whiff_rate >= 0.98", sp_wr, sp_wr >= 0.98))
    ape = (joined["n_swings"] - joined["n_swings_lb"]).abs() / joined["n_swings_lb"]
    checks.append(("median APE n_swings <= 2%", ape.median(), ape.median() <= 0.02))
    checks.append(("p95 APE n_swings <= 5%", ape.quantile(0.95), ape.quantile(0.95) <= 0.05))
    md = (joined["mean_miss"] - joined["miss_distance"]).abs()
    checks.append(("median |mean_miss diff| <= 0.5in", md.median(), md.median() <= 0.5))

    failed = 0
    for name, value, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  (got {value:.4f})")
        failed += 0 if ok else 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
