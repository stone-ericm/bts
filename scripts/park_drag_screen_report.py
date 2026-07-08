#!/usr/bin/env python3
"""park_drag 2026 screen report: controls gate FIRST, then arm deltas.

Usage: uv run python scripts/park_drag_screen_report.py --dir data/validation/park_drag_screen_2026
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHANGE_POINT = "2026-05-24"  # league drag change-point (rollout midpoint)


def _pair_weighted_day_auc(per_day, start=None):
    num = den = 0.0
    for d in per_day:
        if start is not None and d["date"] < start:
            continue
        if d["day_auc"] is None:
            continue
        w = d["n_pos"] * d["n_neg"]
        num += w * d["day_auc"]
        den += w
    return num / den if den else float("nan")


def _mean(per_day, key, start=None):
    vals = [d[key] for d in per_day
            if (start is None or d["date"] >= start) and d[key] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()

    runs = [json.loads(p.read_text()) for p in sorted(args.dir.glob("*_seed*.json"))]
    if not runs:
        raise SystemExit("no result files")

    by_arm: dict[str, list[dict]] = {}
    for r in runs:
        by_arm.setdefault(r["arm"], []).append(r)

    def stat(arm, fn):
        vals = [fn(r) for r in by_arm.get(arm, [])]
        vals = [v for v in vals if not np.isnan(v)]
        return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (float("nan"),) * 2 + (0,)

    full_auc = lambda r: _pair_weighted_day_auc(r["per_day"])           # noqa: E731
    post_auc = lambda r: _pair_weighted_day_auc(r["per_day"], CHANGE_POINT)  # noqa: E731
    top1 = lambda r: _mean(r["per_day"], "top1")                        # noqa: E731

    base_full, _, _ = stat("baseline", full_auc)
    base_post, _, _ = stat("baseline", post_auc)
    base_top1, _, _ = stat("baseline", top1)

    print("=" * 78)
    print("CONTROLS GATE (read this BEFORE any arm result)")
    print("=" * 78)
    g, _, n = stat("ctl_sentinel_gross", full_auc)
    print(f"  gross oracle   day-pair-AUC {g:.4f} (n={n})  -> must be ~1.0 (harness sees leaks)")
    s, ssd, n = stat("ctl_sentinel_soft", full_auc)
    print(f"  soft oracle    {s:.4f} ±{ssd:.4f} (Δ={s - base_full:+.4f})  -> power gate at ~+0.005")
    lk, _, n = stat("ctl_sentinel_leaky", full_auc)
    print(f"  same-day leak  {lk:.4f} (Δ={lk - base_full:+.4f})  -> canary: should sit ABOVE null band")
    pm, pmsd, n = stat("ctl_permuted", full_auc)
    print(f"  permuted       {pm:.4f} ±{pmsd:.4f} (Δ={pm - base_full:+.4f})  -> null band")
    mk, mksd, n = stat("ctl_mask_only", full_auc)
    print(f"  mask-only      {mk:.4f} ±{mksd:.4f} (Δ={mk - base_full:+.4f})  -> null band")

    print()
    print("=" * 78)
    print(f"ARMS (baseline full={base_full:.4f}, post-{CHANGE_POINT}={base_post:.4f}, "
          f"top1={base_top1:.3f})")
    print("=" * 78)
    hdr = f"  {'arm':16s} {'full Δauc':>12s} {'post-5/24 Δauc':>15s} {'Δtop1':>8s}  n"
    print(hdr)
    for arm in ("pd_anchored", "pd_expanding", "outcome_pf", "pd_plus_outcome"):
        f, fsd, n = stat(arm, full_auc)
        p, psd, _ = stat(arm, post_auc)
        t, _, _ = stat(arm, top1)
        print(f"  {arm:16s} {f - base_full:+.4f}±{fsd:.4f} {p - base_post:+.4f}±{psd:.4f} "
              f"{t - base_top1:+.3f}  {n}")
    print()
    print("Read: null band = permuted/mask Δ; an arm counts only if it clears the")
    print("band AND the soft-oracle power gate passed. outcome_pf is the ship bar:")
    print("if it matches pd_anchored, rolling outcomes suffice — no new dependency.")


if __name__ == "__main__":
    main()
