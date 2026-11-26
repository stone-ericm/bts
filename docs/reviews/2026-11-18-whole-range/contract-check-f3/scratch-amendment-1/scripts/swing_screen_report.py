#!/usr/bin/env python3
"""Aggregate screen results -> gate verdicts + frozen-bundle proposal (markdown).

Amendment #2 semantics (spec 2026-06-12):
- PRIMARY screen stat: paired per-day rank-AUC delta vs same-seed baseline,
  seed-AVERAGED per day before inference, week-blocked sign-permutation test.
- NDCG@10 delta reported as directional secondary (confirmation primary on 2025).
- GATES, checked in order before any family reading:
    1. ctl_sentinel_gross must be positive on EVERY seed with a seed-averaged
       delta exceeding every null arm's by a wide margin (>= 3x).
    2. ctl_sentinel_m3 (shift(0) off-by-one) must exceed the empirical null
       band on the aggregate stat.
    3. Null arms (ctl_mask_only, ctl_permuted) define the practical-null band
       [min, max of their |seed-avg delta|]; they must look unremarkable
       (no permutation p < 0.05 with positive delta).
- Candidates judged on the primary stat vs the null band + permutation p;
  family verdicts: alive unless coverage failure or consistently negative.

Usage: .venv/bin/python scripts/swing_screen_report.py \
           --results data/validation/swing_screen_2024 --out docs/audit/
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

NULL_ARMS = ("ctl_mask_only", "ctl_permuted")
SENTINELS = ("ctl_sentinel_gross", "ctl_sentinel_m3")


def load_results(results_dir: Path) -> dict:
    out = {}
    for f in sorted(results_dir.glob("*_seed*.json")):
        r = json.loads(f.read_text())
        out[(r["arm"], r["seed"])] = r
    return out


def paired_daily_deltas(res: dict, arm: str, seeds: list, stat: str) -> pd.DataFrame:
    """date -> seed-averaged paired delta for `stat` ('day_auc' or 'ndcg')."""
    rows = []
    for s in seeds:
        if (arm, s) not in res or ("baseline", s) not in res:
            continue
        base = {d["date"]: d.get(stat) for d in res[("baseline", s)]["per_day"]}
        for d in res[(arm, s)]["per_day"]:
            if d["date"] in base and d.get(stat) is not None and base[d["date"]] is not None:
                rows.append({"date": d["date"], "seed": s, "delta": d[stat] - base[d["date"]]})
    if not rows:
        return pd.DataFrame(columns=["date", "delta"])
    df = pd.DataFrame(rows)
    return df.groupby("date", as_index=False)["delta"].mean()  # seed-average per day


def week_block_sign_permutation_p(daily: pd.DataFrame, n_perm: int = 5000, seed: int = 7) -> float:
    """One-sided p for mean(delta) > 0, flipping signs by ISO-week blocks."""
    if daily.empty:
        return float("nan")
    d = daily.copy()
    d["week"] = pd.to_datetime(d["date"]).dt.isocalendar().week.astype(int)
    week_means = d.groupby("week")["delta"].mean().to_numpy()
    obs = week_means.mean()
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(week_means)))
    perm = (flips * week_means).mean(axis=1)
    return float((perm >= obs).mean())


def per_seed_deltas(res: dict, arm: str, seeds: list, agg: str) -> list:
    out = []
    for s in seeds:
        if (arm, s) in res and ("baseline", s) in res:
            a, b = res[(arm, s)], res[("baseline", s)]
            if agg == "auc_mean":
                av = np.mean([d["day_auc"] for d in a["per_day"] if d.get("day_auc") is not None])
                bv = np.mean([d["day_auc"] for d in b["per_day"] if d.get("day_auc") is not None])
            else:
                av, bv = a[agg], b[agg]
            out.append(float(av - bv))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/audit"))
    args = ap.parse_args()

    res = load_results(args.results)
    seeds = sorted({s for (_, s) in res})
    arms = sorted({a for (a, _) in res if a != "baseline"})

    stats = {}
    for arm in arms:
        daily = paired_daily_deltas(res, arm, seeds, "day_auc")
        ndcg_daily = paired_daily_deltas(res, arm, seeds, "ndcg")
        stats[arm] = {
            "family": res[(arm, seeds[0])]["family"],
            "auc_delta": float(daily["delta"].mean()) if not daily.empty else float("nan"),
            "auc_p": week_block_sign_permutation_p(daily),
            "auc_per_seed": [round(x, 5) for x in per_seed_deltas(res, arm, seeds, "auc_mean")],
            "ndcg_delta": float(ndcg_daily["delta"].mean()) if not ndcg_daily.empty else float("nan"),
            "top1_delta": float(np.mean(per_seed_deltas(res, arm, seeds, "top1_hit"))),
            "n_days": int(len(daily)),
        }

    # --- gates ---
    null_mags = [abs(stats[a]["auc_delta"]) for a in NULL_ARMS if a in stats]
    null_band = max(null_mags) if null_mags else float("nan")
    gross = stats.get("ctl_sentinel_gross")
    m3 = stats.get("ctl_sentinel_m3")
    gate_lines = []
    gate_pass = True

    if gross:
        every_seed_pos = all(x > 0 for x in gross["auc_per_seed"])
        margin_ok = gross["auc_delta"] >= 5 * null_band if null_band == null_band else False
        p_ok = gross["auc_p"] <= 0.001
        ok = every_seed_pos and margin_ok and p_ok
        gate_pass &= ok
        gate_lines.append(
            f"- GATE 1 gross sentinel: {'PASS' if ok else 'FAIL'} — delta "
            f"{gross['auc_delta']:+.5f} (every seed positive: {every_seed_pos}; "
            f">=5x null band {null_band:.5f}: {margin_ok}; p<=0.001: {p_ok} (p={gross['auc_p']:.4f}))"
        )
    else:
        gate_pass = False
        gate_lines.append("- GATE 1 gross sentinel: MISSING")

    # GATE 2 = POWER gate: the soft-oracle is tuned to ~+0.005 (the candidate
    # effect size). If it can't clear the null band with significance, the
    # screen cannot resolve +0.005 candidate effects — a null candidate result
    # would be UNINTERPRETABLE (underpowered), not "no signal".
    soft = stats.get("ctl_sentinel_soft")
    if soft:
        ok = soft["auc_delta"] > null_band and soft["auc_p"] <= 0.05
        gate_pass &= ok
        gate_lines.append(
            f"- GATE 2 soft-oracle POWER (~+0.005 target): {'PASS' if ok else 'FAIL'} — delta "
            f"{soft['auc_delta']:+.5f} vs null band {null_band:.5f}, p={soft['auc_p']:.3f}. "
            f"{'Screen can resolve candidate-size effects.' if ok else 'UNDERPOWERED — null candidates uninterpretable; go to all-2024 folds or 2025.'}"
        )
    else:
        gate_pass = False
        gate_lines.append("- GATE 2 soft-oracle: MISSING")

    # M3 = natural subtle-leak probe (reported, not a hard gate — Codex r4/r5)
    if m3:
        ok = m3["auc_delta"] > null_band
        gate_lines.append(
            f"- GATE 3 M3 natural-leak probe (report-only): {'above' if ok else 'below'} null — "
            f"delta {m3['auc_delta']:+.5f} vs {null_band:.5f} (p={m3['auc_p']:.3f})"
        )

    null_ok = all(
        not (stats[a]["auc_p"] < 0.05 and stats[a]["auc_delta"] > 0)
        for a in NULL_ARMS if a in stats
    )
    gate_pass &= null_ok
    gate_lines.append(f"- GATE 4 nulls unremarkable: {'PASS' if null_ok else 'FAIL'} "
                      f"({ {a: round(stats[a]['auc_delta'], 5) for a in NULL_ARMS if a in stats} })")

    lines = [f"# Swing campaign Stage-1 screen report (amendment #3, residual stacking) — {date.today()}", ""]
    lines.append("> Soft-oracle caveat: it is a CONCENTRATED, sparse leak (few rows reveal "
                 "the exact outcome) calibrated to ~+0.005 mean daily rank-AUC. A real "
                 "candidate is a DIFFUSE weak signal — same mean delta, different per-day "
                 "variance — so the soft gate is a floor test of resolving power, not an "
                 "exact candidate analog.")
    lines.append(f"Seeds: {seeds}; screen days: {stats[arms[0]]['n_days'] if arms else 0}; "
                 f"primary stat: paired daily rank-AUC delta (seed-averaged), "
                 f"week-blocked sign-permutation p")
    lines.append("")
    lines.append(f"## GATES — {'ALL PASS' if gate_pass else 'FAILED (stop; do not read families as signal)'}")
    lines.extend(gate_lines)
    lines.append("")
    lines.append(f"Empirical practical-null band (max |null arm delta|): {null_band:.5f}")
    lines.append("")

    by_family = defaultdict(list)
    for a, st in stats.items():
        by_family[st["family"]].append((a, st))

    lines.append("## Families (primary stat; > null band AND p<0.05 marked ***)")
    bundle = []
    for fam in ("P", "B", "T", "S", "M"):
        lines.append(f"### {fam}")
        fam_rows = sorted(by_family.get(fam, []), key=lambda kv: -(kv[1]["auc_delta"]))
        for a, st in fam_rows:
            star = " ***" if (st["auc_delta"] > null_band and st["auc_p"] < 0.05) else ""
            lines.append(
                f"- `{a}`: aucΔ {st['auc_delta']:+.5f} (p={st['auc_p']:.3f}, "
                f"seeds {st['auc_per_seed']}), ndcgΔ {st['ndcg_delta']:+.5f}, "
                f"top1Δ {st['top1_delta']:+.4f}{star}"
            )
        if fam_rows:
            best_arm, best = fam_rows[0]
            all_neg = all(st["auc_delta"] < 0 and st["ndcg_delta"] < 0 for _, st in fam_rows)
            verdict = "DEAD (consistently negative)" if all_neg else "alive"
            lines.append(f"- **verdict: {verdict}; best `{best_arm}`**")
            if not all_neg:
                bundle.append(best_arm)
        lines.append("")

    lines.append("## Omnibus")
    for a, st in sorted(by_family.get("omnibus", []), key=lambda kv: -(kv[1]["auc_delta"])):
        lines.append(f"- `{a}`: aucΔ {st['auc_delta']:+.5f} (p={st['auc_p']:.3f}), "
                     f"ndcgΔ {st['ndcg_delta']:+.5f}")
    lines.append("")
    lines.append(f"## PROPOSED FROZEN BUNDLE (pending human review): {bundle}")
    if not gate_pass:
        lines.append("\n**GATES FAILED — bundle proposal void; family lines are diagnostics only.**")

    out_path = args.out / f"{date.today()}-swing-screen-report.md"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
