#!/usr/bin/env python3
"""Aggregate screen results -> verdicts + frozen-bundle proposal (markdown).

Verdict rules (pre-registered, spec 2026-06-12):
- Controls: ctl_sentinel ndcg/auc MUST exceed baseline conspicuously (else
  the harness can't detect leakage -> STOP). ctl_placebo and ctl_permuted
  must be ~indistinguishable from baseline (else era-marker confounding).
- Variants ranked within family by paired daily NDCG delta vs baseline
  (same-seed pairing, mean across seeds); best variant per family proposed
  for the bundle.
- Families: alive unless coverage failure or consistently negative across
  ALL variants, seeds, and metrics (kill requires unanimity, not p-values).

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


def load_results(results_dir: Path) -> dict:
    out = {}
    for f in sorted(results_dir.glob("*_seed*.json")):
        r = json.loads(f.read_text())
        out[(r["arm"], r["seed"])] = r
    return out


def paired_ndcg_delta(arm_res: dict, base_res: dict) -> float:
    base_by_date = {d["date"]: d["ndcg"] for d in base_res["per_day"]}
    ds = [d["ndcg"] - base_by_date[d["date"]]
          for d in arm_res["per_day"] if d["date"] in base_by_date]
    return float(np.mean(ds)) if ds else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/audit"))
    args = ap.parse_args()

    res = load_results(args.results)
    seeds = sorted({s for (_, s) in res})
    arms = sorted({a for (a, _) in res})

    rows = []
    for arm in arms:
        if arm == "baseline":
            continue
        deltas, top1s, aucs = [], [], []
        for s in seeds:
            if (arm, s) in res and ("baseline", s) in res:
                deltas.append(paired_ndcg_delta(res[(arm, s)], res[("baseline", s)]))
                top1s.append(res[(arm, s)]["top1_hit"] - res[("baseline", s)]["top1_hit"])
                aucs.append((res[(arm, s)]["auc"] or 0) - (res[("baseline", s)]["auc"] or 0))
        rows.append({
            "arm": arm, "family": res[(arm, seeds[0])]["family"],
            "ndcg_delta": float(np.mean(deltas)),
            "ndcg_delta_per_seed": [round(d, 5) for d in deltas],
            "top1_delta": float(np.mean(top1s)),
            "auc_delta": float(np.mean(aucs)),
        })

    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)

    lines = [f"# Swing campaign Stage-1 screen report — {date.today()}", ""]
    lines.append("## Controls")
    for r in rows:
        if r["family"] == "control":
            lines.append(f"- `{r['arm']}`: ndcg Δ {r['ndcg_delta']:+.5f}, "
                         f"auc Δ {r['auc_delta']:+.5f}, per-seed {r['ndcg_delta_per_seed']}")
    lines.append("")
    lines.append("## Families (variants ranked by paired NDCG delta)")
    bundle = []
    for fam in ("P", "B", "T", "S", "M"):
        lines.append(f"### {fam}")
        fam_rows = sorted(by_family.get(fam, []), key=lambda r: -r["ndcg_delta"])
        for r in fam_rows:
            lines.append(f"- `{r['arm']}`: ndcg Δ {r['ndcg_delta']:+.5f} "
                         f"(seeds {r['ndcg_delta_per_seed']}), top1 Δ {r['top1_delta']:+.4f}, "
                         f"auc Δ {r['auc_delta']:+.5f}")
        if fam_rows:
            best = fam_rows[0]
            all_negative = all(
                d < 0 for r in fam_rows for d in r["ndcg_delta_per_seed"]
            ) and all(r["top1_delta"] < 0 and r["auc_delta"] < 0 for r in fam_rows)
            verdict = "DEAD (consistently negative everywhere)" if all_negative else "alive"
            lines.append(f"- **family verdict: {verdict}; best variant `{best['arm']}`**")
            if not all_negative:
                bundle.append(best["arm"])
        lines.append("")
    lines.append("## Omnibus arms")
    for r in sorted(by_family.get("omnibus", []), key=lambda r: -r["ndcg_delta"]):
        lines.append(f"- `{r['arm']}`: ndcg Δ {r['ndcg_delta']:+.5f}, "
                     f"top1 Δ {r['top1_delta']:+.4f}, auc Δ {r['auc_delta']:+.5f}")
    lines.append("")
    lines.append(f"## PROPOSED FROZEN BUNDLE (pending human review): {bundle}")
    out_path = args.out / f"{date.today()}-swing-screen-report.md"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
