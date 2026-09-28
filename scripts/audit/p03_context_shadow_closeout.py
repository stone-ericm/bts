"""P-03 closeout read: context-stack shadow v2 vs production, on the frozen W0.7 snapshot.

Protocol (docs/audit/2026-09-22-exposure-register.md §B P-03): the `shadow_eval` status
rule — only `context_stack_shadow_v2` shadow files count, each paired with the production
pick file of the same date; paired-day evaluation with Wilson intervals and a two-sided
sign test on the discordant days. W1.4b obligation: report agreement, discordant outcomes,
coverage and interval on the frozen eligibility; equal aggregates != equivalence; the
no-promote disposition stands unless the protocol says otherwise.

Primary numbers use the RECORDED results (what `check-results` reconciled nightly),
exactly as the status artifact computes them, plus the code's own paired bootstrap
(compute_shadow_quality defaults: 10,000 draws, seed 57). Cross-check: the code's
DD-aware dry-run recompute from cached raw game feeds (build_shadow_backfill_manifest);
every recorded-vs-recomputed disagreement is listed. Nothing is written except --out.

Run on the box:  .venv/bin/python scripts/audit/p03_context_shadow_closeout.py \
    --out /tmp/p03_context_shadow_closeout.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bts.picks import load_pick, load_shadow_pick
from bts.shadow_eval import (
    SHADOW_MODEL_NAME,
    SHADOW_STATUS_DEFAULT_MIN_DAYS,
    _recorded_quality_row,
    build_shadow_backfill_manifest,
    build_shadow_cycle_status,
    compute_shadow_quality,
)

DEFAULT_SNAPSHOT = Path("data/hetzner_results/season_2026_snapshot/final-20260928")


def _primary(daily) -> dict | None:
    if daily is None or daily.pick is None:
        return None
    p = daily.pick
    return {"batter": p.batter_name, "team": p.team, "p_game_hit": p.p_game_hit}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P-03 context-shadow closeout read")
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                    help="cached raw game feeds for the recompute cross-check (read-only)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    picks = a.snapshot / "data" / "picks"
    manifest_path = a.snapshot.parent / f"{a.snapshot.name}.sha256"

    status = build_shadow_cycle_status(picks, generated_at="p03-closeout-read", git_commit="n/a")

    rows, discordant = [], []
    for r in status["rows"]:
        date = r["date"]
        prod = load_pick(date, picks) if r["production_file"] else None
        shadow = load_shadow_pick(date, picks)
        rows.append(_recorded_quality_row(date=date, production=prod, shadow=shadow))
        pr, sr = (prod.result if prod else None), (shadow.result if shadow else None)
        if pr in ("hit", "miss") and sr in ("hit", "miss") and pr != sr:
            discordant.append({"date": date, "production_result": pr, "shadow_result": sr,
                               "production_primary": _primary(prod), "shadow_primary": _primary(shadow),
                               "primary_agree": r["primary_agree"]})
    recorded = compute_shadow_quality(rows)  # code defaults: n_bootstrap=10_000, seed=57

    recompute = build_shadow_backfill_manifest(picks, raw_dir=a.raw_dir)
    shadow_changes = [{"date": x["date"], "recorded": x["old_shadow_result"], "recomputed": x["new_shadow_result"],
                       "status": x["shadow"]["status"]}
                      for x in recompute["rows"] if x["old_shadow_result"] != x["new_shadow_result"]]
    q = recompute["quality_if_applied"]

    dates = [r["date"] for r in status["rows"]]
    report = {
        "read": "P-03 context-stack shadow v2 closeout (plan W1.4b)",
        "snapshot": str(a.snapshot),
        "snapshot_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None,
        "eligibility": {
            "shadow_model": SHADOW_MODEL_NAME,
            "rule": "v2 *.shadow.json files paired with <date>.json; evaluable = both results in {hit, miss}",
            "min_days_for_review": SHADOW_STATUS_DEFAULT_MIN_DAYS,
        },
        "coverage": {
            "cycle_state": status["cycle_state"],
            "first_shadow_date": status["coverage"]["first_shadow_date"],
            "latest_shadow_date": status["coverage"]["latest_shadow_date"],
            "counts": status["counts"],
            "unresolved_shadow_dates": status["coverage"]["unresolved_shadow_dates"],
            "missing_production_dates": status["coverage"]["missing_production_dates"],
            "days_on_or_after_2026_09_14_private_production": sum(1 for d in dates if d >= "2026-09-14"),
        },
        "recorded": recorded,
        "discordant_days": discordant,
        "recompute_crosscheck": {
            "raw_dir": str(a.raw_dir),
            "counts": recompute["counts"],
            "api_calls": sum(len(x["api_calls"]) for x in recompute["rows"]),
            "shadow_recorded_vs_recomputed_changes": shadow_changes,
            "production_recorded_vs_recomputed_mismatches": q["production_recorded_mismatches"],
            "quality_if_recomputed": {k: q[k] for k in ("n_evaluable_days", "production_day_hit_rate",
                                                         "shadow_day_hit_rate", "shadow_minus_production_hit_rate",
                                                         "paired_outcomes", "decision_agreement")},
        },
    }
    if a.out:
        a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")

    rq, po, ag = recorded, recorded["paired_outcomes"], recorded["decision_agreement"]
    print(f"coverage {report['coverage']['first_shadow_date']} -> {report['coverage']['latest_shadow_date']}; "
          f"cycle_state {status['cycle_state']}; counts {status['counts']}")
    print(f"evaluable paired days {rq['n_evaluable_days']} of {rq['n_days']}")
    print(f"production {rq['production_day_hit_rate']['hits']}/{rq['n_evaluable_days']} "
          f"wilson {rq['production_day_hit_rate']['wilson_95']}; shadow {rq['shadow_day_hit_rate']['hits']}/"
          f"{rq['n_evaluable_days']} wilson {rq['shadow_day_hit_rate']['wilson_95']}")
    print(f"gap shadow-prod {rq['shadow_minus_production_hit_rate']['value']} paired bootstrap 95% "
          f"{rq['shadow_minus_production_hit_rate']['paired_bootstrap_95']}")
    print(f"paired outcomes {po}")
    print(f"agreement primary {ag['primary']} pair {ag['pair_unordered']}")
    print(f"discordant days: {[(d['date'], d['production_result'], d['shadow_result']) for d in discordant]}")
    print(f"recompute cross-check counts {recompute['counts']}; shadow changes {shadow_changes}; "
          f"production mismatches {q['production_recorded_mismatches']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
