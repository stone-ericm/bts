#!/usr/bin/env python3
"""Retrospective FDR baseline over Phase 1 audit verdicts.

This is SOTA tracker #7 P1: a p-value randomization/FDR layer over existing
audit verdict artifacts. It is not e-BH or online FDR. The p-values are exact
paired sign-flip permutation p-values over per-season P@1 deltas from
``experiments/results/phase1/*/diff.json``.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bts.validate.fdr import (
    bh_qvalues,
    by_qvalues,
    sign_flip_permutation_pvalue,
)


DEFAULT_DIFF_GLOB = "experiments/results/phase1/*/diff.json"
DEFAULT_OUTPUT = Path("data/validation/audit_verdict_fdr_2026-05-06.json")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return None


def _read_summary(path: Path) -> str | None:
    summary_path = path.with_name("summary.txt")
    if summary_path.exists():
        return summary_path.read_text().strip()
    return None


def _extract_season_deltas(diff: dict[str, Any]) -> dict[str, float]:
    section = diff.get("p_at_1_by_season")
    if not isinstance(section, dict):
        return {}
    deltas: dict[str, float] = {}
    for season, row in section.items():
        if not isinstance(row, dict) or "delta" not in row:
            continue
        value = row["delta"]
        if value is None:
            continue
        deltas[str(season)] = float(value)
    return dict(sorted(deltas.items()))


def audit_record_from_diff(path: Path) -> dict[str, Any] | None:
    """Parse one phase1 diff artifact into a testable audit record."""
    diff = json.loads(path.read_text())
    season_deltas = _extract_season_deltas(diff)
    if not season_deltas:
        return None
    deltas = np.array(list(season_deltas.values()), dtype=float)
    pvals = sign_flip_permutation_pvalue(deltas)
    experiment = path.parent.name
    p57_mdp = diff.get("p_57_mdp") if isinstance(diff.get("p_57_mdp"), dict) else {}
    p57_exact = diff.get("p_57_exact") if isinstance(diff.get("p_57_exact"), dict) else {}
    streak_metrics = diff.get("streak_metrics") if isinstance(diff.get("streak_metrics"), dict) else {}
    return {
        "experiment": experiment,
        "diff_path": path.as_posix(),
        "diff_file_sha256": _file_sha256(path),
        "summary": _read_summary(path),
        "season_deltas": season_deltas,
        "mean_p_at_1_delta": float(deltas.mean()),
        "p_two_sided": pvals["p_two_sided"],
        "p_one_sided_positive": pvals["p_one_sided_positive"],
        "p_one_sided_negative": pvals["p_one_sided_negative"],
        "n_seasons": pvals["n"],
        "direction": (
            "positive" if pvals["observed_mean_delta"] > 0
            else "negative" if pvals["observed_mean_delta"] < 0
            else "zero"
        ),
        "p_57_mdp_delta": p57_mdp.get("delta"),
        "p_57_exact_delta": p57_exact.get("delta"),
        "mean_max_streak_delta": (
            streak_metrics.get("mean_max_streak", {}).get("delta")
            if isinstance(streak_metrics.get("mean_max_streak"), dict)
            else None
        ),
    }


def collect_audit_records(patterns: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
            continue
        path = Path(pattern)
        if path.exists():
            paths.append(path)
    unique_paths = sorted({p.resolve() for p in paths})
    records = []
    skipped = []
    for path in unique_paths:
        record = audit_record_from_diff(path)
        if record is None:
            skipped.append(path.as_posix())
        else:
            records.append(record)
    return records, skipped


def build_report(patterns: list[str], *, q: float) -> dict[str, Any]:
    records, skipped = collect_audit_records(patterns)
    m = len(records)
    if m:
        pvalues = np.array([float(row["p_two_sided"]) for row in records])
        q_bh = bh_qvalues(pvalues)
        q_by = by_qvalues(pvalues)
    else:
        q_bh = q_by = np.zeros(0, dtype=float)

    for idx, row in enumerate(records):
        row["q_bh"] = float(q_bh[idx])
        row["q_by"] = float(q_by[idx])
        row["positive_survives_bh_q"] = bool(
            row["direction"] == "positive" and row["q_bh"] <= q
        )
        row["positive_survives_by_q"] = bool(
            row["direction"] == "positive" and row["q_by"] <= q
        )

    records_sorted = sorted(records, key=lambda r: (r["p_two_sided"], r["experiment"]))
    return {
        "schema_version": "audit_verdict_fdr_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "method": "p-value FDR baseline (BH + BY) over Phase 1 audit verdicts",
            "p_value_construction": (
                "Exact paired sign-flip permutation p-values over per-season "
                "P@1 deltas from phase1 diff.json artifacts."
            ),
            "family_scope": "experiments/results/phase1/*/diff.json with p_at_1_by_season deltas",
            "q_threshold": float(q),
            "bh_dependence_assumption": "PRDS",
            "by_dependence_assumption": "arbitrary (with c(m) harmonic penalty)",
            "deploy_gate": None,
            "git_head_sha": _git_head_sha(),
            "notes": (
                "This is an ordinary p-value FDR and randomization-test baseline. "
                "It does NOT close e-BH or online FDR because no valid e-values "
                "or e-processes are constructed here."
            ),
        },
        "m": m,
        "n_skipped": len(skipped),
        "skipped_diff_paths": skipped,
        "n_positive_survive_bh_q": int(sum(r["positive_survives_bh_q"] for r in records)),
        "n_positive_survive_by_q": int(sum(r["positive_survives_by_q"] for r in records)),
        "records": records_sorted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diff-glob",
        action="append",
        default=None,
        help="diff.json path/glob. May be repeated. Default: %(default)s",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--q", type=float, default=0.05)
    args = parser.parse_args()

    patterns = args.diff_glob or [DEFAULT_DIFF_GLOB]
    report = build_report(patterns, q=args.q)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        "audit verdict FDR: m={m} positive_BH={bh} positive_BY={by} wrote {path}".format(
            m=report["m"],
            bh=report["n_positive_survive_bh_q"],
            by=report["n_positive_survive_by_q"],
            path=args.output,
        )
    )
    if report["records"]:
        best = report["records"][0]
        print(
            "smallest p: {exp} p={p:.4f} q_BH={q:.4f} mean_delta={d:+.6f}".format(
                exp=best["experiment"],
                p=best["p_two_sided"],
                q=best["q_bh"],
                d=best["mean_p_at_1_delta"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
