#!/usr/bin/env python3
"""Aggregate every multi-seed verdict on file into one unified per-experiment table.

Sources (auto-detected when present):
  data/validation/screen_pooled_n10_2026-04-28.json
      → 32 experiments × 10 canonical seeds (Phase 1, post-determinism)
  data/lambdarank_only/{hetzner,vultr,oci}/<box>/phase1_seed<N>/<exp>/diff.json
      → 1 experiment × 10 seeds (Phase 1)
  data/borderline_n10_2026-04-29/<box>/phase1_seed<N>/<exp>/diff.json
      → 4 experiments × 10 new10 seeds (Phase 1, when borderline driver finishes)
  data/screen_postcutover/phase2_n10_set2_2026-04-28/set2_result.json
      → 2 experiments forward+backward × 10 orthogonal seeds (Phase 2)
  data/screen_postcutover/phase2_n10_2026-04-28/...
      → set 1 (Phase 2, when CPX62 finishes)

Output:
  data/validation/aggregate_verdicts_<TODAY>.json — JSON keyed by experiment
  stdout — pretty per-experiment table

For each (experiment, seed-corpus) pair, computes mean / SE / t-stat / pooled-keep
verdict under the post-2026-04-28 rule (mean > 0 AND |t| >= 1.5).
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from statistics import mean as _mean


def _t_stat(deltas: list[float]) -> tuple[float, float, float, float]:
    """Returns (mean, sd, se, t). SD=0 → t=inf."""
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    m = _mean(deltas)
    if n < 2:
        return m, 0.0, 0.0, float("inf") if m != 0 else 0.0
    var = sum((d - m) ** 2 for d in deltas) / (n - 1)
    if var == 0:
        return m, 0.0, 0.0, float("inf") if m != 0 else 0.0
    sd = var ** 0.5
    se = sd / (n ** 0.5)
    return m, sd, se, m / se


def _verdict(t: float, m: float, threshold: float = 1.5) -> str:
    """Apply post-2026-04-28 keep rule to a single (mean, t) summary."""
    if m <= 0:
        return "DROP"
    if abs(t) >= threshold:
        return "KEEP"
    return "BORDERLINE"


def load_screen_pooled_n10(repo_root: Path) -> dict[str, dict]:
    """Phase 1 canonical-n10 pooled JSON. One row per experiment."""
    p = repo_root / "data/validation/screen_pooled_n10_2026-04-28.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    out = {}
    for name, body in raw["results"].items():
        per_seed_p57 = [v["d_p57_mdp"] for v in body["per_seed"].values()]
        per_seed_p1_2024 = [v["d_p1_2024"] for v in body["per_seed"].values()]
        per_seed_p1_2025 = [v["d_p1_2025"] for v in body["per_seed"].values()]
        m, sd, se, t = _t_stat(per_seed_p57)
        out[name] = {
            "source": "screen_pooled_n10_2026-04-28",
            "phase": "1",
            "n_seeds": len(per_seed_p57),
            "metric": "d_p57_mdp",
            "mean": m,
            "sd": sd,
            "se": se,
            "t": t,
            "verdict": _verdict(t, m),
            "extras": {
                "mean_d_p1_2024": _mean(per_seed_p1_2024),
                "mean_d_p1_2025": _mean(per_seed_p1_2025),
            },
        }
    return out


def load_lambdarank_only(repo_root: Path) -> dict[str, dict]:
    """1 experiment × 10 seeds spread across 3 providers."""
    base = repo_root / "data/lambdarank_only"
    if not base.exists():
        return {}
    deltas = []
    for prov_dir in base.iterdir():
        if prov_dir.name not in ("hetzner", "vultr", "oci"):
            continue
        for box_dir in prov_dir.iterdir():
            if not box_dir.is_dir():
                continue
            for seed_dir in box_dir.iterdir():
                diff_p = seed_dir / "lambdarank_top1" / "diff.json"
                if not diff_p.exists():
                    continue
                d = json.loads(diff_p.read_text())
                deltas.append((d.get("p_57_mdp", {}) or {}).get("delta", 0))
    if not deltas:
        return {}
    m, sd, se, t = _t_stat(deltas)
    return {
        "lambdarank_top1": {
            "source": "lambdarank_only/{hetzner,vultr,oci}",
            "phase": "1",
            "n_seeds": len(deltas),
            "metric": "d_p57_mdp",
            "mean": m,
            "sd": sd,
            "se": se,
            "t": t,
            "verdict": _verdict(t, m),
            "extras": {},
        }
    }


def load_borderline_n10(repo_root: Path) -> dict[str, dict]:
    """4 experiments × 10 new10 seeds. Only present after the audit_driver run finishes."""
    base = repo_root / "data/borderline_n10_2026-04-29"
    if not base.exists():
        return {}
    by_exp: dict[str, list[float]] = {}
    for box_dir in base.iterdir():
        if not box_dir.is_dir() or box_dir.name == "boxes.json":
            continue
        for seed_dir in box_dir.iterdir():
            if not seed_dir.name.startswith("phase1_seed"):
                continue
            for exp_dir in seed_dir.iterdir():
                diff_p = exp_dir / "diff.json"
                if not diff_p.exists():
                    continue
                d = json.loads(diff_p.read_text())
                delta = (d.get("p_57_mdp", {}) or {}).get("delta", 0)
                by_exp.setdefault(exp_dir.name, []).append(delta)
    out = {}
    for name, deltas in by_exp.items():
        m, sd, se, t = _t_stat(deltas)
        out[name] = {
            "source": "borderline_n10_2026-04-29",
            "phase": "1",
            "n_seeds": len(deltas),
            "metric": "d_p57_mdp",
            "mean": m, "sd": sd, "se": se, "t": t,
            "verdict": _verdict(t, m),
            "extras": {},
        }
    return out


def load_phase2_set_results(repo_root: Path) -> dict[str, dict]:
    """Phase 2 forward+backward verdicts from set 1 / set 2 result JSONs."""
    out = {}
    for label, p in (
        ("phase2_set1", repo_root / "data/screen_postcutover/phase2_n10_2026-04-28/set1_result.json"),
        ("phase2_set2", repo_root / "data/screen_postcutover/phase2_n10_set2_2026-04-28/set2_result.json"),
    ):
        if not p.exists():
            continue
        raw = json.loads(p.read_text())
        for step in raw.get("forward_log", []):
            name = step.get("name")
            mean_d = step.get("delta", 0)
            per_seed = step.get("delta_per_seed", {})
            deltas = list(per_seed.values()) if per_seed else [mean_d]
            m, sd, se, t = _t_stat(deltas)
            out[f"{name}__{label}"] = {
                "source": p.name,
                "phase": "2-forward",
                "n_seeds": len(deltas),
                "metric": "d_p57_mdp",
                "mean": m, "sd": sd, "se": se, "t": t,
                "verdict": "KEEP" if step.get("kept") else "DROP",
                "extras": {"step_kept": step.get("kept"), "experiment": name},
            }
    return out


def load_phase2_targeted(repo_root: Path) -> dict[str, dict]:
    """Per-experiment Phase 2 audits — directory pattern phase2_<exp>_<set>[_<split>]_<date>/result.json.

    Set label is one of {set1_canonical, set2_orthogonal}. Optional split label
    is one of {splita, splitb} for parallel seed-split runs (5 seeds each on
    separate boxes); when both splits exist for the same (exp, set, date), their
    per-seed deltas are POOLED into a single n=10 verdict row. When only one
    split or no split is present, that row stands alone.
    """
    import re
    base = repo_root / "data/screen_postcutover"
    if not base.exists():
        return {}
    pattern = re.compile(
        r"^phase2_(?P<exp>.+?)_(?P<set>set[12]_(?:canonical|orthogonal))"
        r"(?:_(?P<split>split[ab]))?_(?P<date>\d{4}-\d{2}-\d{2})$"
    )
    # First pass: collect per-seed deltas grouped by (exp, set_label, date, log_kind, step_name).
    grouped: dict[tuple, dict] = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        exp_name = m.group("exp")
        set_label = m.group("set")
        run_date = m.group("date")
        split = m.group("split") or "single"
        result_p = d / "result.json"
        if not result_p.exists():
            continue
        raw = json.loads(result_p.read_text())
        for log_kind in ("forward_log", "backward_log"):
            for step in raw.get(log_kind, []):
                sname = step.get("name")
                if not sname:
                    continue
                per_seed = step.get("delta_per_seed", {}) or {}
                kept = step.get("kept", False)
                if not per_seed:
                    # Single-seed mode emits delta but no per_seed dict; synthesize one.
                    per_seed = {"default": step.get("delta", 0)}
                key = (exp_name, set_label, run_date, log_kind, sname)
                if key not in grouped:
                    grouped[key] = {"deltas": {}, "kept_any": False, "sources": []}
                # Tag each seed with its split origin so duplicate seed keys don't clobber.
                for seed_k, seed_d in per_seed.items():
                    grouped[key]["deltas"][f"{split}:{seed_k}"] = seed_d
                grouped[key]["kept_any"] = grouped[key]["kept_any"] or bool(kept)
                grouped[key]["sources"].append(d.name)

    # Second pass: emit one row per group, with deltas pooled across splits.
    out: dict[str, dict] = {}
    for (exp_name, set_label, run_date, log_kind, sname), group in grouped.items():
        deltas = list(group["deltas"].values())
        m_, sd, se, t = _t_stat(deltas)
        log_phase = "2-forward" if log_kind == "forward_log" else "2-backward"
        # Use a deterministic source label that lists every contributing dir.
        sources_str = "+".join(sorted(group["sources"]))
        key = f"{sname}__{set_label}_{run_date}_{log_kind}"
        out[key] = {
            "source": sources_str,
            "phase": log_phase,
            "n_seeds": len(deltas),
            "metric": "d_p57_mdp",
            "mean": m_, "sd": sd, "se": se, "t": t,
            "verdict": _verdict(t, m_) if log_phase == "2-forward" else (
                "KEEP" if group["kept_any"] else "DROP"
            ),
            "extras": {
                "step_kept_any": group["kept_any"],
                "experiment": sname,
                "set_label": set_label,
                "date": run_date,
                "audit_dir_experiment": exp_name,
                "n_splits_pooled": len(group["sources"]),
            },
        }
    return out


def main(repo_root: Path) -> None:
    rows: dict[str, dict] = {}
    for loader in (load_screen_pooled_n10, load_lambdarank_only,
                   load_borderline_n10, load_phase2_set_results,
                   load_phase2_targeted):
        for k, v in loader(repo_root).items():
            # Suffix duplicate names by source so we don't clobber across corpora.
            key = k if k not in rows else f"{k}__{v['source']}"
            rows[key] = v

    out = {
        "generated_at": str(date.today()),
        "n_rows": len(rows),
        "rule": "post-2026-04-28: KEEP if mean > 0 AND |t| >= 1.5; DROP if mean <= 0; BORDERLINE otherwise",
        "rows": rows,
    }
    out_path = repo_root / f"data/validation/aggregate_verdicts_{date.today()}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))

    print(f"\nWrote {out_path}\n")
    print(f"{'experiment':<60} {'phase':>7} {'n':>3} {'mean':>9} {'se':>8} {'t':>7}  {'verdict':<11} {'source':<35}")
    print("-" * 150)
    for name, r in sorted(rows.items(), key=lambda kv: (kv[1]["verdict"], -kv[1]["mean"])):
        t_str = f"{r['t']:+7.2f}" if r['t'] not in (float("inf"), float("-inf")) else "    inf"
        print(f"{name:<60} {r['phase']:>7} {r['n_seeds']:>3} {r['mean']:+9.5f} {r['se']:>8.5f} {t_str}  {r['verdict']:<11} {r['source']:<35}")


if __name__ == "__main__":
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/stone/projects/bts")
    main(repo)
