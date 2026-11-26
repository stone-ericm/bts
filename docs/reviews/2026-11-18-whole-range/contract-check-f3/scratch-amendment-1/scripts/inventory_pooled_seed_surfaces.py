#!/usr/bin/env python3
"""Inventory pooled-seed BTS validation and profile surfaces.

This script intentionally records metadata only. It does not read large parquet
payloads, rebuild policies, or launch compute. The output is a provenance map
for deciding which pooled-seed surfaces are reproducible enough to use in the
next SOTA measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("data/validation/pooled_seed_inventory_2026-05-06.json")

VALIDATION_ARTIFACTS = [
    Path("data/validation/pooled_policy_ab.json"),
    Path("data/validation/pooled_policy_ab_24seed_consolidated.json"),
    Path("data/validation/pooled_policy_ab_trackd_crosspath.json"),
    Path("data/validation/pooled_policy_mc_replay_ab.json"),
    Path("data/validation/screen_pooled_n10_2026-04-28.json"),
    Path("data/validation/baseline_n100_deterministic_2026-04-27.json"),
]

PROFILE_SURFACES = [
    {
        "id": "canonical_single_seed_backtests",
        "path": Path("data/simulation"),
        "role": "single-seed canonical backtest files; not a pooled seed surface",
    },
    {
        "id": "pooled_bins_run_default16",
        "path": Path("data/hetzner_results/pooled_bins_run"),
        "role": "raw per-seed policy-bin profiles used by pooled_policy_ab.json",
    },
    {
        "id": "pooled_bins_run_trackc8",
        "path": Path("data/hetzner_results/pooled_bins_run_trackc"),
        "role": "additional raw per-seed policy-bin profiles used in the 24-seed consolidation",
    },
    {
        "id": "pooled_bins_run_trackd8_crosspath",
        "path": Path("data/hetzner_results/pooled_bins_run_trackd"),
        "role": "cross-path raw per-seed policy-bin profiles used by pooled_policy_ab_trackd_crosspath.json",
    },
    {
        "id": "audit_phase1_score_json_16",
        "path": Path("data/hetzner_results/audit_phase1"),
        "role": "phase-1 per-seed score JSONs; profile directories exist but do not contain backtest parquets",
    },
    {
        "id": "audit_full_48seed_v2_scorecards",
        "path": Path("data/hetzner_results/audit_full_48seed_v2"),
        "role": "48-seed experiment scorecards; no raw policy-bin profile parquets",
    },
    {
        "id": "det_baselines_n100",
        "path": Path("data/det_baselines_n100"),
        "role": "100 deterministic baseline scorecards; no raw policy-bin profile parquets",
    },
]

SEED_DIR_RE = re.compile(r"^(?:simulation_seed|profiles_seed|phase1_seed|seed)(\d+)$")
SEED_JSON_RE = re.compile(r"^seed(\d+)\.json$")
BACKTEST_RE = re.compile(r"backtest_(\d{4})\.parquet$")


def _rel(path: Path) -> str:
    return path.as_posix()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_tracked_paths() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.splitlines())


def _extract_validation_summary(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "top_level_keys": sorted(body.keys()),
    }
    for key in [
        "generated_at",
        "corpus",
        "flags",
        "providers",
        "seed_pool_size",
        "n_seeds",
        "n_experiments",
        "summary",
        "winners",
        "losers",
        "within_pool_summary",
        "leave_one_out_summary",
        "mc_summary",
        "replay_summary",
    ]:
        if key in body:
            summary[key] = body[key]
    if "metrics" in body:
        summary["metrics"] = {
            key: body["metrics"].get(key)
            for key in ["p_at_1_avg", "p_57_mdp", "p_57_exact", "mean_max_streak"]
            if key in body["metrics"]
        }
    return summary


def _seed_from_dir(path: Path) -> int | None:
    match = SEED_DIR_RE.match(path.name)
    return int(match.group(1)) if match else None


def _seed_from_json(path: Path) -> int | None:
    match = SEED_JSON_RE.match(path.name)
    return int(match.group(1)) if match else None


def _inventory_profile_surface(entry: dict[str, Any]) -> dict[str, Any]:
    root = entry["path"]
    out: dict[str, Any] = {
        "id": entry["id"],
        "path": _rel(root),
        "role": entry["role"],
        "exists": root.exists(),
    }
    if not root.exists():
        return out

    seed_dirs = [p for p in root.rglob("*") if p.is_dir() and _seed_from_dir(p) is not None]
    seed_jsons = [p for p in root.rglob("seed*.json") if _seed_from_json(p) is not None]
    backtests = sorted(root.rglob("backtest_*.parquet"))
    scorecards = sorted(root.rglob("scorecard.json"))
    diffs = sorted(root.rglob("diff.json"))

    seeds_from_dirs = sorted({int(_seed_from_dir(p)) for p in seed_dirs if _seed_from_dir(p) is not None})
    seeds_from_json = sorted({int(_seed_from_json(p)) for p in seed_jsons if _seed_from_json(p) is not None})
    seasons = sorted({
        int(match.group(1))
        for p in backtests
        for match in [BACKTEST_RE.match(p.name)]
        if match
    })

    per_seed_backtest_counts: dict[str, int] = {}
    for seed_dir in seed_dirs:
        seed = _seed_from_dir(seed_dir)
        if seed is None:
            continue
        count = len(list(seed_dir.glob("backtest_*.parquet")))
        if count:
            per_seed_backtest_counts[str(seed)] = count

    out.update({
        "seed_dir_count": len(seed_dirs),
        "seeds_from_dirs": seeds_from_dirs,
        "seed_json_count": len(seed_jsons),
        "seeds_from_json": seeds_from_json,
        "backtest_parquet_count": len(backtests),
        "has_raw_backtest_parquets": bool(backtests),
        "backtest_seasons": seasons,
        "per_seed_backtest_counts": dict(sorted(
            per_seed_backtest_counts.items(),
            key=lambda kv: int(kv[0]),
        )),
        "seed_identity_source": (
            "path_seed_dir" if backtests and seed_dirs else
            "none_detected" if backtests else
            "not_applicable"
        ),
        "usable_for_pooled_bin_rebuild": bool(backtests and seed_dirs),
        "scorecard_json_count": len(scorecards),
        "diff_json_count": len(diffs),
    })
    return out


def build_inventory() -> dict[str, Any]:
    tracked = _git_tracked_paths()
    validation = []
    for path in VALIDATION_ARTIFACTS:
        item: dict[str, Any] = {
            "path": _rel(path),
            "exists": path.exists(),
            "git_tracked": _rel(path) in tracked,
        }
        if path.exists():
            body = json.loads(path.read_text())
            item.update(_extract_validation_summary(path, body))
        validation.append(item)

    profiles = []
    for entry in PROFILE_SURFACES:
        item = _inventory_profile_surface(entry)
        item["git_tracked"] = _rel(entry["path"]) in tracked
        profiles.append(item)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/inventory_pooled_seed_surfaces.py",
        "validation_artifacts": validation,
        "profile_surfaces": profiles,
        "notes": [
            "Raw parquet payloads are counted but not read.",
            "Determinism state is recorded only where the artifact embeds flags.",
            "Some source surfaces are local untracked research artifacts; git_tracked records this.",
            "Git tracking is reported for files; directories are normally untracked as paths.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    inventory = build_inventory()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
