#!/usr/bin/env python3
"""Export live-forward artifact profiles as ranked surfaces.

The output parquet is intentionally shaped for
``scripts/leaderboard_mechanism_mining.py --surface NAME=PATH``. Its required
columns match the `load_ranked_surfaces` contract: `date`, `rank`, `batter_id`,
`p_game_hit`, and `actual_hit`, with `game_pk` and `n_pas` preserved when
available.

This is read-only with respect to production state. It reads existing
live-forward artifacts and writes only the requested output files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bts.experiment.artifacts import PROFILE_SCHEMA_COLUMNS  # noqa: E402
from scripts.leaderboard_backfilled_model_audit import load_ranked_surfaces  # noqa: E402
from scripts.leaderboard_candidate_join_audit import normalize_date_key  # noqa: E402
from scripts.live_forward_provenance_inventory import (  # noqa: E402
    _artifact_dirs,
    _read_json,
    inspect_artifact_dir,
    utc_now_iso,
    write_json,
)


SCHEMA_VERSION = "live_forward_ranked_surface_export_v1"
VALID_VARIANTS = {"production", "candidate"}
SURFACE_REQUIRED_COLUMNS = ["date", "rank", "batter_id", "p_game_hit", "actual_hit"]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def parse_dates(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {normalize_date_key(part.strip()) for part in raw.split(",") if part.strip()}


def _date_selected(
    date_key: str | None,
    *,
    dates: set[str] | None,
    min_date: str | None,
    max_date: str | None,
) -> bool:
    if date_key is None:
        return False
    normalized = normalize_date_key(date_key)
    if dates is not None and normalized not in dates:
        return False
    if min_date is not None and normalized < normalize_date_key(min_date):
        return False
    if max_date is not None and normalized > normalize_date_key(max_date):
        return False
    return True


def _coerce_surface_frame(
    frame: pd.DataFrame,
    *,
    artifact_dir: Path,
    manifest: dict[str, Any],
    variant: str,
    rel_path: str,
    readiness: dict[str, Any],
) -> pd.DataFrame:
    if list(frame.columns) != PROFILE_SCHEMA_COLUMNS:
        raise ValueError(
            f"{artifact_dir / rel_path} columns do not match PROFILE_SCHEMA_COLUMNS"
        )
    out = frame.copy()
    out["date"] = out["date"].map(normalize_date_key)
    out["rank"] = pd.to_numeric(out["rank"], errors="raise").astype(int)
    out["batter_id"] = pd.to_numeric(out["batter_id"], errors="raise").astype("Int64")
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out["p_game_hit"] = pd.to_numeric(out["p_game_hit"], errors="coerce")
    out["actual_hit"] = pd.to_numeric(out["actual_hit"], errors="coerce")
    out["n_pas"] = pd.to_numeric(out["n_pas"], errors="coerce").astype("Int64")
    out["surface_variant"] = variant
    out["source_artifact_dir"] = str(artifact_dir)
    out["source_profile_path"] = str(artifact_dir / rel_path)
    out["source_manifest_path"] = str(artifact_dir / "manifest.json")
    out["source_run_kind"] = manifest.get("run_kind")
    out["source_git_commit"] = manifest.get("git_commit")
    out["source_generated_at"] = manifest.get("generated_at")
    out["candidate_name"] = manifest.get("candidate_name")
    out["baseline_name"] = manifest.get("baseline_name")
    out["fresh_target_claim"] = manifest.get("fresh_target_claim")
    out["production_deploy_claim"] = manifest.get("production_deploy_claim")
    out["production_pick_snapshot_present"] = bool(manifest.get("production_pick_snapshot"))
    out["verification_ok"] = readiness.get("verification", {}).get("ok")
    out["at_lock_ranked_surface_joinable"] = readiness.get(
        "at_lock_ranked_surface_joinable"
    )
    out["official_fresh_target_ready"] = readiness.get("official_fresh_target_ready")
    return out


def _load_variant_frames(
    *,
    artifact_dir: Path,
    manifest: dict[str, Any],
    readiness: dict[str, Any],
    variant: str,
) -> list[pd.DataFrame]:
    paths = manifest.get("profile_paths", {}).get(variant, {}) or {}
    frames: list[pd.DataFrame] = []
    for _key, rel_path in sorted(paths.items()):
        path = artifact_dir / rel_path
        if not path.exists():
            raise FileNotFoundError(f"missing {variant} profile path: {path}")
        frames.append(
            _coerce_surface_frame(
                pd.read_parquet(path),
                artifact_dir=artifact_dir,
                manifest=manifest,
                variant=variant,
                rel_path=rel_path,
                readiness=readiness,
            )
        )
    return frames


def build_surface(
    *,
    artifact_root: Path,
    variant: str,
    require_official_ready: bool,
    output_path: Path,
    manifest_output_path: Path | None,
    resolved_root: Path | None,
    expected_candidate: str | None,
    expected_top_n: int | None,
    require_production_pick_snapshot: bool,
    dates: set[str] | None,
    min_date: str | None,
    max_date: str | None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VALID_VARIANTS)}, got {variant!r}")
    artifact_dirs = _artifact_dirs(artifact_root)
    frames: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    included: list[dict[str, Any]] = []
    for artifact_dir in artifact_dirs:
        readiness = inspect_artifact_dir(
            artifact_dir,
            resolved_root=resolved_root,
            expected_candidate=expected_candidate,
            expected_top_n=expected_top_n,
            require_production_pick_snapshot=require_production_pick_snapshot,
        )
        date_key = readiness.get("date")
        if not _date_selected(date_key, dates=dates, min_date=min_date, max_date=max_date):
            skipped.append({
                "artifact_dir": str(artifact_dir),
                "date": date_key,
                "reason": "date_filter",
            })
            continue
        eligible = (
            readiness.get("official_fresh_target_ready")
            if require_official_ready
            else readiness.get("at_lock_ranked_surface_joinable")
        )
        if not eligible:
            skipped.append({
                "artifact_dir": str(artifact_dir),
                "date": date_key,
                "reason": (
                    "not_official_fresh_target_ready"
                    if require_official_ready
                    else "not_at_lock_ranked_surface_joinable"
                ),
            })
            continue
        manifest = _read_json(artifact_dir / "manifest.json")
        if manifest is None:
            skipped.append({
                "artifact_dir": str(artifact_dir),
                "date": date_key,
                "reason": "missing_manifest",
            })
            continue
        variant_frames = _load_variant_frames(
            artifact_dir=artifact_dir,
            manifest=manifest,
            readiness=readiness,
            variant=variant,
        )
        if not variant_frames:
            skipped.append({
                "artifact_dir": str(artifact_dir),
                "date": date_key,
                "reason": f"missing_{variant}_profiles",
            })
            continue
        frames.extend(variant_frames)
        included.append({
            "artifact_dir": str(artifact_dir),
            "date": date_key,
            "run_kind": readiness.get("run_kind"),
            "git_commit": readiness.get("git_commit"),
            "official_fresh_target_ready": readiness.get("official_fresh_target_ready"),
            "at_lock_ranked_surface_joinable": readiness.get(
                "at_lock_ranked_surface_joinable"
            ),
        })

    if not frames:
        raise ValueError(
            "no eligible live-forward profile rows found; "
            f"require_official_ready={require_official_ready}"
        )
    surface = pd.concat(frames, ignore_index=True)
    duplicates = surface.duplicated(["date", "rank"], keep=False)
    if duplicates.any():
        examples = (
            surface.loc[duplicates, ["date", "rank", "source_artifact_dir"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"duplicate date/rank rows in exported surface: {examples}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    surface.to_parquet(output_path, index=False)

    # Contract smoke: the exported parquet must be loadable by the mechanism
    # mining surface reader without special casing.
    _ranked, _joinable, inventory = load_ranked_surfaces(
        {"contract_check": output_path}
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "research_only": True,
        "production_deploy_claim": False,
        "mutation_free_export": True,
        "surface_contract": "scripts.leaderboard_backfilled_model_audit.load_ranked_surfaces",
        "artifact_root": str(artifact_root),
        "resolved_root": str(resolved_root) if resolved_root is not None else None,
        "variant": variant,
        "require_official_ready": bool(require_official_ready),
        "expected_candidate": expected_candidate,
        "expected_top_n": expected_top_n,
        "output_path": str(output_path),
        "rows": int(len(surface)),
        "dates": int(surface["date"].nunique()),
        "date_min": surface["date"].min(),
        "date_max": surface["date"].max(),
        "max_rank": int(surface["rank"].max()) if len(surface) else None,
        "required_columns": SURFACE_REQUIRED_COLUMNS,
        "included_artifacts": included,
        "skipped_artifacts": skipped,
        "contract_check_inventory": inventory["contract_check"],
    }
    if manifest_output_path is not None:
        write_json(report, manifest_output_path)
        report["manifest_output_path"] = str(manifest_output_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("data/validation/decision_weighted_lgbm_v0_live_forward"),
    )
    parser.add_argument(
        "--resolved-root",
        type=Path,
        default=Path("data/validation/decision_weighted_lgbm_v0_live_forward_resolved"),
    )
    parser.add_argument("--variant", choices=sorted(VALID_VARIANTS), default="production")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--expected-candidate", default="decision_weighted_lgbm_v0")
    parser.add_argument("--expected-top-n", type=int, default=10)
    parser.add_argument(
        "--require-official-ready",
        action="store_true",
        help="Export only artifacts that pass the official fresh-target gate.",
    )
    parser.add_argument(
        "--no-require-production-pick-snapshot",
        action="store_true",
        help="Do not require production_pick_snapshot when computing official readiness.",
    )
    parser.add_argument("--dates", default=None, help="Comma-separated date filter.")
    parser.add_argument("--min-date", default=None)
    parser.add_argument("--max-date", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_surface(
        artifact_root=args.artifact_root,
        variant=args.variant,
        require_official_ready=args.require_official_ready,
        output_path=args.output,
        manifest_output_path=args.manifest_output,
        resolved_root=args.resolved_root,
        expected_candidate=args.expected_candidate,
        expected_top_n=args.expected_top_n,
        require_production_pick_snapshot=not args.no_require_production_pick_snapshot,
        dates=parse_dates(args.dates),
        min_date=args.min_date,
        max_date=args.max_date,
    )
    print(json.dumps({
        "schema_version": report["schema_version"],
        "output_path": report["output_path"],
        "rows": report["rows"],
        "dates": report["dates"],
        "date_min": report["date_min"],
        "date_max": report["date_max"],
        "require_official_ready": report["require_official_ready"],
    }, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
