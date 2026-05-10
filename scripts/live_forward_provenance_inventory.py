#!/usr/bin/env python3
"""Inventory live-forward ranked-slate artifact provenance.

This read-only tool answers whether existing live-forward artifacts are usable
for at-lock leaderboard coverage/miscalibration joins. It does not export,
resolve, mutate, or deploy artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bts.experiment.artifacts import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    PRODUCTION_PICK_SNAPSHOT_VERSION,
    PROFILE_SCHEMA_COLUMNS,
)


SCHEMA_VERSION = "live_forward_provenance_inventory_v1"


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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _artifact_dirs(root: Path) -> list[Path]:
    if (root / "manifest.json").exists():
        return [root]
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    )


def _first_scalar(frame: pd.DataFrame, column: str) -> Any:
    if column not in frame.columns or frame.empty:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return values.iloc[0]


def _top_rank_row(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty or "rank" not in frame.columns:
        return None
    ranked = frame.sort_values("rank").head(1)
    if ranked.empty:
        return None
    row = ranked.iloc[0]
    return {
        "rank": int(row["rank"]) if pd.notna(row.get("rank")) else None,
        "batter_id": int(row["batter_id"]) if pd.notna(row.get("batter_id")) else None,
        "game_pk": int(row["game_pk"]) if pd.notna(row.get("game_pk")) else None,
        "p_game_hit": (
            float(row["p_game_hit"]) if pd.notna(row.get("p_game_hit")) else None
        ),
        "actual_hit": (
            bool(row["actual_hit"]) if pd.notna(row.get("actual_hit")) else None
        ),
        "n_pas": int(row["n_pas"]) if pd.notna(row.get("n_pas")) else None,
    }


def _variant_profile_report(
    *,
    artifact_dir: Path,
    manifest: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    variant_paths = manifest.get("profile_paths", {}).get(variant, {}) or {}
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for key, rel_path in sorted(variant_paths.items()):
        path = artifact_dir / rel_path
        row: dict[str, Any] = {
            "key": str(key),
            "path": str(path),
            "exists": path.exists(),
            "readable": False,
            "columns_match_schema": False,
            "rows": 0,
            "rank_min": None,
            "rank_max": None,
            "actual_hit_null_rows": None,
            "n_pas_null_rows": None,
            "top_rank": None,
        }
        if path.exists():
            try:
                frame = pd.read_parquet(path)
            except Exception as exc:  # pragma: no cover - defensive report path
                row["read_error"] = str(exc)
            else:
                row["readable"] = True
                row["columns_match_schema"] = list(frame.columns) == PROFILE_SCHEMA_COLUMNS
                row["rows"] = int(len(frame))
                row["rank_min"] = int(frame["rank"].min()) if "rank" in frame and len(frame) else None
                row["rank_max"] = int(frame["rank"].max()) if "rank" in frame and len(frame) else None
                row["actual_hit_null_rows"] = (
                    int(frame["actual_hit"].isna().sum()) if "actual_hit" in frame else None
                )
                row["n_pas_null_rows"] = (
                    int(frame["n_pas"].isna().sum()) if "n_pas" in frame else None
                )
                row["top_rank"] = _top_rank_row(frame)
                frames.append(frame)
        rows.append(row)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    manifest_rows = manifest.get("row_counts", {}).get(variant, {}) or {}
    expected_rows = sum(int(value) for value in manifest_rows.values()) if manifest_rows else None
    all_readable = bool(rows) and all(row["readable"] for row in rows)
    return {
        "paths": rows,
        "path_count": int(len(rows)),
        "all_paths_readable": all_readable,
        "all_columns_match_schema": bool(rows) and all(
            row["columns_match_schema"] for row in rows
        ),
        "rows": int(len(combined)),
        "expected_rows_from_manifest": expected_rows,
        "row_count_matches_manifest": (
            int(len(combined)) == expected_rows if expected_rows is not None else None
        ),
        "dates": sorted(str(d) for d in combined["date"].dropna().unique())
        if "date" in combined
        else [],
        "run_kinds": sorted(str(x) for x in combined["run_kind"].dropna().unique())
        if "run_kind" in combined
        else [],
        "model_names": sorted(str(x) for x in combined["model_name"].dropna().unique())
        if "model_name" in combined
        else [],
        "actual_hit_null_rows": (
            int(combined["actual_hit"].isna().sum()) if "actual_hit" in combined else None
        ),
        "n_pas_null_rows": (
            int(combined["n_pas"].isna().sum()) if "n_pas" in combined else None
        ),
        "max_rank": int(combined["rank"].max()) if "rank" in combined and len(combined) else None,
        "top_rank": _top_rank_row(combined),
        "generated_at_values": sorted(
            str(x) for x in combined["generated_at"].dropna().unique()
        )
        if "generated_at" in combined
        else [],
        "git_commit_values": sorted(str(x) for x in combined["git_commit"].dropna().unique())
        if "git_commit" in combined
        else [],
    }


def _snapshot_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    snapshot = manifest.get("production_pick_snapshot")
    if not isinstance(snapshot, dict):
        return {
            "present": False,
            "version": None,
            "version_ok": False,
            "date": None,
            "source_sha256": None,
            "slot_batter_ids": {},
        }
    slots = snapshot.get("slots") or {}
    slot_batter_ids = {
        str(slot): slot_payload.get("batter_id")
        for slot, slot_payload in slots.items()
        if isinstance(slot_payload, dict)
    }
    return {
        "present": True,
        "version": snapshot.get("snapshot_version"),
        "version_ok": snapshot.get("snapshot_version") == PRODUCTION_PICK_SNAPSHOT_VERSION,
        "date": snapshot.get("date"),
        "source_sha256": snapshot.get("source_sha256"),
        "slot_batter_ids": slot_batter_ids,
        "has_inline_json": isinstance(snapshot.get("production_pick_json"), dict),
    }


def _verification_summary(artifact_dir: Path) -> dict[str, Any]:
    verification = _read_json(artifact_dir / "verification.json")
    if verification is None:
        return {
            "present": False,
            "ok": None,
            "failure_count": None,
        }
    return {
        "present": True,
        "ok": verification.get("ok"),
        "failure_count": verification.get("failure_count"),
        "schema_version": verification.get("schema_version"),
    }


def _resolved_summary(*, date_key: str | None, resolved_root: Path | None) -> dict[str, Any]:
    if resolved_root is None or date_key is None:
        return {
            "present": False,
            "artifact_dir": None,
            "run_kind": None,
            "verification": {"present": False, "ok": None, "failure_count": None},
        }
    artifact_dir = resolved_root / date_key
    manifest = _read_json(artifact_dir / "manifest.json")
    if manifest is None:
        return {
            "present": False,
            "artifact_dir": str(artifact_dir),
            "run_kind": None,
            "verification": _verification_summary(artifact_dir),
        }
    variants = {
        variant: _variant_profile_report(
            artifact_dir=artifact_dir,
            manifest=manifest,
            variant=variant,
        )
        for variant in ("production", "candidate")
    }
    return {
        "present": True,
        "artifact_dir": str(artifact_dir),
        "run_kind": manifest.get("run_kind"),
        "source_run_kind": manifest.get("source_run_kind"),
        "verification": _verification_summary(artifact_dir),
        "resolution": _read_json(artifact_dir / "resolution.json") or {},
        "variants": variants,
        "outcome_joinable": _resolved_outcome_joinable(manifest, variants),
    }


def _resolved_outcome_joinable(
    manifest: dict[str, Any],
    variants: dict[str, dict[str, Any]],
) -> bool:
    if manifest.get("run_kind") != "live_forward_resolved":
        return False
    for variant in ("production", "candidate"):
        report = variants.get(variant, {})
        if not report.get("all_paths_readable"):
            return False
        if report.get("actual_hit_null_rows") not in (0, None):
            return False
        if report.get("n_pas_null_rows") not in (0, None):
            return False
    return True


def inspect_artifact_dir(
    artifact_dir: Path,
    *,
    resolved_root: Path | None,
    expected_candidate: str | None,
    expected_top_n: int | None,
    require_production_pick_snapshot: bool,
) -> dict[str, Any]:
    manifest = _read_json(artifact_dir / "manifest.json")
    if manifest is None:
        return {
            "artifact_dir": str(artifact_dir),
            "manifest_exists": False,
            "at_lock_ranked_surface_joinable": False,
            "official_fresh_target_ready": False,
        }
    date_key = manifest.get("date")
    variants = {
        variant: _variant_profile_report(
            artifact_dir=artifact_dir,
            manifest=manifest,
            variant=variant,
        )
        for variant in ("production", "candidate")
    }
    snapshot = _snapshot_summary(manifest)
    verification = _verification_summary(artifact_dir)
    resolved = _resolved_summary(date_key=date_key, resolved_root=resolved_root)

    live_preoutcome = manifest.get("run_kind") == "live_forward_preoutcome"
    research_only = (
        manifest.get("fresh_target_claim") is True
        and manifest.get("production_deploy_claim") is False
    )
    expected_candidate_ok = (
        True if expected_candidate is None else manifest.get("candidate_name") == expected_candidate
    )
    expected_top_n_ok = (
        True if expected_top_n is None else manifest.get("top_n") == expected_top_n
    )
    profiles_joinable = all(
        variants[variant].get("all_paths_readable")
        and variants[variant].get("all_columns_match_schema")
        and variants[variant].get("row_count_matches_manifest") is not False
        for variant in ("production", "candidate")
    )
    preoutcome_nulls = all(
        variants[variant].get("actual_hit_null_rows") == variants[variant].get("rows")
        and variants[variant].get("n_pas_null_rows") == variants[variant].get("rows")
        for variant in ("production", "candidate")
    )
    snapshot_ok = snapshot["present"] and snapshot["version_ok"] and snapshot["has_inline_json"]
    verifier_ok_or_missing = verification["ok"] is True or verification["present"] is False

    at_lock_ranked_surface_joinable = (
        live_preoutcome
        and research_only
        and expected_candidate_ok
        and expected_top_n_ok
        and profiles_joinable
        and preoutcome_nulls
    )
    official_fresh_target_ready = (
        at_lock_ranked_surface_joinable
        and (snapshot_ok or not require_production_pick_snapshot)
        and verification["ok"] is True
    )
    return {
        "artifact_dir": str(artifact_dir),
        "manifest_exists": True,
        "date": date_key,
        "schema_version": manifest.get("schema_version"),
        "schema_version_ok": manifest.get("schema_version") == ARTIFACT_SCHEMA_VERSION,
        "generated_at": manifest.get("generated_at"),
        "git_commit": manifest.get("git_commit"),
        "run_kind": manifest.get("run_kind"),
        "fresh_target_claim": manifest.get("fresh_target_claim"),
        "production_deploy_claim": manifest.get("production_deploy_claim"),
        "candidate_name": manifest.get("candidate_name"),
        "baseline_name": manifest.get("baseline_name"),
        "top_n": manifest.get("top_n"),
        "environment": manifest.get("environment") or {},
        "production_pick_snapshot": snapshot,
        "verification": verification,
        "variants": variants,
        "resolved": resolved,
        "at_lock_ranked_surface_joinable": bool(at_lock_ranked_surface_joinable),
        "resolved_outcome_joinable": bool(resolved.get("outcome_joinable", False)),
        "official_fresh_target_ready": bool(official_fresh_target_ready),
        "requires_verifier_before_official_use": bool(verifier_ok_or_missing and not verification["present"]),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dates = [row.get("date") for row in rows if row.get("date")]
    ready = [row for row in rows if row.get("official_fresh_target_ready")]
    at_lock = [row for row in rows if row.get("at_lock_ranked_surface_joinable")]
    resolved = [row for row in rows if row.get("resolved_outcome_joinable")]
    return {
        "artifact_count": int(len(rows)),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "at_lock_ranked_surface_joinable_count": int(len(at_lock)),
        "resolved_outcome_joinable_count": int(len(resolved)),
        "official_fresh_target_ready_count": int(len(ready)),
        "missing_verification_count": int(
            sum(1 for row in rows if not row.get("verification", {}).get("present"))
        ),
        "missing_production_pick_snapshot_count": int(
            sum(
                1
                for row in rows
                if not row.get("production_pick_snapshot", {}).get("present")
            )
        ),
        "git_commit_counts": _value_counts(row.get("git_commit") for row in rows),
        "run_kind_counts": _value_counts(row.get("run_kind") for row in rows),
        "candidate_counts": _value_counts(row.get("candidate_name") for row in rows),
    }


def _value_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = "missing" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_inventory(
    *,
    artifact_root: Path,
    output_path: Path,
    rows_output_path: Path | None,
    resolved_root: Path | None,
    expected_candidate: str | None,
    expected_top_n: int | None,
    require_production_pick_snapshot: bool,
    generated_at: str | None = None,
) -> dict[str, Any]:
    artifact_dirs = _artifact_dirs(artifact_root)
    rows = [
        inspect_artifact_dir(
            artifact_dir,
            resolved_root=resolved_root,
            expected_candidate=expected_candidate,
            expected_top_n=expected_top_n,
            require_production_pick_snapshot=require_production_pick_snapshot,
        )
        for artifact_dir in artifact_dirs
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "research_only": True,
        "production_deploy_claim": False,
        "mutation_free_inventory": True,
        "artifact_root": str(artifact_root),
        "resolved_root": str(resolved_root) if resolved_root is not None else None,
        "expected_candidate": expected_candidate,
        "expected_top_n": expected_top_n,
        "require_production_pick_snapshot": bool(require_production_pick_snapshot),
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    write_json(report, output_path)
    if rows_output_path is not None:
        rows_output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.json_normalize(rows).to_parquet(rows_output_path, index=False)
        report["rows_output_path"] = str(rows_output_path)
        write_json(report, output_path)
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, default=None)
    parser.add_argument("--expected-candidate", default="decision_weighted_lgbm_v0")
    parser.add_argument("--expected-top-n", type=int, default=10)
    parser.add_argument(
        "--no-require-production-pick-snapshot",
        action="store_true",
        help="Do not require production_pick_snapshot for official-ready flag.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_inventory(
        artifact_root=args.artifact_root,
        output_path=args.output,
        rows_output_path=args.rows_output,
        resolved_root=args.resolved_root,
        expected_candidate=args.expected_candidate,
        expected_top_n=args.expected_top_n,
        require_production_pick_snapshot=not args.no_require_production_pick_snapshot,
    )
    print(json.dumps({
        "schema_version": report["schema_version"],
        "output": str(args.output),
        "artifact_count": report["summary"]["artifact_count"],
        "official_fresh_target_ready_count": report["summary"][
            "official_fresh_target_ready_count"
        ],
        "at_lock_ranked_surface_joinable_count": report["summary"][
            "at_lock_ranked_surface_joinable_count"
        ],
        "resolved_outcome_joinable_count": report["summary"][
            "resolved_outcome_joinable_count"
        ],
    }, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
