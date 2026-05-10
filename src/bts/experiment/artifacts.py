"""Candidate-vs-production ranked-slate artifact helpers.

This module freezes the v1 artifact surface for #16 historical screens:
paired production and candidate top-N ranked slate parquets plus a manifest
that records provenance and launch posture. The artifact is intentionally
research-only; it does not write production picks or deployment assets.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bts.experiment.base import ExperimentDef
from bts.experiment.runner import compose_blend_args


ARTIFACT_SCHEMA_VERSION = "bts_candidate_ranked_slate_pair_v1"
PRODUCTION_PICK_SNAPSHOT_VERSION = "production_pick_snapshot_v1"
PROFILE_REQUIRED_COLUMNS = {
    "date",
    "rank",
    "batter_id",
    "game_pk",
    "p_game_hit",
    "actual_hit",
    "n_pas",
}
PROFILE_SCHEMA_COLUMNS = [
    "artifact_schema_version",
    "run_kind",
    "variant",
    "model_name",
    "generated_at",
    "git_commit",
    "date",
    "season",
    "rank",
    "batter_id",
    "game_pk",
    "p_game_hit",
    "actual_hit",
    "n_pas",
]


def current_git_commit(cwd: str | Path = ".") -> str | None:
    """Return the current git commit SHA, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _profile_path_sort_key(key: str) -> tuple[int, int | str]:
    if key.isdigit():
        return (0, int(key))
    return (1, key)


def validate_ranked_profiles(frame: pd.DataFrame, *, label: str) -> None:
    """Validate the ranked-slate profile columns used by the artifact schema."""
    missing = sorted(PROFILE_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing ranked profile columns: {missing}")
    if frame.empty:
        raise ValueError(f"{label} contains no ranked profile rows")
    if frame["date"].isna().any():
        raise ValueError(f"{label} contains null dates")
    if (frame["rank"] < 1).any():
        raise ValueError(f"{label} contains non-positive ranks")
    duplicate_keys = frame.duplicated(["date", "rank"])
    if duplicate_keys.any():
        raise ValueError(f"{label} contains duplicate date/rank rows")


def tag_ranked_profiles(
    profiles: pd.DataFrame,
    *,
    variant: str,
    model_name: str,
    season: int,
    run_kind: str,
    generated_at: str,
    git_commit: str | None,
) -> pd.DataFrame:
    """Return profiles with frozen artifact metadata columns."""
    validate_ranked_profiles(profiles, label=f"{variant} season {season}")
    tagged = profiles.copy()
    tagged["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    tagged["run_kind"] = run_kind
    tagged["variant"] = variant
    tagged["model_name"] = model_name
    tagged["generated_at"] = generated_at
    tagged["git_commit"] = git_commit
    tagged["season"] = int(season)
    return tagged[PROFILE_SCHEMA_COLUMNS]


def save_ranked_profiles(profiles: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(path, index=False)
    return path


def predictions_to_ranked_profiles(
    predictions: pd.DataFrame,
    *,
    date: str,
    top_n: int,
) -> pd.DataFrame:
    """Convert live prediction rows to the ranked profile shape pre-outcome."""
    required = {"batter_id", "game_pk", "p_game_hit"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"live predictions missing columns: {missing}")

    profile = predictions.dropna(subset=["p_game_hit"]).head(top_n).copy()
    if profile.empty:
        raise ValueError(f"live predictions for {date} have no scored rows")
    profile["date"] = pd.Timestamp(date).date()
    profile["rank"] = range(1, len(profile) + 1)
    profile["actual_hit"] = pd.NA
    profile["n_pas"] = pd.NA
    return profile[["date", "rank", "batter_id", "game_pk", "p_game_hit", "actual_hit", "n_pas"]]


def _normalize_pick_slot(slot: Any) -> dict[str, Any] | None:
    if not isinstance(slot, dict):
        return None
    keep = [
        "batter_id",
        "batter_name",
        "team",
        "lineup_position",
        "pitcher_id",
        "pitcher_name",
        "pitcher_team",
        "game_pk",
        "game_time",
        "p_game_hit",
        "projected_lineup",
        "flags",
    ]
    return {key: slot.get(key) for key in keep if key in slot}


def load_production_pick_snapshot(
    pick_file: str | Path,
    *,
    expected_date: str,
) -> dict[str, Any]:
    """Load a locked production pick JSON into a manifest-safe snapshot.

    The live-forward artifact is a paired ranked-slate surface, while the
    production decision is locked in ``data/picks``. Keeping a compact,
    read-only snapshot in the manifest preserves parity evidence without
    writing production picks, model caches, or deploy assets.
    """
    path = Path(pick_file)
    body = json.loads(path.read_text())
    pick_date = str(body.get("date"))
    if pick_date != str(expected_date):
        raise ValueError(
            f"production pick date mismatch: expected {expected_date!r}, "
            f"found {pick_date!r} in {path}"
        )

    return {
        "snapshot_version": PRODUCTION_PICK_SNAPSHOT_VERSION,
        "source_path": str(path),
        "source_sha256": _file_sha256(path),
        "date": pick_date,
        "run_time": body.get("run_time"),
        "result": body.get("result"),
        "slot_results": body.get("slot_results"),
        "model_git_sha": body.get("model_git_sha"),
        "model_pickle_sha256": body.get("model_pickle_sha256"),
        "policy_npz_sha256": body.get("policy_npz_sha256"),
        "production_lgbm_deterministic": body.get("production_lgbm_deterministic"),
        "production_pick_json": body,
        "slots": {
            "pick": _normalize_pick_slot(body.get("pick")),
            "double_down": _normalize_pick_slot(body.get("double_down")),
        },
    }


def materialize_candidate_profile_pair(
    *,
    pa_df: pd.DataFrame,
    candidate: ExperimentDef,
    seasons: list[int],
    output_dir: str | Path,
    retrain_every: int = 7,
    top_n: int = 10,
    baseline_name: str = "production_lgbm_v0",
    run_kind: str = "historical_local_screen",
    data_dir: str | Path | None = None,
    git_commit: str | None = None,
    generated_at: str | None = None,
) -> dict:
    """Materialize paired production/candidate ranked-slate artifacts.

    Args:
        pa_df: Feature-enriched PA dataframe.
        candidate: ExperimentDef to compare against the production blend.
        seasons: Historical seasons to score.
        output_dir: Directory that will receive manifest.json and profiles/.
        retrain_every: Walk-forward retraining cadence.
        top_n: Ranked slate size to retain per day.
        baseline_name: Human-readable production reference label.
        run_kind: Artifact role; v1 uses "historical_local_screen".
        data_dir: Source data directory for manifest provenance.
        git_commit: Optional commit SHA override for tests.
        generated_at: Optional timestamp override for tests.

    Returns:
        Manifest dict written to output_dir/manifest.json.
    """
    from bts.simulate.backtest_blend import blend_walk_forward

    if not seasons:
        raise ValueError("seasons must include at least one season")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or utc_timestamp()
    git_commit = git_commit if git_commit is not None else current_git_commit()

    baseline_configs, baseline_params, baseline_capture = compose_blend_args([])
    candidate_df = (
        candidate.modify_features(pa_df.copy())
        if candidate.touches_features()
        else pa_df
    )
    candidate_configs, candidate_params, candidate_capture = compose_blend_args([candidate])

    profile_paths: dict[str, dict[str, str]] = {"production": {}, "candidate": {}}
    row_counts: dict[str, dict[str, int]] = {"production": {}, "candidate": {}}
    day_counts: dict[str, dict[str, int]] = {"production": {}, "candidate": {}}

    for season in seasons:
        production_profiles = blend_walk_forward(
            pa_df,
            season,
            retrain_every=retrain_every,
            top_n=top_n,
            blend_configs=baseline_configs,
            lgb_params=baseline_params,
            capture_per_model=baseline_capture,
        )
        candidate_profiles = blend_walk_forward(
            candidate_df,
            season,
            retrain_every=retrain_every,
            top_n=top_n,
            blend_configs=candidate_configs,
            lgb_params=candidate_params,
            capture_per_model=candidate_capture,
        )

        tagged_production = tag_ranked_profiles(
            production_profiles,
            variant="production",
            model_name=baseline_name,
            season=season,
            run_kind=run_kind,
            generated_at=generated_at,
            git_commit=git_commit,
        )
        tagged_candidate = tag_ranked_profiles(
            candidate_profiles,
            variant="candidate",
            model_name=candidate.name,
            season=season,
            run_kind=run_kind,
            generated_at=generated_at,
            git_commit=git_commit,
        )

        prod_path = output_root / "profiles" / "production" / f"backtest_{season}.parquet"
        cand_path = output_root / "profiles" / "candidate" / f"backtest_{season}.parquet"
        save_ranked_profiles(tagged_production, prod_path)
        save_ranked_profiles(tagged_candidate, cand_path)

        season_key = str(season)
        profile_paths["production"][season_key] = _relative(prod_path, output_root)
        profile_paths["candidate"][season_key] = _relative(cand_path, output_root)
        row_counts["production"][season_key] = int(len(tagged_production))
        row_counts["candidate"][season_key] = int(len(tagged_candidate))
        day_counts["production"][season_key] = int(tagged_production["date"].nunique())
        day_counts["candidate"][season_key] = int(tagged_candidate["date"].nunique())

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "git_commit": git_commit,
        "run_kind": run_kind,
        "production_deploy_claim": False,
        "fresh_target_claim": run_kind == "live_forward_preoutcome",
        "candidate_name": candidate.name,
        "baseline_name": baseline_name,
        "seasons": [int(s) for s in seasons],
        "top_n": int(top_n),
        "retrain_every": int(retrain_every),
        "data_dir": str(data_dir) if data_dir is not None else None,
        "environment": {
            "BTS_LGBM_RANDOM_STATE": os.environ.get("BTS_LGBM_RANDOM_STATE"),
            "BTS_LGBM_DETERMINISTIC": os.environ.get("BTS_LGBM_DETERMINISTIC"),
        },
        "profile_schema_columns": list(PROFILE_SCHEMA_COLUMNS),
        "profile_paths": profile_paths,
        "row_counts": row_counts,
        "day_counts": day_counts,
    }
    _write_json(manifest, output_root / "manifest.json")
    return manifest


def materialize_live_candidate_profile_pair(
    *,
    date: str,
    candidate: ExperimentDef,
    output_dir: str | Path,
    data_dir: str | Path = "data/processed",
    top_n: int = 10,
    refresh_data: bool = False,
    production_pick_file: str | Path | None = None,
    baseline_name: str = "production_lgbm_v0",
    git_commit: str | None = None,
    generated_at: str | None = None,
) -> dict:
    """Materialize pre-outcome production/candidate ranked slates for one date.

    This is the fresh-target logging path. It intentionally writes only
    research artifacts under ``output_dir``: no pick JSON, no model cache, no
    posting side effects, and no deploy files.
    """
    from bts.model.predict import run_pipeline

    if candidate.touches_features() or candidate.feature_cols() is not None:
        raise ValueError(
            "live candidate artifact logging currently supports training/blend "
            "config experiments only"
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or utc_timestamp()
    git_commit = git_commit if git_commit is not None else current_git_commit()

    candidate_configs, candidate_params, _ = compose_blend_args([candidate])
    production_predictions = run_pipeline(
        date,
        data_dir=str(data_dir),
        refresh_data=refresh_data,
    )
    candidate_predictions = run_pipeline(
        date,
        data_dir=str(data_dir),
        refresh_data=False,
        blend_configs_override=candidate_configs,
        lgb_params_override=candidate_params,
    )

    production_profiles = predictions_to_ranked_profiles(
        production_predictions,
        date=date,
        top_n=top_n,
    )
    candidate_profiles = predictions_to_ranked_profiles(
        candidate_predictions,
        date=date,
        top_n=top_n,
    )
    production_pick_snapshot = None
    if production_pick_file is not None:
        production_pick_snapshot = load_production_pick_snapshot(
            production_pick_file,
            expected_date=date,
        )
    season = pd.Timestamp(date).year
    tagged_production = tag_ranked_profiles(
        production_profiles,
        variant="production",
        model_name=baseline_name,
        season=season,
        run_kind="live_forward_preoutcome",
        generated_at=generated_at,
        git_commit=git_commit,
    )
    tagged_candidate = tag_ranked_profiles(
        candidate_profiles,
        variant="candidate",
        model_name=candidate.name,
        season=season,
        run_kind="live_forward_preoutcome",
        generated_at=generated_at,
        git_commit=git_commit,
    )

    date_key = str(date)
    prod_path = output_root / "profiles" / "production" / f"live_{date_key}.parquet"
    cand_path = output_root / "profiles" / "candidate" / f"live_{date_key}.parquet"
    save_ranked_profiles(tagged_production, prod_path)
    save_ranked_profiles(tagged_candidate, cand_path)

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "git_commit": git_commit,
        "run_kind": "live_forward_preoutcome",
        "production_deploy_claim": False,
        "fresh_target_claim": True,
        "candidate_name": candidate.name,
        "baseline_name": baseline_name,
        "date": date_key,
        "dates": [date_key],
        "seasons": [int(season)],
        "top_n": int(top_n),
        "retrain_every": None,
        "data_dir": str(data_dir),
        "refresh_data": bool(refresh_data),
        "production_pick_snapshot": production_pick_snapshot,
        "environment": {
            "BTS_LGBM_RANDOM_STATE": os.environ.get("BTS_LGBM_RANDOM_STATE"),
            "BTS_LGBM_DETERMINISTIC": os.environ.get("BTS_LGBM_DETERMINISTIC"),
        },
        "profile_schema_columns": list(PROFILE_SCHEMA_COLUMNS),
        "profile_paths": {
            "production": {date_key: _relative(prod_path, output_root)},
            "candidate": {date_key: _relative(cand_path, output_root)},
        },
        "row_counts": {
            "production": {date_key: int(len(tagged_production))},
            "candidate": {date_key: int(len(tagged_candidate))},
        },
        "day_counts": {
            "production": {date_key: int(tagged_production["date"].nunique())},
            "candidate": {date_key: int(tagged_candidate["date"].nunique())},
        },
    }
    _write_json(manifest, output_root / "manifest.json")
    return manifest


def load_manifest(artifact_dir: str | Path) -> dict:
    path = Path(artifact_dir) / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported candidate artifact schema: "
            f"{manifest.get('schema_version')!r}"
        )
    return manifest


def _load_variant_profiles(
    artifact_dir: Path,
    manifest: dict,
    variant: str,
) -> pd.DataFrame:
    frames = []
    paths = manifest.get("profile_paths", {}).get(variant, {})
    for season_key in sorted(paths, key=_profile_path_sort_key):
        path = artifact_dir / paths[season_key]
        frame = pd.read_parquet(path)
        validate_ranked_profiles(frame, label=f"{variant} {season_key}")
        frames.append(frame)
    if not frames:
        raise ValueError(f"manifest has no {variant} profile paths")
    return pd.concat(frames, ignore_index=True)


def _append_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str | None = None,
) -> None:
    row = {"name": name, "status": "pass" if passed else "fail"}
    if detail:
        row["detail"] = detail
    checks.append(row)


def _series_all_equal(frame: pd.DataFrame, column: str, expected: Any) -> bool:
    if column not in frame.columns:
        return False
    values = frame[column].dropna().unique().tolist()
    if expected is None:
        return len(values) == 0
    return values == [expected]


def _candidate_verification_report(
    *,
    artifact_root: Path,
    manifest: dict,
    variant_reports: dict[str, dict[str, Any]],
    checks: list[dict[str, Any]],
    generated_at: str | None,
    save_path: str | Path | None,
) -> dict:
    failed = [check for check in checks if check["status"] != "pass"]
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": generated_at or utc_timestamp(),
        "artifact_dir": str(artifact_root),
        "ok": not failed,
        "failure_count": len(failed),
        "manifest": {
            "schema_version": manifest.get("schema_version"),
            "git_commit": manifest.get("git_commit"),
            "run_kind": manifest.get("run_kind"),
            "production_deploy_claim": manifest.get("production_deploy_claim"),
            "fresh_target_claim": manifest.get("fresh_target_claim"),
            "candidate_name": manifest.get("candidate_name"),
            "baseline_name": manifest.get("baseline_name"),
            "date": manifest.get("date"),
            "dates": manifest.get("dates"),
            "top_n": manifest.get("top_n"),
            "has_production_pick_snapshot": bool(manifest.get("production_pick_snapshot")),
        },
        "variants": variant_reports,
        "checks": checks,
    }

    if save_path is not None:
        _write_json(report, Path(save_path))
        report["verification_path"] = str(save_path)
    return report


def verify_candidate_artifact_pair(
    *,
    artifact_dir: str | Path,
    expected_run_kind: str | None = None,
    expected_candidate: str | None = None,
    expected_date: str | None = None,
    expected_git_commit: str | None = None,
    expected_top_n: int | None = None,
    require_live_preoutcome: bool = False,
    require_production_pick_snapshot: bool = False,
    generated_at: str | None = None,
    save_path: str | Path | None = None,
) -> dict:
    """Verify paired production/candidate ranked-slate artifact integrity.

    This is a read-only post-export gate. It validates manifest fields,
    referenced profile parquets, row-count metadata, and the stricter null
    outcome contract for live pre-outcome artifacts.
    """
    artifact_root = Path(artifact_dir)
    checks: list[dict[str, Any]] = []
    manifest_path = artifact_root / "manifest.json"
    _append_check(checks, "manifest_exists", manifest_path.exists(), str(manifest_path))
    if not manifest_path.exists():
        return _candidate_verification_report(
            artifact_root=artifact_root,
            manifest={},
            variant_reports={},
            checks=checks,
            generated_at=generated_at,
            save_path=save_path,
        )

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        _append_check(checks, "manifest_json_readable", False, str(exc))
        return _candidate_verification_report(
            artifact_root=artifact_root,
            manifest={},
            variant_reports={},
            checks=checks,
            generated_at=generated_at,
            save_path=save_path,
        )
    _append_check(checks, "manifest_json_readable", True)
    _append_check(
        checks,
        "manifest_schema_version",
        manifest.get("schema_version") == ARTIFACT_SCHEMA_VERSION,
        f"expected {ARTIFACT_SCHEMA_VERSION!r}, found {manifest.get('schema_version')!r}",
    )

    run_kind = manifest.get("run_kind")
    candidate_name = manifest.get("candidate_name")
    date_key = expected_date or manifest.get("date")
    if expected_run_kind is not None:
        _append_check(
            checks,
            "expected_run_kind",
            run_kind == expected_run_kind,
            f"expected {expected_run_kind!r}, found {run_kind!r}",
        )
    if expected_candidate is not None:
        _append_check(
            checks,
            "expected_candidate",
            candidate_name == expected_candidate,
            f"expected {expected_candidate!r}, found {candidate_name!r}",
        )
    if expected_date is not None:
        _append_check(
            checks,
            "expected_date",
            manifest.get("date") == expected_date
            and manifest.get("dates") == [expected_date],
            f"expected {expected_date!r}, found date={manifest.get('date')!r} "
            f"dates={manifest.get('dates')!r}",
        )
    if expected_git_commit is not None:
        _append_check(
            checks,
            "expected_git_commit",
            manifest.get("git_commit") == expected_git_commit,
            f"expected {expected_git_commit!r}, found {manifest.get('git_commit')!r}",
        )
    if expected_top_n is not None:
        _append_check(
            checks,
            "expected_top_n",
            manifest.get("top_n") == expected_top_n,
            f"expected {expected_top_n}, found {manifest.get('top_n')!r}",
        )

    if require_live_preoutcome:
        _append_check(
            checks,
            "live_run_kind",
            run_kind == "live_forward_preoutcome",
            f"found {run_kind!r}",
        )
        _append_check(
            checks,
            "live_fresh_target_claim",
            manifest.get("fresh_target_claim") is True,
            f"found {manifest.get('fresh_target_claim')!r}",
        )
        _append_check(
            checks,
            "live_production_deploy_claim",
            manifest.get("production_deploy_claim") is False,
            f"found {manifest.get('production_deploy_claim')!r}",
        )
        _append_check(
            checks,
            "live_git_commit_present",
            bool(manifest.get("git_commit")),
            f"found {manifest.get('git_commit')!r}",
        )
        _append_check(
            checks,
            "live_candidate_name_present",
            bool(candidate_name),
            f"found {candidate_name!r}",
        )
        _append_check(
            checks,
            "live_date_present",
            bool(manifest.get("date")),
            f"found {manifest.get('date')!r}",
        )
        _append_check(
            checks,
            "live_top_n_present",
            manifest.get("top_n") is not None,
            f"found {manifest.get('top_n')!r}",
        )

    production_pick_snapshot = manifest.get("production_pick_snapshot")
    if require_production_pick_snapshot:
        _append_check(
            checks,
            "production_pick_snapshot_present",
            isinstance(production_pick_snapshot, dict),
            f"found {type(production_pick_snapshot).__name__}",
        )
        if isinstance(production_pick_snapshot, dict):
            _append_check(
                checks,
                "production_pick_snapshot_version",
                production_pick_snapshot.get("snapshot_version")
                == PRODUCTION_PICK_SNAPSHOT_VERSION,
                f"expected {PRODUCTION_PICK_SNAPSHOT_VERSION!r}, "
                f"found {production_pick_snapshot.get('snapshot_version')!r}",
            )
            _append_check(
                checks,
                "production_pick_snapshot_date",
                production_pick_snapshot.get("date") == date_key,
                f"expected {date_key!r}, found {production_pick_snapshot.get('date')!r}",
            )
            _append_check(
                checks,
                "production_pick_snapshot_json_inline",
                isinstance(production_pick_snapshot.get("production_pick_json"), dict),
                "locked production pick JSON must be embedded, not path-only",
            )
            slots = production_pick_snapshot.get("slots") or {}
            has_any_slot = any(
                isinstance(slots.get(slot_key), dict)
                and slots[slot_key].get("batter_id") is not None
                and slots[slot_key].get("game_pk") is not None
                for slot_key in ("pick", "double_down")
            )
            _append_check(
                checks,
                "production_pick_snapshot_slots",
                has_any_slot,
                f"slots={slots!r}",
            )

    profile_paths = manifest.get("profile_paths", {})
    row_counts = manifest.get("row_counts", {})
    day_counts = manifest.get("day_counts", {})
    row_top_n = expected_top_n
    if row_top_n is None and require_live_preoutcome:
        row_top_n = manifest.get("top_n")
    variant_reports: dict[str, dict[str, Any]] = {}

    for variant in ("production", "candidate"):
        variant_paths = profile_paths.get(variant, {})
        _append_check(
            checks,
            f"{variant}_paths_present",
            bool(variant_paths),
            f"paths={variant_paths!r}",
        )
        frames = []
        paths_report = {}
        for key in sorted(variant_paths, key=_profile_path_sort_key):
            rel_path = variant_paths[key]
            path = artifact_root / rel_path
            path_exists = path.exists()
            _append_check(
                checks,
                f"{variant}_{key}_path_exists",
                path_exists,
                str(path),
            )
            if not path_exists:
                continue
            try:
                frame = pd.read_parquet(path)
            except Exception as exc:
                _append_check(checks, f"{variant}_{key}_parquet_readable", False, str(exc))
                continue
            _append_check(checks, f"{variant}_{key}_parquet_readable", True)
            frames.append(frame)
            paths_report[key] = {
                "path": str(path),
                "rows": int(len(frame)),
                "dates": sorted(str(d) for d in frame["date"].dropna().unique())
                if "date" in frame.columns else [],
            }

            _append_check(
                checks,
                f"{variant}_{key}_columns",
                list(frame.columns) == PROFILE_SCHEMA_COLUMNS,
                f"found {list(frame.columns)!r}",
            )
            try:
                validate_ranked_profiles(frame, label=f"{variant} {key}")
            except ValueError as exc:
                _append_check(checks, f"{variant}_{key}_ranked_schema", False, str(exc))
            else:
                _append_check(checks, f"{variant}_{key}_ranked_schema", True)

            _append_check(
                checks,
                f"{variant}_{key}_artifact_schema_column",
                _series_all_equal(frame, "artifact_schema_version", ARTIFACT_SCHEMA_VERSION),
                f"expected {ARTIFACT_SCHEMA_VERSION!r}",
            )
            _append_check(
                checks,
                f"{variant}_{key}_run_kind_column",
                _series_all_equal(frame, "run_kind", run_kind),
                f"expected {run_kind!r}",
            )
            _append_check(
                checks,
                f"{variant}_{key}_variant_column",
                _series_all_equal(frame, "variant", variant),
                f"expected {variant!r}",
            )
            expected_model = (
                manifest.get("baseline_name")
                if variant == "production"
                else manifest.get("candidate_name")
            )
            _append_check(
                checks,
                f"{variant}_{key}_model_name_column",
                _series_all_equal(frame, "model_name", expected_model),
                f"expected {expected_model!r}",
            )
            _append_check(
                checks,
                f"{variant}_{key}_git_commit_column",
                _series_all_equal(frame, "git_commit", manifest.get("git_commit")),
                f"expected {manifest.get('git_commit')!r}",
            )

            expected_rows = row_counts.get(variant, {}).get(key)
            _append_check(
                checks,
                f"{variant}_{key}_row_count",
                expected_rows == len(frame),
                f"manifest={expected_rows!r}, actual={len(frame)}",
            )
            expected_days = day_counts.get(variant, {}).get(key)
            actual_days = int(frame["date"].nunique()) if "date" in frame.columns else None
            _append_check(
                checks,
                f"{variant}_{key}_day_count",
                expected_days == actual_days,
                f"manifest={expected_days!r}, actual={actual_days!r}",
            )

            if row_top_n is not None and "date" in frame.columns:
                per_date_counts = frame.groupby("date").size()
                _append_check(
                    checks,
                    f"{variant}_{key}_top_n_rows_per_date",
                    (per_date_counts == row_top_n).all(),
                    f"counts={per_date_counts.to_dict()!r}",
                )
            if require_live_preoutcome:
                _append_check(
                    checks,
                    f"{variant}_{key}_actual_hit_null",
                    "actual_hit" in frame.columns and frame["actual_hit"].isna().all(),
                )
                _append_check(
                    checks,
                    f"{variant}_{key}_n_pas_null",
                    "n_pas" in frame.columns and frame["n_pas"].isna().all(),
                )

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            variant_reports[variant] = {
                "rows": int(len(combined)),
                "dates": sorted(str(d) for d in combined["date"].dropna().unique())
                if "date" in combined.columns else [],
                "paths": paths_report,
            }
        else:
            variant_reports[variant] = {"rows": 0, "dates": [], "paths": paths_report}

    if require_live_preoutcome and date_key is not None:
        for variant, summary in variant_reports.items():
            _append_check(
                checks,
                f"{variant}_live_date_only",
                summary["dates"] == [date_key],
                f"expected {[date_key]!r}, found {summary['dates']!r}",
            )

    return _candidate_verification_report(
        artifact_root=artifact_root,
        manifest=manifest,
        variant_reports=variant_reports,
        checks=checks,
        generated_at=generated_at,
        save_path=save_path,
    )


def compare_candidate_profile_pair(
    *,
    artifact_dir: str | Path,
    mc_trials: int = 10_000,
    season_length: int = 180,
    save_path: str | Path | None = None,
    generated_at: str | None = None,
) -> dict:
    """Compute scorecards and deltas for a frozen profile-pair artifact."""
    from bts.validate.scorecard import compute_full_scorecard, diff_scorecards

    artifact_root = Path(artifact_dir)
    manifest = load_manifest(artifact_root)
    production_profiles = _load_variant_profiles(artifact_root, manifest, "production")
    candidate_profiles = _load_variant_profiles(artifact_root, manifest, "candidate")

    production_scorecard = compute_full_scorecard(
        production_profiles,
        mc_trials=mc_trials,
        season_length=season_length,
    )
    candidate_scorecard = compute_full_scorecard(
        candidate_profiles,
        mc_trials=mc_trials,
        season_length=season_length,
    )
    diff = diff_scorecards(production_scorecard, candidate_scorecard)
    primary_delta = None
    if isinstance(diff.get("p_57_mdp"), dict):
        primary_delta = diff["p_57_mdp"].get("delta")

    comparison = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": generated_at or utc_timestamp(),
        "source_manifest": str(artifact_root / "manifest.json"),
        "source_git_commit": manifest.get("git_commit"),
        "run_kind": manifest.get("run_kind"),
        "production_deploy_claim": False,
        "fresh_target_claim": bool(manifest.get("fresh_target_claim", False)),
        "candidate_name": manifest.get("candidate_name"),
        "baseline_name": manifest.get("baseline_name"),
        "mc_trials": int(mc_trials),
        "season_length": int(season_length),
        "primary_metric": "p_57_mdp",
        "primary_delta": primary_delta,
        "scorecards": {
            "production": production_scorecard,
            "candidate": candidate_scorecard,
        },
        "diff": diff,
    }

    output_path = Path(save_path) if save_path is not None else artifact_root / "comparison.json"
    _write_json(comparison, output_path)
    comparison["comparison_path"] = str(output_path)
    return comparison
