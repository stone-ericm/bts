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
RESOLVED_ARTIFACT_SCHEMA_VERSION = "bts_candidate_ranked_slate_pair_v2"
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = {
    ARTIFACT_SCHEMA_VERSION,
    RESOLVED_ARTIFACT_SCHEMA_VERSION,
}
PRODUCTION_PICK_SNAPSHOT_VERSION = "production_pick_snapshot_v1"
OUTCOME_STATUS_RESOLVED = "resolved"
OUTCOME_STATUS_VOID_POSTPONEMENT = "void_postponement"
OUTCOME_STATUS_VOID_CANCELLATION = "void_cancellation"
OUTCOME_STATUS_PENDING = "pending"
OUTCOME_STATUS_VALUES = (
    OUTCOME_STATUS_RESOLVED,
    OUTCOME_STATUS_VOID_POSTPONEMENT,
    OUTCOME_STATUS_VOID_CANCELLATION,
    OUTCOME_STATUS_PENDING,
)
VOID_OUTCOME_STATUSES = {
    OUTCOME_STATUS_VOID_POSTPONEMENT,
    OUTCOME_STATUS_VOID_CANCELLATION,
}
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
RESOLVED_PROFILE_SCHEMA_COLUMNS = [
    *PROFILE_SCHEMA_COLUMNS,
    "outcome_status",
]
VOID_DETAILED_STATES = {
    "postponed": OUTCOME_STATUS_VOID_POSTPONEMENT,
    "cancelled": OUTCOME_STATUS_VOID_CANCELLATION,
    "canceled": OUTCOME_STATUS_VOID_CANCELLATION,
}


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
    if manifest.get("schema_version") not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError(
            "unsupported candidate artifact schema: "
            f"{manifest.get('schema_version')!r}"
        )
    return manifest


def _expected_profile_columns(manifest: dict) -> list[str]:
    configured = manifest.get("profile_schema_columns")
    if (
        isinstance(configured, list)
        and all(isinstance(column, str) for column in configured)
    ):
        return list(configured)
    if manifest.get("schema_version") == RESOLVED_ARTIFACT_SCHEMA_VERSION:
        return list(RESOLVED_PROFILE_SCHEMA_COLUMNS)
    return list(PROFILE_SCHEMA_COLUMNS)


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


def _scorecard_profiles(
    profiles: pd.DataFrame,
    *,
    variant: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return rows eligible for scorecard denominators.

    v2 resolved live-forward artifacts can contain terminal void rows. Those
    rows are real artifact rows, but they are not observations of model quality.
    """
    report: dict[str, Any] = {
        "variant": variant,
        "input_rows": int(len(profiles)),
        "input_dates": int(profiles["date"].nunique()) if "date" in profiles else None,
        "applied": False,
    }
    if "outcome_status" not in profiles.columns:
        report["scorecard_rows"] = int(len(profiles))
        report["scorecard_dates"] = (
            int(profiles["date"].nunique()) if "date" in profiles else None
        )
        return profiles, report

    statuses = set(profiles["outcome_status"].dropna().unique())
    unknown_statuses = sorted(statuses - set(OUTCOME_STATUS_VALUES))
    if unknown_statuses:
        raise ValueError(
            f"{variant} has unsupported outcome_status values: {unknown_statuses}"
        )

    all_dates = set(profiles["date"].dropna().unique())
    resolved = profiles[profiles["outcome_status"] == OUTCOME_STATUS_RESOLVED].copy()
    rank12 = resolved[resolved["rank"].isin([1, 2])]
    complete_rank12 = rank12.groupby("date")["rank"].nunique()
    scorecard_dates = set(complete_rank12[complete_rank12 == 2].index)
    scorecard = resolved[resolved["date"].isin(scorecard_dates)].copy()

    if scorecard.empty:
        raise ValueError(
            f"{variant} has no scorecard-eligible dates after excluding "
            "non-resolved outcome_status rows"
        )

    excluded = profiles.loc[~profiles.index.isin(scorecard.index)]
    excluded_status_counts = _outcome_status_counts(excluded)
    report.update(
        {
            "applied": True,
            "scorecard_rows": int(len(scorecard)),
            "scorecard_dates": int(scorecard["date"].nunique()),
            "excluded_rows": int(len(profiles) - len(scorecard)),
            "excluded_status_counts": excluded_status_counts,
            "dropped_dates_missing_resolved_rank_1_or_2": [
                str(date) for date in sorted(all_dates - scorecard_dates)
            ],
        }
    )
    return scorecard, report


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


def _manifest_date_keys(manifest: dict) -> list[str]:
    dates = manifest.get("dates")
    if dates is None:
        date = manifest.get("date")
        dates = [date] if date else []
    date_keys = [str(date) for date in dates if date is not None]
    if not date_keys:
        raise ValueError("manifest has no live-forward dates to resolve")
    return date_keys


def _load_outcomes_from_pa(
    *,
    data_dir: str | Path,
    date_keys: list[str],
) -> pd.DataFrame:
    """Load batter-game outcomes for the artifact dates from processed PA data."""
    data_root = Path(data_dir)
    years = sorted({pd.Timestamp(date_key).year for date_key in date_keys})
    frames = []
    for year in years:
        path = data_root / f"pa_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing processed PA parquet: {path}")
        frames.append(
            pd.read_parquet(
                path,
                columns=["date", "batter_id", "game_pk", "is_hit"],
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=["_outcome_date_key", "batter_id", "game_pk", "actual_hit", "n_pas"]
        )

    pa = pd.concat(frames, ignore_index=True)
    pa["_outcome_date_key"] = pd.to_datetime(pa["date"]).dt.strftime("%Y-%m-%d")
    pa = pa[pa["_outcome_date_key"].isin(date_keys)]
    if pa.empty:
        return pd.DataFrame(
            columns=["_outcome_date_key", "batter_id", "game_pk", "actual_hit", "n_pas"]
        )

    # Absence from processed PA data is treated as missing outcome evidence, not
    # as a 0-PA game, because game finality is checked outside this artifact join.
    outcomes = (
        pa.groupby(["_outcome_date_key", "batter_id", "game_pk"], as_index=False)
        .agg(actual_hit=("is_hit", "max"), n_pas=("is_hit", "count"))
    )
    outcomes["actual_hit"] = outcomes["actual_hit"].astype(int)
    outcomes["n_pas"] = outcomes["n_pas"].astype(int)
    return outcomes


def _terminal_void_outcome_status(detailed: str | None) -> str | None:
    return VOID_DETAILED_STATES.get((detailed or "").strip().lower())


def _is_terminal_void_detailed_state(detailed: str | None) -> bool:
    return _terminal_void_outcome_status(detailed) is not None


def _load_terminal_void_statuses(
    *,
    date_keys: list[str],
) -> dict[str, dict[int, dict[str, str]]]:
    from bts.picks import get_game_statuses_detailed

    statuses_by_date: dict[str, dict[int, dict[str, str]]] = {}
    for date_key in date_keys:
        statuses_by_date[date_key] = get_game_statuses_detailed(date_key)
    return statuses_by_date


def _missing_row_terminal_void_status(
    row: pd.Series,
    *,
    terminal_void_statuses: dict[str, dict[int, dict[str, str]]],
) -> str | None:
    date_key = str(row.get("_outcome_date_key") or row.get("date"))
    try:
        game_pk = int(row["game_pk"])
    except (TypeError, ValueError):
        return None
    status = terminal_void_statuses.get(date_key, {}).get(game_pk)
    if not status:
        return None
    return _terminal_void_outcome_status(status.get("detailed"))


def _empty_outcome_status_counts() -> dict[str, int]:
    return {status: 0 for status in OUTCOME_STATUS_VALUES}


def _outcome_status_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = _empty_outcome_status_counts()
    if "outcome_status" not in frame.columns:
        return counts
    observed = frame["outcome_status"].value_counts(dropna=False).to_dict()
    for status in OUTCOME_STATUS_VALUES:
        counts[status] = int(observed.get(status, 0))
    return counts


def _add_outcome_status_counts(
    target: dict[str, int],
    source: dict[str, int],
) -> None:
    for status in OUTCOME_STATUS_VALUES:
        target[status] += int(source.get(status, 0))


def _missing_row_example(
    row: pd.Series,
    *,
    variant: str,
    profile_key: str,
) -> dict[str, Any]:
    return {
        "date": str(row.get("date")),
        "rank": int(row["rank"]) if pd.notna(row.get("rank")) else None,
        "batter_id": int(row["batter_id"]) if pd.notna(row.get("batter_id")) else None,
        "game_pk": int(row["game_pk"]) if pd.notna(row.get("game_pk")) else None,
        "variant": variant,
        "profile_key": profile_key,
    }


def resolve_live_candidate_artifact_pair(
    *,
    artifact_dir: str | Path,
    output_dir: str | Path,
    data_dir: str | Path = "data/processed",
    allow_partial: bool = False,
    treat_void_games_as_terminal: bool = False,
    detailed_statuses_by_date: dict[str, dict[int, dict[str, str]]] | None = None,
    overwrite: bool = False,
    generated_at: str | None = None,
    save_path: str | Path | None = None,
) -> dict:
    """Join post-game outcomes onto a live-forward pre-outcome artifact copy.

    The source artifact is left unchanged. The resolved copy can be passed to
    ``compare-candidate-artifacts`` once all non-void outcome rows are present.
    """
    artifact_root = Path(artifact_dir)
    output_root = Path(output_dir)
    if artifact_root.resolve() == output_root.resolve():
        raise ValueError("output_dir must differ from artifact_dir")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise ValueError(
            f"resolved output_dir is not empty: {output_root}; "
            "pass overwrite=True to replace it"
        )

    manifest = load_manifest(artifact_root)
    if manifest.get("run_kind") != "live_forward_preoutcome":
        raise ValueError(
            "resolve-live-candidate-artifacts requires run_kind "
            f"'live_forward_preoutcome', found {manifest.get('run_kind')!r}"
        )

    date_keys = _manifest_date_keys(manifest)
    outcomes = _load_outcomes_from_pa(data_dir=data_dir, date_keys=date_keys)
    terminal_void_statuses = (
        detailed_statuses_by_date
        if detailed_statuses_by_date is not None
        else (
            _load_terminal_void_statuses(date_keys=date_keys)
            if treat_void_games_as_terminal
            else {}
        )
    )
    generated_at = generated_at or utc_timestamp()

    resolved_items: list[tuple[Path, pd.DataFrame]] = []
    variant_reports: dict[str, dict[str, Any]] = {}
    total_missing = 0
    total_terminal_void = 0
    total_outcome_status_counts = _empty_outcome_status_counts()
    missing_examples: list[dict[str, Any]] = []
    terminal_void_examples: list[dict[str, Any]] = []

    for variant in ("production", "candidate"):
        variant_paths = manifest.get("profile_paths", {}).get(variant, {})
        paths_report = {}
        variant_missing = 0
        variant_terminal_void = 0
        variant_outcome_status_counts = _empty_outcome_status_counts()
        for key in sorted(variant_paths, key=_profile_path_sort_key):
            rel_path = variant_paths[key]
            source_path = artifact_root / rel_path
            frame = pd.read_parquet(source_path)
            validate_ranked_profiles(frame, label=f"{variant} {key}")

            joinable = frame.drop(
                columns=["actual_hit", "n_pas", "outcome_status"],
                errors="ignore",
            ).copy()
            joinable["_outcome_date_key"] = pd.to_datetime(joinable["date"]).dt.strftime(
                "%Y-%m-%d"
            )
            resolved = joinable.merge(
                outcomes,
                on=["_outcome_date_key", "batter_id", "game_pk"],
                how="left",
                indicator=True,
            )
            missing_mask = resolved["_merge"] == "left_only"
            resolved["outcome_status"] = OUTCOME_STATUS_RESOLVED
            resolved.loc[missing_mask, "outcome_status"] = OUTCOME_STATUS_PENDING
            if treat_void_games_as_terminal and missing_mask.any():
                void_statuses = resolved.loc[missing_mask].apply(
                    _missing_row_terminal_void_status,
                    axis=1,
                    terminal_void_statuses=terminal_void_statuses,
                )
                for index, outcome_status in void_statuses.dropna().items():
                    resolved.loc[index, "outcome_status"] = outcome_status
            terminal_void_mask = resolved["outcome_status"].isin(VOID_OUTCOME_STATUSES)
            unresolved_missing_mask = resolved["outcome_status"] == OUTCOME_STATUS_PENDING

            missing_count = int(unresolved_missing_mask.sum())
            terminal_void_count = int(terminal_void_mask.sum())
            path_outcome_status_counts = _outcome_status_counts(resolved)
            total_missing += missing_count
            total_terminal_void += terminal_void_count
            variant_missing += missing_count
            variant_terminal_void += terminal_void_count
            _add_outcome_status_counts(
                total_outcome_status_counts,
                path_outcome_status_counts,
            )
            _add_outcome_status_counts(
                variant_outcome_status_counts,
                path_outcome_status_counts,
            )
            if missing_count:
                for _, row in resolved.loc[unresolved_missing_mask].head(10).iterrows():
                    missing_examples.append(
                        _missing_row_example(row, variant=variant, profile_key=key)
                    )
            if terminal_void_count:
                for _, row in resolved.loc[terminal_void_mask].head(10).iterrows():
                    example = _missing_row_example(row, variant=variant, profile_key=key)
                    date_key = str(row.get("_outcome_date_key") or row.get("date"))
                    status = terminal_void_statuses.get(date_key, {}).get(
                        int(row["game_pk"]), {}
                    )
                    example["detailed_state"] = status.get("detailed")
                    terminal_void_examples.append(example)

            resolved = resolved.drop(columns=["_outcome_date_key", "_merge"])
            resolved["run_kind"] = "live_forward_resolved"
            resolved["artifact_schema_version"] = RESOLVED_ARTIFACT_SCHEMA_VERSION
            resolved["actual_hit"] = resolved["actual_hit"].astype("Int64")
            resolved["n_pas"] = resolved["n_pas"].astype("Int64")
            resolved = resolved[RESOLVED_PROFILE_SCHEMA_COLUMNS]

            target_path = output_root / rel_path
            resolved_items.append((target_path, resolved))
            paths_report[key] = {
                "source_path": str(source_path),
                "resolved_path": str(target_path),
                "rows": int(len(resolved)),
                "missing_outcomes": missing_count,
                "terminal_void_outcomes": terminal_void_count,
                "outcome_status_counts": path_outcome_status_counts,
            }

        variant_reports[variant] = {
            "rows": int(sum(item["rows"] for item in paths_report.values())),
            "missing_outcomes": variant_missing,
            "terminal_void_outcomes": variant_terminal_void,
            "outcome_status_counts": variant_outcome_status_counts,
            "paths": paths_report,
        }

    if total_missing and not allow_partial:
        raise ValueError(
            f"missing outcomes for {total_missing} live-forward artifact rows; "
            f"examples={missing_examples[:10]!r}"
        )

    for target_path, frame in resolved_items:
        save_ranked_profiles(frame, target_path)

    resolved_manifest = json.loads(json.dumps(manifest, default=_json_default))
    resolved_manifest["schema_version"] = RESOLVED_ARTIFACT_SCHEMA_VERSION
    resolved_manifest["run_kind"] = "live_forward_resolved"
    resolved_manifest["source_run_kind"] = manifest.get("run_kind")
    resolved_manifest["source_schema_version"] = manifest.get("schema_version")
    resolved_manifest["source_manifest"] = str(artifact_root / "manifest.json")
    resolved_manifest["outcomes_resolved_at"] = generated_at
    resolved_manifest["outcome_data_dir"] = str(data_dir)
    resolved_manifest["outcome_allow_partial"] = bool(allow_partial)
    resolved_manifest["outcome_terminal_void_enabled"] = bool(treat_void_games_as_terminal)
    resolved_manifest["outcome_terminal_void_total"] = int(total_terminal_void)
    resolved_manifest["outcome_missing_total"] = int(total_missing)
    resolved_manifest["outcome_status_values"] = list(OUTCOME_STATUS_VALUES)
    resolved_manifest["outcome_status_counts"] = total_outcome_status_counts
    resolved_manifest["profile_schema_columns"] = list(RESOLVED_PROFILE_SCHEMA_COLUMNS)
    resolved_manifest["outcome_missing_semantics"] = (
        "Missing outcome rows mean no PA evidence was available for that "
        "date/batter/game key, including postponed or void games. They are "
        "never coerced to actual_hit=0."
    )
    resolved_manifest["outcome_status_semantics"] = (
        "resolved rows have observed actual_hit/n_pas values. "
        "void_postponement and void_cancellation rows are terminal non-events "
        "with actual_hit/n_pas left null. pending rows mean evidence is still "
        "missing and are not acceptable in official resolved artifacts."
    )
    resolved_manifest["outcome_terminal_void_semantics"] = (
        "When terminal void handling is enabled, missing rows whose original "
        "game was postponed or cancelled remain actual_hit/n_pas null and are "
        "counted separately from transient missing outcomes."
    )
    resolved_manifest["outcome_missing_by_variant"] = {
        variant: report["missing_outcomes"]
        for variant, report in variant_reports.items()
    }
    resolved_manifest["outcome_terminal_void_by_variant"] = {
        variant: report["terminal_void_outcomes"]
        for variant, report in variant_reports.items()
    }
    resolved_manifest["outcome_status_counts_by_variant"] = {
        variant: report["outcome_status_counts"]
        for variant, report in variant_reports.items()
    }
    _write_json(resolved_manifest, output_root / "manifest.json")

    report = {
        "schema_version": RESOLVED_ARTIFACT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_manifest": str(artifact_root / "manifest.json"),
        "resolved_manifest": str(output_root / "manifest.json"),
        "run_kind": "live_forward_resolved",
        "candidate_name": manifest.get("candidate_name"),
        "dates": date_keys,
        "complete": total_missing == 0,
        "missing_count": int(total_missing),
        "terminal_void_count": int(total_terminal_void),
        "outcome_status_counts": total_outcome_status_counts,
        "missing_semantics": resolved_manifest["outcome_missing_semantics"],
        "outcome_status_semantics": resolved_manifest["outcome_status_semantics"],
        "terminal_void_semantics": resolved_manifest["outcome_terminal_void_semantics"],
        "missing_examples": missing_examples[:20],
        "terminal_void_examples": terminal_void_examples[:20],
        "variants": variant_reports,
    }
    if save_path is not None:
        _write_json(report, Path(save_path))
        report["resolution_path"] = str(save_path)
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
    manifest_schema_version = manifest.get("schema_version")
    _append_check(
        checks,
        "manifest_schema_version",
        manifest_schema_version in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
        "expected one of "
        f"{sorted(SUPPORTED_ARTIFACT_SCHEMA_VERSIONS)!r}, "
        f"found {manifest_schema_version!r}",
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
    expected_profile_columns = _expected_profile_columns(manifest)
    observed_outcome_status_counts = _empty_outcome_status_counts()

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
                list(frame.columns) == expected_profile_columns,
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
                _series_all_equal(
                    frame,
                    "artifact_schema_version",
                    manifest_schema_version,
                ),
                f"expected {manifest_schema_version!r}",
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
            if run_kind == "live_forward_resolved":
                if "outcome_status" in frame.columns:
                    path_counts = _outcome_status_counts(frame)
                    _add_outcome_status_counts(
                        observed_outcome_status_counts,
                        path_counts,
                    )
                    observed_statuses = set(frame["outcome_status"].dropna().unique())
                    _append_check(
                        checks,
                        f"{variant}_{key}_outcome_status_values",
                        observed_statuses.issubset(set(OUTCOME_STATUS_VALUES)),
                        f"found {sorted(observed_statuses)!r}",
                    )
                    actual_hit_null = frame["actual_hit"].isna()
                    n_pas_null = frame["n_pas"].isna()
                    resolved_mask = frame["outcome_status"] == OUTCOME_STATUS_RESOLVED
                    void_mask = frame["outcome_status"].isin(VOID_OUTCOME_STATUSES)
                    pending_mask = frame["outcome_status"] == OUTCOME_STATUS_PENDING
                    _append_check(
                        checks,
                        f"{variant}_{key}_resolved_outcomes_observed",
                        (
                            frame.loc[resolved_mask, "actual_hit"].notna().all()
                            and frame.loc[resolved_mask, "n_pas"].notna().all()
                        ),
                        "resolved rows must have observed actual_hit and n_pas",
                    )
                    _append_check(
                        checks,
                        f"{variant}_{key}_void_outcomes_null",
                        (
                            frame.loc[void_mask, "actual_hit"].isna().all()
                            and frame.loc[void_mask, "n_pas"].isna().all()
                        ),
                        "void rows must keep actual_hit and n_pas null",
                    )
                    _append_check(
                        checks,
                        f"{variant}_{key}_pending_outcomes_absent",
                        not pending_mask.any(),
                        f"pending rows={int(pending_mask.sum())}",
                    )
                    _append_check(
                        checks,
                        f"{variant}_{key}_null_outcomes_known_void",
                        (actual_hit_null | n_pas_null).equals(void_mask),
                        "null outcome fields are allowed only for known void rows",
                    )
                else:
                    _append_check(
                        checks,
                        f"{variant}_{key}_resolved_outcomes_not_null",
                        (
                            "actual_hit" in frame.columns
                            and "n_pas" in frame.columns
                            and frame["actual_hit"].notna().all()
                            and frame["n_pas"].notna().all()
                        ),
                        "legacy resolved artifacts cannot contain null outcomes",
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

    if (
        run_kind == "live_forward_resolved"
        and manifest_schema_version == RESOLVED_ARTIFACT_SCHEMA_VERSION
    ):
        manifest_counts = manifest.get("outcome_status_counts")
        _append_check(
            checks,
            "outcome_status_counts",
            manifest_counts == observed_outcome_status_counts,
            f"manifest={manifest_counts!r}, observed={observed_outcome_status_counts!r}",
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
    production_scorecard_profiles, production_filter_report = _scorecard_profiles(
        production_profiles,
        variant="production",
    )
    candidate_scorecard_profiles, candidate_filter_report = _scorecard_profiles(
        candidate_profiles,
        variant="candidate",
    )
    production_dates = set(production_scorecard_profiles["date"].dropna().unique())
    candidate_dates = set(candidate_scorecard_profiles["date"].dropna().unique())
    common_dates = production_dates & candidate_dates
    if not common_dates:
        raise ValueError(
            "no common scorecard-eligible dates remain after outcome_status filtering"
        )
    if production_dates != common_dates:
        production_scorecard_profiles = production_scorecard_profiles[
            production_scorecard_profiles["date"].isin(common_dates)
        ].copy()
    if candidate_dates != common_dates:
        candidate_scorecard_profiles = candidate_scorecard_profiles[
            candidate_scorecard_profiles["date"].isin(common_dates)
        ].copy()
    paired_date_filter = {
        "common_scorecard_dates": len(common_dates),
        "production_only_dates_dropped": [
            str(date) for date in sorted(production_dates - common_dates)
        ],
        "candidate_only_dates_dropped": [
            str(date) for date in sorted(candidate_dates - common_dates)
        ],
        "policy": (
            "Candidate and production scorecards are evaluated on the same "
            "post-filter date set."
        ),
    }

    production_scorecard = compute_full_scorecard(
        production_scorecard_profiles,
        mc_trials=mc_trials,
        season_length=season_length,
    )
    candidate_scorecard = compute_full_scorecard(
        candidate_scorecard_profiles,
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
        "outcome_status_filter": {
            "production": production_filter_report,
            "candidate": candidate_filter_report,
            "paired_date_filter": paired_date_filter,
            "policy": (
                "Rows with outcome_status other than resolved are excluded from "
                "scorecard denominators. Dates without resolved rank 1 and rank "
                "2 rows are excluded from streak scorecards."
            ),
        },
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
