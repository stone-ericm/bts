"""Candidate-vs-production ranked-slate artifact helpers.

This module freezes the v1 artifact surface for #16 historical screens:
paired production and candidate top-N ranked slate parquets plus a manifest
that records provenance and launch posture. The artifact is intentionally
research-only; it does not write production picks or deployment assets.
"""

from __future__ import annotations

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
