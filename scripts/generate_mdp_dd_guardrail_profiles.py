#!/usr/bin/env python3
"""Generate the primary profile surface for the MDP DD guardrail audit.

The output is a ranked daily profile surface with game_pk present at generation
time, not a post-hoc join. The companion manifest freezes provenance before any
guardrail result sweep is interpreted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from bts.features.compute import compute_all_features
from bts.simulate.backtest_blend import (
    GAME_PROBABILITY_ACTUAL_PA,
    GAME_PROBABILITY_MODES,
    blend_walk_forward,
)


DEFAULT_SEASONS = (2021, 2022, 2023, 2024, 2025)
PROFILE_REQUIRED_COLUMNS = (
    "date",
    "season",
    "rank",
    "batter_id",
    "game_pk",
    "p_game_hit",
    "actual_hit",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def _parse_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def input_parquet_manifest(data_dir: Path) -> dict[str, Any]:
    paths = sorted(data_dir.glob("pa_*.parquet"))
    return {
        "data_dir": str(data_dir),
        "count": len(paths),
        "files": {
            path.name: {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        },
    }


def load_feature_data(data_dir: Path) -> pd.DataFrame:
    paths = sorted(data_dir.glob("pa_*.parquet"))
    if not paths:
        raise RuntimeError(f"No pa_*.parquet files found in {data_dir}")
    frames = [pd.read_parquet(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    df = compute_all_features(df)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _season_output_summary(path: Path, required_columns: Sequence[str]) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    date_series = pd.to_datetime(frame["date"]).dt.date if "date" in frame.columns else None
    missing_columns = sorted(set(required_columns).difference(frame.columns))
    null_counts = {
        col: int(frame[col].isna().sum())
        for col in required_columns
        if col in frame.columns
    }
    duplicate_date_batter_rows = None
    duplicate_date_batter_game_rows = None
    ambiguous_date_batter_rows = None
    if {"date", "batter_id"}.issubset(frame.columns):
        duplicate_date_batter_rows = int(frame.duplicated(["date", "batter_id"], keep=False).sum())
        if "game_pk" in frame.columns:
            duplicate_date_batter_game_rows = int(
                frame.duplicated(["date", "batter_id", "game_pk"], keep=False).sum()
            )
            game_counts = (
                frame.groupby(["date", "batter_id"], dropna=False)["game_pk"]
                .nunique(dropna=True)
                .reset_index(name="n_game_pk")
            )
            ambiguous_keys = game_counts[game_counts["n_game_pk"] > 1][["date", "batter_id"]]
            if ambiguous_keys.empty:
                ambiguous_date_batter_rows = 0
            else:
                joined = frame.merge(ambiguous_keys, on=["date", "batter_id"], how="inner")
                ambiguous_date_batter_rows = int(len(joined))

    by_date = {}
    if date_series is not None:
        by_date = {
            str(day): int(count)
            for day, count in frame.assign(_date=date_series).groupby("_date").size().items()
        }

    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "columns": list(frame.columns),
        "missing_columns": missing_columns,
        "row_count": int(len(frame)),
        "date_count": int(date_series.nunique()) if date_series is not None else 0,
        "rows_by_date": by_date,
        "null_counts": null_counts,
        "duplicate_date_batter_rows": duplicate_date_batter_rows,
        "duplicate_date_batter_game_rows": duplicate_date_batter_game_rows,
        "ambiguous_date_batter_rows": ambiguous_date_batter_rows,
    }


def profile_output_manifest(output_dir: Path, seasons: Sequence[int]) -> dict[str, Any]:
    seasons_out = {}
    issues = []
    for season in seasons:
        path = output_dir / f"backtest_{int(season)}.parquet"
        if not path.exists():
            seasons_out[str(season)] = {"path": str(path), "missing": True}
            issues.append(f"missing output for season {season}")
            continue
        summary = _season_output_summary(path, PROFILE_REQUIRED_COLUMNS)
        seasons_out[str(season)] = summary
        if summary["missing_columns"]:
            issues.append(f"season {season} missing columns {summary['missing_columns']}")
        if any(summary["null_counts"].values()):
            issues.append(f"season {season} has nulls in required columns")
    return {
        "output_dir": str(output_dir),
        "required_columns": list(PROFILE_REQUIRED_COLUMNS),
        "valid": not issues,
        "issues": issues,
        "seasons": seasons_out,
    }


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("backtest_*.parquet")) + list(output_dir.glob("manifest*.json"))
    if existing and not force:
        names = ", ".join(path.name for path in existing[:5])
        raise RuntimeError(
            f"{output_dir} already contains guardrail outputs ({names}); use --force to overwrite"
        )


def generate_profiles(
    *,
    data_dir: Path,
    output_dir: Path,
    seasons: Sequence[int] = DEFAULT_SEASONS,
    retrain_every: int = 7,
    top_n: int = 10,
    game_probability_mode: str = GAME_PROBABILITY_ACTUAL_PA,
    force: bool = False,
    command: str | None = None,
) -> dict[str, Any]:
    if game_probability_mode not in GAME_PROBABILITY_MODES:
        raise ValueError(f"unknown game_probability_mode: {game_probability_mode!r}")
    _prepare_output_dir(output_dir, force=force)

    started_at = datetime.now(timezone.utc).isoformat()
    input_manifest = input_parquet_manifest(data_dir)
    df = load_feature_data(data_dir)
    for season in seasons:
        profiles = blend_walk_forward(
            df,
            int(season),
            retrain_every=retrain_every,
            top_n=top_n,
            game_probability_mode=game_probability_mode,
        )
        profiles = profiles.copy()
        profiles["season"] = int(season)
        ordered_columns = [
            col for col in PROFILE_REQUIRED_COLUMNS if col in profiles.columns
        ] + [
            col for col in profiles.columns if col not in PROFILE_REQUIRED_COLUMNS
        ]
        profiles = profiles[ordered_columns]
        path = output_dir / f"backtest_{int(season)}.parquet"
        profiles.to_parquet(path, index=False)
        print(f"saved {path} ({len(profiles)} rows)", file=sys.stderr)

    finished_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "production_deploy_claim": False,
        "writes_policy_artifact": False,
        "artifact_role": "mdp_dd_guardrail_primary_profile_surface",
        "created_at": finished_at,
        "generation_started_at": started_at,
        "generation_finished_at": finished_at,
        "command": command,
        "git_sha": _git_sha(),
        "parameters": {
            "seasons": [int(season) for season in seasons],
            "retrain_every": int(retrain_every),
            "top_n": int(top_n),
            "game_probability_mode": game_probability_mode,
        },
        "environment": {
            "BTS_LGBM_RANDOM_STATE": os.environ.get("BTS_LGBM_RANDOM_STATE", "42(default)"),
            "BTS_LGBM_DETERMINISTIC": os.environ.get("BTS_LGBM_DETERMINISTIC", "0(default)"),
        },
        "inputs": input_manifest,
        "outputs": profile_output_manifest(output_dir, seasons),
        "next_step": (
            "Review/freeze this manifest before running "
            "scripts/evaluate_mdp_dd_guardrail.py on the generated profiles."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}", file=sys.stderr)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seasons", type=_parse_ints, default=list(DEFAULT_SEASONS))
    parser.add_argument("--retrain-every", type=int, default=7)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--game-probability-mode",
        choices=sorted(GAME_PROBABILITY_MODES),
        default=GAME_PROBABILITY_ACTUAL_PA,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    generate_profiles(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seasons=args.seasons,
        retrain_every=args.retrain_every,
        top_n=args.top_n,
        game_probability_mode=args.game_probability_mode,
        force=args.force,
        command=" ".join(sys.argv),
    )


if __name__ == "__main__":
    main()
