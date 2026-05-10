#!/usr/bin/env python3
"""Compare captured leaderboard picks to backfilled ranked model surfaces.

This is a research-only historical audit. It intentionally treats ranked
surfaces as post-hoc model outputs unless their provenance proves otherwise.
It does not write production state, scheduler state, or dashboard inputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.leaderboard_candidate_join_audit import (
    PICK_NUMBERS,
    VALID_RESULTS,
    contiguous_block_bootstrap_ci,
    load_cohort_users,
    load_snapshot_cohort,
    load_user_picks,
    normalize_date_key,
    parse_cutoff,
    pick_feature_tables,
    utc_now_iso,
    write_json,
)


SCHEMA_VERSION = "leaderboard_backfilled_model_audit_v1"
SURFACE_REQUIRED_COLUMNS = {"date", "rank", "batter_id", "p_game_hit", "actual_hit"}
DEFAULT_TOP_K = (1, 2, 5, 10)


def parse_surface_specs(raw_specs: list[str]) -> dict[str, Path]:
    if not raw_specs:
        raise ValueError("at least one --surface NAME=PATH is required")
    specs: dict[str, Path] = {}
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"surface spec must be NAME=PATH, got {raw!r}")
        name, path = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"surface name is empty in {raw!r}")
        if name in specs:
            raise ValueError(f"duplicate surface name: {name}")
        specs[name] = Path(path)
    return specs


def parse_dates(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {normalize_date_key(part.strip()) for part in raw.split(",") if part.strip()}


def parse_top_k(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return DEFAULT_TOP_K
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--top-k must contain positive integers")
    return tuple(values)


def _filter_date_strings(
    frame: pd.DataFrame,
    *,
    date_col: str,
    dates: set[str] | None,
    min_date: str | None,
    max_date: str | None,
) -> pd.DataFrame:
    out = frame.copy()
    if dates is not None:
        out = out[out[date_col].isin(dates)].copy()
    if min_date is not None:
        min_key = normalize_date_key(min_date)
        out = out[out[date_col] >= min_key].copy()
    if max_date is not None:
        max_key = normalize_date_key(max_date)
        out = out[out[date_col] <= max_key].copy()
    return out


def _read_surface(path: Path, *, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"surface {name!r} not found: {path}")
    frame = pd.read_parquet(path)
    missing = sorted(SURFACE_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing ranked-surface columns: {missing}")
    frame = frame.copy()
    frame["surface"] = name
    frame["surface_path"] = str(path)
    frame["date"] = frame["date"].map(normalize_date_key)
    frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
    frame["batter_id"] = pd.to_numeric(frame["batter_id"], errors="raise").astype("Int64")
    frame["p_game_hit"] = pd.to_numeric(frame["p_game_hit"], errors="coerce")
    frame["actual_hit"] = pd.to_numeric(frame["actual_hit"], errors="coerce")
    for col in ("game_pk", "n_pas", "batter_name"):
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame


def load_ranked_surfaces(
    surface_specs: dict[str, Path],
    *,
    dates: set[str] | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ranked_parts: list[pd.DataFrame] = []
    joinable_parts: list[pd.DataFrame] = []
    inventory: dict[str, Any] = {}
    for name, path in surface_specs.items():
        frame = _read_surface(path, name=name)
        frame = _filter_date_strings(
            frame,
            date_col="date",
            dates=dates,
            min_date=min_date,
            max_date=max_date,
        )
        duplicate_rank = frame.duplicated(["date", "rank"], keep=False)
        if duplicate_rank.any():
            examples = (
                frame.loc[duplicate_rank, ["date", "rank"]]
                .drop_duplicates()
                .head(5)
                .to_dict("records")
            )
            raise ValueError(
                f"{path} has duplicate date/rank rows; examples={examples}"
            )

        duplicate_batter = frame.duplicated(["date", "batter_id"], keep=False)
        joinable = (
            frame.sort_values(
                ["date", "batter_id", "rank", "p_game_hit"],
                ascending=[True, True, True, False],
            )
            .drop_duplicates(["date", "batter_id"], keep="first")
            .copy()
        )

        ranked_parts.append(frame)
        joinable_parts.append(joinable)
        inventory[name] = {
            "path": str(path),
            "rows": int(len(frame)),
            "joinable_rows": int(len(joinable)),
            "dates": int(frame["date"].nunique()) if not frame.empty else 0,
            "date_min": frame["date"].min() if not frame.empty else None,
            "date_max": frame["date"].max() if not frame.empty else None,
            "max_rank": int(frame["rank"].max()) if not frame.empty else None,
            "actual_hit_null_rows": int(frame["actual_hit"].isna().sum()),
            "date_batter_duplicate_rows_collapsed_for_leaderboard_join": int(
                duplicate_batter.sum()
            ),
        }

    ranked = pd.concat(ranked_parts, ignore_index=True) if ranked_parts else pd.DataFrame()
    joinable = pd.concat(joinable_parts, ignore_index=True) if joinable_parts else pd.DataFrame()
    return ranked, joinable, inventory


def _filter_picks(
    picks: pd.DataFrame,
    *,
    dates: set[str] | None,
    min_date: str | None,
    max_date: str | None,
) -> pd.DataFrame:
    if picks.empty:
        return picks.copy()
    return _filter_date_strings(
        picks,
        date_col="pick_date",
        dates=dates,
        min_date=min_date,
        max_date=max_date,
    )


def _cohort_picks(
    picks: pd.DataFrame,
    *,
    users: set[str] | None,
) -> pd.DataFrame:
    if users is None:
        return picks.copy()
    return picks[picks["username"].isin(users)].copy()


def build_individual_pick_join(
    picks: pd.DataFrame,
    *,
    users: set[str] | None,
    cohort_name: str,
    surface_joinable: pd.DataFrame,
    surface_names: list[str],
    top_k: tuple[int, ...],
) -> pd.DataFrame:
    base = _cohort_picks(picks, users=users)
    columns = [
        "cohort",
        "surface",
        "username",
        "date",
        "captured_at",
        "pick_number",
        "batter_id",
        "batter_name",
        "leaderboard_result",
        "leaderboard_hit",
        "model_rank",
        "model_p_game_hit",
        "model_actual_hit",
        "model_n_pas",
        "model_game_pk",
        "surface_date_available",
        "in_model_surface",
        *[f"model_top_{k}" for k in top_k],
    ]
    if base.empty:
        return pd.DataFrame(columns=columns)

    base = base.copy()
    base["cohort"] = cohort_name
    base["date"] = base["pick_date"].map(normalize_date_key)
    base["batter_id"] = pd.to_numeric(base["batter_id"], errors="coerce").astype("Int64")
    base = base.rename(columns={"result": "leaderboard_result"})

    names = pd.DataFrame({"surface": surface_names})
    base["_join_key"] = 1
    names["_join_key"] = 1
    expanded = base.merge(names, on="_join_key", how="inner").drop(columns=["_join_key"])
    surface_dates = surface_joinable[["surface", "date"]].drop_duplicates().copy()
    surface_dates["surface_date_available"] = True
    expanded = expanded.merge(surface_dates, on=["surface", "date"], how="left")
    expanded["surface_date_available"] = (
        expanded["surface_date_available"].fillna(False).astype(bool)
    )

    keep = [
        "surface",
        "date",
        "batter_id",
        "rank",
        "p_game_hit",
        "actual_hit",
        "n_pas",
        "game_pk",
    ]
    joined = expanded.merge(
        surface_joinable[keep],
        on=["surface", "date", "batter_id"],
        how="left",
    )
    joined["leaderboard_hit"] = pd.Series(pd.NA, index=joined.index, dtype="Int64")
    joined.loc[joined["leaderboard_result"] == "hit", "leaderboard_hit"] = 1
    joined.loc[joined["leaderboard_result"] == "not_hit", "leaderboard_hit"] = 0
    joined["in_model_surface"] = joined["rank"].notna()
    for k in top_k:
        joined[f"model_top_{k}"] = joined["rank"].le(k).fillna(False).astype(bool)

    joined = joined.rename(
        columns={
            "rank": "model_rank",
            "p_game_hit": "model_p_game_hit",
            "actual_hit": "model_actual_hit",
            "n_pas": "model_n_pas",
            "game_pk": "model_game_pk",
        }
    )
    return joined[columns]


def _num_or_none(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _summarize_individual_frame(
    frame: pd.DataFrame,
    *,
    top_k: tuple[int, ...],
    include_by_slot: bool,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "n_picks": 0,
            "n_resolved": 0,
            "leaderboard_hit_rate": None,
            "n_picks_on_surface_dates": 0,
            "surface_date_coverage_rate": None,
            "in_model_surface_rate": None,
            "in_model_surface_rate_on_surface_dates": None,
            "mean_model_rank": None,
            "median_model_rank": None,
            "mean_model_p_game_hit": None,
            "top_k_share": {str(k): None for k in top_k},
            "top_k_share_on_surface_dates": {str(k): None for k in top_k},
        }
    resolved = frame[frame["leaderboard_result"].isin(VALID_RESULTS)].copy()
    in_surface = frame[frame["in_model_surface"]].copy()
    on_surface_dates = frame[frame["surface_date_available"]].copy()
    out = {
        "n_picks": int(len(frame)),
        "n_resolved": int(len(resolved)),
        "n_users": int(frame["username"].nunique()),
        "n_dates": int(frame["date"].nunique()),
        "leaderboard_hit_rate": (
            float(resolved["leaderboard_hit"].astype(int).mean()) if len(resolved) else None
        ),
        "n_picks_on_surface_dates": int(len(on_surface_dates)),
        "surface_date_coverage_rate": float(frame["surface_date_available"].mean()),
        "in_model_surface_rate": float(frame["in_model_surface"].mean()),
        "in_model_surface_rate_on_surface_dates": (
            float(on_surface_dates["in_model_surface"].mean()) if len(on_surface_dates) else None
        ),
        "mean_model_rank": _num_or_none(in_surface["model_rank"].mean()),
        "median_model_rank": _num_or_none(in_surface["model_rank"].median()),
        "mean_model_p_game_hit": _num_or_none(in_surface["model_p_game_hit"].mean()),
        "top_k_share": {
            str(k): float(frame[f"model_top_{k}"].mean()) for k in top_k
        },
        "top_k_share_on_surface_dates": {
            str(k): (
                float(on_surface_dates[f"model_top_{k}"].mean())
                if len(on_surface_dates)
                else None
            )
            for k in top_k
        },
    }
    if include_by_slot:
        out["by_pick_number"] = {
            str(int(slot)): _summarize_individual_frame(
                group,
                top_k=top_k,
                include_by_slot=False,
            )
            for slot, group in frame.groupby("pick_number")
        }
    return out


def summarize_individual_pick_overlap(
    joined: pd.DataFrame,
    *,
    top_k: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if joined.empty:
        return result
    for (cohort, surface), group in joined.groupby(["cohort", "surface"]):
        result.setdefault(str(cohort), {})[str(surface)] = _summarize_individual_frame(
            group,
            top_k=top_k,
            include_by_slot=True,
        )
    return result


def build_consensus_surface_units(
    consensus: pd.DataFrame,
    *,
    cohort_name: str,
    ranked_surfaces: pd.DataFrame,
    surface_joinable: pd.DataFrame,
    surface_names: list[str],
    top_k: tuple[int, ...],
) -> pd.DataFrame:
    columns = [
        "cohort",
        "surface",
        "date",
        "pick_number",
        "n_public_users",
        "consensus_batter_id",
        "consensus_batter_name",
        "consensus_pick_count",
        "consensus_pick_share",
        "consensus_result",
        "model_batter_id",
        "model_game_pk",
        "model_p_game_hit",
        "model_actual_hit",
        "model_n_pas",
        "consensus_model_rank",
        "consensus_model_p_game_hit",
        "consensus_model_actual_hit",
        "consensus_model_game_pk",
        "surface_date_available",
        "consensus_in_model_surface",
        *[f"consensus_model_top_{k}" for k in top_k],
        "agree",
        "consensus_hit",
        "model_hit",
        "delta",
    ]
    if consensus.empty:
        return pd.DataFrame(columns=columns)
    cohort_consensus = consensus[consensus["cohort"] == cohort_name].copy()
    if cohort_consensus.empty:
        return pd.DataFrame(columns=columns)

    names = pd.DataFrame({"surface": surface_names})
    cohort_consensus["_join_key"] = 1
    names["_join_key"] = 1
    units = cohort_consensus.merge(names, on="_join_key", how="inner").drop(columns=["_join_key"])
    surface_dates = ranked_surfaces[["surface", "date"]].drop_duplicates().copy()
    surface_dates["surface_date_available"] = True
    units = units.merge(surface_dates, on=["surface", "date"], how="left")
    units["surface_date_available"] = (
        units["surface_date_available"].fillna(False).astype(bool)
    )

    slot_rows = ranked_surfaces[ranked_surfaces["rank"].isin(PICK_NUMBERS)].copy()
    slot_rows["pick_number"] = slot_rows["rank"].astype(int)
    slot_rows = slot_rows.rename(
        columns={
            "batter_id": "model_batter_id",
            "game_pk": "model_game_pk",
            "p_game_hit": "model_p_game_hit",
            "actual_hit": "model_actual_hit",
            "n_pas": "model_n_pas",
        }
    )
    units = units.merge(
        slot_rows[
            [
                "surface",
                "date",
                "pick_number",
                "model_batter_id",
                "model_game_pk",
                "model_p_game_hit",
                "model_actual_hit",
                "model_n_pas",
            ]
        ],
        on=["surface", "date", "pick_number"],
        how="left",
    )

    consensus_rank = surface_joinable.rename(
        columns={
            "batter_id": "consensus_batter_id",
            "rank": "consensus_model_rank",
            "p_game_hit": "consensus_model_p_game_hit",
            "actual_hit": "consensus_model_actual_hit",
            "game_pk": "consensus_model_game_pk",
        }
    )
    units = units.merge(
        consensus_rank[
            [
                "surface",
                "date",
                "consensus_batter_id",
                "consensus_model_rank",
                "consensus_model_p_game_hit",
                "consensus_model_actual_hit",
                "consensus_model_game_pk",
            ]
        ],
        on=["surface", "date", "consensus_batter_id"],
        how="left",
    )

    units["consensus_in_model_surface"] = units["consensus_model_rank"].notna()
    for k in top_k:
        units[f"consensus_model_top_{k}"] = (
            units["consensus_model_rank"].le(k).fillna(False).astype(bool)
        )
    units["agree"] = (
        units["model_batter_id"].notna()
        & units["consensus_batter_id"].notna()
        & (
            units["model_batter_id"].astype("Int64")
            == units["consensus_batter_id"].astype("Int64")
        )
    )
    units["consensus_hit"] = pd.Series(pd.NA, index=units.index, dtype="Int64")
    units.loc[units["consensus_result"] == "hit", "consensus_hit"] = 1
    units.loc[units["consensus_result"] == "not_hit", "consensus_hit"] = 0
    units["model_hit"] = pd.Series(pd.NA, index=units.index, dtype="Int64")
    valid_model_hit = units["model_actual_hit"].notna()
    units.loc[valid_model_hit, "model_hit"] = (
        units.loc[valid_model_hit, "model_actual_hit"].astype(bool).astype(int)
    )
    units["delta"] = units["consensus_hit"] - units["model_hit"]
    return units[columns]


def _summarize_consensus_rank_frame(
    frame: pd.DataFrame,
    *,
    top_k: tuple[int, ...],
    include_by_slot: bool,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "n_units": 0,
            "n_units_on_surface_dates": 0,
            "surface_date_coverage_rate": None,
            "consensus_in_surface_rate": None,
            "consensus_in_surface_rate_on_surface_dates": None,
            "mean_consensus_model_rank": None,
            "median_consensus_model_rank": None,
            "mean_consensus_model_p_game_hit": None,
            "top_k_coverage": {str(k): None for k in top_k},
            "top_k_coverage_on_surface_dates": {str(k): None for k in top_k},
        }
    in_surface = frame[frame["consensus_in_model_surface"]].copy()
    on_surface_dates = frame[frame["surface_date_available"]].copy()
    out = {
        "n_units": int(len(frame)),
        "n_units_on_surface_dates": int(len(on_surface_dates)),
        "n_resolved_consensus_units": int(frame["consensus_result"].isin(VALID_RESULTS).sum()),
        "surface_date_coverage_rate": float(frame["surface_date_available"].mean()),
        "consensus_in_surface_rate": float(frame["consensus_in_model_surface"].mean()),
        "consensus_in_surface_rate_on_surface_dates": (
            float(on_surface_dates["consensus_in_model_surface"].mean())
            if len(on_surface_dates)
            else None
        ),
        "mean_consensus_model_rank": _num_or_none(in_surface["consensus_model_rank"].mean()),
        "median_consensus_model_rank": _num_or_none(in_surface["consensus_model_rank"].median()),
        "mean_consensus_model_p_game_hit": _num_or_none(
            in_surface["consensus_model_p_game_hit"].mean()
        ),
        "top_k_coverage": {
            str(k): float(frame[f"consensus_model_top_{k}"].mean()) for k in top_k
        },
        "top_k_coverage_on_surface_dates": {
            str(k): (
                float(on_surface_dates[f"consensus_model_top_{k}"].mean())
                if len(on_surface_dates)
                else None
            )
            for k in top_k
        },
    }
    if include_by_slot:
        out["by_pick_number"] = {
            str(int(slot)): _summarize_consensus_rank_frame(
                group,
                top_k=top_k,
                include_by_slot=False,
            )
            for slot, group in frame.groupby("pick_number")
        }
    return out


def summarize_consensus_rank_coverage(
    units: pd.DataFrame,
    *,
    top_k: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if units.empty:
        return result
    for (cohort, surface), group in units.groupby(["cohort", "surface"]):
        result.setdefault(str(cohort), {})[str(surface)] = _summarize_consensus_rank_frame(
            group,
            top_k=top_k,
            include_by_slot=True,
        )
    return result


def _summarize_consensus_vs_model_frame(
    units: pd.DataFrame,
    *,
    expected_block_length: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    valid = units[
        units["consensus_result"].isin(VALID_RESULTS)
        & units["model_hit"].notna()
        & units["model_batter_id"].notna()
    ].copy()
    if valid.empty:
        return {
            "n_units": 0,
            "n_disagreements": 0,
            "agreement_rate": None,
            "model_hit_rate": None,
            "consensus_hit_rate": None,
            "mean_delta": None,
            "disagreement_model_hit_rate": None,
            "disagreement_consensus_hit_rate": None,
            "disagreement_mean_delta": None,
            "bootstrap": None,
            "disagreement_bootstrap": None,
        }
    valid["delta"] = valid["delta"].astype(float)
    valid["model_hit"] = valid["model_hit"].astype(int)
    valid["consensus_hit"] = valid["consensus_hit"].astype(int)
    disagreements = valid[~valid["agree"]].copy()
    return {
        "n_units": int(len(valid)),
        "n_disagreements": int(len(disagreements)),
        "agreement_rate": float(valid["agree"].mean()),
        "model_hit_rate": float(valid["model_hit"].mean()),
        "consensus_hit_rate": float(valid["consensus_hit"].mean()),
        "mean_delta": float(valid["delta"].mean()),
        "disagreement_model_hit_rate": (
            float(disagreements["model_hit"].mean()) if len(disagreements) else None
        ),
        "disagreement_consensus_hit_rate": (
            float(disagreements["consensus_hit"].mean()) if len(disagreements) else None
        ),
        "disagreement_mean_delta": (
            float(disagreements["delta"].mean()) if len(disagreements) else None
        ),
        "bootstrap": contiguous_block_bootstrap_ci(
            valid,
            expected_block_length=expected_block_length,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "disagreement_bootstrap": contiguous_block_bootstrap_ci(
            disagreements,
            expected_block_length=expected_block_length,
            n_bootstrap=n_bootstrap,
            seed=seed + 1,
        ) if len(disagreements) else None,
    }


def summarize_consensus_vs_model(
    units: pd.DataFrame,
    *,
    expected_block_length: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if units.empty:
        return result
    offset = 0
    for (cohort, surface), group in units.groupby(["cohort", "surface"]):
        result.setdefault(str(cohort), {})[str(surface)] = _summarize_consensus_vs_model_frame(
            group,
            expected_block_length=expected_block_length,
            n_bootstrap=n_bootstrap,
            seed=seed + offset,
        )
        offset += 17
    return result


def _summarize_rank_popularity_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"n_model_rows": 0, "by_rank": {}}
    by_rank: dict[str, Any] = {}
    for rank, group in frame.groupby("rank"):
        by_rank[str(int(rank))] = {
            "n_model_rows": int(len(group)),
            "mean_any_public_pick_share": _num_or_none(group["any_public_pick_share"].mean()),
            "median_any_public_pick_share": _num_or_none(
                group["any_public_pick_share"].median()
            ),
            "share_public_consensus_any_slot": _num_or_none(
                group["is_public_consensus_any_slot"].mean()
            ),
            "mean_slot1_public_pick_share": _num_or_none(
                group["slot1_public_pick_share"].mean()
            ),
            "mean_slot2_public_pick_share": _num_or_none(
                group["slot2_public_pick_share"].mean()
            ),
        }
    return {
        "n_model_rows": int(len(frame)),
        "by_rank": by_rank,
    }


def build_model_rank_popularity(
    ranked_surfaces: pd.DataFrame,
    *,
    features: pd.DataFrame,
    cohort_name: str,
    max_rank: int,
) -> pd.DataFrame:
    columns = [
        "cohort",
        "surface",
        "date",
        "rank",
        "batter_id",
        "p_game_hit",
        "slot1_public_pick_share",
        "slot2_public_pick_share",
        "any_public_pick_share",
        "is_public_consensus_any_slot",
    ]
    if ranked_surfaces.empty:
        return pd.DataFrame(columns=columns)
    base = ranked_surfaces[ranked_surfaces["rank"] <= max_rank].copy()
    base["cohort"] = cohort_name
    for pick_number in PICK_NUMBERS:
        prefix = f"slot{pick_number}_"
        if features.empty:
            base[prefix + "public_pick_share"] = pd.NA
            base[prefix + "is_public_consensus"] = pd.NA
            continue
        slot_features = features[features["pick_number"] == pick_number][
            ["date", "batter_id", "public_pick_share", "is_public_consensus"]
        ].copy()
        slot_features = slot_features.rename(
            columns={
                "public_pick_share": prefix + "public_pick_share",
                "is_public_consensus": prefix + "is_public_consensus",
            }
        )
        base = base.merge(slot_features, on=["date", "batter_id"], how="left")
    share_cols = [f"slot{pick_number}_public_pick_share" for pick_number in PICK_NUMBERS]
    consensus_cols = [
        f"slot{pick_number}_is_public_consensus" for pick_number in PICK_NUMBERS
    ]
    base["any_public_pick_share"] = base[share_cols].max(axis=1)
    base["is_public_consensus_any_slot"] = (
        base[consensus_cols].fillna(0).astype(float).max(axis=1)
    )
    return base[columns]


def summarize_model_rank_popularity(popularity: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if popularity.empty:
        return result
    for (cohort, surface), group in popularity.groupby(["cohort", "surface"]):
        result.setdefault(str(cohort), {})[str(surface)] = _summarize_rank_popularity_frame(
            group
        )
    return result


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def build_audit(
    *,
    leaderboard_dir: Path,
    surface_specs: dict[str, Path],
    output_path: Path,
    joined_output_path: Path | None,
    consensus_units_output_path: Path | None,
    cohort_as_of_iso: str | None,
    cohort_users_json: Path | None,
    dates: set[str] | None,
    min_date: str | None,
    max_date: str | None,
    top_k: tuple[int, ...],
    n_bootstrap: int,
    expected_block_length: int,
    seed: int,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or utc_now_iso()
    cohort_as_of = parse_cutoff(cohort_as_of_iso) if cohort_as_of_iso else None
    ranked_surfaces, surface_joinable, surface_inventory = load_ranked_surfaces(
        surface_specs,
        dates=dates,
        min_date=min_date,
        max_date=max_date,
    )
    surface_names = list(surface_specs)

    picks, pick_inventory = load_user_picks(
        leaderboard_dir,
        decision_cutoff=None,
    )
    picks = _filter_picks(
        picks,
        dates=dates,
        min_date=min_date,
        max_date=max_date,
    )
    all_users = set(picks["username"].dropna().astype(str)) if not picks.empty else set()
    if cohort_users_json is not None:
        fixed_users, fixed_cohort_meta = load_cohort_users(cohort_users_json)
    else:
        fixed_users, fixed_cohort_meta = load_snapshot_cohort(
            leaderboard_dir,
            cohort_as_of=cohort_as_of,
        )
    fixed_cohort_meta["cohort_as_of_iso"] = cohort_as_of_iso

    all_features, all_consensus, all_meta = pick_feature_tables(
        picks,
        users=None,
        cohort_name="all_tracked",
    )
    fixed_features, fixed_consensus, fixed_meta = pick_feature_tables(
        picks,
        users=fixed_users,
        cohort_name="fixed_cohort",
    )
    consensus = pd.concat(
        [frame for frame in (all_consensus, fixed_consensus) if not frame.empty],
        ignore_index=True,
    ) if (not all_consensus.empty or not fixed_consensus.empty) else pd.DataFrame()

    individual_join = pd.concat(
        [
            build_individual_pick_join(
                picks,
                users=None,
                cohort_name="all_tracked",
                surface_joinable=surface_joinable,
                surface_names=surface_names,
                top_k=top_k,
            ),
            build_individual_pick_join(
                picks,
                users=fixed_users,
                cohort_name="fixed_cohort",
                surface_joinable=surface_joinable,
                surface_names=surface_names,
                top_k=top_k,
            ),
        ],
        ignore_index=True,
    )

    consensus_units = pd.concat(
        [
            build_consensus_surface_units(
                consensus,
                cohort_name="all_tracked",
                ranked_surfaces=ranked_surfaces,
                surface_joinable=surface_joinable,
                surface_names=surface_names,
                top_k=top_k,
            ),
            build_consensus_surface_units(
                consensus,
                cohort_name="fixed_cohort",
                ranked_surfaces=ranked_surfaces,
                surface_joinable=surface_joinable,
                surface_names=surface_names,
                top_k=top_k,
            ),
        ],
        ignore_index=True,
    )

    popularity = pd.concat(
        [
            build_model_rank_popularity(
                ranked_surfaces,
                features=all_features,
                cohort_name="all_tracked",
                max_rank=max(top_k),
            ),
            build_model_rank_popularity(
                ranked_surfaces,
                features=fixed_features,
                cohort_name="fixed_cohort",
                max_rank=max(top_k),
            ),
        ],
        ignore_index=True,
    )

    if joined_output_path is None:
        joined_output_path = output_path.with_suffix(".individual_picks.parquet")
    if consensus_units_output_path is None:
        consensus_units_output_path = output_path.with_suffix(".consensus_units.parquet")
    _write_parquet(individual_join, joined_output_path)
    _write_parquet(consensus_units, consensus_units_output_path)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "research_only": True,
        "production_deploy_claim": False,
        "no_policy_edit_supported": True,
        "leaderboard_dir": str(leaderboard_dir),
        "surface_specs": {name: str(path) for name, path in surface_specs.items()},
        "joined_individual_picks_path": str(joined_output_path),
        "joined_consensus_units_path": str(consensus_units_output_path),
        "filters": {
            "dates": sorted(dates) if dates else None,
            "min_date": normalize_date_key(min_date) if min_date else None,
            "max_date": normalize_date_key(max_date) if max_date else None,
            "top_k": list(top_k),
        },
        "pre_registered_primary_comparison": {
            "registered_before_numeric_results_in_this_artifact": True,
            "primary_cohort": "fixed_cohort",
            "primary_unit": "resolved (pick_date, pick_number) date-slot",
            "primary_outcome": (
                "disagreement-conditional mean(consensus_hit - model_rank_slot_hit) "
                "for each named surface"
            ),
            "primary_descriptive": (
                "fixed-cohort consensus top-k coverage and individual tracked-pick "
                "top-k share for k in filters.top_k"
            ),
            "comparison_scope": (
                "Backfilled surfaces are current/post-hoc model surfaces unless "
                "their own provenance proves a frozen historical information set."
            ),
            "first_forward_eval_gate": (
                "Do not support production policy edits until a future fixed-cohort "
                "protocol has at least 30 resolved disagreement date-slot units, or "
                "a separately validated mechanism passes its own gate."
            ),
        },
        "inventory": {
            "leaderboard": {
                **pick_inventory,
                "dedup_rows_after_date_filter": int(len(picks)),
                "users_after_date_filter": int(len(all_users)),
            },
            "surfaces": surface_inventory,
            "ranked_surface_rows": int(len(ranked_surfaces)),
            "joinable_surface_rows": int(len(surface_joinable)),
            "individual_join_rows": int(len(individual_join)),
            "consensus_unit_rows": int(len(consensus_units)),
        },
        "cohorts": {
            "all_tracked": {
                "source": "user_picks",
                "n_users": int(len(all_users)),
                **all_meta,
            },
            "fixed_cohort": {
                **fixed_cohort_meta,
                **fixed_meta,
            },
        },
        "comparison": {
            "individual_pick_overlap": summarize_individual_pick_overlap(
                individual_join,
                top_k=top_k,
            ),
            "consensus_rank_coverage": summarize_consensus_rank_coverage(
                consensus_units,
                top_k=top_k,
            ),
            "consensus_vs_model": summarize_consensus_vs_model(
                consensus_units,
                expected_block_length=expected_block_length,
                n_bootstrap=n_bootstrap,
                seed=seed,
            ),
            "model_rank_public_popularity": summarize_model_rank_popularity(popularity),
        },
        "methodology_constraints": {
            "post_hoc_observation_not_pre_lock_signal": True,
            "leaderboard_capture_time_is_scrape_time_not_user_decision_time": True,
            "backfilled_surface_not_historical_production_truth_by_default": True,
            "production_model_evolution_caveat": (
                "A 2026 backtest generated with current code and parameters is a "
                "current-model backfill. It should not be read as the exact policy "
                "that shipped on earlier dates."
            ),
            "candidate_training_cutoff_must_be_verified_per_surface": True,
            "surface_info_set_verdict_must_be_supplied_by_provenance": (
                "This script validates surface shape and joins. It does not prove "
                "the surface was generated with a leak-free training cutoff."
            ),
            "leaderboard_join_key": "date + batter_id",
            "leaderboard_join_key_caveat": (
                "Leaderboard pick rows do not carry game_pk, so same-date "
                "doubleheader context can only be collapsed to a batter-date "
                "best-ranked model row."
            ),
            "subgroup_reads_are_diagnostic_without_fdr_or_preregistration": True,
            "day_block_bootstrap_expected_block_length": int(expected_block_length),
            "day_block_bootstrap_seed": int(seed),
        },
    }
    write_json(report, output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard-dir", default="data/leaderboard")
    parser.add_argument(
        "--surface",
        action="append",
        default=[],
        help="Repeatable NAME=PATH ranked surface, e.g. production=data/simulation/backtest_2026.parquet",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--joined-output", default=None)
    parser.add_argument("--consensus-units-output", default=None)
    parser.add_argument("--cohort-as-of-iso", default=None)
    parser.add_argument("--cohort-users-json", default=None)
    parser.add_argument("--dates", default=None, help="Comma-separated YYYY-MM-DD filter")
    parser.add_argument("--min-date", default=None)
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--top-k", default="1,2,5,10")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--expected-block-length", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    report = build_audit(
        leaderboard_dir=Path(args.leaderboard_dir),
        surface_specs=parse_surface_specs(args.surface),
        output_path=Path(args.output),
        joined_output_path=Path(args.joined_output) if args.joined_output else None,
        consensus_units_output_path=(
            Path(args.consensus_units_output) if args.consensus_units_output else None
        ),
        cohort_as_of_iso=args.cohort_as_of_iso,
        cohort_users_json=Path(args.cohort_users_json) if args.cohort_users_json else None,
        dates=parse_dates(args.dates),
        min_date=args.min_date,
        max_date=args.max_date,
        top_k=parse_top_k(args.top_k),
        n_bootstrap=args.n_bootstrap,
        expected_block_length=args.expected_block_length,
        seed=args.seed,
    )
    print(f"wrote {args.output}")
    print(f"joined individual picks: {report['joined_individual_picks_path']}")
    print(f"joined consensus units: {report['joined_consensus_units_path']}")
    fixed = report["comparison"]["consensus_vs_model"].get("fixed_cohort", {})
    for surface, stats in fixed.items():
        print(
            "fixed_cohort "
            f"{surface}: n={stats['n_units']} "
            f"disagreements={stats['n_disagreements']} "
            f"disagreement_delta={stats['disagreement_mean_delta']}"
        )


if __name__ == "__main__":
    main()
