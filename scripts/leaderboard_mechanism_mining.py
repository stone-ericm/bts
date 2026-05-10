#!/usr/bin/env python3
"""Pre-registered leaderboard mechanism-mining artifact builder.

This is research-only. It joins captured BTS leaderboard consensus picks to
locked realized production picks, optionally annotates consensus batters with
ranked model-surface placement, and writes validation artifacts only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bts.validate.fdr import bh_qvalues, by_qvalues
from scripts.leaderboard_backfilled_model_audit import (
    load_ranked_surfaces,
    parse_surface_specs,
    parse_top_k,
)
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


SCHEMA_VERSION = "leaderboard_mechanism_mining_v1"
PREREG_PATH = "docs/sota_audit/2026-05-10-leaderboard-mechanism-mining-prereg.md"
DEFAULT_FDR_MIN_N = 15
DEFAULT_MECHANISM_MIN_N = 30
DEFAULT_MECHANISM_MIN_LIFT = 0.05
DEFAULT_Q_THRESHOLD = 0.10
DEFAULT_BOOTSTRAP_SEED = 20260510

DECOMPOSITION_VARIABLES = [
    "cohort",
    "pick_number",
    "consensus_pick_share_bin",
    "production_p_game_hit_bin",
    "agreement_state",
    "production_batter_skill_quartile",
    "production_batter_skill_prior_pa_bin",
    "production_projected_lineup",
    "production_regime",
    "production_is_park_driven",
    "production_is_indoor",
    "production_weather_temp_bin",
    "consensus_model_rank_bin",
    "consensus_model_probability_bin",
]

REALIZED_PRODUCTION_REQUIRED_COLUMNS = {
    "date",
    "slot",
    "batter_id",
    "p_game_hit",
    "actual_hit",
}

PRODUCTION_OPTIONAL_RENAMES = {
    "source_file": "production_source_file",
    "run_time": "production_run_time",
    "batter_name": "production_batter_name",
    "pitcher_id": "production_pitcher_id",
    "game_pk": "production_game_pk",
    "result_status": "production_result_status",
    "projected_lineup": "production_projected_lineup",
    "pick_file_result": "production_pick_file_result",
    "regime": "production_regime",
    "model_cutoff_label": "production_model_cutoff_label",
    "cutoff_iso": "production_cutoff_iso",
    "attribution_source": "production_attribution_source",
    "pick_venue_id": "production_pick_venue_id",
    "pick_roof_type": "production_pick_roof_type",
    "pick_weather_temp": "production_weather_temp",
    "pick_is_indoor": "production_is_indoor",
    "is_park_driven": "production_is_park_driven",
    "batter_skill_prior_pa": "production_batter_skill_prior_pa",
    "batter_skill_prior_hit_rate": "production_batter_skill_prior_hit_rate",
    "batter_skill_quartile": "production_batter_skill_quartile",
}

SLOT_TO_PICK_NUMBER = {
    "primary": 1,
    "pick": 1,
    "double_down": 2,
    "dd": 2,
}


def parse_dates(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {normalize_date_key(part.strip()) for part in raw.split(",") if part.strip()}


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
        out = out[out[date_col] >= normalize_date_key(min_date)].copy()
    if max_date is not None:
        out = out[out[date_col] <= normalize_date_key(max_date)].copy()
    return out


def _actual_hit_to_numeric(series: pd.Series) -> pd.Series:
    mapped = series.map({True: 1, False: 0, "True": 1, "False": 0})
    mapped = mapped.where(mapped.notna(), series)
    return pd.to_numeric(mapped, errors="coerce")


def _hit_from_result(series: pd.Series) -> pd.Series:
    hit = pd.Series(pd.NA, index=series.index, dtype="Int64")
    hit.loc[series == "hit"] = 1
    hit.loc[series == "not_hit"] = 0
    return hit


def _bin_probability(value: Any) -> str:
    if value is None or pd.isna(value):
        return "missing_surface"
    val = float(value)
    if val < 0.68:
        return "<0.68"
    if val < 0.74:
        return "0.68-0.74"
    if val < 0.80:
        return "0.74-0.80"
    return ">=0.80"


def _bin_production_probability(value: Any) -> str:
    result = _bin_probability(value)
    return "missing" if result == "missing_surface" else result


def _bin_consensus_share(value: Any) -> str:
    if value is None or pd.isna(value):
        return "missing"
    val = float(value)
    if val < 0.15:
        return "<0.15"
    if val < 0.25:
        return "0.15-0.25"
    return ">=0.25"


def _bin_prior_pa(value: Any) -> str:
    if value is None or pd.isna(value):
        return "missing"
    val = float(value)
    if val < 100:
        return "<100"
    if val < 300:
        return "100-299"
    if val < 600:
        return "300-599"
    return ">=600"


def _bin_weather_temp(value: Any, is_indoor: Any) -> str:
    if _truthy_or_missing(is_indoor) == "true" or value is None or pd.isna(value):
        return "indoor_or_missing"
    val = float(value)
    if val < 60:
        return "<60"
    if val < 75:
        return "60-74"
    if val < 85:
        return "75-84"
    return ">=85"


def _bin_model_rank(rank: Any, surface_available: Any) -> str:
    if _truthy_or_missing(surface_available) != "true":
        return "missing_surface"
    if rank is None or pd.isna(rank):
        return "off_top10"
    val = int(rank)
    if val == 1:
        return "rank1"
    if val == 2:
        return "rank2"
    if val <= 5:
        return "rank3_5"
    if val <= 10:
        return "rank6_10"
    return "off_top10"


def _truthy_or_missing(value: Any) -> str:
    if value is None or pd.isna(value):
        return "missing"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return "true"
        if lowered in {"false", "0", "no", "n"}:
            return "false"
        return lowered or "missing"
    return "true" if bool(value) else "false"


def _string_or_missing(value: Any) -> str:
    if value is None or pd.isna(value):
        return "missing"
    return str(value)


def _quartile_or_missing(value: Any) -> str:
    if value is None or pd.isna(value):
        return "missing"
    return str(int(value)) if float(value).is_integer() else str(value)


def load_realized_production_picks(
    path: Path,
    *,
    dates: set[str] | None,
    min_date: str | None,
    max_date: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"realized production surface not found: {path}")
    raw = pd.read_parquet(path)
    missing = sorted(REALIZED_PRODUCTION_REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(f"{path} missing realized-production columns: {missing}")

    frame = raw.copy()
    frame["pick_number"] = frame["slot"].astype(str).map(SLOT_TO_PICK_NUMBER)
    frame = frame[frame["pick_number"].isin(PICK_NUMBERS)].copy()
    if frame.empty:
        raise ValueError(f"{path} has no primary/double_down realized-production rows")

    frame["date"] = frame["date"].map(normalize_date_key)
    frame = _filter_date_strings(
        frame,
        date_col="date",
        dates=dates,
        min_date=min_date,
        max_date=max_date,
    )
    duplicate_slot = frame.duplicated(["date", "pick_number"], keep=False)
    if duplicate_slot.any():
        examples = (
            frame.loc[duplicate_slot, ["date", "pick_number"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(f"{path} has duplicate date/pick_number rows: {examples}")

    frame["production_batter_id"] = pd.to_numeric(
        frame["batter_id"], errors="raise"
    ).astype("Int64")
    frame["production_p_game_hit"] = pd.to_numeric(
        frame["p_game_hit"], errors="coerce"
    )
    frame["production_actual_hit"] = _actual_hit_to_numeric(frame["actual_hit"])
    frame["production_hit"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    valid_actual = frame["production_actual_hit"].notna()
    frame.loc[valid_actual, "production_hit"] = (
        frame.loc[valid_actual, "production_actual_hit"].astype(int).astype("Int64")
    )
    frame["production_slot"] = frame["slot"].astype(str)

    for source_col, output_col in PRODUCTION_OPTIONAL_RENAMES.items():
        if source_col in frame.columns:
            frame[output_col] = frame[source_col]
        else:
            frame[output_col] = pd.NA

    status_counts = (
        frame["production_result_status"]
        .astype("string")
        .fillna("missing")
        .value_counts()
        .sort_index()
    )
    inventory = {
        "path": str(path),
        "rows": int(len(frame)),
        "dates": int(frame["date"].nunique()) if not frame.empty else 0,
        "date_min": frame["date"].min() if not frame.empty else None,
        "date_max": frame["date"].max() if not frame.empty else None,
        "actual_hit_null_rows": int(frame["production_actual_hit"].isna().sum()),
        "result_status_counts": {
            str(status): int(count) for status, count in status_counts.items()
        },
    }
    keep = [
        "date",
        "pick_number",
        "production_slot",
        "production_batter_id",
        "production_p_game_hit",
        "production_actual_hit",
        "production_hit",
        *PRODUCTION_OPTIONAL_RENAMES.values(),
    ]
    return frame[keep].copy(), inventory


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


def build_consensus(
    *,
    leaderboard_dir: Path,
    decision_cutoff_iso: str | None,
    cohort_as_of_iso: str | None,
    cohort_users_json: Path | None,
    dates: set[str] | None,
    min_date: str | None,
    max_date: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    decision_cutoff = parse_cutoff(decision_cutoff_iso)
    cohort_as_of = parse_cutoff(cohort_as_of_iso) if cohort_as_of_iso else decision_cutoff
    picks, pick_inventory = load_user_picks(
        leaderboard_dir,
        decision_cutoff=decision_cutoff,
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
    fixed_cohort_meta["cohort_as_of_iso"] = cohort_as_of_iso or decision_cutoff_iso

    _all_features, all_consensus, all_meta = pick_feature_tables(
        picks,
        users=None,
        cohort_name="all_tracked",
    )
    _fixed_features, fixed_consensus, fixed_meta = pick_feature_tables(
        picks,
        users=fixed_users,
        cohort_name="fixed_cohort",
    )
    consensus = pd.concat(
        [frame for frame in (all_consensus, fixed_consensus) if not frame.empty],
        ignore_index=True,
    ) if (not all_consensus.empty or not fixed_consensus.empty) else pd.DataFrame()
    inventory = {
        "leaderboard": {
            **pick_inventory,
            "dedup_rows_after_date_filter": int(len(picks)),
            "all_tracked_users_after_date_filter": int(len(all_users)),
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
    }
    return consensus, inventory


def _surface_names_from_ranked(ranked_surfaces: pd.DataFrame) -> list[str]:
    if ranked_surfaces.empty:
        return ["realized_production_only"]
    return sorted(str(surface) for surface in ranked_surfaces["surface"].dropna().unique())


def _model_surface_dates(ranked_surfaces: pd.DataFrame) -> pd.DataFrame:
    if ranked_surfaces.empty:
        return pd.DataFrame(columns=["surface", "date", "surface_date_available"])
    dates = ranked_surfaces[["surface", "date"]].drop_duplicates().copy()
    dates["surface_date_available"] = True
    return dates


def build_mechanism_units(
    *,
    production: pd.DataFrame,
    consensus: pd.DataFrame,
    ranked_joinable: pd.DataFrame,
    ranked_surfaces: pd.DataFrame,
) -> pd.DataFrame:
    base_columns = [
        "cohort",
        "surface",
        "date",
        "pick_number",
        "production_slot",
        "production_batter_id",
        "production_batter_name",
        "production_p_game_hit",
        "production_hit",
        "production_actual_hit",
        "production_result_status",
        "consensus_batter_id",
        "consensus_batter_name",
        "consensus_pick_count",
        "consensus_pick_share",
        "n_public_users",
        "consensus_result",
        "consensus_hit",
        "consensus_model_rank",
        "consensus_model_p_game_hit",
        "surface_date_available",
        "resolved_for_outcome",
        "agreement_state",
        "delta",
    ]
    columns = base_columns + [
        col for col in DECOMPOSITION_VARIABLES if col not in set(base_columns)
    ]
    if production.empty or consensus.empty:
        return pd.DataFrame(columns=columns)

    base = consensus.merge(production, on=["date", "pick_number"], how="inner")
    if base.empty:
        return pd.DataFrame(columns=columns)
    base["consensus_batter_id"] = pd.to_numeric(
        base["consensus_batter_id"], errors="coerce"
    ).astype("Int64")
    base["consensus_hit"] = _hit_from_result(base["consensus_result"])

    surface_names = pd.DataFrame({"surface": _surface_names_from_ranked(ranked_surfaces)})
    base["_join_key"] = 1
    surface_names["_join_key"] = 1
    expanded = base.merge(surface_names, on="_join_key", how="inner").drop(
        columns=["_join_key"]
    )

    surface_dates = _model_surface_dates(ranked_surfaces)
    if not surface_dates.empty:
        expanded = expanded.merge(surface_dates, on=["surface", "date"], how="left")
        expanded["surface_date_available"] = (
            expanded["surface_date_available"].fillna(False).astype(bool)
        )
    else:
        expanded["surface_date_available"] = False

    if ranked_joinable.empty:
        expanded["consensus_model_rank"] = pd.NA
        expanded["consensus_model_p_game_hit"] = pd.NA
    else:
        keep = ["surface", "date", "batter_id", "rank", "p_game_hit"]
        model_join = ranked_joinable[keep].rename(
            columns={
                "batter_id": "consensus_batter_id",
                "rank": "consensus_model_rank",
                "p_game_hit": "consensus_model_p_game_hit",
            }
        )
        expanded = expanded.merge(
            model_join,
            on=["surface", "date", "consensus_batter_id"],
            how="left",
        )

    expanded["resolved_for_outcome"] = (
        expanded["consensus_result"].isin(VALID_RESULTS)
        & expanded["production_hit"].notna()
    )
    same_batter = (
        expanded["production_batter_id"].astype("Int64")
        == expanded["consensus_batter_id"].astype("Int64")
    )
    expanded["agreement_state"] = np.where(same_batter, "same_batter", "different_batter")
    expanded["delta"] = pd.Series(pd.NA, index=expanded.index, dtype="Float64")
    resolved = expanded["resolved_for_outcome"]
    expanded.loc[resolved, "delta"] = (
        expanded.loc[resolved, "consensus_hit"].astype(int)
        - expanded.loc[resolved, "production_hit"].astype(int)
    )

    expanded["consensus_pick_share_bin"] = expanded["consensus_pick_share"].map(
        _bin_consensus_share
    )
    expanded["production_p_game_hit_bin"] = expanded["production_p_game_hit"].map(
        _bin_production_probability
    )
    expanded["production_batter_skill_quartile"] = expanded[
        "production_batter_skill_quartile"
    ].map(_quartile_or_missing)
    expanded["production_batter_skill_prior_pa_bin"] = expanded[
        "production_batter_skill_prior_pa"
    ].map(_bin_prior_pa)
    expanded["production_projected_lineup"] = expanded[
        "production_projected_lineup"
    ].map(_truthy_or_missing)
    expanded["production_regime"] = expanded["production_regime"].map(_string_or_missing)
    expanded["production_is_park_driven"] = expanded["production_is_park_driven"].map(
        _truthy_or_missing
    )
    expanded["production_is_indoor"] = expanded["production_is_indoor"].map(
        _truthy_or_missing
    )
    expanded["production_weather_temp_bin"] = [
        _bin_weather_temp(temp, indoor)
        for temp, indoor in zip(
            expanded["production_weather_temp"],
            expanded["production_is_indoor"],
            strict=False,
        )
    ]
    expanded["consensus_model_rank_bin"] = [
        _bin_model_rank(rank, available)
        for rank, available in zip(
            expanded["consensus_model_rank"],
            expanded["surface_date_available"],
            strict=False,
        )
    ]
    expanded["consensus_model_probability_bin"] = expanded[
        "consensus_model_p_game_hit"
    ].map(_bin_probability)

    for col in columns:
        if col not in expanded.columns:
            expanded[col] = pd.NA
    return expanded[columns].sort_values(
        ["cohort", "surface", "date", "pick_number"]
    ).reset_index(drop=True)


def _rate(frame: pd.DataFrame, col: str) -> float | None:
    if frame.empty:
        return None
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def summarize_units(
    units: pd.DataFrame,
    *,
    top_k: tuple[int, ...],
    n_bootstrap: int,
    expected_block_length: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if units.empty:
        return result
    for (cohort, surface), group in units.groupby(["cohort", "surface"]):
        resolved = group[group["resolved_for_outcome"]].copy()
        disagreements = resolved[resolved["agreement_state"] == "different_batter"].copy()
        surface_dates = group[group["surface_date_available"]].copy()
        coverage = {}
        coverage_on_surface_dates = {}
        for k in top_k:
            if group["surface_date_available"].any():
                coverage[str(k)] = float(group["consensus_model_rank"].le(k).fillna(False).mean())
                coverage_on_surface_dates[str(k)] = (
                    float(surface_dates["consensus_model_rank"].le(k).fillna(False).mean())
                    if len(surface_dates)
                    else None
                )
            else:
                coverage[str(k)] = None
                coverage_on_surface_dates[str(k)] = None
        result.setdefault(str(cohort), {})[str(surface)] = {
            "n_units_total": int(len(group)),
            "n_resolved_units": int(len(resolved)),
            "n_unresolved_or_void_units": int(len(group) - len(resolved)),
            "n_disagreement_units": int(len(disagreements)),
            "agreement_rate_resolved": (
                float((resolved["agreement_state"] == "same_batter").mean())
                if len(resolved)
                else None
            ),
            "production_hit_rate": _rate(resolved, "production_hit"),
            "consensus_hit_rate": _rate(resolved, "consensus_hit"),
            "mean_delta": _rate(resolved, "delta"),
            "disagreement_production_hit_rate": _rate(disagreements, "production_hit"),
            "disagreement_consensus_hit_rate": _rate(disagreements, "consensus_hit"),
            "disagreement_mean_delta": _rate(disagreements, "delta"),
            "top_k_coverage": coverage,
            "top_k_coverage_on_surface_dates": coverage_on_surface_dates,
            "bootstrap": contiguous_block_bootstrap_ci(
                resolved,
                value_col="delta",
                expected_block_length=expected_block_length,
                n_bootstrap=n_bootstrap,
                seed=seed,
            ),
            "disagreement_bootstrap": (
                contiguous_block_bootstrap_ci(
                    disagreements,
                    value_col="delta",
                    expected_block_length=expected_block_length,
                    n_bootstrap=n_bootstrap,
                    seed=seed + 1,
                )
                if len(disagreements)
                else None
            ),
        }
    return result


def exact_positive_sign_test_pvalue(deltas: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(deltas, errors="coerce").dropna()
    nonzero = values[values != 0]
    n_positive = int((nonzero > 0).sum())
    n_negative = int((nonzero < 0).sum())
    n_nonzero = n_positive + n_negative
    if n_nonzero == 0:
        p_value = 1.0
    else:
        p_value = float(
            stats.binomtest(
                n_positive,
                n_nonzero,
                p=0.5,
                alternative="greater",
            ).pvalue
        )
    return {
        "test": "exact_positive_sign_test",
        "n_nonzero": int(n_nonzero),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "p_one_sided_positive": p_value,
    }


def decomposition_fdr_table(
    units: pd.DataFrame,
    *,
    min_n: int = DEFAULT_FDR_MIN_N,
    q_threshold: float = DEFAULT_Q_THRESHOLD,
    mechanism_min_n: int = DEFAULT_MECHANISM_MIN_N,
    mechanism_min_lift: float = DEFAULT_MECHANISM_MIN_LIFT,
) -> dict[str, Any]:
    if units.empty:
        return {
            "method": "exact_positive_sign_test_plus_BH_BY",
            "decomposition_variables": DECOMPOSITION_VARIABLES,
            "min_resolved_disagreement_units": int(min_n),
            "q_threshold": float(q_threshold),
            "n_testable_cells": 0,
            "rows": [],
            "actionable_mechanism_found": False,
            "falsification_state": "power_limited_no_testable_cells",
        }
    resolved_disagreements = units[
        units["resolved_for_outcome"]
        & (units["agreement_state"] == "different_batter")
    ].copy()
    rows: list[dict[str, Any]] = []
    group_cols = ["surface", *DECOMPOSITION_VARIABLES]
    for key, group in resolved_disagreements.groupby(group_cols, dropna=False):
        if len(group) < min_n:
            continue
        key_dict = dict(zip(group_cols, key, strict=False))
        test = exact_positive_sign_test_pvalue(group["delta"])
        rows.append({
            **key_dict,
            "n_resolved_disagreement_units": int(len(group)),
            "production_hit_rate": _rate(group, "production_hit"),
            "consensus_hit_rate": _rate(group, "consensus_hit"),
            "mean_delta": _rate(group, "delta"),
            **test,
        })

    if not rows:
        return {
            "method": "exact_positive_sign_test_plus_BH_BY",
            "decomposition_variables": DECOMPOSITION_VARIABLES,
            "min_resolved_disagreement_units": int(min_n),
            "q_threshold": float(q_threshold),
            "n_testable_cells": 0,
            "rows": [],
            "actionable_mechanism_found": False,
            "falsification_state": "power_limited_no_testable_cells",
        }

    table = pd.DataFrame(rows)
    pvalues = table["p_one_sided_positive"].to_numpy(dtype=float)
    table["q_BH"] = bh_qvalues(pvalues)
    table["q_BY"] = by_qvalues(pvalues)
    table["survives_BH_0_10"] = table["q_BH"] <= q_threshold
    table["survives_BY_0_10"] = table["q_BY"] <= q_threshold

    fixed = table["cohort"] == "fixed_cohort"
    enough_n = table["n_resolved_disagreement_units"] >= mechanism_min_n
    enough_lift = table["mean_delta"] >= mechanism_min_lift
    table["all_tracked_direction_not_contradicted"] = [
        _all_tracked_direction_not_contradicted(table, row)
        for _, row in table.iterrows()
    ]
    table["mechanism_candidate"] = (
        fixed
        & enough_n
        & enough_lift
        & table["survives_BH_0_10"]
        & table["all_tracked_direction_not_contradicted"].fillna(True)
    )
    out_rows = table.sort_values(
        ["q_BH", "p_one_sided_positive", "surface", "cohort"],
        ascending=[True, True, True, True],
    ).to_dict("records")
    actionable = bool(table["mechanism_candidate"].any())
    return {
        "method": "exact_positive_sign_test_plus_BH_BY",
        "decomposition_variables": DECOMPOSITION_VARIABLES,
        "min_resolved_disagreement_units": int(min_n),
        "q_threshold": float(q_threshold),
        "mechanism_min_resolved_disagreement_units": int(mechanism_min_n),
        "mechanism_min_absolute_lift": float(mechanism_min_lift),
        "n_testable_cells": int(len(table)),
        "n_survive_BH_0_10": int(table["survives_BH_0_10"].sum()),
        "n_survive_BY_0_10": int(table["survives_BY_0_10"].sum()),
        "rows": out_rows,
        "actionable_mechanism_found": actionable,
        "falsification_state": (
            "actionable_mechanism_found"
            if actionable
            else "no_actionable_mechanism_for_current_iteration"
        ),
    }


def _all_tracked_direction_not_contradicted(table: pd.DataFrame, row: pd.Series) -> bool | None:
    if row["cohort"] != "fixed_cohort":
        return None
    mask = table["cohort"] == "all_tracked"
    for col in ["surface", *[c for c in DECOMPOSITION_VARIABLES if c != "cohort"]]:
        mask &= table[col] == row[col]
    matches = table[mask]
    if matches.empty:
        return None
    return bool((matches["mean_delta"] >= 0).all())


def write_units(units: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    units.to_parquet(path, index=False)
    return path


def build_audit(
    *,
    leaderboard_dir: Path,
    realized_production_surface: Path,
    output_path: Path,
    units_output_path: Path | None,
    surface_specs: dict[str, Path],
    decision_cutoff_iso: str | None,
    cohort_as_of_iso: str | None,
    cohort_users_json: Path | None,
    dates: set[str] | None,
    min_date: str | None,
    max_date: str | None,
    top_k: tuple[int, ...],
    n_bootstrap: int,
    expected_block_length: int,
    seed: int,
    fdr_min_n: int,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or utc_now_iso()
    production, production_inventory = load_realized_production_picks(
        realized_production_surface,
        dates=dates,
        min_date=min_date,
        max_date=max_date,
    )
    consensus, consensus_inventory = build_consensus(
        leaderboard_dir=leaderboard_dir,
        decision_cutoff_iso=decision_cutoff_iso,
        cohort_as_of_iso=cohort_as_of_iso,
        cohort_users_json=cohort_users_json,
        dates=dates,
        min_date=min_date,
        max_date=max_date,
    )
    if surface_specs:
        ranked_surfaces, ranked_joinable, ranked_inventory = load_ranked_surfaces(
            surface_specs,
            dates=dates,
            min_date=min_date,
            max_date=max_date,
        )
    else:
        ranked_surfaces = pd.DataFrame()
        ranked_joinable = pd.DataFrame()
        ranked_inventory = {}

    units = build_mechanism_units(
        production=production,
        consensus=consensus,
        ranked_joinable=ranked_joinable,
        ranked_surfaces=ranked_surfaces,
    )
    if units_output_path is None:
        units_output_path = output_path.with_suffix(".units.parquet")
    write_units(units, units_output_path)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "research_only": True,
        "production_deploy_claim": False,
        "no_policy_edit_supported": True,
        "pre_registration": {
            "path": PREREG_PATH,
            "status": "committed_before_first_real_data_artifact_run",
            "primary_estimand": (
                "fixed_cohort_consensus_top_N_coverage_by_ranked_model_surfaces"
            ),
            "top_k": list(top_k),
            "realized_production_degrades_to_same_slot_agreement": True,
            "no_interim_cell_result_inspection": True,
        },
        "inputs": {
            "leaderboard_dir": str(leaderboard_dir),
            "realized_production_surface": str(realized_production_surface),
            "surface_specs": {name: str(path) for name, path in surface_specs.items()},
            "decision_cutoff_iso": decision_cutoff_iso,
            "cohort_as_of_iso": cohort_as_of_iso or decision_cutoff_iso,
            "dates": sorted(dates) if dates is not None else None,
            "min_date": min_date,
            "max_date": max_date,
        },
        "outputs": {
            "json": str(output_path),
            "units_parquet": str(units_output_path),
        },
        "decomposition_variables": DECOMPOSITION_VARIABLES,
        "decomposition_variable_source": PREREG_PATH,
        "inventory": {
            **consensus_inventory,
            "realized_production": production_inventory,
            "ranked_surfaces": ranked_inventory,
            "units_total": int(len(units)),
        },
        "summary": summarize_units(
            units,
            top_k=top_k,
            n_bootstrap=n_bootstrap,
            expected_block_length=expected_block_length,
            seed=seed,
        ),
        "fdr_method": {
            "method": "exact_positive_sign_test_plus_BH_BY",
            "q_threshold": DEFAULT_Q_THRESHOLD,
            "min_resolved_disagreement_units": int(fdr_min_n),
            "bootstrap_summary_expected_block_length": int(expected_block_length),
            "bootstrap_summary_n_bootstrap": int(n_bootstrap),
            "day_block_bootstrap_seed": int(seed),
        },
        "decomposition_fdr": decomposition_fdr_table(
            units,
            min_n=fdr_min_n,
            q_threshold=DEFAULT_Q_THRESHOLD,
        ),
        "mechanism_found_threshold": {
            "min_resolved_disagreement_units": DEFAULT_MECHANISM_MIN_N,
            "min_absolute_lift": DEFAULT_MECHANISM_MIN_LIFT,
            "q_BH_threshold": DEFAULT_Q_THRESHOLD,
            "requires_all_tracked_direction_not_contradicted": True,
            "requires_lock_time_variables": True,
            "production_claim_supported": False,
        },
        "methodology_constraints": {
            "historical_leaderboard_mining_is_post_hoc": True,
            "latest_snapshot_cohort_used_retrospectively": True,
            "survivorship_and_right_truncation_bias": True,
            "captured_public_picks_are_behavior_observations_not_pre_lock_proof": True,
            "backfilled_ranked_surfaces_are_not_at_lock_without_manifest_proof": True,
            "power_limited_at_current_sample_size": True,
            "absence_of_found_mechanism_falsifies_current_data_nomination_capacity_only": True,
        },
        "falsification_rule": {
            "if_no_cell_satisfies_all_mechanism_found_conditions": (
                "no actionable mechanism for this iteration"
            ),
            "future_iterations_require": [
                "new_data",
                "new_decomposition_variables_pre_registered_before_inspection",
                "different_cohort_definition",
            ],
        },
    }
    write_json(report, output_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaderboard-dir", type=Path, default=Path("data/leaderboard"))
    parser.add_argument(
        "--realized-production-surface",
        type=Path,
        required=True,
        help="Canonical realized-picks parquet with locked production slots.",
    )
    parser.add_argument(
        "--surface",
        action="append",
        default=[],
        help="Optional ranked model surface as NAME=PATH.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--units-output", type=Path, default=None)
    parser.add_argument("--decision-cutoff-iso", default=None)
    parser.add_argument("--cohort-as-of-iso", default=None)
    parser.add_argument("--cohort-users-json", type=Path, default=None)
    parser.add_argument("--dates", default=None, help="Comma-separated date filter.")
    parser.add_argument("--min-date", default=None)
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--top-k", default="1,2,5,10")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--expected-block-length", type=int, default=7)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--fdr-min-n", type=int, default=DEFAULT_FDR_MIN_N)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = build_audit(
        leaderboard_dir=args.leaderboard_dir,
        realized_production_surface=args.realized_production_surface,
        output_path=args.output,
        units_output_path=args.units_output,
        surface_specs=parse_surface_specs(args.surface, require=False),
        decision_cutoff_iso=args.decision_cutoff_iso,
        cohort_as_of_iso=args.cohort_as_of_iso,
        cohort_users_json=args.cohort_users_json,
        dates=parse_dates(args.dates),
        min_date=args.min_date,
        max_date=args.max_date,
        top_k=parse_top_k(args.top_k),
        n_bootstrap=args.n_bootstrap,
        expected_block_length=args.expected_block_length,
        seed=args.seed,
        fdr_min_n=args.fdr_min_n,
    )
    print(json.dumps({
        "schema_version": report["schema_version"],
        "output": report["outputs"]["json"],
        "units_parquet": report["outputs"]["units_parquet"],
        "units_total": report["inventory"]["units_total"],
        "actionable_mechanism_found": report["decomposition_fdr"][
            "actionable_mechanism_found"
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
