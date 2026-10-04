"""Production-led unit table (review r1 findings 4 and 6, required change 4).

One unit per (cohort, locked production date-slot): every locked slot is retained in each cohort whatever its
resolution or consensus availability; no slot-2 unit exists without a production DD. Four masks stay distinct:
``production_resolved``, ``consensus_available`` (an identified legal id), ``consensus_resolved`` and their conjunction
``joint_resolved``; ``delta`` exists only on joint resolution. Decomposition values use the original 14 ordered axes
and bins (helpers imported from the historical script, which review found correct); every missing value is the
string ``missing`` / ``missing_surface`` so cells match consistently.
"""
from __future__ import annotations

import pandas as pd

from scripts.audit.mining87.surfaces import consensus_surface
from scripts.leaderboard_mechanism_mining import (
    DECOMPOSITION_VARIABLES,
    _bin_consensus_share,
    _bin_model_rank,
    _bin_prior_pa,
    _bin_probability,
    _bin_production_probability,
    _bin_weather_temp,
    _quartile_or_missing,
    _string_or_missing,
    _truthy_or_missing,
)

COHORTS = ("fixed_cohort", "all_tracked")


def _check_binary(frame: pd.DataFrame, value: str, mask: str) -> None:
    vals = frame.loc[frame[mask], value]
    if vals.isna().any() or not vals.isin([0, 1]).all():
        raise ValueError(f"{value} must be binary 0/1 wherever {mask}")
    if frame.loc[~frame[mask], value].notna().any():
        raise ValueError(f"{value} must be null wherever not {mask}")


def build_units(production: pd.DataFrame, consensus: pd.DataFrame, *, surfaces: dict,
                unit_games: dict | None) -> pd.DataFrame:
    """``production``: locked slots (+ context); ``consensus``: both cohorts' consensus tables; ``surfaces``:
    {date: {"admitted", "reason", "batters"}}; ``unit_games``: {consensus unit_id: game_pk} or None."""
    if production.duplicated(["date", "pick_number"]).any():
        raise ValueError("duplicate production (date, pick_number) slots")
    _check_binary(production, "production_hit", "production_resolved")
    parts = []
    for cohort in COHORTS:
        cons = consensus[consensus["cohort"] == cohort].drop(columns=["cohort"])
        part = production.merge(cons, on=["date", "pick_number"], how="left", validate="one_to_one")
        part.insert(0, "cohort", cohort)
        parts.append(part)
    units = pd.concat(parts, ignore_index=True)
    absent = units["consensus_status"].isna()
    units.loc[absent, "consensus_status"] = "no_public_votes"
    units.loc[absent, "consensus_settlement"] = "unavailable"
    units.loc[absent, "consensus_settlement_reason"] = "no_consensus_id"
    for col in ("consensus_available", "consensus_resolved", "consensus_tie_direct", "consensus_tie_dependent",
                "consensus_tie_broken", "consensus_tie_exposed_by_exclusion",
                "consensus_legal_differs_from_unconstrained"):
        units[col] = units[col].astype("boolean").fillna(False).astype(bool)
    units["n_public_users"] = units["n_public_users"].fillna(0).astype(int)
    for col in ("consensus_batter_id", "consensus_hit", "consensus_unit_id", "consensus_unconstrained_batter_id"):
        units[col] = units[col].astype("Int64")
    _check_binary(units, "consensus_hit", "consensus_resolved")

    surface_rows = []
    for date, cid, available, unit in zip(units["date"], units["consensus_batter_id"], units["consensus_available"],
                                          units["consensus_unit_id"]):
        info = surfaces.get(date) or {"admitted": False, "reason": "no_served_slate", "batters": None}
        unit_game = None if unit_games is None or pd.isna(unit) else unit_games.get(int(unit))
        if available:
            got = consensus_surface(int(cid), info["batters"], admitted=info["admitted"], unit_game=unit_game)
        else:
            got = {"rank": None, "p_game_hit": None, "probability_target_bound": False,
                   "status": "no_consensus_id" if info["admitted"] else "no_admitted_surface"}
        surface_rows.append({"surface_admitted": bool(info["admitted"]), "surface_admission_reason": info["reason"],
                             "consensus_model_rank": got["rank"], "consensus_model_p_game_hit": got["p_game_hit"],
                             "consensus_model_probability_status": got["status"],
                             "consensus_probability_target_bound": got["probability_target_bound"]})
    units = pd.concat([units, pd.DataFrame(surface_rows, index=units.index)], axis=1)
    units["consensus_model_rank"] = units["consensus_model_rank"].astype("Int64")
    units["consensus_model_p_game_hit"] = units["consensus_model_p_game_hit"].astype("Float64")

    same = units["consensus_available"] & (units["consensus_batter_id"] == units["production_batter_id"]).fillna(False)
    units["agreement_state"] = "different_batter"
    units.loc[same, "agreement_state"] = "same_batter"
    units.loc[~units["consensus_available"], "agreement_state"] = "consensus_unavailable"
    units["joint_resolved"] = units["production_resolved"] & units["consensus_available"] & units["consensus_resolved"]
    units["resolved_disagreement"] = units["joint_resolved"] & (units["agreement_state"] == "different_batter")
    units["delta"] = pd.Series(pd.NA, index=units.index, dtype="Int64")
    jr = units["joint_resolved"]
    units.loc[jr, "delta"] = units.loc[jr, "consensus_hit"] - units.loc[jr, "production_hit"]

    units["consensus_pick_share_bin"] = units["consensus_pick_share"].map(_bin_consensus_share)
    units["production_p_game_hit_bin"] = units["production_p_game_hit"].map(_bin_production_probability)
    units["production_batter_skill_quartile"] = units["production_batter_skill_quartile"].map(_quartile_or_missing)
    units["production_batter_skill_prior_pa_bin"] = units["production_batter_skill_prior_pa"].map(_bin_prior_pa)
    units["production_weather_temp_bin"] = [_bin_weather_temp(t, i) for t, i in
                                            zip(units["production_weather_temp"], units["production_is_indoor"])]
    for col in ("production_projected_lineup", "production_is_park_driven", "production_is_indoor"):
        units[col] = units[col].map(_truthy_or_missing)
    units["production_regime"] = units["production_regime"].map(_string_or_missing)
    units["consensus_model_rank_bin"] = [
        _bin_model_rank(r, adm and avail) if (adm and avail) else "missing_surface"
        for r, adm, avail in zip(units["consensus_model_rank"], units["surface_admitted"],
                                 units["consensus_available"])]
    units["consensus_model_probability_bin"] = units["consensus_model_p_game_hit"].map(_bin_probability)
    units["pick_number"] = units["pick_number"].astype(int)
    units["unit_key"] = units["cohort"] + "|" + units["date"] + "|" + units["pick_number"].astype(str)

    rest = [c for c in units.columns if c not in ("unit_key", "date") and c not in DECOMPOSITION_VARIABLES]
    ordered = units[["unit_key", "date", *rest, *DECOMPOSITION_VARIABLES]].copy()
    if ordered.duplicated(["cohort", "date", "pick_number"]).any():
        raise ValueError("duplicate (cohort, date, pick_number) units")
    return ordered.sort_values(["cohort", "date", "pick_number"]).reset_index(drop=True)
