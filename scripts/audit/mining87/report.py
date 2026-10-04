"""Summaries and the registered report, in amendment A4 order: coverage and denominators first, then the primary
top-N estimand (or its unavailability) with the same-slot fallback, the secondary reads, both streams' complete cell
inventories, the sensitivity comparison, the nomination record, the separately labelled served-slate diagnostic,
and the methodology and limits (review r1 findings 6 and 9, required changes 6 and 8–10).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.audit.mining87.inference import (BLOCK_SEMANTICS, CONDITIONS, SIGN_TEST_ASSUMPTION,
                                              date_block_bootstrap_ratio)
from scripts.audit.mining87.units import COHORTS
from scripts.leaderboard_mechanism_mining import DECOMPOSITION_VARIABLES, _bin_consensus_share

SCHEMA_VERSION = "mining87_report_v1"
PRODUCTION_STATUS = {"hit": "resolved", "not_hit": "resolved"}
CONSENSUS_STATUS = {"hit": "resolved", "not_hit": "resolved"}


def jsonable(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def _counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def _rate(series: pd.Series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else None


def coverage(units: pd.DataFrame, *, production_inventory: dict, admission: list[dict], public_inventory: dict,
             cohort_meta: dict, context_meta: dict) -> dict:
    decisions = pd.Series([a["reason"] for a in admission], dtype=object)
    per_cohort = {}
    for cohort in COHORTS:
        u = units[units["cohort"] == cohort]
        resolved_prod = u["production_resolved"]
        with_id = resolved_prod & u["consensus_available"]
        per_cohort[cohort] = {
            "n_units": int(len(u)), "n_units_by_slot": _counts(u["pick_number"]),
            "production_settlement": _counts(u["production_settlement"].replace(PRODUCTION_STATUS)),
            "consensus_status": _counts(u["consensus_status"]),
            "consensus_settlement": _counts(u["consensus_settlement"].replace(CONSENSUS_STATUS)),
            "consensus_settlement_reason": _counts(u["consensus_settlement_reason"]),
            "ties": {"direct": int(u["consensus_tie_direct"].sum()),
                     "exposed_by_excluding_primary": int(u["consensus_tie_exposed_by_exclusion"].sum()),
                     "dependent_on_tied_primary": int(u["consensus_tie_dependent"].sum()),
                     "flagged_for_sensitivity": int(u["consensus_tie_broken"].sum()),
                     "legal_dd_differs_from_unconstrained": int(u["consensus_legal_differs_from_unconstrained"].sum())},
            "agreement_state": _counts(u["agreement_state"]),
            "denominators": {
                "top_n_coverage": int((with_id & u["surface_admitted"]).sum()),
                "top_n_exclusions": {"production_unresolved": int((~resolved_prod).sum()),
                                     "consensus_unavailable": int((resolved_prod & ~u["consensus_available"]).sum()),
                                     "no_admitted_surface": int((with_id & ~u["surface_admitted"]).sum())},
                "same_slot_agreement": int(with_id.sum()),
                "paired_outcomes_joint_resolved": int(u["joint_resolved"].sum()),
                "resolved_disagreement": int(u["resolved_disagreement"].sum()),
                "miscalibration_target_bound": int((u["consensus_resolved"] & u["surface_admitted"]
                                                    & u["consensus_probability_target_bound"]).sum())}}
    return {
        "production": production_inventory,
        "public_picks": public_inventory,
        "cohort": cohort_meta,
        "production_context": context_meta,
        "surfaces": {"dates_considered": len(admission), "admitted_dates": int(sum(a["admitted"] for a in admission)),
                     "decisions": _counts(decisions) if len(decisions) else {},
                     "selection_consistency": _counts(pd.Series([a["selection_consistency"] or "not_evaluated"
                                                                 for a in admission], dtype=object))
                     if admission else {},
                     "per_date": admission},
        "cohorts": per_cohort}


def top_n(units: pd.DataFrame, top_k: tuple) -> dict:
    out = {}
    for cohort in COHORTS:
        u = units[(units["cohort"] == cohort) & units["production_resolved"] & units["consensus_available"]]
        adm = u[u["surface_admitted"]]
        if adm.empty:
            out[cohort] = {"available": False, "denominator": 0,
                           "reason": "no_admitted_surface" if not u["surface_admitted"].any() else "no_eligible_units",
                           "note": "missing surfaces are not off-top-N failures"}
            continue
        rank = adm["consensus_model_rank"]
        out[cohort] = {"available": True, "denominator": int(len(adm)), "n_admitted_dates": int(adm["date"].nunique()),
                       "n_off_ranking": int(rank.isna().sum()),
                       "coverage": {str(k): float(rank.le(k).fillna(False).mean()) for k in top_k}}
    out["primary_cohort"] = "fixed_cohort"
    return out


def same_slot_agreement(units: pd.DataFrame) -> dict:
    out = {}
    for cohort in COHORTS:
        u = units[(units["cohort"] == cohort) & units["production_resolved"] & units["consensus_available"]]
        same = u["agreement_state"] == "same_batter"
        out[cohort] = {"denominator": int(len(u)), "agreement_rate": float(same.mean()) if len(u) else None,
                       "by_slot": {str(s): {"denominator": int(len(g)),
                                            "agreement_rate": float((g["agreement_state"] == "same_batter").mean())}
                                   for s, g in u.groupby("pick_number")},
                       "note": "production decision diagnostic (protocol fallback), not top-N model coverage"}
    return out


def _paired(frame: pd.DataFrame, params) -> dict:
    return {"n_units": int(len(frame)), "production_hit_rate": _rate(frame["production_hit"]),
            "consensus_hit_rate": _rate(frame["consensus_hit"]), "mean_delta": _rate(frame["delta"]),
            "bootstrap": date_block_bootstrap_ratio(frame, "delta", expected_block_length=params.expected_block_length,
                                                    n_bootstrap=params.n_bootstrap, seed=params.seed)}


def paired_outcomes(units: pd.DataFrame, params) -> dict:
    out = {}
    for cohort in COHORTS:
        u = units[(units["cohort"] == cohort) & units["joint_resolved"]]
        out[cohort] = {"all_joint_resolved": _paired(u, params),
                       "resolved_disagreement": _paired(u[u["agreement_state"] == "different_batter"], params)}
    return out


def miscalibration(units: pd.DataFrame) -> dict:
    out = {}
    for cohort in COHORTS:
        u = units[(units["cohort"] == cohort) & units["consensus_resolved"] & units["surface_admitted"]
                  & units["consensus_probability_target_bound"] & units["consensus_model_p_game_hit"].notna()]
        if u.empty:
            out[cohort] = {"available": False, "reason": "no admitted surface row uniquely bound to the consensus "
                                                         "outcome target"}
            continue
        p = u["consensus_model_p_game_hit"].astype(float)
        y = u["consensus_hit"].astype(float)
        out[cohort] = {"available": True, "n_units": int(len(u)), "mean_probability": float(p.mean()),
                       "mean_outcome": float(y.mean()), "mean_residual_probability_minus_outcome":
                       float((p - y).mean()), "descriptive_only": True}
    return out


def concentration(consensus: pd.DataFrame) -> dict:
    """From the frozen choice table: every window date-slot with a legal consensus id, production-matched or not."""
    out = {}
    for cohort in COHORTS:
        u = consensus[(consensus["cohort"] == cohort) & consensus["consensus_available"].astype(bool)]
        out[cohort] = {str(s): {"n_slots": int(len(g)), "mean_share": _rate(g["consensus_pick_share"]),
                                "median_share": float(g["consensus_pick_share"].median()) if len(g) else None,
                                "share_bins": _counts(g["consensus_pick_share"].map(_bin_consensus_share)),
                                "median_public_users": float(g["n_public_users"].median()) if len(g) else None}
                       for s, g in u.groupby("pick_number")}
    out["caveat"] = ("consensus concentration may indicate a publicly obvious batter class or a stale blind spot; it "
                     "is not a success metric")
    return out


def served_slate_diagnostic(units: pd.DataFrame, slate_tables: dict, admission: list[dict], top_k: tuple) -> dict:
    """Unproven served-slate batter coverage on NOT-admitted dates; never enters cells, family or nomination."""
    state = {a["date"]: a["selection_consistency"] for a in admission}
    admitted = {a["date"] for a in admission if a["admitted"]}
    out = {"label": "unproven_served_slate_diagnostic", "outside_registered_family": True,
           "admission": "dates here failed the A3 admission rule; this is not the protocol's top-N estimand"}
    for cohort in COHORTS:
        u = units[(units["cohort"] == cohort) & units["production_resolved"] & units["consensus_available"]
                  & units["date"].isin(set(slate_tables) - admitted)]
        ranks = [slate_tables[d].get(int(b), {}).get("rank") for d, b in zip(u["date"], u["consensus_batter_id"])]
        frame = pd.DataFrame({"state": [state.get(d) for d in u["date"]], "rank": pd.array(ranks, dtype="Int64")})
        out[cohort] = {str(st): {"denominator": int(len(g)),
                                 "coverage": {str(k): float(g["rank"].le(k).fillna(False).mean()) for k in top_k}}
                       for st, g in frame.groupby("state", dropna=False)}
    return out


def cells_records(cells: pd.DataFrame) -> list[dict]:
    return [jsonable(r) for r in cells.to_dict("records")]


def nomination(primary: dict, comparison: list[dict], units: pd.DataFrame, params) -> dict:
    cells = primary["cells"]
    nominated = cells[cells["nominated"]]
    fixed = cells[cells["cohort"] == "fixed_cohort"]
    fu = units[units["cohort"] == "fixed_cohort"]
    out = {"nominating_stream": "primary", "run_mode": params.mode, "can_nominate": params.can_nominate,
           "outcome_state": primary["summary"]["outcome_state"],
           "conditions": list(CONDITIONS), "conditions_reported_separately_in": "streams.primary.cells",
           "nominated_cells": [{"cell": {v: jsonable(r[v]) for v in DECOMPOSITION_VARIABLES},
                                "robustness": r["robustness"],
                                "target": "feature hypothesis for W4 rank 7; not a production edit"}
                               for r in nominated.to_dict("records")],
           "primary_candidates_failing_sensitivity": [c for c in comparison if not c["passes_sensitivity"]],
           "information_limits": {
               "fixed_testable_condition_status": primary["summary"]["fixed_testable_condition_status"],
               "fixed_cells_sparse_support": int(((fixed["n_resolved_disagreement"] > 0) & ~fixed["testable"]).sum()),
               "fixed_units_without_admitted_surface": int((~fu["surface_admitted"]).sum()),
               "fixed_units_production_unresolved": int((~fu["production_resolved"]).sum()),
               "fixed_units_consensus_unavailable": int((~fu["consensus_available"]).sum()),
               "fixed_units_consensus_unresolved": int((fu["consensus_available"] & ~fu["consensus_resolved"]).sum())}}
    if out["outcome_state"] in ("no_actionable_mechanism", "power_limited_no_testable_cells"):
        out["statement"] = ("No cell met all five conditions: no actionable mechanism for this cohort and window. This "
                            "does not disprove signal in a larger or better-instrumented sample; missing provenance, "
                            "unresolved settlement, unknown lock availability and sparse support are listed "
                            "separately from measured negative effects.")
    return out


METHODOLOGY = {
    "historical_leaderboard_mining_is_post_hoc": True,
    "fixed_cohort_chosen_retrospectively_from_the_2026_07_04_snapshot": True,
    "survivorship_and_right_truncation_bias": True,
    "prior_exposure": "X-10 (7/03 me-vs-leaderboard) overlaps this window; 2026 is not an untouched holdout",
    "captured_public_picks_are_behavior_observations_not_pre_lock_proof": True,
    "no_pre_lock_visibility_claim": True,
    "served_slates_are_not_admitted_without_an_independent_witness": True,
    "missing_surfaces_are_not_off_top_n_failures": True,
    "cell_test_assumption": SIGN_TEST_ASSUMPTION,
    "bootstrap_semantics": BLOCK_SEMANTICS,
    "condition_5": "requires an explicit mechanism record whose operational variables carry lock-availability "
                   "evidence; missing evidence is unknown, never passage",
    "by_failure": "a BH survivor failing BY is BH-only exploratory, not a robust claim",
    "sensitivity": "the tie-excluded stream is diagnostic and cannot nominate",
    "power_limited_at_current_sample_size": True,
    "partial_or_source_invalid_runs_cannot_falsify_nomination_capacity": True,
    "retries": "a technical retry uses identical frozen inputs and methods (--expect-inputs) without results "
               "inspection; changed dates, cohort or methods after inspection are exploratory",
    "absence_of_found_mechanism_falsifies_current_data_nomination_capacity_only": True,
    "public_window_ends_2026_07_03": "half a season (the daily corpus ends 7/04)",
}
