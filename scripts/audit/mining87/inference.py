"""Intervals, the complete cell inventory, BH/BY families, the five nomination conditions and the analysis streams
(amendment A2/A4; review r1 findings 3, 5, 7, 9; required changes 7, 8 and 9).

* **Intervals** resample whole observed dates with their delta sums and unit counts (circular geometric blocks over
  the ordered observed dates, not calendar days), so they target the reported unit-weighted mean. Every calculation
  uses the frozen seed itself.
* **Cells** are every observed nonempty cell of the 14-variable cross product over all retained units. A cell is
  testable at ≥15 resolved disagreement units; sparse cells stay descriptive with unavailable p/q.
* **Test** (amended A4 extension, frozen): the exact one-sided paired sign test for every testable cell, regardless of
  count or observed lift. It assumes the nonzero signs are exchangeable under H0; day-level dependence violates that,
  and BH/BY do not repair invalid cell p-values.
* **Family**: BH and BY over the complete testable family of the stream (both cohorts), no significance pruning.
* **Conditions** (fixed-cohort cells, each reported separately, tri-state): c1 ≥30 units; c2 lift ≥0.05 (exact);
  c3 BH q ≤0.10; c4 all-tracked same-cell direction from the pre-filter inventory (absent support unknown, negative
  support fails, sparse cells count); c5 an explicit mechanism record whose operational variables carry evidence of
  availability at lock (missing evidence is unknown, never passage; public-consensus variables cannot be operational).
  Only a stream allowed to nominate (the registered primary) may set ``nominated``.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from bts.validate.fdr import bh_qvalues, by_qvalues
from scripts.leaderboard_mechanism_mining import DECOMPOSITION_VARIABLES, exact_positive_sign_test_pvalue

CELL_TEST = "exact_one_sided_paired_sign_test"
SIGN_TEST_ASSUMPTION = (
    "Amended A4 extension: the exact one-sided paired sign test is used for every testable cell (the original "
    "protocol admitted it only as a small-count fallback). It assumes nonzero consensus-minus-production signs are "
    "exchangeable (independent and symmetric) under H0; serial or same-day dependence violates that, and BH/BY "
    "control multiplicity, not the validity of the cell p-values.")
BLOCK_SEMANTICS = ("circular geometric blocks over the ordered observed dates (not calendar days); each draw "
                   "resamples whole dates with their delta sums and unit counts; the replicate is sum/count, the "
                   "unit-weighted mean")
MECHANISM_SCHEMA = "mining87_mechanism_records_v1"
CONSENSUS_DERIVED = frozenset({"cohort", "consensus_pick_share_bin", "agreement_state", "consensus_model_rank_bin",
                               "consensus_model_probability_bin"})
CONDITIONS = ("c1_min_units", "c2_min_lift", "c3_bh", "c4_all_tracked_direction", "c5_lock_available_mechanism")
NON_COHORT = [v for v in DECOMPOSITION_VARIABLES if v != "cohort"]
CONDITION_COLUMNS = [*CONDITIONS, "c5_reason", "all_tracked_n_resolved_disagreement", "all_tracked_mean_delta",
                     "all_five_conditions", "statistical_candidate", "nominated"]


# --- intervals -------------------------------------------------------------------------------------------------------

def date_block_bootstrap_ratio(frame: pd.DataFrame, value_col: str, *, expected_block_length: int, n_bootstrap: int,
                               seed: int, date_col: str = "date", return_samples: bool = False) -> dict | None:
    vals = frame[[date_col, value_col]].copy()
    vals[value_col] = pd.to_numeric(vals[value_col], errors="coerce")
    vals = vals.dropna(subset=[value_col])
    if vals.empty:
        return None
    by_date = vals.groupby(date_col)[value_col].agg(["sum", "count"]).sort_index()
    sums, counts = by_date["sum"].to_numpy(float), by_date["count"].to_numpy(float)
    n_dates = len(sums)
    out = {"kind": "circular_geometric_day_block_bootstrap", "estimand": "unit_weighted_mean",
           "block_semantics": BLOCK_SEMANTICS, "n_bootstrap": int(n_bootstrap),
           "expected_block_length": int(expected_block_length), "seed": int(seed), "n_dates": int(n_dates),
           "n_units": int(counts.sum()), "mean": float(sums.sum() / counts.sum()), "ci_lower": None, "ci_upper": None}
    if n_dates < 2 or n_bootstrap <= 0:
        return out
    rng = np.random.default_rng(seed)
    p_stop = 1.0 / max(1, int(expected_block_length))
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx: list[int] = []
        while len(idx) < n_dates:
            start, length = int(rng.integers(0, n_dates)), int(rng.geometric(p_stop))
            idx.extend((start + k) % n_dates for k in range(min(length, n_dates - len(idx))))
        samples[i] = sums[idx].sum() / counts[idx].sum()
    out.update(ci_lower=float(np.quantile(samples, 0.025)), ci_upper=float(np.quantile(samples, 0.975)))
    if return_samples:
        out["samples"] = samples.tolist()
    return out


# --- cells -----------------------------------------------------------------------------------------------------------

def _cell_id(values) -> str:
    return json.dumps([int(v) if isinstance(v, (int, np.integer)) else str(v) for v in values])


def cell_inventory(units: pd.DataFrame) -> pd.DataFrame:
    rd = units["resolved_disagreement"].astype(bool)
    delta = pd.to_numeric(units["delta"], errors="coerce").astype(float)
    work = units[DECOMPOSITION_VARIABLES].copy()
    work["_unit"] = 1
    work["_prod_resolved"] = units["production_resolved"].astype(bool).astype(int)
    work["_joint"] = units["joint_resolved"].astype(bool).astype(int)
    work["_rd"] = rd.astype(int)
    work["_pos"] = (rd & (delta > 0)).astype(int)
    work["_neg"] = (rd & (delta < 0)).astype(int)
    work["_d"] = delta.where(rd)
    work["_ph"] = pd.to_numeric(units["production_hit"], errors="coerce").astype(float).where(rd)
    work["_ch"] = pd.to_numeric(units["consensus_hit"], errors="coerce").astype(float).where(rd)
    cells = (work.groupby(DECOMPOSITION_VARIABLES, dropna=False, sort=True)
             .agg(n_units=("_unit", "sum"), n_production_resolved=("_prod_resolved", "sum"),
                  n_joint_resolved=("_joint", "sum"), n_resolved_disagreement=("_rd", "sum"),
                  n_positive=("_pos", "sum"), n_negative=("_neg", "sum"), sum_delta=("_d", "sum"),
                  production_hit_rate=("_ph", "mean"), consensus_hit_rate=("_ch", "mean"),
                  mean_delta=("_d", "mean"))
             .reset_index())
    cells["cell_id"] = [_cell_id(v) for v in cells[DECOMPOSITION_VARIABLES].itertuples(index=False)]
    return cells


def load_mechanism_records(path: Path | None) -> tuple[list[dict], dict]:
    """Mechanism-record file (an input; hashed and pinned, so it must be frozen before the registered run)::

        {"schema": "mining87_mechanism_records_v1",
         "records": [{"cell": {<all 14 decomposition variables, cohort included: value>},
                      "mechanism": "<the stated feature mechanism>",
                      "operational_variables": ["<variable the mechanism would use at lock>", ...],
                      "lock_evidence": {"<variable>": {"status": "available_at_lock" | "not_available_at_lock",
                                                       "evidence": "<reference>"}}}]}

    A record matches a cell only when its ``cell`` names every axis with the cell's exact value."""
    if path is None:
        return [], {"source": "none_supplied", "n_records": 0}
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict) or doc.get("schema") != MECHANISM_SCHEMA or not isinstance(doc.get("records"), list):
        raise ValueError(f"mechanism record file schema is not {MECHANISM_SCHEMA}")
    return doc["records"], {"source": str(path), "n_records": len(doc["records"])}


def _norm_cell(cell) -> dict | None:
    if not isinstance(cell, dict) or set(cell) != set(DECOMPOSITION_VARIABLES):
        return None
    return {k: str(cell[k]) for k in DECOMPOSITION_VARIABLES}


def lock_mechanism_status(cell: dict, records: list[dict]) -> tuple[str, str]:
    want = _norm_cell(cell)
    matching = [r for r in records if isinstance(r, dict) and _norm_cell(r.get("cell")) == want]
    if not matching:
        return "unknown", "no_mechanism_record"
    if len(matching) > 1:
        return "unknown", "conflicting_mechanism_records"
    rec = matching[0]
    if not isinstance(rec.get("mechanism"), str) or not rec["mechanism"].strip():
        return "unknown", "mechanism_not_stated"
    ops = rec.get("operational_variables")
    if not isinstance(ops, list) or not ops:
        return "unknown", "no_operational_variables"
    if any(v in CONSENSUS_DERIVED for v in ops):
        return "fail", "operational_variable_is_public_consensus_observation"
    evidence = rec.get("lock_evidence") if isinstance(rec.get("lock_evidence"), dict) else {}
    for var in ops:
        ev = evidence.get(var)
        if isinstance(ev, dict) and ev.get("status") == "not_available_at_lock":
            return "fail", f"not_available_at_lock:{var}"
    for var in ops:
        ev = evidence.get(var)
        if not isinstance(ev, dict) or ev.get("status") != "available_at_lock" \
                or not isinstance(ev.get("evidence"), str) or not ev["evidence"].strip():
            return "unknown", f"lock_evidence_missing:{var}"
    return "pass", "lock_available_mechanism_evidenced"


def _none(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def evaluate_stream(units: pd.DataFrame, *, stream: str, mechanism_records: list[dict], can_nominate: bool,
                    min_n: int, q_threshold: float, mechanism_min_n: int, mechanism_min_lift: float) -> dict:
    cells = cell_inventory(units)
    cells["testable"] = cells["n_resolved_disagreement"] >= min_n
    tests = [exact_positive_sign_test_pvalue(pd.Series([1] * int(p) + [-1] * int(n), dtype=float)) if t else None
             for t, p, n in zip(cells["testable"], cells["n_positive"], cells["n_negative"])]
    cells["test"] = [CELL_TEST if t else None for t in tests]
    cells["p_one_sided_positive"] = [t["p_one_sided_positive"] if t else np.nan for t in tests]
    for col in ("q_BH", "q_BY"):
        cells[col] = np.nan
    fam = cells["testable"].to_numpy()
    if fam.any():
        p = cells.loc[fam, "p_one_sided_positive"].to_numpy(float)
        cells.loc[fam, "q_BH"], cells.loc[fam, "q_BY"] = bh_qvalues(p), by_qvalues(p)
    cells["survives_BH"] = cells["testable"] & (cells["q_BH"] <= q_threshold)
    cells["survives_BY"] = cells["testable"] & (cells["q_BY"] <= q_threshold)
    cells["robustness"] = [("BH_and_BY" if by else "BH_only_exploratory") if bh else None
                           for bh, by in zip(cells["survives_BH"], cells["survives_BY"])]

    tracked = {tuple(r[v] for v in NON_COHORT): r for r in cells[cells["cohort"] == "all_tracked"].to_dict("records")}
    lift = Fraction(str(mechanism_min_lift))
    rows = []
    for r in cells.to_dict("records"):
        out = {c: None for c in (*CONDITIONS, "c5_reason", "all_tracked_n_resolved_disagreement",
                                 "all_tracked_mean_delta")}
        out.update(all_five_conditions=False, statistical_candidate=False, nominated=False)
        if r["cohort"] == "fixed_cohort":
            n = int(r["n_resolved_disagreement"])
            match = tracked.get(tuple(r[v] for v in NON_COHORT))
            t_n = int(match["n_resolved_disagreement"]) if match else 0
            out["all_tracked_n_resolved_disagreement"] = t_n
            out["all_tracked_mean_delta"] = _none(match["mean_delta"]) if match else None
            out["c1_min_units"] = "pass" if n >= mechanism_min_n else "fail"
            out["c2_min_lift"] = ("unknown" if n == 0 else
                                  "pass" if Fraction(int(r["sum_delta"]), n) >= lift else "fail")
            out["c3_bh"] = "not_testable" if not r["testable"] else "pass" if r["survives_BH"] else "fail"
            out["c4_all_tracked_direction"] = ("unknown" if t_n == 0 else
                                               "fail" if match["mean_delta"] < 0 else "pass")
            cell = {v: r[v] for v in DECOMPOSITION_VARIABLES}
            out["c5_lock_available_mechanism"], out["c5_reason"] = lock_mechanism_status(cell, mechanism_records)
            out["statistical_candidate"] = all(out[c] == "pass" for c in CONDITIONS[:4])
            out["all_five_conditions"] = all(out[c] == "pass" for c in CONDITIONS)
            out["nominated"] = bool(out["all_five_conditions"] and can_nominate)
        rows.append(out)
    cells = pd.concat([cells, pd.DataFrame(rows, index=cells.index, columns=CONDITION_COLUMNS)], axis=1)
    for col in ("all_five_conditions", "statistical_candidate", "nominated"):
        cells[col] = cells[col].astype(bool)

    fixed_testable = cells[(cells["cohort"] == "fixed_cohort") & cells["testable"]]
    n_testable = int(cells["testable"].sum())
    n_nominated = int(cells["nominated"].sum())
    state = ("diagnostic_cannot_nominate" if not can_nominate else
             "power_limited_no_testable_cells" if n_testable == 0 else
             "feature_hypothesis_nominated" if n_nominated else "no_actionable_mechanism")
    summary = {
        "stream": stream, "can_nominate": bool(can_nominate), "outcome_state": state,
        "n_nonempty_cells": int(len(cells)),
        "n_nonempty_cells_by_cohort": {k: int(v) for k, v in cells["cohort"].value_counts().sort_index().items()},
        "n_testable_cells": n_testable,
        "n_testable_by_cohort": {k: int(v) for k, v in cells.loc[cells["testable"], "cohort"].value_counts()
                                 .sort_index().items()},
        "n_survive_BH": int(cells["survives_BH"].sum()), "n_survive_BY": int(cells["survives_BY"].sum()),
        "n_BH_only_exploratory": int((cells["robustness"] == "BH_only_exploratory").sum()),
        "n_statistical_candidates": int(cells["statistical_candidate"].sum()),
        "n_all_five_conditions": int(cells["all_five_conditions"].sum()), "n_nominated": n_nominated,
        "fixed_testable_condition_status": {c: {k: int(v) for k, v in fixed_testable[c].value_counts().sort_index()
                                                .items()} for c in CONDITIONS},
        "test": CELL_TEST, "test_assumption": SIGN_TEST_ASSUMPTION, "family": "BH and BY over every testable cell "
        "of this stream (both cohorts)", "min_resolved_disagreement_units": int(min_n),
        "q_threshold": float(q_threshold), "mechanism_min_resolved_disagreement_units": int(mechanism_min_n),
        "mechanism_min_absolute_lift": float(mechanism_min_lift)}
    return {"stream": stream, "cells": cells, "summary": summary}


def tie_excluded(units: pd.DataFrame) -> pd.DataFrame:
    """The sole sensitivity: drop the already-selected flagged units within each cohort; no new winner is chosen."""
    return units[~units["consensus_tie_broken"].astype(bool)].copy()


def compare_streams(primary_cells: pd.DataFrame, sensitivity_cells: pd.DataFrame) -> list[dict]:
    """For each primary statistical candidate (c1–c4 pass): does it pass the sensitivity, and what was lost."""
    sens = {r["cell_id"]: r for r in sensitivity_cells.to_dict("records")}
    out = []
    for r in primary_cells[primary_cells["statistical_candidate"]].to_dict("records"):
        s = sens.get(r["cell_id"])
        if s is None or not s["n_resolved_disagreement"]:
            losses = ["support"]
        else:
            losses = [name for name, lost in (
                ("support", not s["testable"] or s["c1_min_units"] != "pass"),
                ("effect", s["c2_min_lift"] != "pass"),
                ("q_value", bool(s["testable"]) and s["c3_bh"] != "pass"),
                ("direction", s["c4_all_tracked_direction"] != "pass")) if lost]
        out.append({"cell": {v: (int(r[v]) if v == "pick_number" else r[v]) for v in DECOMPOSITION_VARIABLES},
                    "cell_id": r["cell_id"], "primary_nominated": bool(r["nominated"]),
                    "primary_robustness": r["robustness"], "passes_sensitivity": not losses, "losses": losses,
                    "sensitivity_n_resolved_disagreement": int(s["n_resolved_disagreement"]) if s else 0,
                    "sensitivity_testable": bool(s["testable"]) if s else False,
                    "sensitivity_q_BH": _none(s["q_BH"]) if s else None,
                    "sensitivity_mean_delta": _none(s["mean_delta"]) if s else None})
    return out
