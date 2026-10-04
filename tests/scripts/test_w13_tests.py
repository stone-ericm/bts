"""W1.3 distinguishing tests: predeclared component diagnostics over the frozen W1.2 run (design rev 2)."""
import math

import numpy as np
import pandas as pd

from scripts.audit.w13_tests import core


def _row(bid, gpk, p):
    return {"batter_id": bid, "game_pk": gpk, "p_game_hit": p}


def test_an_authoritative_skip_blocks_the_pick_fallback_and_keeps_the_declined_candidate_separate():
    skip = {"action": "skip", "primary": _row(1, 10, 0.7)}
    out = core.selected_primary(skip, {"pick": _row(1, 10, 0.7)})
    assert out["selected"] is None and out["action"] == "skip" and out["declined"] == (1, 10)
    single = {"action": "single", "primary": _row(2, 20, 0.8)}
    assert core.selected_primary(single, {"pick": _row(3, 30, 0.6)})["selected"] == (2, 20)
    assert core.selected_primary(single, None)["source"] == "decision"
    nodec = core.selected_primary(None, {"pick": _row(3, 30, 0.6)})
    assert nodec["selected"] == (3, 30) and nodec["source"] == "pick" and nodec["action"] == "unknown"
    assert core.selected_primary(None, None)["selected"] is None


def _pair_frame():
    # d1: A picks row0 (hit), B picks row1 (no_hit); row2 has A only (missing B score) and is a hit
    return pd.DataFrame({
        "date": ["d1", "d1", "d1", "d2", "d2"], "row_order": [0, 1, 2, 0, 1],
        "batter_id": [1, 2, 3, 4, 5], "game_pk": [10, 20, 30, 40, 50],
        "A": [0.8, 0.7, 0.99, 0.6, 0.5], "B": [0.6, 0.9, np.nan, 0.4, 0.7],
        "outcome": ["hit", "no_hit", "hit", "no_hit", "hit"], "pool": [True] * 5,
    })


def test_g_is_earlier_minus_later_on_the_common_finite_pool_never_native_winners():
    g = core.paired_reduction(_pair_frame(), "A", "B", "pool", n_resamples=200)
    # common pool drops row 2 (B missing): d1 A->row0 hit, B->row1 no_hit; d2 A->row3 no_hit, B->row4 hit
    assert g["n_common_dates"] == 2 and g["estimate"] == 0.0
    df = _pair_frame()
    df.loc[3, "outcome"] = "hit"          # d2: both hit -> A rate 1.0, B rate 0.5 -> G = +0.5
    assert core.paired_reduction(df, "A", "B", "pool", n_resamples=200)["estimate"] == 0.5


def test_a_unilateral_unknown_winner_drops_the_date_from_the_paired_contrast_and_is_counted():
    df = _pair_frame()
    df.loc[0, "outcome"] = "unknown"
    g = core.paired_reduction(df, "A", "B", "pool", n_resamples=200)
    assert g["n_common_dates"] == 1 and g["excluded"]["A"] == {"unknown": 1}


def test_bridge_sign_is_negated_with_interval_flipped():
    assert core.from_bridge_b_minus_a(-0.1, (-0.3, 0.05)) == (0.1, (-0.05, 0.3))


def test_half_split_puts_the_median_date_first_and_resamples_each_half_at_its_own_size():
    dates = [f"2026-07-{d:02d}" for d in range(1, 8)]          # median 07-04 goes to the first half
    first, second = core.half_split(dates)
    assert first == dates[:4] and second == dates[4:]
    per = pd.DataFrame({"v": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]}, index=dates)
    out = core.half_contrast_bootstrap(per, first, second, {"d": lambda a, b: b["v"].mean() - a["v"].mean()},
                                       n_resamples=300, seed=1)
    assert out["d"]["lo"] == 1.0 and out["d"]["hi"] == 1.0 and out["d"]["n_failed"] == 0


def test_label_requires_every_draw_defined_and_an_interval_wholly_on_one_side():
    full = {"lo": 0.01, "hi": 0.2, "n_ok": 10_000, "n_failed": 0}
    assert core.directional_flag(full) == "consistent"
    assert core.directional_flag({**full, "lo": -0.01}) == "undetermined"
    assert core.directional_flag({"lo": -0.3, "hi": -0.1, "n_ok": 10_000, "n_failed": 0}) == "contradicting"
    assert core.directional_flag({**full, "n_ok": 9_999, "n_failed": 1}) == "unavailable"


def test_reproduction_class_follows_the_runner_precedence_on_serialized_scores():
    row = {"C_served": 0.7123456789012, "C_served_wblank": 0.70, "D": core.serialized(0.7123456789012)}
    assert core.repro_class(row) == "exact_final_feed"
    row2 = {"C_served": 0.71, "C_served_wblank": 0.7023, "D": core.serialized(0.7023)}
    assert core.repro_class(row2) == "inferred_weather_absent_at_serve"
    assert core.repro_class({"C_served": 0.71, "C_served_wblank": 0.72, "D": 0.70}) == "unexplained"
    assert core.repro_class({"C_served": float("nan"), "C_served_wblank": 0.72, "D": 0.70}) == "not_reproducible"


def test_recalibration_fit_clips_p_before_the_logit_and_reports_invalid_fits():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.55, 0.9, 3000)
    y = (rng.uniform(size=3000) < p).astype(int)
    fit = core.recalibration_fit(p, y)
    assert fit["valid"] and abs(fit["intercept"]) < 0.4 and abs(fit["slope"] - 1) < 0.4
    sep = core.recalibration_fit(np.array([0.2, 0.2, 0.2, 0.8, 0.8, 0.8]), np.array([0, 0, 0, 1, 1, 1]))
    assert not sep["valid"] and sep["reason"]
    const = core.recalibration_fit(np.full(6, 0.7), np.array([0, 1, 0, 1, 1, 0]))
    assert not const["valid"] and const["reason"] == "rank_deficient"
    one_class = core.recalibration_fit(np.array([0.6, 0.7, 0.8]), np.array([1, 1, 1]))
    assert not one_class["valid"] and one_class["reason"] == "one_class"
    edge = core.recalibration_fit(np.array([0.0, 1.0, 1.0, 0.5, 0.6] * 50), np.array([0, 1, 1, 0, 1] * 50))
    assert edge["clipped_low"] == 50 and edge["clipped_high"] == 100


def test_drag_date_is_available_only_when_every_contributing_venue_has_an_exact_value():
    cands = pd.DataFrame({"date": ["d1", "d1", "d1", "d2", "d2"], "venue_id": [1, 1, 2, 1, 3]})
    drag = pd.DataFrame({"venue_id": [1, 2, 1], "date": ["d1", "d1", "d2"], "park_drag_delta": [0.01, 0.03, 0.02]})
    out = core.slate_drag(cands, drag)
    assert math.isclose(out.loc["d1", "drag_mean"], 0.02)       # unique venues, equal weight: (0.01 + 0.03) / 2
    assert out.loc["d1", "complete"] and not out.loc["d2", "complete"] and math.isnan(out.loc["d2", "drag_mean"])
    assert out.loc["d2", "missing_venues"] == 1


def test_main_runs_end_to_end_on_synthetic_inputs(tmp_path, monkeypatch):
    import json
    from scripts.audit.w13_tests import run
    monkeypatch.setattr(run, "x29_gate", lambda: "e" * 40)
    rng = np.random.default_rng(9)
    dates = [f"2026-07-{d:02d}" for d in range(1, 21)]
    rows, meta = [], []
    data = tmp_path / "data"
    (data / "raw" / "2026").mkdir(parents=True)
    for k, d in enumerate(dates):
        (data / "picks" / d).mkdir(parents=True)
        meta.append({"date": d, "action": "skip" if k % 7 == 6 else "single", "n_rows": 8 + k % 3})
        for j in range(8):
            gpk = 5000 + 10 * k + j
            p = float(rng.uniform(0.55, 0.85))
            out = "hit" if rng.uniform() < p else "no_hit"
            rows.append({"date": d, "row_order": j, "batter_id": 100 + j, "game_pk": gpk, "projected": j % 2 == 0,
                         "D": core.serialized(p), "C_served": p, "C_served_wblank": p, "C_frozen": p - 0.01,
                         "A26": min(p + 0.05, 0.99), "A26_count": p + 0.02, "B26": p, "n_pa": 4, "n_pas_actual": 4,
                         "outcome": out, "sel_state": "selection_consistent", "pool_verified": True,
                         "pool_surrogate": True, "pool_all": True})
            (data / "raw" / "2026" / f"{gpk}.json").write_text(json.dumps({"gameData": {"venue": {"id": 1 + j % 4}}}))
        top = max((r for r in rows if r["date"] == d), key=lambda r: r["D"])
        dec = {"action": meta[-1]["action"], "primary": {"batter_id": top["batter_id"], "game_pk": top["game_pk"],
                                                         "p_game_hit": top["D"]}}
        (data / "picks" / d / "decision.json").write_text(json.dumps(dec))
    w12 = tmp_path / "w12"; w12.mkdir()
    pd.DataFrame(rows).to_parquet(w12 / "table.parquet")
    pairs = [{"pair": [a, b], "exclusions": {}, "top1_diff_b_minus_a": 0.0, "top1_diff_ci95": [-0.1, 0.1],
              "paired_top1_dates": 20, "rank1_changed_dates": 0}
             for a, b in (("A26", "A26_count"), ("A26_count", "B26"), ("B26", "C_frozen"), ("C_frozen", "C_served"),
                          ("C_served", "D"))]
    surf = [{"surface": s, "brier": 0.2, "log_loss": 0.6} for s in run.SURFACES]
    (w12 / "summary.json").write_text(json.dumps({"registered_dates": dates, "day_meta": meta, "metrics": {
        "selection_consistent": {"pool_verified": {"pairs": pairs, "surfaces": surf}}}}))
    import hashlib
    decs = {d: hashlib.sha256((data / "picks" / d / "decision.json").read_bytes()).hexdigest() for d in dates}
    (w12 / "manifest.json").write_text(json.dumps({"decisions": decs, "picks": {}}))
    files = {n: hashlib.sha256((w12 / n).read_bytes()).hexdigest() for n in ("table.parquet", "summary.json", "manifest.json")}
    (w12 / "ACCEPTED.json").write_text(json.dumps({"files": files, "memo": "synthetic"}))
    drag = pd.DataFrame([{"venue_id": v, "date": d, "park_drag_delta": 0.001 * (k % 5)}
                         for k, d in enumerate(dates) for v in (1, 2, 3, 4)])
    drag.to_csv(tmp_path / "park_drag_export.csv", index=False)
    (tmp_path / "park_drag_manifest.json").write_text("{}")
    assert run.main(["--w12-run", str(w12), "--data-root", str(data), "--drag", str(tmp_path / "park_drag_export.csv"),
                     "--out", str(tmp_path / "out"), "--n-resamples", "40"]) == 0
    res = json.loads(next((tmp_path / "out").glob("*/results.json")).read_text())
    assert res["E2"]["primary"]["classes"] == {"exact_final_feed": 160} and res["E2"]["flag"] == "numerical_parity_conditional"
    assert (next((tmp_path / "out").glob("*/per_test.parquet"))).exists()
    assert set(res["sensitivities"]) == {"all_dates/pool_verified", "selection_consistent/pool_surrogate",
                                         "selection_consistent/pool_all"}
    assert res["sensitivities"]["all_dates/pool_verified"]["E1"]["flag"] == "sensitivity (no flag)"
    assert res["E3"]["full_explanation"].startswith("not testable")
    assert res["E8"]["n"] > 0 and "skip" in res["selections"]["actions"]
    assert res["E6"]["support"]["dates_complete"] == 20
    for e in ("E1", "E4", "E5", "E6", "E7"):
        assert "flag" in res[e]


def _e46_frame():
    rows = []
    for k, d in enumerate([f"2026-07-{x:02d}" for x in range(1, 9)]):
        for j in range(4):
            p = 0.6 + 0.05 * j
            exact = j < 2                                   # rows 0-1 reproduce exactly; rows 2-3 are unexplained
            rows.append({"date": d, "row_order": j, "batter_id": j, "game_pk": 100 * k + j,
                         "D": core.serialized(p), "C_served": p if exact else p + 0.1, "C_served_wblank": p + 0.2,
                         "C_frozen": p, "B26": p, "outcome": "hit" if (j + k) % 2 else "no_hit",
                         "sel_state": "selection_consistent", "pool_verified": True, "projected": False})
    return pd.DataFrame(rows)


def test_e4_flag_uses_only_reproduced_rows_and_each_halfs_supported_dates():
    from scripts.audit.w13_tests import run
    df = _e46_frame()
    df.loc[(df["date"] >= "2026-07-05") & (df["row_order"] >= 2), "outcome"] = "no_hit"   # change only unexplained rows
    dates = sorted(df["date"].unique())
    res = run.e4(df, df, dates, n=200)
    assert res["reproduced_support"]["first"]["dates"] == 4 and res["reproduced_support"]["second"]["dates"] == 4
    assert res["reproduced_support"]["first"]["classes"] == {"exact_final_feed": 8}
    full = res["estimates"]["resid_C_frozen"]
    repro = res["estimates"]["resid_C_frozen_reproduced"]
    assert full != repro                     # the full-population contrast is a separate description
    none = df.assign(C_served=df["C_served"] + 0.3)            # nothing reproduces
    assert run.e4(none, none, dates, n=50)["flag"] == "unavailable"


def test_e6_venue_membership_is_frozen_before_outcome_filtering():
    from scripts.audit.w13_tests import run
    df = _e46_frame()
    venues = {pk: 1 + (pk % 100) for pk in df["game_pk"]}          # venue = row index + 1
    df.loc[df["row_order"] == 3, "outcome"] = "unknown"             # venue 4 has no known outcome on any date
    drag = pd.DataFrame([{"venue_id": v, "date": d, "park_drag_delta": 0.01 * v}
                         for d in df["date"].unique() for v in (1, 2, 3)])   # venue 4 lacks drag
    res = run.e6(df, venues, drag, n=50)
    assert res["support"]["dates_complete"] == 0 and res["flag"] == "unavailable"   # venue 4 is in the pool, so no date is complete


# --- code review r1 (F6, F10) ----------------------------------------------------------------------------------------

def test_out_of_range_scores_are_excluded_before_ranking():
    df = pd.DataFrame({"date": ["d1", "d1"], "row_order": [0, 1], "batter_id": [1, 2], "game_pk": [10, 20],
                       "A": [1.2, 0.7], "B": [0.4, 0.8], "outcome": ["hit", "no_hit"], "pool": [True, True]})
    g = core.paired_reduction(df, "A", "B", "pool", n_resamples=50)
    assert g["estimate"] == 0.0 and g["invalid_scores"] == {"A": 1, "B": 0}
    assert core.repro_class({"C_served": 1.4, "C_served_wblank": 0.7, "D": 1.4}) == "not_reproducible"


def test_duplicate_drag_keys_collapse_when_identical_and_void_the_venue_when_conflicting():
    cands = pd.DataFrame({"date": ["d1", "d1"], "venue_id": [1, 2]})
    base = [{"venue_id": 1, "date": "d1", "park_drag_delta": 0.01}, {"venue_id": 2, "date": "d1", "park_drag_delta": 0.03}]
    same = pd.DataFrame(base + [{"venue_id": 1, "date": "d1", "park_drag_delta": 0.01}])
    assert math.isclose(core.slate_drag(cands, same).loc["d1", "drag_mean"], 0.02)
    conflict = pd.DataFrame(base + [{"venue_id": 1, "date": "d1", "park_drag_delta": 0.09}])
    out = core.slate_drag(cands, conflict)
    assert not out.loc["d1", "complete"] and out.loc["d1", "conflicting_keys"] == 1


# --- code review r1 (F3, F4, F5, F7, F8) -----------------------------------------------------------------------------

def _tab(rows):
    base = {"sel_state": "selection_consistent", "pool_verified": True, "pool_surrogate": True, "pool_all": True,
            "projected": False, "n_pa": 4, "n_pas_actual": 4, "C_frozen": 0.7, "A26": 0.75, "A26_count": 0.72,
            "B26": 0.7, "row_order": 0}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_e2_flag_uses_only_primary_rows_and_a_subthreshold_gap_is_not_parity():
    from scripts.audit.w13_tests import run
    rows = [{"date": "d1", "batter_id": 1, "game_pk": 1, "D": core.serialized(0.7), "C_served": 0.8,
             "C_served_wblank": 0.81, "outcome": "hit"}]
    rows += [{"date": "d1", "batter_id": 10 + i, "game_pk": 10 + i, "D": core.serialized(0.6), "C_served": 0.6,
              "C_served_wblank": 0.6, "outcome": "hit", "pool_verified": False} for i in range(99)]
    res = run.e2(_tab(rows), {})
    assert res["flag"] == "unexplained_reconstruction_gap" and res["primary"]["unexplained_share"] == 1.0
    mixed = [{"date": "d1", "batter_id": i, "game_pk": i, "D": core.serialized(0.6), "C_served": 0.6,
              "C_served_wblank": 0.6, "outcome": "hit"} for i in range(99)] + rows[:1]
    assert run.e2(_tab(mixed), {})["flag"] == "subthreshold_unexplained"


def test_e8_uses_the_primary_cohort_only():
    from scripts.audit.w13_tests import run
    rows = [{"date": f"d{i}", "batter_id": 1, "game_pk": i, "D": 0.9, "C_served": 0.9, "C_served_wblank": 0.9,
             "outcome": "hit" if i == 0 else "no_hit", "pool_verified": i == 0} for i in range(5)]
    tab = _tab(rows)
    sels = {f"d{i}": {"selected": (1, i), "p": 0.9, "action": "single"} for i in range(5)}
    prim = run.stratum(tab, *run.PRIMARY)
    res = run.e8(prim, sels, n=500)
    assert res["n"] == 1 and res["observed_hits"] == 1 and res["flag"] == "null_compatible"
    assert run.e8(tab, sels, n=500, primary=False)["flag"] == "sensitivity (no flag)"


def test_selection_bytes_are_verified_once_and_unmanifested_or_missing_decisions_block_fallback(tmp_path):
    import hashlib
    import json
    import pytest
    from scripts.audit.w13_tests import run
    for d in ("d1", "d2", "d3"):
        (tmp_path / "picks" / d).mkdir(parents=True)
    dec = {"action": "single", "primary": {"batter_id": 1, "game_pk": 10, "p_game_hit": 0.8}}
    pick = {"pick": {"batter_id": 2, "game_pk": 20, "p_game_hit": 0.7}}
    for d in ("d1", "d2"):
        (tmp_path / "picks" / d / "decision.json").write_text(json.dumps(dec))
    for d in ("d1", "d2", "d3"):
        (tmp_path / "picks" / f"{d}.json").write_text(json.dumps(pick))
    h = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    manifest = {"decisions": {"d1": h(tmp_path / "picks/d1/decision.json"), "d3": "0" * 64},
                "picks": {d: h(tmp_path / f"picks/{d}.json") for d in ("d1", "d2", "d3")}}
    sels, files = run.load_selections(tmp_path, ["d1", "d2", "d3"], manifest)
    assert sels["d1"]["selected"] == (1, 10) and sels["d1"]["decision_pick_conflict"] is True
    assert sels["d2"]["action"] == "unavailable" and "decision_unmanifested" in sels["d2"]["evidence_status"]
    assert sels["d3"]["action"] == "unavailable" and "decision_missing" in sels["d3"]["evidence_status"]
    manifest["decisions"]["d1"] = "f" * 64
    with pytest.raises(SystemExit):
        run.load_selections(tmp_path, ["d1"], manifest)


def test_registered_inventory_violations_stop_the_run():
    import pytest
    from scripts.audit.w13_tests import run
    tab = _tab([{"date": "d1", "batter_id": 1, "game_pk": 1, "D": 0.7, "C_served": 0.7, "C_served_wblank": 0.7,
                 "outcome": "hit"}])
    good = {"registered_dates": ["d1"], "day_meta": [{"date": "d1"}]}
    assert run.validate_inventory(tab, good) == ["d1"]
    with pytest.raises(SystemExit):
        run.validate_inventory(pd.concat([tab, tab.assign(date="d9")]), good)
    with pytest.raises(SystemExit):
        run.validate_inventory(pd.concat([tab, tab]), good)
    with pytest.raises(SystemExit):
        run.validate_inventory(tab, {"registered_dates": ["d1"], "day_meta": [{"date": "d2"}]})


def test_empty_primary_support_reports_unavailable_components_instead_of_crashing():
    from scripts.audit.w13_tests import run
    tab = _tab([{"date": "d1", "batter_id": 1, "game_pk": 1, "D": 0.7, "C_served": 0.7, "C_served_wblank": 0.7,
                 "outcome": "hit", "pool_verified": False},
                {"date": "d2", "batter_id": 2, "game_pk": 2, "D": 0.7, "C_served": 0.7, "C_served_wblank": 0.7,
                 "outcome": "no_hit", "pool_verified": False}])
    prim = run.stratum(tab, *run.PRIMARY)
    assert prim.empty
    assert run.e4(prim, tab, ["d1", "d2"], n=20)["flag"] == "unavailable"
    assert run.e6(prim, {}, pd.DataFrame(columns=["venue_id", "date", "park_drag_delta"]), n=20)["flag"] == "unavailable"
    assert run.e7(prim, tab, [{"date": "d1", "action": "single", "n_rows": 1}], n=20)["flag"] == "descriptive"
    assert run.e8(prim, {"d1": {"selected": (1, 1), "p": 0.7, "action": "single"}}, n=20)["flag"] == "not_testable"
    assert run.e5(prim, {}, n=20)["flag"] == "not_testable"
    assert run.e1(prim, "pool_verified", {}, n=20)["flag"] == "unavailable"


# --- code review r2 (N2, N3) -----------------------------------------------------------------------------------------

def test_e7_probability_means_use_only_valid_scores_with_counts():
    from scripts.audit.w13_tests import run
    tab = _tab([{"date": "d1", "batter_id": 1, "game_pk": 1, "D": 1.2, "C_served": 0.7, "C_served_wblank": 0.7,
                 "outcome": "hit"},
                {"date": "d2", "batter_id": 2, "game_pk": 2, "D": 0.8, "C_served": 0.7, "C_served_wblank": 0.7,
                 "outcome": "no_hit"},
                {"date": "d2", "batter_id": 3, "game_pk": 3, "D": 0.6, "C_served": 0.7, "C_served_wblank": 0.7,
                 "outcome": "no_pa"}])
    meta = [{"date": "d1", "action": "single", "n_rows": 1}, {"date": "d2", "action": "single", "n_rows": 2}]
    res = run.e7(run.stratum(tab, *run.PRIMARY), tab, meta, n=20)
    conf = res["strata_primary"]["confirmed"]
    assert math.isclose(conf["mean_p"], 0.7) and conf["p_counts"] == {"valid": 2, "missing": 0, "invalid": 1}
    act = res["by_action_all_dates"]["single"]
    assert math.isclose(act["mean_p"], 0.7) and act["p_counts"]["invalid"] == 1
    bad = tab.assign(D=1.5)
    assert run.e7(run.stratum(bad, *run.PRIMARY), bad, meta, n=20)["strata_primary"]["confirmed"]["mean_p"] is None


def test_per_date_selection_dispositions_are_saved_with_the_frozen_metadata_and_discrepancies():
    from scripts.audit.w13_tests import run
    sels = {"d1": {"selected": None, "declined": (1, 10), "action": "skip", "source": "decision", "p": None,
                   "evidence_status": [], "decision_pick_conflict": False},
            "d2": {"selected": (2, 20), "declined": None, "action": "single", "source": "decision", "p": 0.8,
                   "evidence_status": [], "decision_pick_conflict": False},
            "d3": {"selected": None, "declined": None, "action": "unavailable", "source": None, "p": None,
                   "evidence_status": ["decision_missing"], "decision_pick_conflict": False}}
    meta = [{"date": "d1", "sel_state": "selection_consistent", "sel_source": "decision", "sel_conflict": False},
            {"date": "d2", "sel_state": "selection_consistent", "sel_source": "decision", "sel_conflict": False},
            {"date": "d3", "sel_state": "selection_consistent", "sel_source": "decision", "sel_conflict": False}]
    recs = run.selection_records(sels, meta)
    by = {r["date"]: r for r in recs}
    assert by["d1"]["declined"] == [1, 10] and "frozen_consistent_but_no_genuine_selection" in by["d1"]["discrepancies"]
    assert by["d2"]["discrepancies"] == [] and by["d2"]["frozen_sel_state"] == "selection_consistent"
    assert "decision_evidence_unavailable" in by["d3"]["discrepancies"]
    import json
    assert json.loads(json.dumps(recs))[0]["declined"] == [1, 10]
