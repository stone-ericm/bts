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
    (w12 / "manifest.json").write_text(json.dumps({"decisions": {}, "picks": {}}))
    drag = pd.DataFrame([{"venue_id": v, "date": d, "park_drag_delta": 0.001 * (k % 5)}
                         for k, d in enumerate(dates) for v in (1, 2, 3, 4)])
    drag.to_csv(tmp_path / "park_drag_export.csv", index=False)
    assert run.main(["--w12-run", str(w12), "--data-root", str(data), "--drag", str(tmp_path / "park_drag_export.csv"),
                     "--out", str(tmp_path / "out"), "--n-resamples", "40"]) == 0
    res = json.loads(next((tmp_path / "out").glob("*/results.json")).read_text())
    assert res["E2"]["classes"] == {"exact_final_feed": 160} and res["E2"]["flag"] == "numerical_parity_conditional"
    assert res["E3"]["full_explanation"].startswith("not testable")
    assert res["E8"]["n"] > 0 and "skip" in res["selections"]
    assert res["E6"]["dates_complete"] == 20
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
    assert res["dates_complete"] == 0 and res["flag"] == "unavailable"   # venue 4 is in the pool, so no date is complete
