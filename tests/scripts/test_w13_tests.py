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
    edge = core.recalibration_fit(np.array([0.0, 1.0, 0.5, 0.6] * 50), np.array([0, 1, 0, 1] * 50))
    assert edge["boundary_rows"] == 100


def test_drag_date_is_available_only_when_every_contributing_venue_has_an_exact_value():
    cands = pd.DataFrame({"date": ["d1", "d1", "d1", "d2", "d2"], "venue_id": [1, 1, 2, 1, 3]})
    drag = pd.DataFrame({"venue_id": [1, 2, 1], "date": ["d1", "d1", "d2"], "park_drag_delta": [0.01, 0.03, 0.02]})
    out = core.slate_drag(cands, drag)
    assert math.isclose(out.loc["d1", "drag_mean"], 0.02)       # unique venues, equal weight: (0.01 + 0.03) / 2
    assert out.loc["d1", "complete"] and not out.loc["d2", "complete"] and math.isnan(out.loc["d2", "drag_mean"])
    assert out.loc["d2", "missing_venues"] == 1
