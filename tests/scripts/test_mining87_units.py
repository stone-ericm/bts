"""#87 adapter: production-led unit retention, independent masks and decomposition bins (review r1 findings 4, 6;
required changes 4 and 11)."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit.mining87 import consensus as c
from scripts.audit.mining87 import surfaces as s
from scripts.audit.mining87 import units as u
from scripts.leaderboard_mechanism_mining import DECOMPOSITION_VARIABLES


def prod(date, slot, batter, settlement="hit", **kw):
    row = {"date": date, "pick_number": slot, "production_slot": "primary" if slot == 1 else "double_down",
           "production_selection_id": f"{date}|{slot}|{batter}", "production_batter_id": batter,
           "production_batter_name": f"P{batter}", "production_game_pk": 1000 + batter,
           "production_p_game_hit": 0.77, "production_projected_lineup": False,
           "production_predicted_at": f"{date}T15:00:00.000000Z", "production_locked_at": f"{date}T16:00:00.000000Z",
           "production_lock_basis": "decision:committed_evidenced", "production_regime": "post_bpm",
           "production_game_eligibility": "unknown", "production_settlement": settlement,
           "production_settlement_reason": "contest_graded",
           "production_resolved": settlement in ("hit", "not_hit"),
           "production_hit": {"hit": 1, "not_hit": 0}.get(settlement),
           "production_batter_skill_prior_pa": 350, "production_batter_skill_quartile": 2,
           "production_weather_temp": 72.0, "production_is_indoor": False, "production_is_park_driven": False}
    row.update(kw)
    return row


def production(rows):
    frame = pd.DataFrame(rows)
    frame["production_hit"] = frame["production_hit"].astype("Int64")
    return frame


def obs(user, date, slot, batter, result="hit", unit=None):
    return {"username": user, "pick_date": date, "pick_number": slot, "batter_id": batter, "batter_name": f"B{batter}",
            "unit_id": unit or 5000 + batter, "result": result, "captured_at": pd.Timestamp("2026-05-02T12:00:00")}


def consensus(rows, fixed_users):
    latest = pd.DataFrame(rows)
    fixed, _ = c.consensus_table(latest, users=fixed_users, cohort="fixed_cohort")
    allt, _ = c.consensus_table(latest, users=None, cohort="all_tracked")
    return pd.concat([fixed, allt], ignore_index=True)


def test_units_are_production_led_and_absent_consensus_is_retained_as_unavailable():
    p = production([prod("2026-04-01", 1, 11), prod("2026-04-01", 2, 12), prod("2026-04-02", 1, 13)])
    cons = consensus([obs("a", "2026-04-01", 1, 10), obs("z", "2026-04-02", 1, 13)], fixed_users={"a"})
    units = u.build_units(p, cons, surfaces={}, unit_games=None)
    assert len(units) == 6 and set(units["cohort"]) == {"fixed_cohort", "all_tracked"}
    fixed = units[units["cohort"] == "fixed_cohort"].set_index(["date", "pick_number"])
    assert fixed.loc[("2026-04-01", 1), "agreement_state"] == "different_batter"
    assert fixed.loc[("2026-04-01", 2), "agreement_state"] == "consensus_unavailable"
    assert fixed.loc[("2026-04-01", 2), "consensus_status"] == "no_public_votes"
    assert fixed.loc[("2026-04-02", 1), "agreement_state"] == "consensus_unavailable"
    allt = units[units["cohort"] == "all_tracked"].set_index(["date", "pick_number"])
    assert allt.loc[("2026-04-02", 1), "agreement_state"] == "same_batter"
    assert list(units.columns[-len(DECOMPOSITION_VARIABLES):]) == DECOMPOSITION_VARIABLES


def test_resolution_masks_are_independent_and_void_or_unknown_never_enter_deltas():
    p = production([prod("2026-04-01", 1, 11, settlement="void"), prod("2026-04-02", 1, 12, settlement="not_hit"),
                    prod("2026-04-03", 1, 13, settlement="hit"), prod("2026-04-04", 1, 14, settlement="unknown")])
    cons = consensus([obs("a", "2026-04-01", 1, 10), obs("a", "2026-04-02", 1, 20),
                      obs("a", "2026-04-03", 1, 30, result=""), obs("a", "2026-04-04", 1, 40)], fixed_users={"a"})
    units = u.build_units(p, cons, surfaces={}, unit_games=None)
    f = units[units["cohort"] == "fixed_cohort"].set_index("date")
    assert list(f["production_resolved"]) == [False, True, True, False]
    assert list(f["consensus_resolved"]) == [True, True, False, True]
    assert list(f["joint_resolved"]) == [False, True, False, False]
    assert f["delta"].isna().tolist() == [True, False, True, True] and f.loc["2026-04-02", "delta"] == 1
    assert list(f["resolved_disagreement"]) == [False, True, False, False]


def test_no_dd_unit_is_fabricated_when_production_made_no_double_down():
    p = production([prod("2026-04-01", 1, 11)])
    cons = consensus([obs("a", "2026-04-01", 1, 10), obs("a", "2026-04-01", 2, 20)], fixed_users={"a"})
    units = u.build_units(p, cons, surfaces={}, unit_games=None)
    assert set(units["pick_number"]) == {1}


def test_missing_surfaces_are_missing_not_off_top_n_and_admitted_rankings_are_complete():
    p = production([prod("2026-06-20", 1, 11), prod("2026-06-21", 1, 12), prod("2026-06-22", 1, 13)])
    cons = consensus([obs("a", "2026-06-20", 1, 3), obs("a", "2026-06-21", 1, 99, unit=7),
                      obs("a", "2026-06-22", 1, 3)], fixed_users={"a"})
    rows = [{"rank": 1, "batter_id": 1, "game_pk": 100, "p_game_hit": 0.8},
            {"rank": 2, "batter_id": 3, "game_pk": 102, "p_game_hit": 0.79},
            {"rank": 3, "batter_id": 3, "game_pk": 103, "p_game_hit": 0.75}]
    surfaces = {"2026-06-21": {"admitted": True, "reason": "witness_admitted", "batters": s.batter_table(rows)},
                "2026-06-22": {"admitted": True, "reason": "witness_admitted", "batters": s.batter_table(rows)},
                "2026-06-20": {"admitted": False, "reason": "no_independent_witness",
                               "batters": s.batter_table(rows)}}
    units = u.build_units(p, cons, surfaces=surfaces, unit_games=None)
    f = units[units["cohort"] == "fixed_cohort"].set_index("date")
    assert f.loc["2026-06-20", "consensus_model_rank_bin"] == "missing_surface"
    assert pd.isna(f.loc["2026-06-20", "consensus_model_rank"])                    # unadmitted: raw rank null
    assert f.loc["2026-06-21", "consensus_model_rank_bin"] == "off_top10"           # absent from a full ranking
    assert f.loc["2026-06-22", "consensus_model_rank_bin"] == "rank2"
    assert f.loc["2026-06-22", "consensus_model_probability_bin"] == "missing_surface"
    assert f.loc["2026-06-22", "consensus_model_probability_status"] == "ambiguous_multiple_rows"
    assert list(f["surface_admitted"]) == [False, True, True]
    no_surface = u.build_units(p, cons, surfaces={}, unit_games=None)
    assert set(no_surface["consensus_model_rank_bin"]) == {"missing_surface"}
    assert set(no_surface["consensus_model_probability_bin"]) == {"missing_surface"}


def test_unit_validation_rejects_duplicate_slots_and_non_binary_hits():
    p = production([prod("2026-04-01", 1, 11), prod("2026-04-01", 1, 12)])
    with pytest.raises(ValueError, match="duplicate"):
        u.build_units(p, consensus([], fixed_users=set()), surfaces={}, unit_games=None)
    bad = production([prod("2026-04-01", 1, 11)])
    bad["production_hit"] = pd.array([2], dtype="Int64")
    with pytest.raises(ValueError, match="binary"):
        u.build_units(bad, consensus([], fixed_users=set()), surfaces={}, unit_games=None)


def test_bins_follow_the_protocol_definitions():
    p = production([prod("2026-04-01", 1, 11, production_p_game_hit=0.80, production_batter_skill_prior_pa=99,
                         production_weather_temp=None, production_is_indoor=None,
                         production_projected_lineup=None, production_regime=None)])
    units = u.build_units(p, consensus([obs("a", "2026-04-01", 1, 10)], {"a"}), surfaces={}, unit_games=None)
    row = units[units["cohort"] == "fixed_cohort"].iloc[0]
    assert row["production_p_game_hit_bin"] == ">=0.80" and row["production_batter_skill_prior_pa_bin"] == "<100"
    assert row["production_weather_temp_bin"] == "indoor_or_missing" and row["production_is_indoor"] == "missing"
    assert row["production_projected_lineup"] == "missing" and row["production_regime"] == "missing"
    assert row["consensus_pick_share_bin"] == ">=0.25" and row["pick_number"] == 1
