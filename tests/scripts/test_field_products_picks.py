"""W2.1/W2.2 field products: the pick data contract (design 2026-10-04 rev 2, "Data contract"; code review r1 F4, F7
and the conformance notes). Synthetic only."""
from __future__ import annotations

import gzip
import json
from datetime import date, datetime

import pandas as pd
import pytest

from bts.leaderboard.models import PickRow
from bts.leaderboard.storage import append_user_picks
from scripts.audit.field_products import picks as P


def _row(round_id, pick_number, *, cap, result="hit", unit=10, player=20, streak=1, pick_date=None,
         at_bats=4, hits=1, team="NYM", ha="home"):
    return PickRow(captured_at=cap, round_id=round_id, pick_date=pick_date or date(2026, 5, round_id % 28 + 1),
                   pick_number=pick_number, unit_id=unit, bts_player_id=player, result=result, at_bats=at_bats,
                   hits=hits, streak_after=streak, batter_id=player + 1000, batter_name="B", batter_team=team,
                   opponent_team="ATL", home_or_away=ha)


def _obs(rows, user_id=7, file="u.parquet", rank=0):
    df = pd.DataFrame([r.model_dump() for r in rows])
    return P.with_provenance(df, user_id=user_id, source="daily", file=file, file_rank=rank)


def _wit(obs):
    """A verified-raw-response witness for each round's latest batch (stand-in for a final-grab raw profile)."""
    out = {}
    for (u, r), g in obs.groupby(["user_id", "round_id"]):
        last = g[g["captured_at"] == g["captured_at"].max()]
        out[(int(u), int(r))] = {"file": last["file"].iloc[-1], "captured_at": last["captured_at"].iloc[-1],
                                 "slot_numbers": frozenset(int(x) for x in last["pick_number"])}
    return out


T1, T2, T3 = datetime(2026, 5, 2, 11), datetime(2026, 5, 3, 11), datetime(2026, 5, 4, 11)


def test_read_observations_keeps_every_appended_row_and_both_dd_slots(tmp_path):
    """The failure prevented: latest_per_pick_date keeps one row per date and drops a two-hit DD leg."""
    p = tmp_path / "alice.parquet"
    append_user_picks(p, [_row(1, 1, cap=T1, player=20), _row(1, 2, cap=T1, player=21)])
    append_user_picks(p, [_row(1, 1, cap=T2, player=20), _row(1, 2, cap=T2, player=21)])
    obs = P.read_observations([p], user_id=7, source="daily")
    assert len(obs) == 4 and set(obs["file"]) == {"alice.parquet"} and list(obs["file_row"]) == [0, 1, 2, 3]
    res = P.resolve(obs, witness=_wit(obs))
    assert len(res.slots) == 2 and set(res.slots["pick_number"]) == {1, 2}
    rnd = res.rounds.iloc[0]
    assert rnd["complete"] and rnd["is_dd"] and rnd["n_slots"] == 2
    assert res.slots["usable"].all() and res.slots["hit"].sum() == 2


def test_read_observations_parses_supplied_bytes_not_the_disk(tmp_path):
    p = tmp_path / "alice.parquet"
    append_user_picks(p, [_row(1, 1, cap=T1, result="hit")])
    frozen = p.read_bytes()
    p.unlink()
    append_user_picks(p, [_row(1, 1, cap=T1, result="not_hit")])
    obs = P.read_observations([p], user_id=7, source="daily", read=lambda q: frozen)
    assert list(obs["result"]) == ["hit"]


def test_resolved_slots_keep_their_representative_file_and_row():
    a = _obs([_row(1, 1, cap=T1, result="")], file="a.parquet", rank=0)
    b = _obs([_row(1, 1, cap=T2, result="hit"), _row(2, 1, cap=T2)], file="b.parquet", rank=1)
    s = P.resolve(pd.concat([a, b], ignore_index=True)).slots.set_index("round_id")
    assert (s.loc[1, "rep_file"], s.loc[1, "rep_file_row"], s.loc[1, "rep_source"]) == ("b.parquet", 0, "daily")
    assert s.loc[2, "rep_file_row"] == 1 and s.loc[1, "rep_captured_at"] == pd.Timestamp(T2)


def test_without_a_completeness_witness_no_round_is_complete_but_grades_stay_usable():
    obs = _obs([_row(1, 1, cap=T1), _row(1, 2, cap=T1, player=21)])
    res = P.resolve(obs)
    r = res.rounds.iloc[0]
    assert not r["complete"] and not r["is_dd"] and r["incomplete_reason"] == "no_completeness_witness"
    assert res.slots["usable"].all()


def test_complementary_equal_time_files_are_not_unioned_into_a_complete_dd():
    """Review F4 probe (a): file a holds only the primary, file b only the second leg, same user/round/instant."""
    a = _obs([_row(1, 1, cap=T1)], file="a.parquet", rank=0)
    b = _obs([_row(1, 2, cap=T1, player=21)], file="b.parquet", rank=1)
    obs = pd.concat([a, b], ignore_index=True)
    w = {(7, 1): {"file": "b.parquet", "captured_at": pd.Timestamp(T1), "slot_numbers": frozenset({1, 2})}}
    r = P.resolve(obs, witness=w).rounds.iloc[0]
    assert not r["complete"] and not r["is_dd"] and r["incomplete_reason"] == "competing_equal_time_batches"


def test_identical_equal_time_batches_in_two_files_are_one_snapshot():
    a = _obs([_row(1, 1, cap=T1), _row(1, 2, cap=T1, player=21)], file="a.parquet", rank=0)
    b = _obs([_row(1, 1, cap=T1), _row(1, 2, cap=T1, player=21)], file="b.parquet", rank=1)
    obs = pd.concat([a, b], ignore_index=True)
    res = P.resolve(obs, witness=_wit(obs))
    assert res.rounds.iloc[0]["complete"] and res.log["exact_duplicate_rows"] == 2


def test_disappeared_pending_leg_without_a_witness_leaves_the_round_incomplete():
    """Review F4 probe (b): two pending legs, then a later stored timestamp with only a settled primary."""
    obs = _obs([_row(1, 1, cap=T1, result=""), _row(1, 2, cap=T1, result="", player=21),
                _row(1, 1, cap=T2, result="hit")])
    r = P.resolve(obs).rounds.iloc[0]
    assert not r["complete"] and not r["is_dd"] and r["deleted_legs"] == 1
    assert r["incomplete_reason"] == "leg_disappeared_unwitnessed"
    # a verified complete response at the later batch establishes the deletion: a known single-pick round
    r2 = P.resolve(obs, witness=_wit(obs)).rounds.iloc[0]
    assert r2["complete"] and not r2["is_dd"] and r2["deleted_legs"] == 1


def test_witness_for_another_batch_or_slot_set_does_not_complete_the_round():
    obs = _obs([_row(1, 1, cap=T1)])
    other_time = {(7, 1): {"file": "u.parquet", "captured_at": pd.Timestamp(T2), "slot_numbers": frozenset({1})}}
    other_slots = {(7, 1): {"file": "u.parquet", "captured_at": pd.Timestamp(T1), "slot_numbers": frozenset({1, 2})}}
    assert P.resolve(obs, witness=other_time).rounds.iloc[0]["incomplete_reason"] == "witness_mismatch"
    assert P.resolve(obs, witness=other_slots).rounds.iloc[0]["incomplete_reason"] == "witness_mismatch"


def test_witness_from_a_raw_profile_response_lists_each_rounds_slot_numbers():
    body = {"success": {"seasonBestStreak": 3, "activeStreak": 0, "accuracy": 50, "predictions": [
        {"roundId": 1, "streak": 1, "roundPredictions": [{"number": 1, "unitId": 5, "playerId": 3, "result": "hit"},
                                                         {"number": 2, "unitId": 5, "playerId": 4, "result": "hit"}]},
        {"roundId": 2, "streak": 0, "roundPredictions": [{"number": 1, "unitId": 5, "playerId": 3,
                                                          "result": "not_hit"}]},
        {"roundId": 3, "streak": 0, "roundPredictions": [{"number": 1, "unitId": 5, "playerId": 3, "result": "hit"},
                                                         {"number": 1, "unitId": 5, "playerId": 4, "result": "hit"}]}]}}
    w = P.raw_profile_witness(gzip.compress(json.dumps(body).encode()), user_id=7, file="7.parquet",
                              captured_at=pd.Timestamp(T1))
    assert w[(7, 1)]["slot_numbers"] == frozenset({1, 2}) and w[(7, 2)]["slot_numbers"] == frozenset({1})
    assert (7, 3) not in w                     # a repeated slot number is no witness of a complete round


def test_later_pending_observation_is_not_replaced_by_an_older_settled_label():
    obs = _obs([_row(1, 1, cap=T1, result="hit"), _row(1, 1, cap=T2, result="")])
    res = P.resolve(obs, witness=_wit(obs))
    s = res.slots.iloc[0]
    assert s["label"] == "" and not s["graded"] and not s["usable"] and s["later_unsettled_over_settled"]
    assert not res.rounds.iloc[0]["complete"] and res.rounds.iloc[0]["incomplete_reason"] == "unsettled_slot"
    assert res.log["later_unsettled_over_settled"] == 1


def test_only_exact_hit_and_not_hit_are_graded_and_at_bats_never_settle_a_slot():
    obs = _obs([_row(1, 1, cap=T1, result="", at_bats=4, hits=2), _row(2, 1, cap=T1, result="void"),
                _row(3, 1, cap=T1, result="Hit"), _row(4, 1, cap=T1, result="None"),
                _row(5, 1, cap=T1, result="not_hit", hits=0)])
    res = P.resolve(obs, witness=_wit(obs))
    g = res.slots.set_index("round_id")
    assert list(g["graded"]) == [False, False, False, False, True]
    assert res.log["label_counts"] == {"": 1, "void": 1, "Hit": 1, "None": 1, "not_hit": 1}
    r = res.rounds.set_index("round_id")
    assert r.loc[2, "complete"] and not r.loc[1, "complete"] and not r.loc[3, "complete"]


def test_changed_identity_is_recorded_and_the_latest_snapshot_wins():
    obs = _obs([_row(1, 1, cap=T1, result="", player=20), _row(1, 1, cap=T2, result="hit", player=30)])
    res = P.resolve(obs)
    s = res.slots.iloc[0]
    assert s["bts_player_id"] == 30 and s["identity_changed"] and not s["settled_identity_changed"]
    assert res.log["changed_identity_slots"] == 1 and s["n_obs"] == 2


def test_settled_identity_change_is_flagged_as_an_ownership_signal():
    obs = _obs([_row(1, 1, cap=T1, result="hit", player=20), _row(1, 1, cap=T2, result="hit", player=30)])
    res = P.resolve(obs)
    assert res.slots.iloc[0]["settled_identity_changed"] and res.log["settled_identity_changed"] == 1
    assert P.ownership_conflict_users(res) == {7}


def test_equal_time_conflict_is_recorded_and_excludes_the_slot():
    obs = _obs([_row(1, 1, cap=T1, result="hit", player=20), _row(1, 1, cap=T1, result="not_hit", player=21)])
    res = P.resolve(obs)
    s = res.slots.iloc[0]
    assert s["status"] == "equal_time_conflict" and not s["usable"]
    assert res.rounds.iloc[0]["incomplete_reason"] == "equal_time_conflict"
    assert res.log["equal_time_conflicts"] == 1


def test_exact_duplicate_rows_at_equal_time_are_collapsed_not_conflicts():
    obs = _obs([_row(1, 1, cap=T1), _row(1, 1, cap=T1)])
    res = P.resolve(obs)
    assert res.slots.iloc[0]["status"] == "ok" and res.log["exact_duplicate_rows"] == 1


def test_a_settled_leg_that_disappears_is_incomplete_even_with_a_witness_and_flags_ownership():
    obs = _obs([_row(1, 1, cap=T1, result="hit"), _row(1, 2, cap=T1, result="hit", player=21),
                _row(1, 1, cap=T2, result="hit")])
    res = P.resolve(obs, witness=_wit(obs))
    r = res.rounds.iloc[0]
    assert not r["complete"] and r["incomplete_reason"] == "settled_leg_deleted"
    assert P.ownership_conflict_users(res) == {7}


def test_round_missing_from_a_later_capture_is_kept_but_flagged_and_counted():
    obs = _obs([_row(1, 1, cap=T1), _row(2, 1, cap=T1), _row(2, 1, cap=T2)])
    res = P.resolve(obs, witness=_wit(obs))
    r = res.rounds.set_index("round_id")
    assert r.loc[1, "dropped_from_later_capture"] and not r.loc[2, "dropped_from_later_capture"]
    assert r.loc[1, "complete"] and res.log["rounds_dropped_from_later_capture"] == 1
    s = P.dd_summary(res.rounds, date(2026, 5, 1), date(2026, 5, 5))[7]
    assert s["dropped_from_later_capture"] == 1


def test_unknown_slot_two_is_not_a_known_single_pick_day():
    obs = _obs([_row(1, 1, cap=T1, result="")])
    res = P.resolve(obs, witness=_wit(obs))
    r = res.rounds.iloc[0]
    assert not r["complete"] and not r["is_dd"]
    s = P.dd_summary(res.rounds, date(2026, 5, 1), date(2026, 5, 3))
    assert s[7]["complete_rounds"] == 0 and s[7]["dd_frequency"] is None and s[7]["incomplete_rounds"] == 1


def test_null_coerced_identity_is_unresolved():
    obs = _obs([_row(1, 1, cap=T1, unit=0, player=0)])
    res = P.resolve(obs, witness=_wit(obs))
    assert res.slots.iloc[0]["status"] == "identity_unresolved" and not res.slots.iloc[0]["usable"]
    assert res.rounds.iloc[0]["incomplete_reason"] == "identity_unresolved"


def test_missing_primary_slot_is_incomplete():
    obs = _obs([_row(1, 2, cap=T1)])
    assert P.resolve(obs, witness=_wit(obs)).rounds.iloc[0]["incomplete_reason"] == "missing_primary_slot"


def test_captured_at_orders_revisions_whatever_the_concatenation_order():
    a = _obs([_row(1, 1, cap=T1, result="", player=20)], file="a.parquet")
    b = _obs([_row(1, 1, cap=T2, result="hit", player=21)], file="b.parquet", rank=1)
    res = P.resolve(pd.concat([b, a], ignore_index=True))
    assert res.slots.iloc[0]["bts_player_id"] == 21 and res.slots.iloc[0]["files"] == ["a.parquet", "b.parquet"]


def test_equal_capture_ties_break_by_file_rank_then_row():
    ra, rb = _row(1, 1, cap=T1).model_dump(), _row(1, 1, cap=T1).model_dump()
    ra["batter_id"], rb["batter_id"] = 111, 222
    a = P.with_provenance(pd.DataFrame([ra]), user_id=7, source="daily", file="a.parquet", file_rank=0)
    b = P.with_provenance(pd.DataFrame([rb]), user_id=7, source="daily", file="b.parquet", file_rank=1)
    for frames in ([a, b], [b, a]):
        s = P.resolve(pd.concat(frames, ignore_index=True)).slots.iloc[0]
        assert s["status"] == "ok" and s["batter_id"] == 222


def test_nullable_integers_and_pd_na_are_normalized_at_the_boundary():
    df = pd.DataFrame([_row(1, 1, cap=T1).model_dump(), _row(2, 1, cap=T1, result="hit").model_dump()])
    df["batter_id"] = pd.array([pd.NA, 1020], dtype="Int64")
    df["round_id"] = pd.array([1, 2], dtype="Int32")
    df["result"] = pd.array([pd.NA, "hit"], dtype="string")
    obs = P.with_provenance(df, user_id=7, source="daily", file="u.parquet", file_rank=0)
    s = P.resolve(obs).slots.set_index("round_id")
    assert s.loc[1, "batter_id"] is None and s.loc[1, "label"] == "" and s.loc[2, "batter_id"] == 1020


def test_null_required_integer_is_rejected_at_the_boundary():
    df = pd.DataFrame([_row(1, 1, cap=T1).model_dump()])
    df["streak_after"] = pd.array([pd.NA], dtype="Int64")
    with pytest.raises(ValueError, match="streak_after"):
        P.with_provenance(df, user_id=7, source="daily", file="u.parquet", file_rank=0)


def test_tz_aware_capture_times_are_normalized_to_naive_utc():
    df = pd.DataFrame([_row(1, 1, cap=T1).model_dump()])
    df["captured_at"] = pd.to_datetime(df["captured_at"]).dt.tz_localize("America/New_York")
    obs = P.with_provenance(df, user_id=7, source="daily", file="u.parquet", file_rank=0)
    assert obs["captured_at"].dt.tz is None and obs["captured_at"].iloc[0] == pd.Timestamp("2026-05-02 15:00")


def test_dd_summary_counts_pick_days_dd_frequency_and_unobserved_dates_separately():
    rows = [_row(1, 1, cap=T3, pick_date=date(2026, 5, 1)), _row(1, 2, cap=T3, pick_date=date(2026, 5, 1), player=21),
            _row(2, 1, cap=T3, pick_date=date(2026, 5, 2)),
            _row(3, 1, cap=T3, pick_date=date(2026, 5, 3), result="")]
    obs = _obs(rows)
    s = P.dd_summary(P.resolve(obs, witness=_wit(obs)).rounds, date(2026, 5, 1), date(2026, 5, 5))[7]
    assert s["pick_days"] == 3 and s["complete_rounds"] == 2 and s["dd_rounds"] == 1
    assert s["dd_frequency"] == 0.5 and s["incomplete_rounds"] == 1 and s["unobserved_calendar_dates"] == 2
    assert s["incomplete_reasons"] == {"unsettled_slot": 1} and s["calendar_dates"] == 5


def test_graded_summary_reports_slot_denominators():
    rows = [_row(1, 1, cap=T3, pick_date=date(2026, 5, 1)),
            _row(1, 2, cap=T3, pick_date=date(2026, 5, 1), player=21, result="not_hit"),
            _row(2, 1, cap=T3, pick_date=date(2026, 5, 2), result="void")]
    g = P.graded_summary(P.resolve(_obs(rows)).slots, date(2026, 5, 1), date(2026, 5, 3))[7]
    assert g == {"slots": 3, "graded_slots": 2, "hits": 1, "hit_rate": 0.5, "graded_dates": 1, "graded_rounds": 1,
                 "excluded_by_label": {"void": 1}, "excluded_by_status": {}}


def test_composition_needs_an_independent_witness_same_day_lookups_are_unknown():
    """Review F7 probe: a same-day current lookup (BOS/away) is not historical pick-time context."""
    obs = _obs([_row(1, 1, cap=datetime(2026, 5, 1, 23), pick_date=date(2026, 5, 1), team="BOS", ha="away")])
    c = P.composition(obs, P.resolve(obs).slots)[7]
    assert c["slots"] == 1 and c["witnessed"] == 0 and c["unknown"] == 1 and c["away"] == 0
    assert c["top_team_share"] is None and c["basis"] == "no_independent_historical_context_witness"


def test_composition_counts_equal_time_context_conflicts_instead_of_choosing_one():
    obs = _obs([_row(1, 1, cap=T1, team="NYM", ha="home"), _row(1, 1, cap=T1, team="BOS", ha="away")])
    res = P.resolve(obs)
    assert res.slots.iloc[0]["status"] == "ok"                  # the grade survives
    c = P.composition(obs, res.slots)[7]
    assert c["context_conflicts"] == 1 and c["witnessed"] == 0
