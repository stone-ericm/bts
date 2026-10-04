"""W2.1/W2.2 field products: the pick data contract (design 2026-10-04 rev 2, "Data contract"). Synthetic only."""
from __future__ import annotations

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


def _obs(rows, user_id=7, file="u.parquet"):
    df = pd.DataFrame([r.model_dump() for r in rows])
    return P.with_provenance(df, user_id=user_id, source="daily", file=file, file_rank=0)


T1, T2, T3 = datetime(2026, 5, 2, 11), datetime(2026, 5, 3, 11), datetime(2026, 5, 4, 11)


def test_read_observations_keeps_every_appended_row_and_both_dd_slots(tmp_path):
    """The failure prevented: latest_per_pick_date keeps one row per date and drops a two-hit DD leg."""
    p = tmp_path / "alice.parquet"
    append_user_picks(p, [_row(1, 1, cap=T1, player=20), _row(1, 2, cap=T1, player=21)])
    append_user_picks(p, [_row(1, 1, cap=T2, player=20), _row(1, 2, cap=T2, player=21)])
    obs = P.read_observations([p], user_id=7, source="daily")
    assert len(obs) == 4 and set(obs["file"]) == {"alice.parquet"} and list(obs["file_row"]) == [0, 1, 2, 3]
    res = P.resolve(obs)
    assert len(res.slots) == 2 and set(res.slots["pick_number"]) == {1, 2}
    rnd = res.rounds.iloc[0]
    assert rnd["complete"] and rnd["is_dd"] and rnd["n_slots"] == 2
    assert res.slots["usable"].all() and res.slots["hit"].sum() == 2


def test_later_pending_observation_is_not_replaced_by_an_older_settled_label():
    obs = _obs([_row(1, 1, cap=T1, result="hit"), _row(1, 1, cap=T2, result="")])
    res = P.resolve(obs)
    s = res.slots.iloc[0]
    assert s["label"] == "" and not s["graded"] and not s["usable"] and s["later_unsettled_over_settled"]
    assert not res.rounds.iloc[0]["complete"] and res.rounds.iloc[0]["incomplete_reason"] == "unsettled_slot"
    assert res.log["later_unsettled_over_settled"] == 1


def test_only_exact_hit_and_not_hit_are_graded_and_at_bats_never_settle_a_slot():
    obs = _obs([_row(1, 1, cap=T1, result="", at_bats=4, hits=2), _row(2, 1, cap=T1, result="void"),
                _row(3, 1, cap=T1, result="Hit"), _row(4, 1, cap=T1, result="None"),
                _row(5, 1, cap=T1, result="not_hit", hits=0)])
    res = P.resolve(obs)
    g = res.slots.set_index("round_id")
    assert list(g["graded"]) == [False, False, False, False, True]
    assert res.log["label_counts"] == {"": 1, "void": 1, "Hit": 1, "None": 1, "not_hit": 1}
    # void settles the slot SET (round complete) but is not a graded slot
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


def test_deleted_pending_leg_is_not_retained_once_a_later_settled_snapshot_lacks_it():
    obs = _obs([_row(1, 1, cap=T1, result=""), _row(1, 2, cap=T1, result="", player=21),
                _row(1, 1, cap=T2, result="hit")])
    res = P.resolve(obs)
    assert list(res.slots["pick_number"]) == [1]
    r = res.rounds.iloc[0]
    assert r["complete"] and not r["is_dd"] and r["deleted_legs"] == 1


def test_a_settled_leg_that_disappears_makes_the_round_incomplete():
    obs = _obs([_row(1, 1, cap=T1, result="hit"), _row(1, 2, cap=T1, result="hit", player=21),
                _row(1, 1, cap=T2, result="hit")])
    r = P.resolve(obs).rounds.iloc[0]
    assert not r["complete"] and r["incomplete_reason"] == "settled_leg_deleted"


def test_round_missing_from_a_later_capture_is_kept_but_flagged_and_counted():
    """A newer omission never erases the older observation (season_ledger contest.slot_history precedent), but it is
    recorded: the later full-profile capture no longer shows the round."""
    obs = _obs([_row(1, 1, cap=T1), _row(2, 1, cap=T1), _row(2, 1, cap=T2)])
    res = P.resolve(obs)
    r = res.rounds.set_index("round_id")
    assert r.loc[1, "dropped_from_later_capture"] and not r.loc[2, "dropped_from_later_capture"]
    assert r.loc[1, "complete"] and res.log["rounds_dropped_from_later_capture"] == 1
    s = P.dd_summary(res.rounds, date(2026, 5, 1), date(2026, 5, 5))[7]
    assert s["dropped_from_later_capture"] == 1


def test_unknown_slot_two_is_not_a_known_single_pick_day():
    """A pending primary with no second slot may still gain a DD before lock: incomplete, not single."""
    obs = _obs([_row(1, 1, cap=T1, result="")])
    r = P.resolve(obs).rounds.iloc[0]
    assert not r["complete"] and not r["is_dd"]
    s = P.dd_summary(P.resolve(obs).rounds, date(2026, 5, 1), date(2026, 5, 3))
    assert s[7]["complete_rounds"] == 0 and s[7]["dd_frequency"] is None and s[7]["incomplete_rounds"] == 1


def test_null_coerced_identity_is_unresolved():
    obs = _obs([_row(1, 1, cap=T1, unit=0, player=0)])
    res = P.resolve(obs)
    assert res.slots.iloc[0]["status"] == "identity_unresolved" and not res.slots.iloc[0]["usable"]
    assert res.rounds.iloc[0]["incomplete_reason"] == "identity_unresolved"


def test_missing_primary_slot_is_incomplete():
    obs = _obs([_row(1, 2, cap=T1)])
    assert P.resolve(obs).rounds.iloc[0]["incomplete_reason"] == "missing_primary_slot"


def test_captured_at_orders_revisions_whatever_the_concatenation_order():
    a = _obs([_row(1, 1, cap=T1, result="", player=20)], file="a.parquet")
    b = P.with_provenance(pd.DataFrame([_row(1, 1, cap=T2, result="hit", player=21).model_dump()]),
                          user_id=7, source="daily", file="b.parquet", file_rank=1)
    res = P.resolve(pd.concat([b, a], ignore_index=True))
    assert res.slots.iloc[0]["bts_player_id"] == 21 and res.slots.iloc[0]["files"] == "a.parquet;b.parquet"


def test_equal_capture_ties_break_by_file_rank_then_row():
    """Same content at the same instant in two files: no conflict, and the representative is the declared-last row."""
    ra, rb = _row(1, 1, cap=T1).model_dump(), _row(1, 1, cap=T1).model_dump()
    ra["batter_id"], rb["batter_id"] = 111, 222
    a = P.with_provenance(pd.DataFrame([ra]), user_id=7, source="daily", file="a.parquet", file_rank=0)
    b = P.with_provenance(pd.DataFrame([rb]), user_id=7, source="daily", file="b.parquet", file_rank=1)
    for frames in ([a, b], [b, a]):
        s = P.resolve(pd.concat(frames, ignore_index=True)).slots.iloc[0]
        assert s["status"] == "ok" and s["batter_id"] == 222


def test_dd_summary_counts_pick_days_dd_frequency_and_unobserved_dates_separately():
    rows = [_row(1, 1, cap=T3, pick_date=date(2026, 5, 1)), _row(1, 2, cap=T3, pick_date=date(2026, 5, 1), player=21),
            _row(2, 1, cap=T3, pick_date=date(2026, 5, 2)),
            _row(3, 1, cap=T3, pick_date=date(2026, 5, 3), result="")]
    s = P.dd_summary(P.resolve(_obs(rows)).rounds, date(2026, 5, 1), date(2026, 5, 5))[7]
    assert s["pick_days"] == 3 and s["complete_rounds"] == 2 and s["dd_rounds"] == 1
    assert s["dd_frequency"] == 0.5 and s["incomplete_rounds"] == 1 and s["unobserved_calendar_dates"] == 2
    assert s["incomplete_reasons"] == {"unsettled_slot": 1} and s["calendar_dates"] == 5


def test_graded_summary_reports_slot_denominators():
    rows = [_row(1, 1, cap=T3, pick_date=date(2026, 5, 1)),
            _row(1, 2, cap=T3, pick_date=date(2026, 5, 1), player=21, result="not_hit"),
            _row(2, 1, cap=T3, pick_date=date(2026, 5, 2), result="void")]
    g = P.graded_summary(P.resolve(_obs(rows)).slots, date(2026, 5, 1), date(2026, 5, 3))[7]
    assert g == {"slots": 3, "graded_slots": 2, "hits": 1, "hit_rate": 0.5, "graded_dates": 1,
                 "excluded_by_label": {"void": 1}, "excluded_by_status": {}}


def test_composition_uses_only_same_day_witnessed_context():
    """Final-grab/next-day lookup values are not historical team evidence (B-E4)."""
    rows = [_row(1, 1, cap=datetime(2026, 5, 1, 15), pick_date=date(2026, 5, 1), team="NYM", ha="home"),
            _row(2, 1, cap=datetime(2026, 5, 3, 15), pick_date=date(2026, 5, 2), team="BOS", ha="away"),
            # 02:00 UTC on 5/04 is still 5/03 in New York: same-day witness
            _row(3, 1, cap=datetime(2026, 5, 4, 2), pick_date=date(2026, 5, 3), team="NYM", ha="away"),
            _row(4, 1, cap=datetime(2026, 5, 4, 15), pick_date=date(2026, 5, 4), team=None, ha=None)]
    obs = _obs(rows)
    c = P.composition(obs, P.resolve(obs).slots)[7]
    assert c["slots"] == 4 and c["witnessed"] == 2 and c["unknown"] == 2
    assert c["home"] == 1 and c["away"] == 1 and c["distinct_teams"] == 1 and c["top_team_share"] == 1.0
