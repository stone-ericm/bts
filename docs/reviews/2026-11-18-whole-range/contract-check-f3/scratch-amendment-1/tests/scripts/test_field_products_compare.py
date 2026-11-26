"""W2.2 comparison (design r1 edits B-E1/B-E6): our ledger denominator, E's usable graded slots, the all-observed-date
and frozen shared-date tables with a joint whole-date bootstrap, the availability table and the restricted extension."""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from bts.leaderboard.models import PickRow
from scripts.audit.field_products import compare as Q
from scripts.audit.field_products import picks as P
from scripts.audit.mlb_benchmark import metrics as m

W0, W1 = date(2026, 5, 1), date(2026, 7, 3)


def _sel(sid, d, slot, *, commit="committed_evidenced", entry="confirmed", match="evidenced", grade="hit",
         kind="selection", rid=1, unit=1):
    return {"row_kind": kind, "date": d, "round_id": rid, "slot": slot, "selection_id": sid, "batter_id": 5,
            "game_pk": 9, "commit_status": commit, "entry_status": entry, "match": match, "match_reason": "x",
            "unit_id": unit, "contest_slot_grade_raw": grade, "contest_round_result": None, "streak_before": None,
            "streak_after": None}


def test_our_denominator_needs_commit_unique_link_and_exact_grade_and_keeps_both_dd_legs():
    ledger = pd.DataFrame([
        _sel("a", "2026-05-02", "primary", grade="hit", unit=1),
        _sel("b", "2026-05-02", "double_down", grade="not_hit", match="inferred", unit=2),
        _sel("c", "2026-05-03", "primary", commit="unconfirmed", unit=3),
        _sel("d", "2026-05-04", "primary", entry="unknown", match=None, grade=None, unit=4),
        _sel("e", "2026-05-05", "primary", grade="void", unit=5),
        _sel(None, "2026-05-06", None, kind="contest_only", unit=6),
        _sel("f", "2026-07-04", "primary", unit=7),                       # outside the window
        _sel("g", "2026-05-07", "primary", grade="hit", unit=8),
        _sel("h", "2026-05-08", "primary", match="ambiguous", unit=9),
    ])
    contest = pd.DataFrame([{"selection_id": s, "match": "evidenced", "slot_result": g} for s, g in
                            [("a", "hit"), ("b", "not_hit"), ("e", "void"), ("g", "hit"), ("g", "hit"),
                             ("h", "hit")]])
    inc, exc = Q.ours_slots(ledger, contest, W0, W1)
    assert list(inc["selection_id"]) == ["a", "b"] and list(inc["slot"]) == ["primary", "double_down"]
    assert list(inc["hit"]) == [True, False]
    assert exc["reasons"] == {"not_committed_evidenced:unconfirmed": 1, "not_uniquely_confirmed:unknown/None": 1,
                              "grade_not_exact:void": 1, "contest_link_not_unique:2": 1,
                              "not_uniquely_confirmed:confirmed/ambiguous": 1}
    assert exc["non_selection_rows"] == {"contest_only": 1} and exc["included_by_slot"] == {"primary": 1,
                                                                                           "double_down": 1}
    assert exc["included_by_match"] == {"evidenced": 1, "inferred": 1} and exc["window_rows"] == 8


def test_undated_ledger_rows_are_counted_not_crashed_on():
    ledger = pd.DataFrame([_sel("a", "2026-05-02", "primary"), _sel(None, None, None, kind="contest_only")])
    inc, exc = Q.ours_slots(ledger, None, W0, W1)
    assert len(inc) == 1 and exc["undated_rows"] == {"contest_only": 1} and exc["contest_cross_check"] is False


def test_contest_slot_grade_must_agree_with_the_ledger_row():
    ledger = pd.DataFrame([_sel("a", "2026-05-02", "primary", grade="hit")])
    contest = pd.DataFrame([{"selection_id": "a", "match": "evidenced", "slot_result": "not_hit"}])
    inc, exc = Q.ours_slots(ledger, contest, W0, W1)
    assert inc.empty and exc["reasons"] == {"contest_grade_disagrees": 1}


def _arm(rows):
    """(user, date, hit[, round_id]); the round id defaults to the date's day (one round per date)."""
    rows = [r if len(r) == 4 else (*r, int(r[1][-2:])) for r in rows]
    return pd.DataFrame(rows, columns=["user_id", "date", "hit", "round_id"])


def test_all_date_table_uses_the_union_and_each_arms_own_denominator_and_shared_uses_common_dates():
    e = _arm([(1, "2026-05-01", True), (1, "2026-05-01", False), (2, "2026-05-02", True), (2, "2026-05-03", False)])
    o = _arm([(0, "2026-05-02", True), (0, "2026-05-03", True), (0, "2026-05-04", False)])
    t = Q.date_tables(e, o, n_resamples=200, n_members=10, calendar_dates=64)
    a, s = t["all_observed_dates"], t["shared_dates"]
    assert a["dates"] == 4 and a["E"]["slots"] == 4 and a["E"]["ratio"] == 0.5 and a["ours"]["ratio"] == 2 / 3
    assert a["E"]["dates_with_slots"] == 3 and a["ours"]["dates_with_slots"] == 3 and a["E"]["users"] == 2
    assert a["E"]["rounds"] == 3 and a["ours"]["rounds"] == 3 and s["E"]["rounds"] == 2
    assert a["coverage"] == {"E_members": 10, "E_users_contributing": 2, "E_user_share": 0.2,
                             "window_calendar_dates": 64, "dates": 4, "date_share_of_window": 4 / 64}
    assert s["coverage"]["dates"] == 2 and s["coverage"]["E_users_contributing"] == 1   # only user 2 on 5/02-5/03
    assert s["dates"] == 2 and s["date_list"] == ["2026-05-02", "2026-05-03"]
    assert s["E"]["ratio"] == 0.5 and s["ours"]["ratio"] == 1.0 and s["diff_ours_minus_E"] == 0.5
    iv = s["intervals"]["diff_ours_minus_E"]
    assert iv["seed"] == 20261004 and iv["n_resamples"] == 200 and iv["n_ok"] + iv["n_failed"] == 200
    assert "diff_ours_minus_E" not in a["intervals"]


def test_failed_resamples_are_counted_not_dropped_silently():
    e = _arm([(1, "2026-05-01", True)])
    o = _arm([(0, f"2026-05-0{k}", True) for k in range(1, 6)])
    iv = Q.date_tables(e, o, n_resamples=300)["all_observed_dates"]["intervals"]["E_ratio"]
    assert iv["n_failed"] > 0 and iv["n_ok"] + iv["n_failed"] == 300
    # shared failure policy: any failed draw leaves the nominal interval unavailable, conditional one labelled
    assert iv["lo"] is None and iv["hi"] is None and iv["status"] == "unavailable_failed_draws"
    assert iv["conditional_lo"] is not None


def test_no_common_date_makes_the_shared_table_unavailable():
    t = Q.date_tables(_arm([(1, "2026-05-01", True)]), _arm([(0, "2026-05-02", True)]), n_resamples=50)
    assert t["shared_dates"]["available"] is False and t["shared_dates"]["diff_ours_minus_E"] is None


def test_both_arms_are_resampled_on_the_same_date_draws():
    """Identical per-date results in both arms -> the paired difference is 0 in EVERY joint draw."""
    rows = [(1, f"2026-05-{k:02d}", k % 3 == 0) for k in range(1, 15)]
    t = Q.date_tables(_arm(rows), _arm([(0, d, h) for _, d, h in rows]), n_resamples=300)
    iv = t["shared_dates"]["intervals"]["diff_ours_minus_E"]
    assert iv["lo"] == 0.0 and iv["hi"] == 0.0 and iv["n_failed"] == 0


def test_date_summary_bootstrap_equals_resampling_every_drawn_dates_rows_with_repeats():
    rng = np.random.default_rng(3)
    rows = [(u, f"2026-05-{d:02d}", bool(rng.random() < 0.6)) for d in range(1, 12) for u in range(int(rng.integers(1, 5)))]
    e = _arm(rows)
    t = Q.date_tables(e, _arm([(0, f"2026-05-{d:02d}", True) for d in range(1, 12)]), n_resamples=150)
    ref = m.block_bootstrap_many(e.assign(hit=e["hit"].astype(int)),
                                 {"r": lambda s: s["hit"].sum() / len(s)}, n_resamples=150)["r"]
    got = t["all_observed_dates"]["intervals"]["E_ratio"]
    assert (got["lo"], got["hi"]) == (ref["lo"], ref["hi"])


def _res(rows_by_user):
    frames = []
    for uid, rows in rows_by_user.items():
        frames.append(P.with_provenance(pd.DataFrame([r.model_dump() for r in rows]), user_id=uid, source="daily",
                                        file=f"{uid}.parquet", file_rank=0))
    return P.resolve(pd.concat(frames, ignore_index=True))


def _pk(rid, d, result="hit", pn=1, player=20):
    return PickRow(captured_at=datetime(2026, 7, 4, 12), round_id=rid, pick_date=d, pick_number=pn, unit_id=1,
                   bts_player_id=player, result=result, at_bats=4, hits=1, streak_after=1)


def test_e_arm_keeps_e_in_a_members_and_excludes_quarantined_and_out_of_window_slots():
    res = _res({11: [_pk(1, date(2026, 5, 2))], 12: [_pk(1, date(2026, 5, 2), "not_hit")],
                13: [_pk(2, date(2026, 7, 4))], 14: [_pk(1, date(2026, 5, 2), "")]})
    arm = Q.e_arm(res.slots, available={11, 13, 14}, start=W0, end=W1)
    assert list(arm["user_id"]) == [11] and list(arm["date"]) == ["2026-05-02"]


def test_availability_keeps_every_member_and_separates_history_from_activity():
    members = pd.DataFrame({"order": [1, 2, 3, 4, 5], "user_id": [11, 12, 13, 14, 15]})
    labels = members.assign(allocation=["E_in_A", "B", "B", "E_unfetched", "B"])
    binding = members.assign(binding=["bound", "bound", "quarantined_manifest_collision", "no_daily_file", "bound"],
                             files=[["a"], ["b"], ["c"], [], ["e"]], n_files=[1, 1, 1, 0, 1])
    res = _res({11: [_pk(1, date(2026, 5, 2))], 12: [_pk(2, date(2026, 4, 20))],
                15: [_pk(1, date(2026, 5, 2), player=20), _pk(1, date(2026, 5, 2), player=21)]})
    obs_stats = {11: {"rows": 1, "captures": 1}, 12: {"rows": 1, "captures": 1}, 15: {"rows": 2, "captures": 1}}
    av = Q.availability(members, labels, binding, ownership_quarantined={15}, res=res, obs_stats=obs_stats,
                        start=W0, end=W1).set_index("user_id")
    assert len(av) == 5
    assert av.loc[11, "daily_history"] == "available" and av.loc[11, "window_activity"] == "observed"
    assert av.loc[11, "window_graded_slots"] == 1 and av.loc[11, "allocation"] == "E_in_A"
    assert av.loc[11, "attribution_basis"] == "stable_5_01_username_unwitnessed"
    assert pd.isna(av.loc[13, "attribution_basis"])
    assert av.loc[12, "window_activity"] == "none_observed_unknown"
    assert av.loc[13, "daily_history"] == "quarantined_manifest_collision"
    assert av.loc[13, "window_activity"] == "not_assessable"
    assert av.loc[14, "daily_history"] == "no_daily_file"
    assert av.loc[15, "daily_history"] == "quarantined_ownership_conflict"


def test_extension_is_restricted_to_usable_fetched_histories_and_labelled():
    fg = pd.DataFrame([
        {"order": 1, "user_id": 11, "allocation": "E_in_A", "fetched": True, "usable": True, "history": "usable",
         "first_pick_date": "2026-03-26", "last_pick_date": "2026-09-27", "n_rounds": 3},
        {"order": 2, "user_id": 12, "allocation": "B", "fetched": True, "usable": False, "history": "no_history",
         "first_pick_date": None, "last_pick_date": None, "n_rounds": 0},
        {"order": 3, "user_id": 13, "allocation": "E_unfetched", "fetched": False, "usable": False,
         "history": "budget_omission", "first_pick_date": None, "last_pick_date": None, "n_rounds": 0}])
    res = _res({11: [_pk(1, date(2026, 6, 1)), _pk(2, date(2026, 7, 4)), _pk(3, date(2026, 9, 27), "not_hit")]})
    x = Q.extension(fg, res.slots, n_members=3)
    assert x["label"].startswith("final-backfill extension") and x["window"] == ["2026-07-04", "2026-09-27"]
    assert x["pooled"] == {"users": 1, "dates": 2, "rounds": 2, "slots": 2, "hits": 1, "ratio": 0.5}
    assert x["coverage"]["usable"] == 1 and x["coverage"]["E_in_A"] == 1 and x["coverage"]["budget_omissions"] == 1
    assert x["coverage"]["history"] == {"usable": 1, "no_history": 1, "budget_omission": 1}


def test_e_window_exclusions_are_counted_by_label_and_status():
    res = _res({11: [_pk(1, date(2026, 5, 2)), _pk(2, date(2026, 5, 3), "void"), _pk(3, date(2026, 5, 4), ""),
                     _pk(4, date(2026, 7, 4), "void")],
                12: [_pk(1, date(2026, 5, 2), "void")]})
    x = Q.e_exclusions(res.slots, available={11}, start=W0, end=W1)
    assert x == {"window_slots": 3, "usable_graded": 1, "excluded_by_label": {"void": 1, "": 1},
                 "excluded_by_status": {}}


def test_round_denominator_counts_user_round_keys_not_user_dates():
    """Review F8 probe: user 7, rounds 1 and 2 both on May 1, one graded slot each -> rounds 2, not 1."""
    e = _arm([(7, "2026-05-01", True, 1), (7, "2026-05-01", False, 2)])
    a = Q.date_tables(e, _arm([(0, "2026-05-01", True)]), n_resamples=20)["all_observed_dates"]
    assert a["E"]["rounds"] == 2 and a["E"]["slots"] == 2 and a["E"]["user_dates_with_multiple_rounds"] == 1


def test_e_arm_outputs_carry_the_unwitnessed_attribution_basis():
    res = _res({11: [_pk(1, date(2026, 5, 2))]})
    arm = Q.e_arm(res.slots, available={11}, start=W0, end=W1)
    assert set(arm["attribution_basis"]) == {"stable_5_01_username_unwitnessed"}
    t = Q.date_tables(arm, _arm([(0, "2026-05-02", True)]), n_resamples=20)
    assert t["all_observed_dates"]["E"]["attribution_basis"] == "stable_5_01_username_unwitnessed"
    assert t["shared_dates"]["E"]["attribution_basis"] == "stable_5_01_username_unwitnessed"
