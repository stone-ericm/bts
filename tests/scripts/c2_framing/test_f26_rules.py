"""The 2026 framing test's pure rules (design `docs/sota_audit/2026-10-09-prereg-c2-framing-2026-test.md`, frozen at
9b344d9): the starter proxy (§3), the projection and the as-of value (§4), the arm transform and missing reasons (§4),
the calendar and the dispositions (§6)."""
import math

import numpy as np
import pandas as pd
import pytest

from scripts.audit.c2_framing import f26 as F
from scripts.audit.c2_framing import screen as S


# ---------------------------------------------------------------- the starter proxy (§3)

def _player(pid, order=None, codes=("2",), **extra):
    p = {"person": {"id": pid}, **extra}
    if order is not None:
        p["battingOrder"] = order
    if codes is not None:
        p["allPositions"] = [{"code": c} for c in codes]
    return p


def _feed(away_players, home_players, *, pk=777, date="2026-04-01", game_number=1, away_tid=10, home_tid=20,
          season="2026", game_type="R"):
    game = {"pk": pk, "season": season, "type": game_type}
    if game_number is not None:
        game["gameNumber"] = game_number
    return {"gameData": {"game": game, "datetime": {"officialDate": date},
                         "teams": {"away": {"id": away_tid}, "home": {"id": home_tid}},
                         "probablePitchers": {"away": {"id": 501}, "home": {"id": 502}}},
            "liveData": {"boxscore": {"teams": {
                "away": {"players": {f"ID{p['person']['id']}": p for p in away_players}},
                "home": {"players": {f"ID{p['person']['id']}": p for p in home_players}}}}}}


def _nine(first_id, catcher_slot=2, catcher_codes=("2",)):
    out = []
    for slot in range(1, 10):
        codes = catcher_codes if slot == catcher_slot else ("8",)
        out.append(_player(first_id + slot, f"{slot}00", codes))
    return out


def test_the_starter_proxy_is_the_one_starting_nine_player_whose_first_position_is_c():
    feed = _feed(_nine(100) + [_player(199, "201", ("2",))],        # a substitute catcher in slot 2 does not count
                 _nine(200, catcher_slot=9))
    recs = {r["fielding_side"]: r for r in F.starter_proxy(feed, 777, 2026)}
    assert recs["away"]["catcher_id"] == 102 and recs["away"]["reason"] == "identified"
    assert recs["home"]["catcher_id"] == 209
    assert recs["away"]["team_id"] == 10 and recs["home"]["team_id"] == 20
    assert recs["away"]["official_date"] == "2026-04-01" and recs["away"]["game_number"] == 1
    assert recs["away"]["game_number_fallback"] is False


def test_a_starter_who_moved_to_catcher_later_is_not_the_proxy_and_order_is_the_arrays():
    # slot 2 played 1B first, then C: not the proxy; the array's given order decides
    away = _nine(100, catcher_slot=0) + []
    away[1] = _player(102, "200", ("3", "2"))
    recs = {r["fielding_side"]: r for r in F.starter_proxy(_feed(away, _nine(200)), 777, 2026)}
    assert recs["away"]["catcher_id"] is None and recs["away"]["reason"] == "no_candidate"


def test_two_candidates_give_no_proxy():
    away = _nine(100)
    away[4] = _player(105, "500", ("2",))
    recs = {r["fielding_side"]: r for r in F.starter_proxy(_feed(away, _nine(200)), 777, 2026)}
    assert recs["away"]["catcher_id"] is None and recs["away"]["reason"] == "multiple_candidates"


@pytest.mark.parametrize("order", ["1000", "0", "100 ", " 100", "0100", 100, "1"])
def test_only_the_canonical_slot_strings_are_starting_slots(order):
    away = _nine(100, catcher_slot=0)
    away.append(_player(150, order, ("2",)))
    recs = {r["fielding_side"]: r for r in F.starter_proxy(_feed(away, _nine(200)), 777, 2026)}
    if isinstance(order, str):
        assert recs["away"]["reason"] == "no_candidate"
    else:
        assert recs["away"]["reason"] == "malformed_player"       # a non-string battingOrder is malformed


@pytest.mark.parametrize("damage, reason", [
    (lambda p: p["person"].update(id="102"), "malformed_player"),
    (lambda p: p["person"].update(id=True), "malformed_player"),
    (lambda p: p["person"].update(id=0), "malformed_player"),
    (lambda p: p["person"].update(id=102.0), "malformed_player"),
    (lambda p: p.update(allPositions=[]), "malformed_positions"),
    (lambda p: p.update(allPositions="2"), "malformed_positions"),
    (lambda p: p.update(allPositions=[{"code": 2}]), "malformed_positions"),
    (lambda p: p.pop("allPositions"), "malformed_positions"),
])
def test_a_malformed_starting_record_makes_the_side_unidentified(damage, reason):
    away = _nine(100)
    damage(away[1])
    recs = {r["fielding_side"]: r for r in F.starter_proxy(_feed(away, _nine(200)), 777, 2026)}
    assert recs["away"]["catcher_id"] is None and recs["away"]["reason"] == reason
    assert recs["home"]["catcher_id"] == 202


def test_a_malformed_record_elsewhere_on_the_side_is_not_ignored():
    away = _nine(100)
    feed = _feed(away, _nine(200))
    feed["liveData"]["boxscore"]["teams"]["away"]["players"]["IDbad"] = "not a record"
    recs = {r["fielding_side"]: r for r in F.starter_proxy(feed, 777, 2026)}
    assert recs["away"]["reason"] == "malformed_player"


def test_a_missing_game_number_falls_back_to_one_and_says_so():
    recs = F.starter_proxy(_feed(_nine(100), _nine(200), game_number=None), 777, 2026)
    assert all(r["game_number"] == 1 and r["game_number_fallback"] is True for r in recs)


@pytest.mark.parametrize("kw", [dict(pk=778), dict(date="2026-4-1"), dict(date=None), dict(away_tid="10"),
                                dict(home_tid=None), dict(game_number=0), dict(game_number="2"), dict(season="2025"),
                                dict(game_type="S")])
def test_malformed_game_fields_refuse(kw):
    with pytest.raises(F.InputRefused):
        F.starter_proxy(_feed(_nine(100), _nine(200), **kw), 777, 2026)


def test_the_lookup_entry_is_productions_four_keys():
    entry = F.lookup_entry(_feed(_nine(100), _nine(200)), 777)
    assert entry == {"away": 501, "home": 502, "away_tid": 10, "home_tid": 20}


# ---------------------------------------------------------------- the projection (§4)

def _table(rows):
    """rows: (game_pk, side, date, game_number, team_id, catcher_id)."""
    return pd.DataFrame([{"game_pk": pk, "fielding_side": side, "official_date": d, "game_number": gn,
                          "team_id": t, "catcher_id": c} for pk, side, d, gn, t, c in rows])


def test_the_projection_is_the_most_used_catcher_of_the_last_ten_earlier_dated_games():
    rows = [(i, "home", f"2026-04-{i:02d}", 1, 7, 50 if i % 3 else 60) for i in range(1, 16)]
    t = F.TeamHistory(_table(rows))
    last10 = [50 if i % 3 else 60 for i in range(5, 15)]           # games on 04-05 .. 04-14
    assert t.project(7, "2026-04-15") == max(set(last10), key=last10.count)
    assert t.project(7, "2026-04-01") is None                        # no earlier-dated game


def test_the_window_is_taken_first_then_unknowns_are_ignored():
    rows = [(i, "away", f"2026-04-{i:02d}", 1, 7, 50) for i in range(1, 11)]
    rows += [(100 + i, "away", f"2026-04-{10 + i:02d}", 1, 7, None) for i in range(1, 11)]
    assert F.TeamHistory(_table(rows)).project(7, "2026-04-25") is None


def test_same_day_games_never_enter_history_and_ties_go_to_the_latest_start():
    rows = [(1, "home", "2026-04-01", 1, 7, 50), (2, "away", "2026-04-02", 1, 7, 60),
            (3, "home", "2026-04-03", 1, 7, 70), (4, "home", "2026-04-03", 2, 7, 50)]
    t = F.TeamHistory(_table(rows))
    assert t.project(7, "2026-04-03") == 60          # 50 and 60 tie once; 60's start is later; 04-03 is excluded
    assert t.project(7, "2026-04-04") == 50          # 50 twice now (game 2 of the doubleheader is the latest)


def test_the_tie_break_uses_the_history_order_tuple_within_a_date():
    rows = [(9, "home", "2026-04-01", 2, 7, 50), (5, "home", "2026-04-01", 1, 7, 60)]
    assert F.TeamHistory(_table(rows)).project(7, "2026-04-02") == 50     # game 2 starts after game 1
    rows = [(5, "home", "2026-04-01", 1, 7, 50), (9, "home", "2026-04-01", 1, 7, 60)]
    assert F.TeamHistory(_table(rows)).project(7, "2026-04-02") == 60     # equal game numbers: higher game_pk


def test_the_window_crosses_seasons_and_home_and_away():
    rows = [(1, "home", "2025-09-27", 1, 7, 50), (2, "away", "2025-09-28", 1, 7, 50), (3, "away", "2025-09-28", 1, 8, 99)]
    assert F.TeamHistory(_table(rows)).project(7, "2026-03-26") == 50


# ---------------------------------------------------------------- the as-of value (§4)

def _pas(rows):
    """rows: (date, catcher, rate); a pitcher column so the screen's framing_by can be compared."""
    return pd.DataFrame([{"date": pd.Timestamp(d), "fielding_catcher_id": c, "pa_borderline_csr": r, "pitcher_id": 1}
                         for d, c, r in rows])


def test_the_as_of_value_equals_framing_by_for_a_catcher_who_caught_that_day():
    rng = np.random.default_rng(0)
    rows = []
    for k in range(40):
        d = f"2026-04-{(k % 28) + 1:02d}" if k < 28 else f"2026-05-{k - 27:02d}"
        for c in (50, 60):
            for _ in range(3):
                rows.append((d, c, float(rng.uniform()) if rng.uniform() > 0.2 else np.nan))
    df = _pas(rows)
    ref = S.framing_by(df, "fielding_catcher_id", "catcher_framing")
    asof = F.AsOfFraming(df)
    for (c, d), v in ref.groupby(["fielding_catcher_id", "date"])["catcher_framing"].first().items():
        got = asof.value(c, d)
        assert (math.isnan(v) and math.isnan(got)) or got == v          # bit-equal, not approximately


def test_the_minimum_is_five_nonmissing_daily_rates():
    rows = [("2026-04-0%d" % i, 50, 0.5 if i != 3 else np.nan) for i in range(1, 6)]   # five dates, four rates
    a = F.AsOfFraming(_pas(rows))
    assert math.isnan(a.value(50, pd.Timestamp("2026-04-06")))
    a = F.AsOfFraming(_pas(rows + [("2026-04-06", 50, 0.9)]))
    assert a.value(50, pd.Timestamp("2026-04-07")) == pytest.approx((0.5 * 4 + 0.9) / 5)


def test_a_doubleheader_is_one_daily_rate_and_same_day_rates_never_count():
    rows = [("2026-04-0%d" % i, 50, 0.5) for i in range(1, 6)] + [("2026-04-06", 50, 0.6), ("2026-04-06", 50, 0.8)]
    a = F.AsOfFraming(_pas(rows))
    assert a.value(50, pd.Timestamp("2026-04-06")) == pytest.approx(0.5)            # 04-06's games excluded
    assert a.value(50, pd.Timestamp("2026-04-07")) == pytest.approx((0.5 * 5 + 0.7) / 6)


def test_an_absent_catcher_gets_his_full_prior_prefix_not_the_last_shifted_row():
    rows = [("2026-04-0%d" % i, 50, r) for i, r in zip(range(1, 7), (0.1, 0.3, 0.5, 0.2, 0.8, 0.9))]
    a = F.AsOfFraming(_pas(rows))
    # on 04-09 he did not catch; the value includes his last observed date 04-06
    assert a.value(50, pd.Timestamp("2026-04-09")) == pytest.approx(np.mean([0.1, 0.3, 0.5, 0.2, 0.8, 0.9]))
    assert math.isnan(a.value(99, pd.Timestamp("2026-04-09")))                       # never caught
    assert math.isnan(a.value(None, pd.Timestamp("2026-04-09")))


def test_a_future_rate_never_changes_an_earlier_value():
    rows = [("2026-04-0%d" % i, 50, 0.5) for i in range(1, 7)]
    before = F.AsOfFraming(_pas(rows)).value(50, pd.Timestamp("2026-04-07"))
    after = F.AsOfFraming(_pas(rows + [("2026-04-08", 50, 0.99)])).value(50, pd.Timestamp("2026-04-07"))
    assert before == after


# ---------------------------------------------------------------- the arm transform (§4)

def _day():
    # batters of game 1 (home batters face the away catcher; away batters the home catcher)
    return pd.DataFrame({"date": pd.Timestamp("2026-04-09"), "game_pk": [1, 1, 1], "is_home": [True, True, False],
                         "batter_id": [11, 11, 12], "catcher_framing": [0.0, 0.0, 0.0], "other": [1, 2, 3],
                         "is_hit": [1, 0, 1]}, index=[7, 8, 9])


def _arm_world():
    rates = [("2026-04-0%d" % i, c, r) for i in range(1, 9) for c, r in ((50, 0.4), (60, 0.6), (70, 0.2))]
    table = _table([(1, "away", "2026-04-09", 1, 7, 50), (1, "home", "2026-04-09", 1, 8, None),
                    (2, "away", "2026-04-08", 1, 8, 60), (3, "home", "2026-04-07", 1, 7, 70)])
    return F.AsOfFraming(_pas(rates)), table


def test_the_posted_arm_names_the_opposing_sides_proxy():
    asof, table = _arm_world()
    t = F.ArmTransform("A_posted", table, asof)
    out = t(_day(), pd.Timestamp("2026-04-09"))
    assert out.loc[[7, 8], "catcher_framing"].tolist() == pytest.approx([0.4, 0.4])   # away catcher 50
    assert math.isnan(out.loc[9, "catcher_framing"])                                  # home side unidentified
    assert t.counts["side_games"] == {"identified": 1, "no_catcher": 1, "too_few_rates": 0}
    assert t.counts["pa_rows"] == {"identified": 2, "no_catcher": 1, "too_few_rates": 0}


def test_the_projected_arm_names_the_teams_projection():
    asof, table = _arm_world()
    out = F.ArmTransform("A_projected", table, asof)(_day(), pd.Timestamp("2026-04-09"))
    # away team 7's history before 04-09: game 3 (catcher 70); home team 8's: game 2 (catcher 60)
    assert out.loc[[7, 8], "catcher_framing"].tolist() == pytest.approx([0.2, 0.2])
    assert out.loc[9, "catcher_framing"] == pytest.approx(0.6)


def test_the_transform_changes_only_the_catcher_column():
    asof, table = _arm_world()
    day = _day()
    out = F.ArmTransform("A_posted", table, asof)(day, pd.Timestamp("2026-04-09"))
    pd.testing.assert_frame_equal(out.drop(columns=["catcher_framing"]), day.drop(columns=["catcher_framing"]))
    assert day["catcher_framing"].tolist() == [0.0, 0.0, 0.0]          # the input is not modified


def test_a_side_game_missing_from_the_table_refuses():
    asof, table = _arm_world()
    day = _day()
    day.loc[9, "game_pk"] = 4
    with pytest.raises(F.InputRefused, match="table"):
        F.ArmTransform("A_posted", table, asof)(day, pd.Timestamp("2026-04-09"))


def test_an_unknown_arm_refuses():
    asof, table = _arm_world()
    with pytest.raises(ValueError):
        F.ArmTransform("B", table, asof)


# ---------------------------------------------------------------- the calendar and dispositions (§6)

def test_the_expected_calendar_is_the_original_portion_dates():
    pa = pd.DataFrame({"date": pd.to_datetime(["2026-04-02", "2026-04-01", "2026-04-02", "2026-04-03"]),
                       "season": 2026, "is_resumed_portion": [False, False, True, True],
                       "batter_id": [1, 2, 3, 4], "game_pk": [1, 2, 3, 4], "is_hit": [0, 1, 1, 1]})
    assert F.expected_calendar(pa) == ["2026-04-01", "2026-04-02"]


@pytest.mark.parametrize("flag", [None, [False, None, True, True], [0, 0, 1, 1]])
def test_the_calendar_refuses_an_absent_or_malformed_resumed_flag(flag):
    pa = pd.DataFrame({"date": pd.to_datetime(["2026-04-01"] * 4), "season": 2026,
                       "batter_id": [1, 2, 3, 4], "game_pk": [1, 2, 3, 4], "is_hit": 0})
    if flag is not None:
        pa["is_resumed_portion"] = pd.Series(flag, dtype="object" if None in flag else None)
    with pytest.raises(F.InputRefused):
        F.expected_calendar(pa)


def _x(rows):
    return np.array(rows, dtype=float)


def test_the_disposition_quantities():
    rng = np.random.default_rng(1)
    x = (rng.uniform(size=(10, 185)) < 0.06).astype(float) - (rng.uniform(size=(10, 185)) < 0.03).astype(float)
    out = F.dispose(x)
    assert out["m"] == pytest.approx(x.mean(axis=1).mean())
    idx = np.random.default_rng(20261009).integers(0, 185, size=(10000, 185))
    assert out["L"] == pytest.approx(float(np.quantile(x.mean(axis=0)[idx].mean(axis=1), 0.1, method="linear")))
    assert out["d"] == pytest.approx(x.mean(axis=1).tolist())
    assert out["seeds_positive"] == int((x.mean(axis=1) > 0).sum())


def test_positive_needs_the_practical_size_a_positive_l_and_six_positive_seeds():
    x = np.zeros((10, 100))
    x[:, :10] = 1.0                                           # every seed +10pp on the same ten days
    assert F.dispose(x)["disposition"] == "positive"
    y = x.copy()
    y[5:, :] = 0.0                                            # five positive seeds only
    assert F.dispose(y)["disposition"] == "inconclusive"


def test_l_of_zero_fails_and_a_tie_is_negative():
    x = np.zeros((10, 50))
    out = F.dispose(x)
    assert out["L"] == 0 and out["m"] == 0 and out["disposition"] == "negative"
    assert out["bootstrap_constant"] is True


def test_below_the_practical_size_is_inconclusive():
    x = np.zeros((10, 1000))
    x[:, :2] = 1.0                                            # +0.2pp
    assert F.dispose(x)["disposition"] == "inconclusive"


def test_the_block_resampling_is_reported_and_does_not_decide():
    rng = np.random.default_rng(3)
    x = (rng.uniform(size=(10, 185)) < 0.08).astype(float) - (rng.uniform(size=(10, 185)) < 0.02).astype(float)
    out = F.dispose(x)
    n = 185
    starts = np.random.default_rng(20261009).integers(0, n, size=(10000, math.ceil(n / 7)))
    idx = ((starts[:, :, None] + np.arange(7)) % n).reshape(10000, -1)[:, :n]
    assert out["L_block7"] == pytest.approx(float(np.quantile(x.mean(axis=0)[idx].mean(axis=1), 0.1, method="linear")))


@pytest.mark.parametrize("shape", [(9, 50), (11, 50), (10, 0)])
def test_anything_but_ten_seeds_over_a_nonempty_calendar_is_incomplete(shape):
    assert F.dispose(np.zeros(shape))["disposition"] == "incomplete"


def test_a_non_finite_delta_is_incomplete():
    x = np.zeros((10, 20))
    x[3, 4] = np.nan
    assert F.dispose(x)["disposition"] == "incomplete"


def test_daily_counts_are_reported():
    x = np.zeros((10, 4))
    x[:, 0] = 1
    x[:, 1] = -1
    x[0, 2] = 1
    out = F.dispose(x)
    assert out["daily"] == {"positive": 2, "negative": 1, "zero": 1, "discordant_seed_days": 21}


def test_an_l_of_exactly_zero_is_not_positive_even_with_the_size_and_the_seeds():
    x = np.zeros((10, 100))
    x[:, :2] = 1.0                       # m = +2pp, all ten seeds positive; ~13% of draws miss both days, so L = 0
    out = F.dispose(x)
    assert out["m"] >= F.PRACTICAL_MIN and out["seeds_positive"] == 10 and out["L"] == 0
    assert out["disposition"] == "inconclusive"


# ---------------------------------------------------------------- review c1 finding 7: malformed records anywhere

@pytest.mark.parametrize("sub, reason", [
    (_player("1999", "201", ()), "malformed_player"),           # a substitute with a string id
    (_player(1999, "201", ()), "malformed_positions"),          # a substitute who batted, with no positions
    (_player(1999, "201", None), "malformed_positions"),
    ({"person": {"id": "x"}}, "malformed_player"),              # a bench record with a bad id
    ({"person": {"id": 1999}, "allPositions": "1"}, "malformed_positions"),
])
def test_a_malformed_non_starting_record_makes_the_side_unidentified(sub, reason):
    away = _nine(100) + [sub]
    recs = {r["fielding_side"]: r for r in F.starter_proxy(_feed(away, _nine(200)), 777, 2026)}
    assert recs["away"]["catcher_id"] is None and recs["away"]["reason"] == reason


def test_well_formed_bench_and_pitcher_records_do_not_disturb_the_proxy():
    away = _nine(100) + [{"person": {"id": 1998}}, _player(1997, None, ("1",)), _player(1996, "201", ("2",))]
    recs = {r["fielding_side"]: r for r in F.starter_proxy(_feed(away, _nine(200)), 777, 2026)}
    assert recs["away"]["catcher_id"] == 102 and recs["away"]["reason"] == "identified"


def _good_table():
    return [{"game_pk": 1, "season": 2026, "game_type": "R", "official_date": "2026-04-01", "game_number": 1,
             "game_number_fallback": False, "fielding_side": side, "team_id": tid, "catcher_id": cid,
             "reason": "identified" if cid else "no_candidate"} for side, tid, cid in (("away", 7, 50), ("home", 8, None))]


def test_a_well_formed_table_is_accepted():
    assert len(F.table_frame(_good_table())) == 2


@pytest.mark.parametrize("damage", [
    lambda t: t[0].pop("reason"),
    lambda t: t[0].update(extra=1),
    lambda t: t[0].update(game_pk=1.0),
    lambda t: t[0].update(game_pk=True),
    lambda t: t[0].update(season=2024),
    lambda t: t[0].update(game_type="S"),
    lambda t: t[0].update(official_date="2026-4-1"),
    lambda t: t[0].update(official_date="2025-04-01"),
    lambda t: t[0].update(game_number=0),
    lambda t: t[0].update(game_number_fallback="no"),
    lambda t: t[0].update(fielding_side="left"),
    lambda t: t[0].update(team_id="7"),
    lambda t: t[0].update(catcher_id=50.0),
    lambda t: t[0].update(reason="no_candidate"),                    # a catcher with a non-identified reason
    lambda t: t[1].update(reason="identified"),                      # identified without a catcher
    lambda t: t[1].update(reason="made_up"),
    lambda t: t[1].update(official_date="2026-04-02"),               # the two sides disagree on the game
    lambda t: t.pop(),                                               # one side only
    lambda t: t.append(dict(t[0])),                                  # a duplicate side
])
def test_a_malformed_or_inconsistent_table_refuses(damage):
    t = _good_table()
    damage(t)
    with pytest.raises(F.InputRefused):
        F.table_frame(t)


@pytest.mark.parametrize("ids", [[1.0, 2.0], [1.5, 2.0], [1, None], [0, 2], [-1, 2]])
def test_pa_game_ids_must_be_exact_positive_integers(ids):
    with pytest.raises(F.InputRefused, match="game_pk"):
        F.pa_game_ids(pd.DataFrame({"game_pk": pd.Series(ids, dtype="object" if None in ids else None)}))


def test_pa_game_ids_returns_the_sorted_distinct_ids():
    assert F.pa_game_ids(pd.DataFrame({"game_pk": [3, 1, 3, 2]})) == [1, 2, 3]


# ---------------------------------------------------------------- review c1 finding 4: catcher history starts in 2019

def test_a_catcher_id_before_2019_refuses():
    df = pd.DataFrame({"season": [2018] * 5 + [2019], "fielding_catcher_id": [50] * 6,
                       "date": pd.to_datetime([f"2018-09-2{i}" for i in range(5)] + ["2019-04-01"]),
                       "pa_borderline_csr": [1.0] * 5 + [0.5]})
    with pytest.raises(F.InputRefused, match="2019"):
        F.history_start_problem(df)


def test_pre_2019_rows_without_a_catcher_id_are_fine():
    df = pd.DataFrame({"season": [2017, 2018, 2019], "fielding_catcher_id": [np.nan, np.nan, 50.0]})
    F.history_start_problem(df)
    F.history_start_problem(pd.DataFrame({"season": [2017, 2019]}).assign(fielding_catcher_id=[np.nan, 7.0]))
