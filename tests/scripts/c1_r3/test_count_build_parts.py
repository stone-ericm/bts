"""T2 verification/quarantine, T3 count table, T4 starter workload (synthetic feeds)."""
from collections import Counter

import pytest

from scripts.audit.c1_r3 import count_bf as B
from scripts.audit.c1_r3 import count_meta as M
from scripts.audit.c1_r3 import count_table as T
from scripts.audit.c1_r3 import count_verify as V
from tests.scripts.c1_r3.feeds import feed, lineup, play


def parquet_counts(meta):
    """What the production PA parquet would hold for this game: every PA (resumed included) by (is_home, batter)."""
    return Counter((p.side == "home", p.batter) for p in meta.pas)


# ---------- T2 ----------
def test_a_clean_game_is_certified():
    m = M.extract(feed(pk=7))
    assert V.verify_game(m, pk=7, season=2023, parquet=parquet_counts(m)) == []


def test_quarantine_reasons():
    m = M.extract(feed(pk=7))
    assert any("identity" in r for r in V.verify_game(m, pk=8, season=2023, parquet=parquet_counts(m)))
    top = M.extract(feed(pk=7, top_pk=8))
    assert any("identity" in r for r in V.verify_game(top, pk=7, season=2023, parquet=parquet_counts(top)))
    assert any("officialDate" in r for r in V.verify_game(m, pk=7, season=2023, parquet=parquet_counts(m),
                                                            parquet_date="2023-06-02"))
    assert any("season" in r for r in V.verify_game(m, pk=7, season=2022, parquet=parquet_counts(m)))
    short = parquet_counts(m)
    short[(False, 101)] -= 1
    assert any("completeness" in r for r in V.verify_game(m, pk=7, season=2023, parquet=short))
    stranger = M.extract(feed(pk=7, plays=[play(0, "top", 777, 250), play(1, "bottom", 201, 150)]))
    assert any("not in the away lineup" in r for r in V.verify_game(stranger, pk=7, season=2023,
                                                                       parquet=parquet_counts(stranger)))
    broken = M.extract(feed(pk=7, away=lineup(100, n=8)))
    assert any("slots" in r for r in V.verify_game(broken, pk=7, season=2023, parquet=parquet_counts(broken)))


def test_census_counts_every_quarantine_and_stops_above_one_percent():
    results = {pk: [] for pk in range(1, 201)}
    results[5] = ["x"]
    results[6] = ["y"]
    c = V.census(results, eligible=set(range(1, 201)), feeds_without_parquet={999})
    assert c["quarantined"] == {5: ["x"], 6: ["y"]} and c["rate"] == pytest.approx(0.01) and c["stop"] is False
    results[7] = ["z"]
    c = V.census(results, eligible=set(range(1, 201)), feeds_without_parquet=set())
    assert c["stop"] is True and c["eligible"] == 200
    missing = V.census({1: []}, eligible={1, 2}, feeds_without_parquet=set())
    assert missing["quarantined"] == {2: ["no re-acquired feed for an eligible game"]}


# ---------- T3 ----------
def test_starter_pa_counts_exclude_the_resumed_portion_and_substitutes():
    plays = [play(0, "top", 101, 250, start="2023-06-01T23:00:00Z"), play(1, "top", 101, 250, start="2023-06-01T23:30:00Z"),
             play(2, "top", 111, 250, start="2023-06-01T23:40:00Z"),          # a substitute in slot 1
             play(3, "top", 102, 250, start="2023-06-02T20:00:00Z")]          # resumed portion
    m = M.extract(feed(plays=plays, away=lineup(100, subs=[(111, "101")]), resume="2023-06-02T19:00:00Z"))
    rows = {(r["slot"], r["is_home"]): r["n"] for r in T.starter_counts(m)}
    assert rows[(1, False)] == 2 and rows[(2, False)] == 0 and rows[(1, True)] == 0


def test_count_table_conditions_on_n_at_least_one_caps_at_eight_and_smooths():
    rows = [{"slot": 1, "is_home": False, "n": n} for n in (0, 1, 4, 4, 5, 9, 11)]
    tab = T.count_table(rows)
    cell = tab["cells"]["1|away"]
    assert cell["counts"] == {"1": 1, "2": 0, "3": 0, "4": 2, "5": 1, "6": 0, "7": 0, "8": 2}
    assert cell["n_zero_excluded"] == 1 and cell["overflow_gt8"] == 2
    p = cell["p"]
    assert p["4"] == pytest.approx((2 + 1) / (6 + 8)) and sum(p.values()) == pytest.approx(1.0)
    assert set(tab["cells"]) == {f"{s}|{h}" for s in range(1, 10) for h in ("away", "home")}   # empty cells kept
    assert tab["cells"]["9|home"]["p"]["1"] == pytest.approx(1 / 8)                              # add-one only


# ---------- T4 ----------
def test_starter_workload_counts_every_pa_faced_including_the_resumed_portion():
    plays = [play(0, "top", 101, 250, start="2023-06-01T23:00:00Z"), play(1, "top", 102, 250, start="2023-06-01T23:10:00Z"),
             play(2, "top", 103, 260, start="2023-06-01T23:20:00Z"),          # relief
             play(3, "bottom", 201, 150, start="2023-06-01T23:30:00Z"),
             play(4, "top", 104, 250, start="2023-06-02T20:00:00Z")]          # resumed: the starter returns
    m = M.extract(feed(plays=plays, home_pitchers=(250, 260), resume="2023-06-02T19:00:00Z"))
    starts = {s["side"]: s for s in B.starts(m)}
    assert starts["home"]["pitcher"] == 250 and starts["home"]["bf"] == 3
    assert starts["away"]["pitcher"] == 150 and starts["away"]["bf"] == 1
    assert starts["home"]["official_date"] == "2023-06-01"


def test_league_median_bf():
    assert B.league_median([{"bf": b} for b in (20, 25, 30, 18)]) == 22.5
    with pytest.raises(ValueError):
        B.league_median([])


# ---------- review r1 F4: batting order, substitutions and zero-PA starters (independent expected counts) ----------
def full_game(**kw):
    m = M.extract(feed(pk=7, **kw))
    return m, V.verify_game(m, pk=7, season=2023, parquet=parquet_counts(m))


def test_swapped_slot_codes_break_the_batting_order():
    away = lineup(100)
    away["ID101"]["battingOrder"], away["ID102"]["battingOrder"] = "200", "100"
    assert any("batting order" in r for r in full_game(away=away)[1])


def test_a_starter_returning_after_his_substitute_is_refused():
    plays, i = [], 0
    order = [111] + list(range(102, 110)) + [101]          # sub 111 bats in slot 1, then starter 101 returns
    for k, b in enumerate(order):
        plays.append(play(i, "top", b, 250, inning=k // 3 + 1)); i += 1
    for k in range(1, 10):
        plays.append(play(i, "bottom", 200 + k, 150, inning=4)); i += 1
    m, reasons = full_game(plays=plays, away=lineup(100, subs=[(111, "101")]))
    assert any("after a substitute" in r for r in reasons)


def test_a_starter_without_a_pa_needs_replacement_evidence():
    plays, i = [], 0
    for b in [111] + list(range(102, 110)):                # slot 1's sub bats instead of the starter: evidenced
        plays.append(play(i, "top", b, 250, inning=1)); i += 1
    for k in range(1, 10):
        plays.append(play(i, "bottom", 200 + k, 150, inning=1)); i += 1
    assert not any("no PA" in r for r in full_game(plays=plays, away=lineup(100, subs=[(111, "101")]))[1])
    short = [play(0, "top", 101, 250, inning=1), play(1, "top", 102, 250, inning=1), play(2, "bottom", 201, 150, inning=1)]
    m, reasons = full_game(plays=short)                     # the lineup never reached slots 3-9 (away) / 2-9 (home)
    assert not any("no PA" in r for r in reasons)
    skip = [play(0, "top", 101, 250, inning=1), play(1, "top", 103, 250, inning=1), play(2, "bottom", 201, 150, inning=1)]
    assert any("no PA" in r or "batting order" in r for r in full_game(plays=skip)[1])


def test_an_intentional_walk_advances_the_order_but_is_not_a_production_pa():
    plays, i = [], 0
    for k in range(1, 10):
        plays.append(play(i, "top", 100 + k, 250, event="intent_walk" if k == 4 else "single", inning=1)); i += 1
    for k in range(1, 10):
        plays.append(play(i, "bottom", 200 + k, 150, inning=1)); i += 1
    m, reasons = full_game(plays=plays)
    assert reasons == [] and len(m.pas) == 17 and len(m.plays) == 18
    rows = {(r["slot"], r["is_home"]): r["n"] for r in T.starter_counts(m)}
    assert rows[(4, False)] == 0                           # starter 104 batted only an IBB: no production PA


def test_an_inning_ending_caught_stealing_lets_the_batter_lead_off_next_inning():
    plays = [play(0, "top", 101, 250, inning=1), play(1, "top", 102, 250, event="caught_stealing_2b", inning=1),
             play(2, "bottom", 201, 150, inning=1), play(3, "top", 102, 250, inning=2)]
    m, reasons = full_game(plays=plays)
    assert not any("batting order" in r for r in reasons)
