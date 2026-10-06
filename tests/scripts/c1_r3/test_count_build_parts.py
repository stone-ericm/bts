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
    assert any("gamePk" in r for r in V.verify_game(m, pk=8, season=2023, parquet=parquet_counts(m)))
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
