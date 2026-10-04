"""W2.3 MLB forecast benchmark: as-of join core (design docs/superpowers/specs/2026-10-04-mlb-forecast-benchmark-design.md)."""
import pandas as pd

from scripts.audit.mlb_benchmark import core


def _caps():
    return [
        ("2026-07-05T14:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.70, "numberSelections": 5},
                                  {"roundId": 11, "playerId": 1, "probabilityStarter": 0.60, "numberSelections": 1}]),
        ("2026-07-05T17:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.72, "numberSelections": 9},
                                  {"roundId": 10, "playerId": 2, "probabilityStarter": 0.65, "numberSelections": 3}]),
        ("2026-07-05T20:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.75, "numberSelections": 12}]),
    ]


ROUNDS = {10: "2026-07-05", 11: "2026-07-06"}
PLAYERS = {1: 701, 2: 702}


def test_asof_takes_the_latest_capture_at_or_before_the_cutoff_and_never_tomorrows_round():
    fc = core.forecasts_asof(_caps(), ROUNDS, PLAYERS, "2026-07-05", pd.Timestamp("2026-07-05T18:00:00Z"))
    assert fc[701]["p"] == 0.72 and fc[701]["captured_at"] == "2026-07-05T17:00:00Z"
    assert fc[702]["p"] == 0.65
    early = core.forecasts_asof(_caps(), ROUNDS, PLAYERS, "2026-07-05", pd.Timestamp("2026-07-05T15:00:00Z"))
    assert early[701]["p"] == 0.70 and 702 not in early          # round 11 (tomorrow) never used


def test_a_player_missing_from_the_latest_capture_keeps_the_last_earlier_value():
    fc = core.forecasts_asof(_caps(), ROUNDS, PLAYERS, "2026-07-05", pd.Timestamp("2026-07-05T21:00:00Z"))
    assert fc[701]["p"] == 0.75
    assert fc[702]["p"] == 0.65 and fc[702]["captured_at"] == "2026-07-05T17:00:00Z"


def test_join_marks_doubleheader_batters_ambiguous_and_reports_coverage_both_ways():
    slate = pd.DataFrame({"batter_id": [701, 701, 703], "game_pk": [100, 101, 102], "D": [0.8, 0.7, 0.6]})
    fc = {701: {"p": 0.7, "n_sel": 3, "captured_at": "x"}, 702: {"p": 0.6, "n_sel": 1, "captured_at": "x"}}
    joined, cov = core.join_to_slate(slate, fc)
    assert joined["mlb_p"].isna().all()                          # 701 ambiguous (two games), 703 unlisted
    assert cov == {"slate_rows": 3, "mlb_listed": 2, "matched_unique": 0, "ambiguous_doubleheader": 1,
                   "mlb_not_in_slate": 1, "slate_not_listed": 1}
