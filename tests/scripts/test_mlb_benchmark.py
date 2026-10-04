"""W2.3 MLB forecast benchmark: as-of join core (design rev 2, gate 1)."""
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


def test_asof_uses_the_latest_whole_sheet_at_or_before_the_cutoff_and_never_tomorrows_round():
    fc = core.forecasts_asof(_caps(), ROUNDS, PLAYERS, "2026-07-05", pd.Timestamp("2026-07-05T18:00:00Z"))
    assert fc[701]["p"] == 0.72 and fc[701]["captured_at"] == "2026-07-05T17:00:00Z" and fc[702]["p"] == 0.65
    early = core.forecasts_asof(_caps(), ROUNDS, PLAYERS, "2026-07-05", pd.Timestamp("2026-07-05T15:00:00Z"))
    assert early == {701: {"p": 0.70, "n_sel": 5, "captured_at": "2026-07-05T14:00:00Z", "round_id": 10}}


def test_a_player_absent_from_a_newer_stored_sheet_is_absent_not_resurrected():
    fc = core.forecasts_asof(_caps(), ROUNDS, PLAYERS, "2026-07-05", pd.Timestamp("2026-07-05T21:00:00Z"))
    assert set(fc) == {701} and fc[701]["p"] == 0.75


def test_join_links_only_a_unique_scheduled_game_and_labels_it_inferred():
    slate = pd.DataFrame({"batter_id": [701, 702, 703, 704], "team": ["NYM", "ATL", "LAD", "SD"],
                          "game_pk": [100, 200, 300, 400], "D": [0.8, 0.7, 0.6, 0.5]})
    fc = {701: {"p": 0.7, "captured_at": "x"}, 702: {"p": 0.6, "captured_at": "x"},
          704: {"p": 0.5, "captured_at": "x"}, 799: {"p": 0.5, "captured_at": "x"}}
    team_games = {"NYM": {100}, "ATL": {200, 201}, "LAD": {300}, "SD": {401}}   # ATL doubleheader; SD mismatch
    joined, cov = core.join_to_slate(slate, fc, team_games)
    assert list(joined["link_status"]) == ["inferred_unique_game", "multi_or_no_game", "not_listed", "game_mismatch"]
    assert joined["mlb_p"].notna().sum() == 1 and cov["mlb_not_in_slate"] == 1
