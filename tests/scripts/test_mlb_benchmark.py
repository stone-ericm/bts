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
    assert early == {701: {"p": 0.70, "n_sel": 5, "captured_at": "2026-07-05T14:00:00Z", "round_id": 10, "player_id": 1}}


def test_a_player_absent_from_a_newer_stored_sheet_is_absent_not_resurrected():
    fc = core.forecasts_asof(_caps(), ROUNDS, PLAYERS, "2026-07-05", pd.Timestamp("2026-07-05T21:00:00Z"))
    assert set(fc) == {701} and fc[701]["p"] == 0.75


def test_games_come_from_mlbs_own_sheets_never_from_our_slate():
    fc = {701: {"player_id": 1}, 702: {"player_id": 2}, 703: {"player_id": 3}, 704: {"player_id": 4}}
    squads = {1: 11, 2: 12, 3: None, 4: 14}
    units = [{"feedId": 100, "homeSquadId": 11, "awaySquadId": 21},
             {"feedId": 200, "homeSquadId": 12, "awaySquadId": 22}, {"feedId": 201, "homeSquadId": 22, "awaySquadId": 12},
             {"feedId": 300, "homeSquadId": 13, "awaySquadId": 23}]
    assert core.games_by_batter(fc, squads, units) == {701: {100}, 702: {200, 201}, 703: set(), 704: set()}


def test_join_links_only_a_unique_game_and_labels_it_inferred():
    slate = pd.DataFrame({"batter_id": [701, 702, 703, 704, 705], "game_pk": [100, 200, 300, 400, 500],
                          "D": [0.8, 0.7, 0.6, 0.5, 0.4]})
    fc = {b: {"p": 0.5, "captured_at": "x"} for b in (701, 702, 704, 705, 799)}
    games = {701: {100}, 702: {200, 201}, 704: {401}, 705: set(), 799: {900}}   # 702 doubleheader; 704 mismatch
    joined, cov = core.join_to_slate(slate, fc, games)
    assert list(joined["link_status"]) == ["inferred_unique_game", "multi_or_no_game", "not_listed",
                                           "game_mismatch", "multi_or_no_game"]
    assert joined["mlb_p"].notna().sum() == 1 and cov["mlb_not_in_slate"] == 1
