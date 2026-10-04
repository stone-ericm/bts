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
    units = [{"feedId": 100, "roundId": 10, "homeSquadId": 11, "awaySquadId": 21},
             {"feedId": 200, "roundId": 10, "homeSquadId": 12, "awaySquadId": 22},
             {"feedId": 201, "roundId": 10, "homeSquadId": 22, "awaySquadId": 12},
             {"feedId": 300, "roundId": 10, "homeSquadId": 13, "awaySquadId": 23}]
    assert core.games_by_batter(fc, squads, units, round_id=10) == {701: {100}, 702: {200, 201}, 703: set(), 704: set()}


def test_join_links_only_a_unique_game_and_labels_it_inferred():
    slate = pd.DataFrame({"batter_id": [701, 702, 703, 704, 705], "game_pk": [100, 200, 300, 400, 500],
                          "D": [0.8, 0.7, 0.6, 0.5, 0.4]})
    fc = {b: {"p": 0.5, "captured_at": "x"} for b in (701, 702, 704, 705, 799)}
    games = {701: {100}, 702: {200, 201}, 704: {401}, 705: set(), 799: {900}}   # 702 doubleheader; 704 mismatch
    joined, cov = core.join_to_slate(slate, fc, games)
    assert list(joined["link_status"]) == ["inferred_unique_game", "multi_or_no_game", "not_listed",
                                           "game_mismatch", "multi_or_no_game"]
    assert joined["mlb_p"].notna().sum() == 1 and cov["mlb_not_in_slate"] == 1


def test_a_newer_sheet_without_the_round_or_empty_is_no_support_never_a_fallback():
    caps = _caps()[:2] + [("2026-07-05T20:00:00Z", [{"roundId": 11, "playerId": 1, "probabilityStarter": 0.6}])]
    stamp, rows = core.latest_sheet(caps, ROUNDS, "2026-07-05", pd.Timestamp("2026-07-05T21:00:00Z"))
    assert stamp == "2026-07-05T20:00:00Z" and rows == []
    assert core.forecasts_asof(caps, ROUNDS, PLAYERS, "2026-07-05", pd.Timestamp("2026-07-05T21:00:00Z")) == {}
    empty = _caps()[:2] + [("2026-07-05T20:00:00Z", [])]
    assert core.latest_sheet(empty, ROUNDS, "2026-07-05", pd.Timestamp("2026-07-05T21:00:00Z")) == ("2026-07-05T20:00:00Z", [])
    assert core.latest_sheet(_caps(), ROUNDS, "2026-07-05", pd.Timestamp("2026-07-05T13:00:00Z")) == (None, [])


def test_an_unresolved_unit_makes_multiplicity_unknown_and_other_rounds_are_ignored():
    fc = {701: {"player_id": 1}, 702: {"player_id": 2}}
    squads = {1: 11, 2: 12}
    units = [{"feedId": 100, "roundId": 10, "homeSquadId": 11, "awaySquadId": 21},
             {"feedId": None, "roundId": 10, "homeSquadId": 11, "awaySquadId": 22},   # 701's possible second game
             {"feedId": 200, "roundId": 10, "homeSquadId": 12, "awaySquadId": 23},
             {"feedId": 201, "roundId": 11, "homeSquadId": 12, "awaySquadId": 24}]   # tomorrow: not today's game
    assert core.games_by_batter(fc, squads, units, round_id=10) == {701: None, 702: {200}}
    unknown_squad = units[:1] + [{"feedId": 300, "roundId": 10, "homeSquadId": None, "awaySquadId": 25}]
    assert core.games_by_batter(fc, squads, unknown_squad, round_id=10) == {701: None, 702: None}
    no_round = units[:1] + [{"feedId": 400, "roundId": None, "homeSquadId": 30, "awaySquadId": 31}]
    assert core.games_by_batter(fc, squads, no_round, round_id=10) == {701: None, 702: None}
    slate = pd.DataFrame({"batter_id": [701, 702], "game_pk": [100, 200], "D": [0.8, 0.7]})
    fc2 = {701: {"p": 0.5, "captured_at": "x"}, 702: {"p": 0.6, "captured_at": "x"}}
    joined, _ = core.join_to_slate(slate, fc2, {701: None, 702: {200}})
    assert list(joined["link_status"]) == ["multiplicity_unknown", "inferred_unique_game"]


def test_round_resolution_needs_exactly_one_round_for_the_date():
    rounds = [{"id": 10, "date": "2026-07-05T00:00:00Z"}, {"id": 11, "date": "2026-07-06T00:00:00Z"},
              {"id": 12, "date": "2026-07-06T00:00:00Z"}]
    assert core.round_for_date(rounds, "2026-07-05") == (10, None)
    assert core.round_for_date(rounds, "2026-07-06") == (None, "multiple_rounds")
    assert core.round_for_date(rounds, "2026-07-07") == (None, "no_round")


def test_player_lookup_excludes_ids_listed_twice_with_conflicting_values():
    players = [{"id": 1, "feedId": 701, "squadId": 11}, {"id": 2, "feedId": 702, "squadId": 12},
               {"id": 2, "feedId": 799, "squadId": 12}, {"id": 3, "feedId": 703, "squadId": 13},
               {"id": 3, "feedId": 703, "squadId": 13}]
    feed, squad, conflicts = core.player_lookup(players)
    assert feed == {1: 701, 3: 703} and squad == {1: 11, 3: 13} and conflicts == [2]


def test_units_are_complete_only_when_every_scheduled_game_has_a_unit_in_the_round():
    units = [{"feedId": 100, "roundId": 10}, {"feedId": 200, "roundId": 10}, {"feedId": 300, "roundId": 11}]
    assert core.units_complete(units, 10, {100, 200}) is True
    assert core.units_complete(units, 10, {100, 200, 250}) is False
    assert core.units_complete(units, 10, set()) is False        # no schedule evidence: completeness not established


def test_unchanged_since_walks_back_through_stored_sheets_while_the_value_is_identical():
    caps = [("2026-07-05T10:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.70}]),
            ("2026-07-05T12:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.72}]),
            ("2026-07-05T14:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.72}]),
            ("2026-07-05T16:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.72}])]
    assert core.unchanged_since(caps, "2026-07-05T16:00:00Z", 10, 1) == "2026-07-05T12:00:00Z"
    gap = caps[:2] + [("2026-07-05T14:00:00Z", [])] + caps[3:]   # absent in between: the run stops there
    assert core.unchanged_since(gap, "2026-07-05T16:00:00Z", 10, 1) == "2026-07-05T16:00:00Z"
