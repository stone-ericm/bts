import json

from scripts.audit.season_ledger.contest import (players_lookup, rounds_lookup, team_games, unit_status_history,
                                                 units_lookup)
from scripts.audit.season_ledger.sources.static import parse_players, parse_rounds, parse_schedule, parse_units
from tests.scripts.season_ledger.builders import dumps, gz


def _game(pk, away, home, state="Final", number=1):
    return {"gamePk": pk, "gameNumber": number, "officialDate": "2026-05-10",
            "status": {"codedGameState": state[0], "detailedState": state},
            "teams": {"away": {"team": {"abbreviation": away}}, "home": {"team": {"abbreviation": home}}}}


def test_gzip_and_plain_captures_both_parse_and_corrupt_gzip_is_quarantined():
    # Review Focus 4
    body = dumps({"rounds": [{"id": 971, "date": "2026-08-20T08:00:00-04:00", "status": "complete"}]})
    assert parse_rounds("static/rounds/20260704T030011Z.json", body).rows[0]["round_date"] == "2026-08-20"
    assert parse_rounds("static/rounds/20260705T030011Z.json.gz", gz(body)).rows[0]["round_id"] == 971
    bad = parse_rounds("static/rounds/20260706T030011Z.json.gz", b"\x1f\x8bgarbage")
    assert bad.rows == [] and bad.quarantined[0]["reason"].startswith("bad_gzip")


def test_normalized_fields_keep_their_raw_values():
    # Codex plan r2 #2: a wrong-typed feedId is nulled in the typed column but survives in the raw record.
    (row,) = parse_units("static/units/20260801T150000Z.json", dumps({"units": [
        {"id": 2449, "feedId": "RAW_GAME", "roundId": 1009, "status": "scheduled", "lineups": [1, 2]}]})).rows
    assert (row["feed_id"], row["type_mismatch_fields"]) == (None, "feedId")
    assert json.loads(row["record_raw_json"]) == {"id": 2449, "feedId": "RAW_GAME", "roundId": 1009, "status": "scheduled"}


def test_a_capture_without_its_list_is_quarantined_with_its_raw_document():
    # A record that parsed keeps its parsed value; only undecodable bytes stay solely in the sealed bundle.
    (q,) = parse_units("static/units/20260801T150000Z.json", dumps({"error": "RAW_ERROR"})).quarantined
    assert (q["locator"], q["reason"], json.loads(q["record_raw_json"])) == ("file", "missing_units_list",
                                                                             {"error": "RAW_ERROR"})


def test_units_carry_status_and_capture_time_and_empty_lists_have_no_rows():
    rows = parse_units("static/units/20260801T150000Z.json.gz", gz(dumps({"units": [
        {"id": 2449, "feedId": 822679, "roundId": 1009, "status": "postponed"}]}))).rows
    assert (rows[0]["captured_at"], rows[0]["status"], rows[0]["locator"]) == (
        "2026-08-01T15:00:00.000000Z", "postponed", "item=0")
    empty = parse_units("static/units/20260927T230002Z.json.gz", gz(dumps({"units": []})))
    assert empty.rows == [] and empty.quarantined == []
    assert unit_status_history(rows) == {2449: [("2026-08-01T15:00:00.000000Z", "postponed")]}
    grab = parse_units("static/grab_20260927/003_units.json.gz", gz(dumps({"units": [
        {"id": 1, "feedId": 2, "roundId": 3, "status": "scheduled"}]}))).rows
    assert grab[0]["captured_at"] is None


def test_lookups_keep_conflicting_captures():
    units = parse_units("static/units/20260801T150000Z.json", dumps({"units": [{"id": 2449, "feedId": 822679, "roundId": 1009}]})).rows \
        + parse_units("static/units/20260802T150000Z.json", dumps({"units": [{"id": 2449, "feedId": 999999, "roundId": 1009}]})).rows
    assert units_lookup(units)[2449] == {"feed_ids": {822679, 999999}, "round_ids": {1009}}
    players = parse_players("static/players/p.json", dumps({"players": [{"id": 1300, "feedId": 680757, "squadId": 5,
                                                                          "name": "Steven Kwan"}]})).rows
    assert players_lookup(players) == {1300: {680757}}
    rounds = parse_rounds("static/rounds/r.json", dumps({"rounds": [{"id": 1, "date": "2026-03-25T08:00:00-04:00"}]})).rows
    assert rounds_lookup(rounds) == {1: {"2026-03-25"}}


def test_schedule_lists_every_game_including_postponed():
    body = dumps({"dates": [{"date": "2026-05-10", "games": [_game(824765, "TB", "BOS"),
                                                             _game(824999, "TB", "BOS", "Postponed", 2)]}]})
    rows = parse_schedule("schedules/2026-05-10.json", body).rows
    assert team_games(rows)[("2026-05-10", "TB")] == {824765, 824999}
    assert {r["detailed_state"] for r in rows} == {"Final", "Postponed"}


def test_incomplete_schedule_entries_are_quarantined_not_dropped():
    # Codex plan r1 #7: a game without team metadata must never leave the other game looking unique.
    body = dumps({"dates": [{"date": "2026-05-10", "games": [
        _game(824765, "TB", "BOS"),
        {"gamePk": 824999, "teams": {"away": {"team": {}}, "home": {"team": {"abbreviation": "BOS"}}}}]},
        {"date": "2026-05-11"}]})
    parsed = parse_schedule("schedules/2026-05-10.json", body)
    assert [r["game_pk"] for r in parsed.rows] == [824765]
    assert {(q["locator"], q["reason"]) for q in parsed.quarantined} == {
        ("date=0/game=1", "game_missing_team_abbreviation"), ("date=1", "date_without_games_list")}
