import json

from scripts.audit.season_ledger.sources.pick_files import parse_archive, parse_pick_file, pick_file_state
from tests.scripts.season_ledger.builders import dumps, pick_json


def test_double_down_file_yields_two_slot_rows_with_raw_slot_results():
    data = pick_json("2026-08-20", dd={}, result="miss", slot_results={"pick": "miss", "double_down": "hit"},
                     notification_sent=True, notification_id="dm-1")
    parsed = parse_pick_file("picks/2026-08-20.json", data)
    assert parsed.quarantined == []
    rows = {r["slot"]: r for r in parsed.rows}
    assert (rows["primary"]["locator"], rows["double_down"]["locator"]) == ("slot=primary", "slot=double_down")
    assert rows["primary"]["slot_result_raw"] == "miss" and rows["double_down"]["slot_result_raw"] == "hit"
    assert rows["primary"]["day_result_raw"] == "miss" and rows["primary"]["has_double_down"] is True
    assert rows["double_down"]["batter_id"] == 202 and rows["double_down"]["game_pk"] == 5002
    assert rows["primary"]["obs_id"] != rows["double_down"]["obs_id"]


def test_early_file_records_absent_fields_at_every_depth_and_never_defaults_delivery():
    # Review Focus 2 and spec §4 losslessness: the 3/29–3/30 shape — null game, no later fields.
    data = dumps({"date": "2026-03-29", "run_time": "2026-03-29T15:00:00+00:00", "result": "hit",
                  "bluesky_posted": False, "bluesky_uri": None, "runner_up": None, "double_down": None,
                  "pick": {"batter_id": 101, "batter_name": "Ada Batter", "team": "TB", "game_pk": None,
                           "game_time": None, "lineup_position": 3}})
    (row,) = parse_pick_file("picks/2026-03-29.json", data).rows
    assert (row["game_pk"], row["game_time"], row["bluesky_posted"]) == (None, None, False)
    assert row["notification_sent"] is None and row["slot_result_raw"] is None
    absent = set(row["absent_fields"].split(","))
    assert {"notification_sent", "slot_results", "delivered_at", "model_git_sha", "pick.pitcher_id"} <= absent
    assert "bluesky_posted" not in absent and "pick.game_pk" not in absent     # present as null, not absent


def test_provenance_and_the_raw_record_survive():
    policy = {"objective": "emax_season_best", "best_streak": 18}
    data = pick_json("2026-09-05", policy_decision=policy, feature_env={"BTS_ROOKIE_GATE_K": "20"},
                     feature_env_schema_version="v1", runner_up={"batter_name": "R", "p_game_hit": 0.7})
    (row,) = parse_pick_file("picks/2026-09-05.json", data).rows
    assert row["pick_policy_objective"] == "emax_season_best" and json.loads(row["policy_decision_json"]) == policy
    assert json.loads(row["feature_env_json"]) == {"BTS_ROOKIE_GATE_K": "20"}
    assert row["feature_env_schema_version"] == "v1" and row["pitcher_name"] == "P"
    assert json.loads(row["record_raw_json"]) == json.loads(data)


def test_wrong_types_are_nulled_and_flagged_and_a_slot_needs_its_batter_id():
    data = pick_json("2026-05-01", primary={"lineup_position": "1"}, notification_sent="yes",
                     dd={"batter_id": "202"})
    parsed = parse_pick_file("picks/2026-05-01.json", data)
    (row,) = parsed.rows
    assert row["lineup_position"] is None and row["notification_sent"] is None
    assert set(row["type_mismatch_fields"].split(",")) == {"pick.lineup_position", "notification_sent"}
    (q,) = parsed.quarantined
    assert (q["locator"], q["reason"], json.loads(q["record_raw_json"])["batter_id"]) == (
        "slot=double_down", "slot_missing_batter_id", "202")


def test_a_malformed_game_pk_quarantines_the_slot_but_a_source_null_does_not():
    # Codex plan r3 #2: a present game identity that cannot be typed must never read as an unrecorded game.
    parsed = parse_pick_file("picks/2026-05-01.json", pick_json("2026-05-01", primary={"game_pk": "bad"}, dd={}))
    assert [r["slot"] for r in parsed.rows] == ["double_down"]
    (q,) = parsed.quarantined
    assert (q["locator"], q["reason"], json.loads(q["record_raw_json"])["game_pk"]) == (
        "slot=primary", "slot_bad_game_pk", "bad")
    early = parse_pick_file("picks/2026-03-29.json", pick_json("2026-03-29", primary={"game_pk": None, "game_time": None}))
    assert early.quarantined == [] and early.rows[0]["game_pk"] is None


def test_pick_file_state_keeps_file_level_delivery_facts_when_no_slot_parses():
    # Codex plan r3 #5: the file's own delivery fields stay readable when every slot is quarantined.
    data = pick_json("2026-05-01", primary={"batter_id": "bad"}, notification_sent=True, notification_id="dm-1")
    state = pick_file_state(data, parse_pick_file("picks/2026-05-01.json", data))
    assert (state["slots"], state["complete"]) == (frozenset({"primary"}), False)
    assert (state["file_fields"]["notification_sent"], state["file_fields"]["notification_id"]) == (True, "dm-1")
    unreadable = b'{"date": "2026-05-01", "pick": {'
    assert pick_file_state(unreadable, parse_pick_file("picks/2026-05-01.json", unreadable))["file_fields"] is None


def test_non_json_and_truncated_files_are_quarantined():
    for data in (b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X", b'{"date": "2026-05-01", "pick": {'):
        parsed = parse_pick_file("picks/2026-05-01.json", data)
        assert parsed.rows == [] and parsed.quarantined[0]["reason"].startswith("invalid_json")


def test_file_without_pick_object_is_quarantined_with_its_raw_document():
    parsed = parse_pick_file("picks/2026-05-01.json", dumps({"date": "2026-05-01"}))
    assert parsed.quarantined == [{"source_path": "picks/2026-05-01.json", "locator": "file", "reason": "no_pick_object",
                                   "record_raw_json": '{"date":"2026-05-01"}'}]
    assert parse_pick_file("picks/2026-05-01.json", b"null").quarantined[0]["record_raw_json"] == "null"


def test_archive_rows_carry_prefix_reason_and_time():
    data = pick_json("2026-08-30", deferred_fallback={"reason": "gap_blocked", "deferred_at": "2026-08-30T12:01:00-04:00"})
    (row,) = parse_archive("picks/2026-08-30/deferred_fallback_20260830T120100-0400.json", data).rows
    assert (row["source_kind"], row["archive_prefix"], row["archive_reason"], row["archived_at"]) == (
        "archive", "deferred_fallback", "gap_blocked", "2026-08-30T12:01:00-04:00")


def test_repair_and_manual_versions_parse_as_pick_versions():
    before = parse_pick_file("picks/archive_actual_streak_repair_20260527T101500Z/2026-05-24.json.before",
                             pick_json("2026-05-24", result="miss"), kind="repair_archive").rows[0]
    postponed = parse_pick_file("picks/archive/2026-04-11.json.postponed", pick_json("2026-04-11"),
                                kind="manual_archive").rows[0]
    assert (before["source_kind"], before["archive_prefix"], before["day_result_raw"]) == ("repair_archive", None, "miss")
    assert (postponed["source_kind"], postponed["batter_id"]) == ("manual_archive", 101)


def test_unrecognized_archive_name_is_quarantined():
    parsed = parse_archive("picks/2026-08-30/something_else.json", pick_json("2026-08-30"))
    assert parsed.rows == [] and parsed.quarantined[0]["reason"] == "unrecognized_archive_name"
