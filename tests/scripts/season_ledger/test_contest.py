import json

from scripts.audit.season_ledger.contest import entered_rounds, line_round_streaks, slot_history, streak_before
from scripts.audit.season_ledger.sources.contest_ledger import parse_contest_ledger, parse_saver_transitions
from tests.scripts.season_ledger.builders import contest_line, rnd, slot

PATH = "picks/account_state/contest_ledger.jsonl"


def _parse(*lines):
    return parse_contest_ledger(PATH, ("\n".join(lines) + "\n").encode())


def test_null_absent_and_wrong_type_stats_are_kept_distinct():
    parsed = _parse(contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [
        slot(1928, 2513, "hit", hits=None), slot(1927, 1300, "hit", number=2, drop=("atBats",)),
        slot(1926, 1400, "hit", number=3, hits="1"), slot(1925, 1500, {"grade": "hit"}, number=4),
        slot(1924, 1600, "hit", number=5, hits=2 ** 80)])]))
    slots = {r["player_id"]: r for r in parsed.rows if r["row_level"] == "slot"}
    assert (slots[1500]["slot_result"], slots[1500]["slot_result_state"]) == (None, "type_mismatch")
    assert (slots[1600]["hits"], slots[1600]["hits_state"]) == (None, "type_mismatch")     # beyond int64
    assert (slots[2513]["hits"], slots[2513]["hits_state"]) == (None, "null")
    assert (slots[1300]["at_bats"], slots[1300]["at_bats_state"], slots[1300]["absent_fields"]) == (None, "absent", "atBats")
    assert (slots[1300]["hits"], slots[1300]["hits_state"]) == (1, "value")
    assert (slots[1400]["hits"], slots[1400]["hits_state"], slots[1400]["type_mismatch_fields"]) == (
        None, "type_mismatch", "hits")


def test_line_with_a_playerless_slot_is_quarantined_whole():
    # Review Focus 3 / Interpretation I1: identity must be complete.
    playerless = slot(1928, None, None, hits=None, at_bats=None)
    parsed = _parse(contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1927, 1300, "hit"), playerless])]))
    assert parsed.rows == []
    (q,) = parsed.quarantined
    assert (q["locator"], q["reason"]) == ("line=1", "slot_missing_identity_or_result")
    assert json.loads(q["record_raw_json"])["predictions"][0]["roundId"] == 971      # the refused line is kept


def test_in_progress_round_with_null_result_qualifies():
    parsed = _parse(contest_line("2026-08-22T20:00:00Z", [rnd(972, None, None, None,
                                                              [slot(1930, 99, None, hits=None, at_bats=None)])]))
    (s,) = [r for r in parsed.rows if r["row_level"] == "slot"]
    assert parsed.quarantined == [] and (s["slot_result"], s["round_result"], s["round_streak"]) == (None, None, None)


def test_every_line_round_and_slot_is_an_occurrence_with_its_own_raw_record():
    # Codex plan r2 #2: a slotted round's own facts (here a wrong-typed streak) survive in its round row.
    parsed = _parse(contest_line("2026-05-14T14:30:00Z", [rnd(873, "void", 0, 0, []),
                                                          rnd(874, "hit", "RAW_STREAK", 1, [slot(1900, 11, "hit")])]))
    assert [(r["row_level"], r["locator"]) for r in parsed.rows] == [
        ("line", "line=1"), ("round", "line=1/round=0"), ("round", "line=1/round=1"), ("slot", "line=1/round=1/slot=0")]
    line, slotless, slotted, s = parsed.rows
    assert "predictions" not in json.loads(line["record_raw_json"])
    assert json.loads(slotless["record_raw_json"]) == {"roundId": 873, "result": "void", "streak": 0, "streakIncrease": 0}
    assert json.loads(slotted["record_raw_json"])["streak"] == "RAW_STREAK" and slotted["round_streak"] is None
    assert slotted["type_mismatch_fields"] == "streak" and s["round_streak"] is None


def test_round_growth_and_later_drop_are_tracked():
    parsed = _parse(
        contest_line("2026-08-20T20:00:00Z", [rnd(971, "hit", 7, 1, [slot(1928, 2513, "hit")])]),
        contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1928, 2513, "hit"),
                                                                     slot(1927, 1300, "hit", number=2)])]),
        contest_line("2026-08-22T14:30:00Z", [rnd(972, "not_hit", 0, -8, [slot(1930, 99, "not_hit", hits=0)])]))
    hist = {(h["round_id"], h["player_id"]): h for h in slot_history(parsed.rows)}
    a, b = hist[(971, 2513)], hist[(971, 1300)]
    assert (a["first_seen"], a["last_seen"], a["n_observations"]) == (
        "2026-08-20T20:00:00.000000Z", "2026-08-21T14:30:00.000000Z", 2)
    assert a["changed"] is True and b["changed"] is False
    assert b["first_seen"] == "2026-08-21T14:30:00.000000Z"
    assert a["dropped_later"] is True and b["dropped_later"] is True     # round 971 absent from the 8/22 line
    assert hist[(972, 99)]["dropped_later"] is False
    assert (a["slot_result"], a["round_streak"]) == ("hit", 8)           # the older positive is kept


def test_a_malformed_later_line_does_not_mark_older_slots_dropped():
    parsed = _parse(contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1928, 2513, "hit")])]),
                    '{"recorded_at": "2026-08-22T14:30:00Z", "predictions": [{"result": "hit"}]}')
    assert parsed.quarantined[0]["locator"] == "line=2"
    (h,) = slot_history(parsed.rows)
    assert h["dropped_later"] is False and h["last_seen"] == "2026-08-21T14:30:00.000000Z"


def test_streak_before_uses_the_previous_entered_round_in_the_same_line():
    # Interpretation I2: no line ever reports 970, so it was a skip day and 971's previous entered round is 969.
    parsed = _parse(contest_line("2026-08-22T14:30:00Z", [
        rnd(969, "hit", 5, 1, [slot(1900, 11, "hit")]),
        rnd(971, "hit", 7, 2, [slot(1928, 2513, "hit")]),
        rnd(972, None, None, None, [slot(1930, 99, None, hits=None, at_bats=None)])]))
    streaks, entered = line_round_streaks(parsed.rows)[1], entered_rounds(parsed.rows)
    assert streak_before(streaks, 971, entered) == 5
    assert streak_before(streaks, 969, entered) is None
    assert streak_before(streaks, 972, entered) == 7


def test_streak_before_is_unknown_when_the_previous_entered_round_is_missing_from_the_line():
    # Codex plan r1 #6: line 1 reports 969 and 970; line 2 dropped 970, so 971's predecessor is not in line 2.
    parsed = _parse(contest_line("2026-08-21T14:30:00Z", [rnd(969, "hit", 5, 1, [slot(1900, 11, "hit")]),
                                                          rnd(970, "not_hit", 0, -5, [slot(1901, 12, "not_hit", hits=0)])]),
                    contest_line("2026-08-22T14:30:00Z", [rnd(969, "hit", 5, 1, [slot(1900, 11, "hit")]),
                                                          rnd(971, "hit", 1, 1, [slot(1928, 2513, "hit")])]))
    assert streak_before(line_round_streaks(parsed.rows)[2], 971, entered_rounds(parsed.rows)) is None


def test_saver_transitions_are_reported_as_attempts_with_their_raw_record():
    data = (json.dumps({"ts": "2026-08-10T12:00:00+00:00", "source": "auto", "outcome": "rejected",
                        "new_state": "used"}) + "\n").encode()
    (row,) = parse_saver_transitions("picks/account_state/saver_transitions.jsonl", data).rows
    assert (row["attempted_at"], row["attempt_source"], row["attempt_outcome"]) == (
        "2026-08-10T12:00:00.000000Z", "auto", "rejected")
    assert json.loads(row["record_raw_json"])["new_state"] == "used"
