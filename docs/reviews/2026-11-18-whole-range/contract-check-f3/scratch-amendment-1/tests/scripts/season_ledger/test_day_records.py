import json

from scripts.audit.season_ledger.sources.day_records import (ACCEPTED_SCHEMAS, decision_objective, parse_decision,
                                                             parse_lineup_evolution, parse_scheduler_state)
from tests.scripts.season_ledger.builders import cand, decision_json, dumps, evolution_jsonl, state_json

EVO = "picks/lineup_evolution_2026-05-01.jsonl"


def test_v1_decision_reads_as_reach57_and_keeps_raw_objective():
    row = parse_decision("picks/2026-06-23/decision.json",
                         decision_json("2026-06-23", action="single", primary=cand(101, 5001),
                                       schema="bts_daily_decision_v1")).rows[0]
    assert (row["objective"], row["objective_raw"]) == ("reach57", None)
    assert (row["primary_batter_id"], row["primary_game_pk"], row["double_down_batter_id"]) == (101, 5001, None)


def test_invalid_v3_objective_is_unknown():
    row = parse_decision("picks/2026-09-05/decision.json",
                         decision_json("2026-09-05", action="double", primary=cand(101, 5001),
                                       double_down=cand(202, 5002), objective="bogus")).rows[0]
    assert row["objective"] == "unknown" and row["objective_raw"] == "bogus"


def test_vendored_decision_rules_match_production():
    from bts import daily_decision as prod
    assert ACCEPTED_SCHEMAS == prod.ACCEPTED_SCHEMAS
    for schema in (*prod.ACCEPTED_SCHEMAS, "bts_daily_decision_v9"):
        for objective in ("reach57", "emax_season_best", "bogus", None, "<absent>"):
            rec = {"schema_version": schema} if objective == "<absent>" else {"schema_version": schema, "objective": objective}
            assert decision_objective(rec) == prod.decision_objective(rec), (schema, objective)


def test_action_source_keeps_raw_and_normalizes_unknown():
    for raw in ("forced", "unknown"):
        row = parse_decision("picks/2026-08-10/decision.json",
                             decision_json("2026-08-10", action="single", primary=cand(101, 5001), source=raw)).rows[0]
        assert (row["action_source_raw"], row["action_source"]) == (raw, "unknown")


def test_invalid_decision_records_are_quarantined_not_crashed_on():
    for doc in ({"schema_version": "bts_daily_decision_v3", "scoreable": True},
                {"schema_version": "bts_daily_decision_v3", "action": {"x": 1}, "scoreable": True, "date": "d"}):
        parsed = parse_decision("picks/2026-08-10/decision.json", dumps(doc))
        assert parsed.rows == [] and parsed.quarantined[0]["reason"] == "invalid_decision_record"
    # Codex plan r2 #3: a decision that names a selection needs a usable batter identity.
    bad = decision_json("2026-08-10", action="single", primary=dict(cand(101, 5001), batter_id="101"))
    parsed = parse_decision("picks/2026-08-10/decision.json", bad)
    assert parsed.rows == [] and parsed.quarantined[0]["reason"] == "decision_candidate_missing_batter_id"
    assert json.loads(parsed.quarantined[0]["record_raw_json"])["primary"]["batter_id"] == "101"
    # Codex plan r3 #2: a present chosen game identity that cannot be typed quarantines too; a source null does not.
    bad_game = decision_json("2026-08-10", action="single", primary=cand(101, "bad"))
    assert parse_decision("picks/2026-08-10/decision.json", bad_game).quarantined[0]["reason"] == (
        "decision_candidate_bad_game_pk")
    unrecorded = decision_json("2026-08-10", action="single", primary=cand(101, None))
    assert parse_decision("picks/2026-08-10/decision.json", unrecorded).quarantined == []


def test_decision_keeps_its_raw_record_and_absent_candidate_fields():
    primary = {"batter_id": 101, "batter_name": "A", "team": "TB", "game_pk": 5001}     # no p_game_hit
    data = decision_json("2026-08-10", action="single", primary=primary)
    row = parse_decision("picks/2026-08-10/decision.json", data).rows[0]
    assert "primary.p_game_hit" in row["absent_fields"].split(",") and row["primary_p_game_hit"] is None
    assert json.loads(row["record_raw_json"]) == json.loads(data)


def test_scheduler_state_keeps_nested_records_losslessly():
    data = state_json("2026-09-19", final_skip_candidate={"primary": cand(303, 7001), "double": None, "streak": 4},
                      delivery_refusals=[{"at": "2026-09-19T18:00:00-04:00",
                                          "archive": "refused_delivery_20260919T180000-0400.json"}],
                      fallback_refreshes=[{"started": "x", "duration_sec": 3.5}])
    row = parse_scheduler_state("picks/2026-09-19/scheduler_state.json", data).rows[0]
    assert row["final_skip_candidate_present"] is True
    assert (row["skip_candidate_batter_id"], row["skip_candidate_game_pk"]) == (303, 7001)
    assert json.loads(row["final_skip_candidate_json"])["streak"] == 4
    assert json.loads(row["delivery_refusals_json"])[0]["archive"] == "refused_delivery_20260919T180000-0400.json"
    assert json.loads(row["fallback_refreshes_json"]) == [{"started": "x", "duration_sec": 3.5}]


def test_lineup_evolution_rows_per_slot_and_every_bad_record_quarantined():
    good = evolution_jsonl("2026-05-01", [({"batter_id": 101, "game_pk": 5001, "team": "TB"}, None),
                                          ({"batter_id": 111, "game_pk": 5011, "team": "TB"},
                                           {"batter_id": 202, "game_pk": 5002, "team": "NYY"})])
    extra = (b"{not json\n" + dumps({"date": "2026-05-01", "primary": "oops", "double_down": None}) + b"\n"
             + dumps({"date": "2026-05-01", "primary": None}) + b"\n"
             + dumps({"date": "2026-05-01", "primary": {"batter_id": None, "game_pk": 5001}}) + b"\n"
             + dumps({"date": "2026-05-01", "primary": {"batter_id": 101, "game_pk": "bad"}}) + b"\n")
    parsed = parse_lineup_evolution(EVO, good + extra)
    assert [(r["locator"], r["batter_id"], r["source_kind"]) for r in parsed.rows] == [
        ("line=1/slot=primary", 101, "lineup_evolution"), ("line=2/slot=primary", 111, "lineup_evolution"),
        ("line=2/slot=double_down", 202, "lineup_evolution")]
    assert [(q["locator"], q["reason"]) for q in parsed.quarantined] == [
        ("line=3", "invalid_json_line"), ("line=4/slot=primary", "slot_not_object"), ("line=5", "line_without_slots"),
        ("line=6/slot=primary", "slot_missing_batter_id"), ("line=7/slot=primary", "slot_bad_game_pk")]
    assert json.loads(parsed.quarantined[0]["record_raw_json"]) == "{not json"      # unparsed text kept as a string
