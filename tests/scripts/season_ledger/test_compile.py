import json
import random
import shutil

import pyarrow.parquet as pq
import pytest

from scripts.audit.season_ledger.compile import compile_bundle
from tests.scripts.season_ledger.builders import (cand, contest_line, decision_json, dumps, gz, pick_json, rnd,
                                                  seal_bundle, slot, state_json)

ROUNDS = {971: "2026-08-20", 972: "2026-08-21", 973: "2026-08-22", 974: "2026-08-23", 975: "2026-08-24",
          976: "2026-08-25", 977: "2026-08-26", 978: "2026-08-27"}
PLAYERS = {2513: 802415, 1300: 202, 1001: 101, 1777: 777, 1404: 404, 1505: 505, 1606: 606, 1707: 707, 1808: 808,
           1708: 708}


def _game(pk, away, home):
    return {"gamePk": pk, "status": {"codedGameState": "F", "detailedState": "Final"},
            "teams": {"away": {"team": {"abbreviation": away}}, "home": {"team": {"abbreviation": home}}}}


def _schedule(day, *games):
    return dumps({"dates": [{"date": day, "games": list(games)}]})


def _units(*units):
    return gz(dumps({"units": [{"id": u, "feedId": f, "roundId": r, "status": s} for u, f, r, s in units]}))


def _state(day):
    return state_json(day, pick_locked=True, pick_locked_at=f"{day}T17:00:00-04:00", committed_pick_written=True)


def _season_files() -> dict[str, bytes]:
    """8/20 C-03 double; 8/21 entered-but-undelivered preview; 8/22 contest-only; 8/23 Pass on a game postponed
    after lock; 8/24 game postponed before lock; 8/25 saver-absorbed miss; 8/26 partial-void double; 8/27 a
    postponement superseded by a later 'scheduled' capture before lock; 8/28 a single decision against a double
    pick file. Round facts are coherent."""
    ledger = contest_line("2026-08-28T14:30:00Z", [
        rnd(971, "hit", 8, 2, [slot(1927, 1300, "hit", number=1), slot(1928, 2513, "hit", number=2)]),
        rnd(972, "hit", 9, 1, [slot(1929, 1001, "hit")]),
        rnd(973, "hit", 10, 1, [slot(1930, 1777, "hit")]),
        rnd(974, "void", 10, 0, [slot(2404, 1404, "void", hits=0, at_bats=0)]),
        rnd(975, "void", 10, 0, [slot(2505, 1505, "void", hits=0, at_bats=0)]),
        rnd(976, "used_mulligan", 10, 0, [slot(1931, 1606, "not_hit", hits=0)]),
        rnd(977, "hit", 11, 1, [slot(1932, 1707, "void", hits=0, at_bats=0), slot(1933, 1808, "hit", number=2)]),
        rnd(978, "hit", 12, 1, [slot(2708, 1708, "hit")])])
    return {
        "picks/2026-08-20.json": pick_json("2026-08-20", primary={"batter_id": 802415, "game_pk": 822934, "team": "TB"},
                                           dd={}, result="miss", slot_results={"pick": "miss", "double_down": "hit"},
                                           notification_sent=True, notification_id="dm-1"),
        "picks/2026-08-20/decision.json": decision_json("2026-08-20", action="double", primary=cand(802415, 822934),
                                                        double_down=cand(202, 5002, team="NYY")),
        "picks/2026-08-20/scheduler_state.json": _state("2026-08-20"),
        "picks/2026-08-21.json": pick_json("2026-08-21"),
        "picks/2026-08-23/decision.json": decision_json("2026-08-23", action="single", primary=cand(404, 6004, team="SEA")),
        "picks/2026-08-23/scheduler_state.json": _state("2026-08-23"),
        "picks/2026-08-24/decision.json": decision_json("2026-08-24", action="single", primary=cand(505, 6005, team="SEA")),
        "picks/2026-08-24/scheduler_state.json": _state("2026-08-24"),
        "picks/2026-08-25/decision.json": decision_json("2026-08-25", action="single", primary=cand(606, 6006, team="SEA")),
        "picks/2026-08-26/decision.json": decision_json("2026-08-26", action="double", primary=cand(707, 6007),
                                                        double_down=cand(808, 6008, team="NYY")),
        "picks/2026-08-27/decision.json": decision_json("2026-08-27", action="single", primary=cand(708, 6708, team="SEA")),
        "picks/2026-08-27/scheduler_state.json": _state("2026-08-27"),
        "picks/2026-08-28/decision.json": decision_json("2026-08-28", action="single", primary=cand(909, 6909)),
        "picks/2026-08-28.json": pick_json("2026-08-28", primary={"batter_id": 909, "game_pk": 6909},
                                           dd={"batter_id": 910, "game_pk": 6910}),
        "picks/._2026-08-20.json": b"\x00\x05\x16\x07",
        "picks/2026-08-21.shadow.json": pick_json("2026-08-21", result="hit"),      # excluded, yet in the recipe universe
        "picks/account_state/contest_ledger.jsonl": (ledger + "\n").encode(),
        "static/rounds/20260828T120000Z.json.gz": gz(dumps({"rounds": [
            {"id": r, "date": f"{d}T08:00:00-04:00"} for r, d in ROUNDS.items()]})),
        "static/players/20260828T120000Z.json.gz": gz(dumps({"players": [
            {"id": p, "feedId": f} for p, f in PLAYERS.items()]})),
        "static/units/20260823T150000Z.json.gz": _units((2404, 6004, 974, "scheduled")),
        "static/units/20260824T020000Z.json.gz": _units((2404, 6004, 974, "postponed")),   # after the 8/23 lock
        "static/units/20260824T150000Z.json.gz": _units((2505, 6005, 975, "postponed")),   # before the 8/24 lock
        "static/units/20260827T100000Z.json.gz": _units((2708, 6708, 978, "postponed")),
        "static/units/20260827T200000Z.json.gz": _units((2708, 6708, 978, "scheduled")),   # superseded before lock
        "schedules/2026-08-20.json": _schedule("2026-08-20", _game(822934, "TOR", "TB"), _game(5002, "NYY", "BAL")),
        "schedules/2026-08-21.json": _schedule("2026-08-21", _game(5001, "BOS", "TB")),
        "schedules/2026-08-22.json": _schedule("2026-08-22", _game(5777, "LAD", "SD")),
        "schedules/2026-08-25.json": _schedule("2026-08-25", _game(6006, "SEA", "HOU")),
        "schedules/2026-08-26.json": _schedule("2026-08-26", _game(6007, "TB", "CLE"), _game(6008, "NYY", "KC")),
    }


FILES = _season_files()     # one byte set, reused by every test (Codex plan r1 #12)


def _compile(tmp_path, name, files=None):
    bundle = tmp_path / f"bundle_{name}"
    seal_bundle(bundle, FILES if files is None else files, missing=["schedules/2026-08-23.json"])
    out = tmp_path / f"out_{name}"
    compile_bundle(bundle, out, uv_lock_sha256="test-lock")
    return bundle, out


def _ledger(out):
    return {r["row_id"]: r for r in pq.read_table(out / "season_2026_ledger.parquet").to_pylist()}


def test_c03_disagreement_slot_order_and_streaks(tmp_path):
    led = _ledger(_compile(tmp_path, "a")[1])
    p, dd = led["2026-08-20|primary|802415|822934"], led["2026-08-20|double_down|202|5002"]
    assert (p["match"], p["entry_status"], p["bts_outcome"], p["local_slot_result_raw"]) == (
        "inferred", "confirmed", "hit", "miss")
    assert p["local_vs_contest_disagreement"] is True and dd["local_vs_contest_disagreement"] is False
    assert (p["slot_number"], dd["slot_number"]) == (2, 1)            # the contest's order differs from ours
    assert (p["streak_before"], p["streak_after"], p["saver_available_before"]) == (None, 8, None)
    assert p["commit_status"] == "committed_evidenced" and p["locked_at"] == "2026-08-20T21:00:00.000000Z"
    assert (p["scheduled_games"], p["round_id"]) == (2, 971)


def test_entered_preview_contest_only_and_unobserved_days(tmp_path):
    led = _ledger(_compile(tmp_path, "a")[1])
    preview = led["2026-08-21|primary|101|5001"]
    assert (preview["commit_status"], preview["entry_status"], preview["match"]) == ("unconfirmed", "confirmed", "inferred")
    only = led["contest|973|1930|1777"]
    assert (only["row_kind"], only["match"], only["bts_outcome_status"], only["slot"]) == (
        "contest_only", "unmapped", "unmapped", None)
    assert led["day|2026-08-22|unobserved_day"]["scheduled_games"] == 1
    assert led["day|2026-03-25|unobserved_day"]["reason"] == "no_evidence"


def test_a_pass_is_an_outcome_and_eligibility_reads_the_latest_status_before_lock(tmp_path):
    led = _ledger(_compile(tmp_path, "a")[1])
    after, before = led["2026-08-23|primary|404|6004"], led["2026-08-24|primary|505|6005"]
    superseded = led["2026-08-27|primary|708|6708"]
    assert (after["match"], after["bts_outcome"], after["contest_norm"]) == ("evidenced", "void", "HOLD")
    assert after["game_eligibility"] == "unknown"                     # postponed only after lock
    assert (before["game_eligibility"], before["game_eligibility_at"]) == (
        "postponed_evidenced", "2026-08-24T15:00:00.000000Z")
    assert superseded["game_eligibility"] == "unknown"                # Codex plan r1 #8: rescheduled before lock
    assert after["streak_before"] == 10 and after["scheduled_games"] is None   # 8/23 schedule declared missing


def test_round_labels_stay_on_the_round_and_legs_keep_their_own_grades(tmp_path):
    led = _ledger(_compile(tmp_path, "a")[1])
    saver = led["2026-08-25|primary|606|6006"]
    assert (saver["bts_outcome"], saver["contest_round_result"], saver["contest_norm"]) == ("not_hit", "used_mulligan", "NO_HIT")
    assert saver["saver_available_before"] is None
    p, dd = led["2026-08-26|primary|707|6007"], led["2026-08-26|double_down|808|6008"]
    assert (p["bts_outcome"], p["contest_norm"], dd["bts_outcome"]) == ("void", "HOLD", "hit")
    assert p["contest_round_result"] == dd["contest_round_result"] == "hit"


def test_every_occurrence_is_accounted_resolvable_and_joined_to_the_reconciliation(tmp_path):
    out = _compile(tmp_path, "a")[1]
    occ = pq.read_table(out / "season_2026_ledger_occurrences.parquet").to_pylist()
    states = {(o["source_path"], o["state"]) for o in occ if o["locator"] == "file"}
    assert ("picks/._2026-08-20.json", "excluded") in states
    assert ("schedules/2026-08-23.json", "declared_missing") in states
    emitted = {o["obs_id"]: o for o in occ if o["state"] == "emitted"}
    assert all(o["disposition"] and o["fields_json"] for o in emitted.values())
    led = _ledger(out)
    assert all(r[c] in emitted for r in led.values() for c in ("pick_obs_id", "decision_obs_id", "contest_obs_id")
               if r[c] is not None)
    build = json.loads((out / "season_2026_ledger_build.json").read_text())
    assert build["row_kinds"]["contest_only"] == 1 and len(build["rules_fingerprint"]) == 64
    recon = pq.read_table(out / "season_2026_ledger_reconciliation.parquet").to_pylist()
    (row,) = [m for m in recon if m["rule_id"] == "S1" and m["source_path"] == "picks/2026-08-20.json"
              and m["slot"] == "primary"]
    assert (row["recipe_value"], row["canonical_bts_outcome"], row["occurrence_disposition"], row["selection_id"]) == (
        "miss", "hit", "canonical_selection", "2026-08-20|primary|802415|822934")
    assert row["source_obs_id"] in emitted
    occ_ids = {o["obs_id"] for o in occ}
    shadow = [m for m in recon if m["source_path"] == "picks/2026-08-21.shadow.json"]
    assert shadow and all(m["source_obs_id"] in occ_ids and (m["source_state"], m["source_reason"], m["selection_id"])
                          == ("excluded", "shadow_model_out_of_scope", None) for m in shadow)   # Codex plan r2 #5


def test_outputs_are_byte_identical_across_runs_roots_and_discovery_order(tmp_path):
    bundle_a, out_a = _compile(tmp_path, "a")
    items = list(FILES.items())
    random.Random(7).shuffle(items)
    out_b = _compile(tmp_path, "b", dict(items))[1]
    moved = tmp_path / "elsewhere" / "bundle"
    shutil.copytree(bundle_a, moved)
    out_c = tmp_path / "out_c"
    compile_bundle(moved, out_c, uv_lock_sha256="test-lock")
    names = sorted(p.name for p in out_a.iterdir())
    assert len(names) == 6
    for f in names:
        assert (out_a / f).read_bytes() == (out_b / f).read_bytes() == (out_c / f).read_bytes(), f
    manifest = moved / "manifest.json"
    doc = json.loads(manifest.read_text())
    doc["entries"].reverse()                       # a hand-reordered manifest: only its own hash may change
    manifest.write_text(json.dumps(doc))
    out_d = tmp_path / "out_d"
    compile_bundle(moved, out_d, uv_lock_sha256="test-lock")
    for f in (n for n in names if n.endswith(".parquet")):
        assert (out_a / f).read_bytes() == (out_d / f).read_bytes(), f


def test_compile_refuses_any_existing_output_directory(tmp_path):
    bundle, out = _compile(tmp_path, "a")
    with pytest.raises(FileExistsError, match="already exists"):
        compile_bundle(bundle, out, uv_lock_sha256="test-lock")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        compile_bundle(bundle, tmp_path / "empty", uv_lock_sha256="test-lock")


def test_raw_values_round_trip_through_the_occurrence_table(tmp_path):
    # Codex plan r2 #2: wrong-typed values are nulled in typed columns but recoverable from the compiled output.
    files = {"static/units/20260801T150000Z.json": dumps({"units": [{"id": 2449, "feedId": "RAW_GAME", "roundId": 1009,
                                                                    "status": "scheduled"}]}),
             "picks/account_state/contest_ledger.jsonl": (contest_line("2026-08-02T14:30:00Z", [
                 rnd(990, "hit", "RAW_STREAK", 1, [slot(3001, 4001, "hit")])]) + "\n").encode()}
    seal_bundle(tmp_path / "b", files)
    compile_bundle(tmp_path / "b", tmp_path / "o", uv_lock_sha256="test-lock")
    occ = {(o["source_path"], o["locator"]): o for o in
           pq.read_table(tmp_path / "o" / "season_2026_ledger_occurrences.parquet").to_pylist()}
    unit = occ[("static/units/20260801T150000Z.json", "item=0")]
    assert json.loads(unit["record_raw_json"])["feedId"] == "RAW_GAME" and json.loads(unit["fields_json"])["feed_id"] is None
    round_occ = occ[("picks/account_state/contest_ledger.jsonl", "line=1/round=0")]
    assert json.loads(round_occ["record_raw_json"])["streak"] == "RAW_STREAK"


def test_outcome_status_distinguishes_a_malformed_grade_from_a_source_null():
    from scripts.audit.season_ledger.compile import _outcome_status
    assert (_outcome_status("hit", "value"), _outcome_status(None, "null"), _outcome_status(None, "type_mismatch")) == (
        "graded", "matched_ungraded", "unknown")


def test_a_conflicting_pick_file_is_kept_whole_as_an_unresolved_view(tmp_path):
    # Codex plan r1 #4: both legs of the double pick file stay visible; neither is silently "not selected".
    out = _compile(tmp_path, "a")[1]
    occ = pq.read_table(out / "season_2026_ledger_occurrences.parquet").to_pylist()
    assert {o["locator"]: o["disposition"] for o in occ if o["source_path"] == "picks/2026-08-28.json"} == {
        "slot=primary": "unresolved_pick_file_view", "slot=double_down": "unresolved_pick_file_view"}
    (r,) = [r for r in _ledger(out).values() if r["date"] == "2026-08-28"]
    assert (r["finalization"], r["pick_view_action"], r["pick_obs_id"]) == ("unresolved", "double", None)


def test_unusable_evidence_never_finalizes_a_day_or_reads_as_absent(tmp_path):
    # Codex plan r3 #2 and #5: malformed game identities never agree, and quarantined evidence is never absence.
    files = {"picks/2026-08-20.json": pick_json("2026-08-20", primary={"game_pk": "bad_file_game"},
                                                notification_sent=True, notification_id="dm-1"),
             "picks/2026-08-20/decision.json": decision_json("2026-08-20", action="single",
                                                             primary=cand(101, "different_bad_game")),
             "picks/2026-08-21.json": pick_json("2026-08-21"),
             "picks/2026-08-21/decision.json": decision_json("2026-08-21", action="single",
                                                             primary=dict(cand(101, 5001), batter_id="101")),
             "picks/lineup_evolution_2026-08-22.jsonl": b"{not json\n"}
    seal_bundle(tmp_path / "b", files)
    compile_bundle(tmp_path / "b", tmp_path / "o", uv_lock_sha256="test-lock")
    days = {r["date"]: (r["row_kind"], r["reason"]) for r in _ledger(tmp_path / "o").values()
            if r["date"] in ("2026-08-20", "2026-08-21", "2026-08-22")}
    assert days == {"2026-08-20": ("unfinalized_day", "decision_unusable"),
                    "2026-08-21": ("unfinalized_day", "decision_unusable"),
                    "2026-08-22": ("unfinalized_day", "unusable_evidence_only")}
    occ = {(o["source_path"], o["locator"]): (o["state"], o["reason"] or o["disposition"]) for o in
           pq.read_table(tmp_path / "o" / "season_2026_ledger_occurrences.parquet").to_pylist()}
    assert occ[("picks/2026-08-20.json", "slot=primary")] == ("quarantined", "slot_bad_game_pk")
    assert occ[("picks/2026-08-21.json", "slot=primary")] == ("emitted", "unresolved_pick_file_view")


def test_a_refusal_counts_only_with_a_reason_and_a_time_before_a_known_lock():
    from scripts.audit.season_ledger.compile import _eligibility
    row = {"locked_at": "2026-08-28T22:00:00.000000Z", "date": "2026-08-28", "slot": "primary", "batter_id": 1,
           "game_pk": 2}
    key = ("2026-08-28", "primary", 1, 2)
    assert _eligibility(row, None, {}, {key: [("2026-08-28T21:00:00.000000Z", "past_submission_cutoff")]}) == (
        "refused_evidenced", "2026-08-28T21:00:00.000000Z", "past_submission_cutoff")
    assert _eligibility(row, None, {}, {key: [(None, "past_submission_cutoff")]})[0] == "unknown"
    assert _eligibility(row, None, {}, {key: [("2026-08-28T23:00:00.000000Z", "late")]})[0] == "unknown"
    assert _eligibility(row, None, {}, {key: [("2026-08-28T21:00:00.000000Z", None)]})[0] == "unknown"   # no reason
    unlocked = dict(row, locked_at=None)                                     # Codex plan r2 #8: lock unknown
    assert _eligibility(unlocked, None, {}, {key: [("2026-08-28T21:00:00.000000Z", "past_submission_cutoff")]})[0] == "unknown"
