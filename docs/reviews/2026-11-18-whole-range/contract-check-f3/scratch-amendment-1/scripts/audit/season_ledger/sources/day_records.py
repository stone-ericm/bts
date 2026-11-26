"""O3 decision files, O4 scheduler state, O5 lineup-evolution logs (spec §4)."""
from __future__ import annotations

import json

from ..ids import Parsed, canonical_json, is_int, joined, load_json_bytes, obs_id, quarantine, sha256_hex, take, typed

# Vendored from bts.daily_decision so the box run executes only this package's code (Task 13);
# test_vendored_decision_rules_match_production pins them to the production module.
ACCEPTED_SCHEMAS = ("bts_daily_decision_v1", "bts_daily_decision_v2", "bts_daily_decision_v3")
_LEGACY_SCHEMAS = ("bts_daily_decision_v1", "bts_daily_decision_v2")
OBJECTIVES = ("reach57", "emax_season_best")

CANDIDATE_FIELDS = {"batter_id": "int", "batter_name": "str", "team": "str", "game_pk": "int", "p_game_hit": "float"}
DECISION_FIELDS = {"schema_version": "str", "date": "str", "action": "str", "source": "str", "streak": "int",
                   "saver_available": "bool", "state_source": "str", "state_status": "str", "allow_double": "bool",
                   "contest_source_date": "str", "delivery_status": "str", "scoreable": "bool", "objective": "str",
                   "best_streak": "int", "best_status": "str", "effective_best": "int", "tail_policy_sha256": "str",
                   "degraded_reason": "str", "finalized_at": "str"}
ACTION_SOURCES = ("mdp", "heuristic")
STATE_FIELDS = {"date": "str", "schedule_fetched_at": "str", "pick_locked": "bool", "pick_locked_at": "str",
                "committed_pick_written": "bool", "result_status": "str", "skip_notified_at": "str"}
STATE_NESTED = ("final_skip_candidate", "delivery_refusals", "fallback_refreshes")
EVOLUTION_LINE_FIELDS = {"captured_at": "str", "date": "str", "run_time": "str"}
EVOLUTION_FIELDS = {"batter_id": "int", "batter_name": "str", "team": "str", "p_game_hit": "float",
                    "projected_lineup": "bool", "game_pk": "int"}


def decision_objective(rec: dict) -> str:
    """As bts.daily_decision.decision_objective: pre-v3 records are reach57; a v3 record without a valid
    objective is unknown."""
    obj = rec.get("objective")
    if rec.get("schema_version") in _LEGACY_SCHEMAS:
        return obj if obj in OBJECTIVES else "reach57"
    return obj if obj in OBJECTIVES else "unknown"


def parse_decision(rel_path: str, data: bytes) -> Parsed:
    content = sha256_hex(data)
    try:
        doc = load_json_bytes(data)
    except ValueError as exc:
        return Parsed([], [quarantine(rel_path, "file", str(exc))])
    # Same acceptance as bts.daily_decision.load_decision, with container types checked first so an object
    # in any of these keys is quarantined instead of crashing a membership test.
    if (not isinstance(doc, dict) or not isinstance(doc.get("schema_version"), str)
            or doc["schema_version"] not in ACCEPTED_SCHEMAS or not isinstance(doc.get("action"), str)
            or doc["action"] not in {"skip", "single", "double"}
            or not isinstance(doc.get("scoreable"), bool) or "date" not in doc):
        return Parsed([], [quarantine(rel_path, "file", "invalid_decision_record", raw=doc)])
    named = ("primary", "double_down") if doc["action"] == "double" else ("primary",) if doc["action"] == "single" else ()
    for n in named:   # I13: a decision that names a selection must name a usable identity (a null game = unrecorded)
        chosen = doc.get(n)
        if not isinstance(chosen, dict) or not is_int(chosen.get("batter_id")):
            return Parsed([], [quarantine(rel_path, "file", "decision_candidate_missing_batter_id", raw=doc)])
        if chosen.get("game_pk") is not None and not is_int(chosen["game_pk"]):
            return Parsed([], [quarantine(rel_path, "file", "decision_candidate_bad_game_pk", raw=doc)])
    values, absent, bad = take(doc, DECISION_FIELDS)
    objective_raw, source_raw = values.pop("objective"), values.pop("source")
    row = {"obs_id": obs_id(rel_path, "file", content), "locator": "file", "source_kind": "decision",
           "source_path": rel_path, "content_sha256": content, **values, "objective_raw": objective_raw,
           "objective": decision_objective(doc), "action_source_raw": source_raw,
           "action_source": source_raw if source_raw in ACTION_SOURCES else "unknown",
           "record_raw_json": canonical_json(doc)}
    for name in ("primary", "double_down", "second_candidate"):
        cand = doc.get(name)
        if name not in doc:
            absent.append(name)
        if cand is not None and not isinstance(cand, dict):
            bad.append(name)
        cand_values, cand_absent, cand_bad = take(cand if isinstance(cand, dict) else {}, CANDIDATE_FIELDS,
                                                  prefix=f"{name}.")
        if isinstance(cand, dict):
            absent += cand_absent
        bad += cand_bad
        row.update({f"{name}_{k}": v for k, v in cand_values.items()})
    row["absent_fields"], row["type_mismatch_fields"] = joined(absent), joined(bad)
    return Parsed([row], [])


def parse_scheduler_state(rel_path: str, data: bytes) -> Parsed:
    content = sha256_hex(data)
    try:
        doc = load_json_bytes(data)
    except ValueError as exc:
        return Parsed([], [quarantine(rel_path, "file", str(exc))])
    if not isinstance(doc, dict) or "date" not in doc:
        return Parsed([], [quarantine(rel_path, "file", "invalid_scheduler_state", raw=doc)])
    values, absent, bad = take(doc, STATE_FIELDS)
    absent += [f for f in STATE_NESTED if f not in doc]
    skip = doc.get("final_skip_candidate")
    primary = skip.get("primary") if isinstance(skip, dict) else None
    row = {"obs_id": obs_id(rel_path, "file", content), "locator": "file", "source_kind": "scheduler_state",
           "source_path": rel_path, "content_sha256": content, **values,
           "final_skip_candidate_present": isinstance(skip, dict),
           "skip_candidate_batter_id": typed(primary.get("batter_id"), "int")[0] if isinstance(primary, dict) else None,
           "skip_candidate_game_pk": typed(primary.get("game_pk"), "int")[0] if isinstance(primary, dict) else None,
           **{f"{name}_json": None if doc.get(name) is None else canonical_json(doc[name]) for name in STATE_NESTED},
           "absent_fields": joined(absent), "type_mismatch_fields": joined(bad),
           "record_raw_json": canonical_json(doc)}
    return Parsed([row], [])


def parse_lineup_evolution(rel_path: str, data: bytes) -> Parsed:
    out = Parsed()
    content = sha256_hex(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return Parsed([], [quarantine(rel_path, "file", "not_utf8")])
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        line_loc = f"line={line_no}"
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            out.quarantined.append(quarantine(rel_path, line_loc, "invalid_json_line", raw=line))
            continue
        if not isinstance(doc, dict):
            out.quarantined.append(quarantine(rel_path, line_loc, "line_not_object", raw=doc))
            continue
        slots = [(s, doc[s]) for s in ("primary", "double_down") if doc.get(s) is not None]
        if not slots:
            out.quarantined.append(quarantine(rel_path, line_loc, "line_without_slots", raw=doc))
            continue
        line_values, line_absent, line_bad = take(doc, EVOLUTION_LINE_FIELDS)
        for slot, s in slots:
            locator = f"{line_loc}/slot={slot}"
            if not isinstance(s, dict) or not is_int(s.get("batter_id")):
                reason = "slot_not_object" if not isinstance(s, dict) else "slot_missing_batter_id"
                out.quarantined.append(quarantine(rel_path, locator, reason, raw=s))
                continue
            if s.get("game_pk") is not None and not is_int(s["game_pk"]):
                out.quarantined.append(quarantine(rel_path, locator, "slot_bad_game_pk", raw=s))
                continue
            values, absent, bad = take(s, EVOLUTION_FIELDS, prefix=f"{slot}.")
            out.rows.append({"obs_id": obs_id(rel_path, locator, content), "locator": locator,
                             "source_kind": "lineup_evolution", "source_path": rel_path, "content_sha256": content,
                             "line_no": line_no, **line_values, "slot": slot, **values,
                             "absent_fields": joined(line_absent + absent),
                             "type_mismatch_fields": joined(line_bad + bad), "record_raw_json": canonical_json(doc)})
    return out
