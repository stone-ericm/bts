"""O6 contest ledger (`account_state/contest_ledger.jsonl`), lossless, and the saver-transition attempts log
(spec §4, §6). Each line, round and slot keeps its own raw JSON."""
from __future__ import annotations

import json

from ..ids import Parsed, canonical_json, is_int, joined, obs_id, quarantine, sha256_hex, take, typed, utc_iso

LINE_FIELDS = {"source_date": "str", "active_streak": "int", "best_streak": "int"}
ROUND_FIELDS = {"result": "str", "streak": "int", "streakIncrease": "int"}
SLOT_FIELDS = {"number": "int"}


def _state(record: dict, key: str, kind: str) -> tuple[object, str]:
    """A typed value plus how the source held it: value / null / absent / type_mismatch."""
    if key not in record:
        return None, "absent"
    value, bad = typed(record[key], kind)
    if bad:
        return None, "type_mismatch"
    return (value, "value") if value is not None else (None, "null")


def _disqualify(doc) -> str | None:
    """Interpretation I1: identity integers required; `result` keys present (null = in progress)."""
    if not isinstance(doc, dict) or utc_iso(doc.get("recorded_at")) is None or not isinstance(doc.get("predictions"), list):
        return "line_missing_recorded_at_or_predictions"
    for rnd in doc["predictions"]:
        if not isinstance(rnd, dict) or not is_int(rnd.get("roundId")) or "result" not in rnd:
            return "round_missing_roundId_or_result"
        slots = rnd.get("roundPredictions")
        if slots is not None and not isinstance(slots, list):
            return "round_predictions_not_list"
        for s in slots or []:
            if (not isinstance(s, dict) or not is_int(s.get("unitId")) or not is_int(s.get("playerId"))
                    or "result" not in s):
                return "slot_missing_identity_or_result"
    return None


def _lines(data: bytes):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return [(n, line) for n, line in enumerate(text.splitlines(), 1) if line.strip()]


def parse_contest_ledger(rel_path: str, data: bytes) -> Parsed:
    out = Parsed()
    content = sha256_hex(data)
    lines = _lines(data)
    if lines is None:
        return Parsed([], [quarantine(rel_path, "file", "not_utf8")])
    for line_no, line in lines:
        line_loc = f"line={line_no}"
        try:
            doc = json.loads(line)
            reason, raw = _disqualify(doc), doc
        except json.JSONDecodeError:
            reason, raw = "invalid_json_line", line
        if reason:
            out.quarantined.append(quarantine(rel_path, line_loc, reason, raw=raw))
            continue
        line_values, line_absent, line_bad = take(doc, LINE_FIELDS)
        base = {"source_kind": "contest_ledger", "source_path": rel_path, "content_sha256": content,
                "line_no": line_no, "recorded_at": utc_iso(doc["recorded_at"]), **line_values}
        out.rows.append(dict(base, row_level="line", locator=line_loc, obs_id=obs_id(rel_path, line_loc, content),
                             absent_fields=joined(line_absent), type_mismatch_fields=joined(line_bad),
                             record_raw_json=canonical_json({k: v for k, v in doc.items() if k != "predictions"})))
        for i, rnd in enumerate(doc["predictions"]):
            round_values, round_absent, round_bad = take(rnd, ROUND_FIELDS)
            rbase = dict(base, round_id=rnd["roundId"], round_result=round_values["result"],
                         round_streak=round_values["streak"], round_streak_increase=round_values["streakIncrease"])
            round_loc = f"{line_loc}/round={i}"
            # Every round is its own occurrence with its own raw record (Codex plan r2 #2).
            out.rows.append(dict(rbase, row_level="round", locator=round_loc,
                                 obs_id=obs_id(rel_path, round_loc, content), absent_fields=joined(round_absent),
                                 type_mismatch_fields=joined(round_bad),
                                 record_raw_json=canonical_json({k: v for k, v in rnd.items()
                                                                 if k != "roundPredictions"})))
            for j, s in enumerate(rnd.get("roundPredictions") or []):
                loc = f"{round_loc}/slot={j}"
                slot_values, slot_absent, slot_bad = take(s, SLOT_FIELDS)
                result, result_state = _state(s, "result", "str")
                hits, hits_state = _state(s, "hits", "int")
                at_bats, at_bats_state = _state(s, "atBats", "int")
                states = (("result", result_state), ("hits", hits_state), ("atBats", at_bats_state))
                out.rows.append(dict(rbase, row_level="slot", locator=loc, obs_id=obs_id(rel_path, loc, content),
                                     slot_number=slot_values["number"], unit_id=s["unitId"], player_id=s["playerId"],
                                     slot_result=result, slot_result_state=result_state, hits=hits,
                                     hits_state=hits_state, at_bats=at_bats, at_bats_state=at_bats_state,
                                     absent_fields=joined([f"round.{f}" for f in round_absent] + slot_absent
                                                          + [k for k, st in states if st == "absent"]),
                                     type_mismatch_fields=joined([f"round.{f}" for f in round_bad] + slot_bad
                                                                 + [k for k, st in states if st == "type_mismatch"]),
                                     record_raw_json=canonical_json(s)))
    return out


def parse_saver_transitions(rel_path: str, data: bytes) -> Parsed:
    """Rows are attempts (including rejected ones), never consumption times (spec §6)."""
    out = Parsed()
    content = sha256_hex(data)
    lines = _lines(data)
    if lines is None:
        return Parsed([], [quarantine(rel_path, "file", "not_utf8")])
    for line_no, line in lines:
        loc = f"line={line_no}"
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            out.quarantined.append(quarantine(rel_path, loc, "invalid_json_line", raw=line))
            continue
        if not isinstance(doc, dict):
            out.quarantined.append(quarantine(rel_path, loc, "line_not_object", raw=doc))
            continue
        out.rows.append({"obs_id": obs_id(rel_path, loc, content), "locator": loc, "source_kind": "saver_transitions",
                         "source_path": rel_path, "content_sha256": content, "line_no": line_no,
                         "attempted_at": utc_iso(doc.get("ts")),
                         "attempt_source": None if doc.get("source") is None else str(doc.get("source")),
                         "attempt_outcome": None if doc.get("outcome") is None else str(doc.get("outcome")),
                         "record_raw_json": canonical_json(doc)})
    return out
