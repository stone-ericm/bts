"""O1 pick files (`picks/<date>.json`), O2 scheduler archives (`picks/<date>/<prefix>_<stamp>.json`) and the
manual / streak-repair versions of pick files (spec §4). Typed values follow Interpretation I13, absent
keys are listed at every depth, and each row keeps the file's raw JSON, so nothing written is lost."""
from __future__ import annotations

import re

from ..ids import Parsed, canonical_json, is_int, joined, load_json_bytes, obs_id, quarantine, sha256_hex, take, typed

SLOTS = (("primary", "pick"), ("double_down", "double_down"))   # ledger slot, JSON key (= slot_results key)
PICK_FIELDS = {"batter_id": "int", "batter_name": "str", "team": "str", "game_pk": "int", "game_time": "str",
               "lineup_position": "int", "projected_lineup": "bool", "pitcher_id": "int", "pitcher_name": "str",
               "pitcher_team": "str", "p_game_hit": "float"}
FILE_FIELDS = {"date": "str", "run_time": "str", "result": "str", "bluesky_posted": "bool", "bluesky_uri": "str",
               "notification_sent": "bool", "notification_id": "str", "notification_channel": "str",
               "delivery_attempted": "bool", "delivered_at": "str", "model_git_sha": "str",
               "model_pickle_sha256": "str", "policy_npz_sha256": "str", "feature_env_schema_version": "str",
               "feature_env_hash": "str", "tail_policy_sha256": "str", "shadow_model_version": "str"}
NESTED_FIELDS = ("slot_results", "policy_decision", "feature_env", "runner_up")     # kept as JSON
ARCHIVE_TIME_KEYS = {"deferred_fallback": "deferred_at", "refused_delivery": "refused_at", "stale_pick": "staled_at"}
_ARCHIVE_NAME = re.compile(r"^(deferred_fallback|refused_delivery|stale_pick)_\d{8}T\d{6}[+-]\d{4}\.json$")


def _json_or_none(value) -> str | None:
    return None if value is None else canonical_json(value)


def parse_pick_file(rel_path: str, data: bytes, *, kind: str = "pick_file") -> Parsed:
    out = Parsed()
    content = sha256_hex(data)
    try:
        doc = load_json_bytes(data)
    except ValueError as exc:
        out.quarantined.append(quarantine(rel_path, "file", str(exc)))
        return out
    if not isinstance(doc, dict) or not isinstance(doc.get("pick"), dict):
        out.quarantined.append(quarantine(rel_path, "file", "no_pick_object", raw=doc))
        return out
    values, file_absent, file_bad = take(doc, FILE_FIELDS)
    file_absent += [f for f in NESTED_FIELDS if f not in doc]
    slot_results, policy = doc.get("slot_results"), doc.get("policy_decision")
    day_result = values.pop("result")
    base = {"source_kind": kind, "source_path": rel_path, "content_sha256": content, **values,
            "day_result_raw": day_result, "has_double_down": doc.get("double_down") is not None,
            "slot_results_json": _json_or_none(slot_results), "policy_decision_json": _json_or_none(policy),
            "feature_env_json": _json_or_none(doc.get("feature_env")),
            "pick_policy_objective": typed(policy.get("objective"), "str")[0] if isinstance(policy, dict) else None,
            "archive_prefix": None, "archive_reason": None, "archived_at": None,
            "record_raw_json": canonical_json(doc)}
    for slot, key in SLOTS:
        pick, locator = doc.get(key), f"slot={slot}"
        if pick is None:
            continue
        if not isinstance(pick, dict):
            out.quarantined.append(quarantine(rel_path, locator, "slot_not_object", raw=pick))
            continue
        if not is_int(pick.get("batter_id")):
            out.quarantined.append(quarantine(rel_path, locator, "slot_missing_batter_id", raw=pick))
            continue
        if pick.get("game_pk") is not None and not is_int(pick["game_pk"]):
            # I13: a present game identity that cannot be typed is not an unrecorded (null) game.
            out.quarantined.append(quarantine(rel_path, locator, "slot_bad_game_pk", raw=pick))
            continue
        slot_values, absent, bad = take(pick, PICK_FIELDS, prefix=f"{key}.")
        slot_result, slot_bad = typed(slot_results.get(key), "str") if isinstance(slot_results, dict) else (None, False)
        out.rows.append({**base, **slot_values, "slot": slot, "locator": locator,
                         "obs_id": obs_id(rel_path, locator, content), "slot_result_raw": slot_result,
                         "absent_fields": joined(file_absent + absent),
                         "type_mismatch_fields": joined(file_bad + bad + ([f"slot_results.{key}"] if slot_bad else []))})
    return out


def pick_file_state(data: bytes, parsed: Parsed) -> dict:
    """What a surviving pick file claims, independent of how much of it parsed (Codex plan r2 #4, r3 #5): the
    slots its raw structure holds, whether every one of them parsed into a usable row, and its typed file-level
    fields (delivery signals included), None when the file is not a readable object. Only a complete file can
    agree with a decision (Interpretation I14)."""
    try:
        doc = load_json_bytes(data)
    except ValueError:
        doc = None
    slots = frozenset(slot for slot, key in SLOTS if isinstance(doc, dict) and doc.get(key) is not None)
    return {"slots": slots, "complete": not parsed.quarantined and {r["slot"] for r in parsed.rows} == set(slots),
            "file_fields": take(doc, FILE_FIELDS)[0] if isinstance(doc, dict) else None}


def parse_archive(rel_path: str, data: bytes) -> Parsed:
    match = _ARCHIVE_NAME.match(rel_path.rsplit("/", 1)[-1])
    if not match:
        return Parsed([], [quarantine(rel_path, "file", "unrecognized_archive_name")])
    parsed = parse_pick_file(rel_path, data, kind="archive")
    prefix = match.group(1)
    if parsed.rows:
        doc = load_json_bytes(data)
        info = doc.get(prefix) if isinstance(doc.get(prefix), dict) else {}
        for row in parsed.rows:
            row.update(archive_prefix=prefix, archive_reason=typed(info.get("reason"), "str")[0],
                       archived_at=typed(info.get(ARCHIVE_TIME_KEYS[prefix]), "str")[0])
    return parsed
