"""Register record validation (design v3.1 §2, §3, §7; plan rev 2 Task 8 amendment).

``validate(records)`` returns a list of error strings (empty = publishable): the JSON schema in
``record_schema.json`` plus the cross-field rules a schema cannot state. A record with any error is
not published. Needs ``jsonschema`` (``uv run --with jsonschema==4.23.0 ...``); it is deliberately
not in the project lock, which the box syncs on deploy.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

SCHEMA_PATH = Path(__file__).with_name("record_schema.json")
ET = ZoneInfo("America/New_York")
_OBSERVATION_KINDS = {"machine_observation", "contemporaneous_operator_report"}


def _parse(t: str) -> datetime:
    """Date-only values are ET midnight of that day; datetimes carry their own offset (schema)."""
    if "T" not in t:
        return datetime.fromisoformat(t).replace(tzinfo=ET)
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def _bounds(obj):
    """Yield every bound-shaped dict (has 'evidence' and at/not_before/not_after) inside obj."""
    if isinstance(obj, dict):
        if "evidence" in obj and any(k in obj for k in ("at", "not_before", "not_after")):
            yield obj
        for v in obj.values():
            yield from _bounds(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _bounds(v)


def _ev_refs(obj):
    if isinstance(obj, dict):
        refs = obj.get("evidence")
        if isinstance(refs, list) and all(isinstance(x, str) for x in refs):
            yield from refs
        for k, v in obj.items():
            if k != "evidence":
                yield from _ev_refs(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _ev_refs(v)


def _earliest(bound) -> tuple[datetime, bool] | None:
    """(earliest instant, value was date-only) for a bound, else None."""
    if not isinstance(bound, dict):
        return None
    t = bound.get("at") or bound.get("not_before") or bound.get("not_after")
    return (_parse(t), "T" not in t) if t else None


def _semantic(r: dict, ids: set[str]) -> list[str]:
    rid = r.get("id", "?")
    errs: list[str] = []
    ev_ids = [e["id"] for e in r.get("evidence", [])]
    if len(ev_ids) != len(set(ev_ids)):
        errs.append(f"{rid}: duplicate evidence id")
    known = set(ev_ids)
    for ref in _ev_refs({k: v for k, v in r.items() if k != "evidence"}):
        if ref not in known:
            errs.append(f"{rid}: unknown evidence id {ref}")
    for rel in r.get("related", []):
        if rel not in ids:
            errs.append(f"{rid}: unknown related record {rel}")

    disp = r["disposition"]
    if disp == "observed_incident" and not any(
            e["kind"] in _OBSERVATION_KINDS and e["strength"] == "primary" for e in r["evidence"]):
        errs.append(f"{rid}: observed_incident needs a primary machine observation or contemporaneous operator report")
    onsets = [_earliest(o["onset"]) for o in r.get("occurrences", [])]
    onsets = [t for t in onsets if t is not None]
    for e in r["evidence"]:
        if e["kind"] != "contemporaneous_operator_report":
            continue
        if "written_at" not in e:
            errs.append(f"{rid}: {e['id']} operator report needs written_at")
        elif onsets:
            first, date_only = min(onsets)
            # §3: written ≤ 48 h after the event; a date-only onset may be any time that day
            limit = timedelta(hours=72 if date_only else 48)
            if _parse(e["written_at"]) - first > limit:
                errs.append(f"{rid}: {e['id']} is not contemporaneous (written > 48 h after the onset)")
    if disp == "unresolved_candidate" and not r.get("missing_evidence"):
        errs.append(f"{rid}: unresolved_candidate must name its missing_evidence")
    if (disp == "pre_ship_exclusion") == r["counted"]:
        errs.append(f"{rid}: counted must be false exactly for pre_ship_exclusion")

    links = {m["n"] for m in r["mechanism"]}
    for f in r["fix"]:
        if f["link"] not in links:
            errs.append(f"{rid}: fix names link {f['link']}, which the mechanism does not have")
    fx = r["fixtures"]
    fixed_links = [f["link"] for f in r["fix"] if isinstance(f["implemented"], dict)]
    if disp == "observed_incident" and r["tier"] == "A" and r["counted"] and fixed_links:
        if not fx["historical_replay"]:
            errs.append(f"{rid}: fixed Tier-A incident needs a historical_replay entry (certified or unavailable)")
        covered = {d["link"] for d in fx["current_defence"]}
        for n in fixed_links:
            if n not in covered:
                errs.append(f"{rid}: current_defence has no entry for link {n}")
    has_fixture = (any(h["status"] == "certified" for h in fx["historical_replay"])
                   or any(d["status"] == "certified" for d in fx["current_defence"])
                   or fx["expected_failure"] or fx["characterization"])
    if r["plan_named"] and not has_fixture and not fx.get("deferred"):
        errs.append(f"{rid}: plan-named record needs a fixture or an explicit deferral")
    unfixed = any(f["implemented"] == "unfixed" for f in r["fix"])
    if unfixed and "§10" in r["contract"]["source"] and not (fx["expected_failure"] or fx["characterization"]):
        errs.append(f"{rid}: unfixed defect with a fixed (§10) contract needs an expected-failure fixture")

    if r["tier"] == "B" and any(x["reaches_production"] for x in r["residual"]):
        errs.append(f"{rid}: residual reaches production (B → A)")
    if r["tier"] == "tier_pending" and not r.get("tier_reason"):
        errs.append(f"{rid}: tier_pending needs a tier_reason")

    for b in _bounds(r):
        if "not_before" in b and "not_after" in b and _parse(b["not_before"]) > _parse(b["not_after"]):
            errs.append(f"{rid}: bound has not_before after not_after")
    for f in r["fix"]:
        d = f["deployed"]
        if isinstance(d, dict) and d.get("live_by") and d.get("not_live_before") and \
                _parse(d["not_live_before"]) > _parse(d["live_by"]):
            errs.append(f"{rid}: deployed not_live_before is after live_by")
    for o in r.get("occurrences", []):
        lat = o["latencies"]
        det = o["first_machine_detection"]
        if isinstance(lat["detection"], dict) and (not isinstance(o["onset"], dict) or not isinstance(det, dict)
                                                   or not isinstance(det["at"], dict)):
            errs.append(f"{rid}: numeric detection latency needs bounded onset and detection times")
        if isinstance(lat["notification"], dict) and not isinstance(o["alert"]["confirmed"], dict):
            errs.append(f"{rid}: numeric notification latency needs a bounded confirmed alert")
        if isinstance(lat["recovery"], dict) and not (isinstance(o["onset"], dict)
                                                      and isinstance(o["restored_verification"], dict)):
            errs.append(f"{rid}: numeric recovery latency needs bounded onset and verification times")
    return errs


def validate(records: list[dict], *, schema: dict | None = None) -> list[str]:
    import jsonschema

    schema = schema or json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    errors: list[str] = []
    seen: set[str] = set()
    for r in records:
        rid = r.get("id", "?") if isinstance(r, dict) else "?"
        if rid in seen:
            errors.append(f"{rid}: duplicate id")
        seen.add(rid)
    ids = {r.get("id") for r in records if isinstance(r, dict)}
    for r in records:
        rid = r.get("id", "?") if isinstance(r, dict) else "?"
        schema_errs = [f"{rid}: schema: {e.json_path}: {e.message}" for e in validator.iter_errors(r)]
        errors.extend(schema_errs)
        if not schema_errs:
            errors.extend(_semantic(r, ids))
    return errors
