"""Register record validation (design v3.1 §2, §3, §7; plan Task 8; Codex phase-1 r3 #6).

``validate(records)`` returns a list of error strings (empty = publishable): the JSON schema in
``record_schema.json`` plus the cross-field rules a schema cannot state. A record with any error is
not published. With ``evidence_root`` (publication), every CERTIFIED replay/defence entry is bound to
its acceptance artifact: the file exists under the root, its sha256 matches, the runner verdict is
``accepted``, and it covers the claimed nodes / patch / fix set. Needs ``jsonschema``
(``uv run --with jsonschema==4.23.0 ...``); it is deliberately not in the project lock.

Contemporaneity (design §3, ruling of 2026-09-29): an operator report is contemporaneous for an
occurrence that cites it when it was written after the occurrence began and within 48 h of the time it
was last observable — the event itself for a one-shot occurrence; for a ``continuing`` condition its
mitigation / verified recovery / fix install, or at any time while none of those is known (the report
then describes a condition still present). A date-only time covers its whole ET day.
"""
from __future__ import annotations

import hashlib
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


def _span(bound) -> tuple[datetime | None, datetime | None]:
    """(earliest, latest) instants a bound allows; a date-only value covers its whole ET day."""
    if not isinstance(bound, dict):
        return None, None

    def lo(t):
        return _parse(t)

    def hi(t):
        return _parse(t) + (timedelta(days=1) if "T" not in t else timedelta(0))

    if "at" in bound:
        return lo(bound["at"]), hi(bound["at"])
    return (lo(bound["not_before"]) if "not_before" in bound else None,
            hi(bound["not_after"]) if "not_after" in bound else None)


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


def _feasible_minutes(start_bound, end_bound):
    """(min, max) minutes between two bounds, or None when either side is unbounded."""
    s_lo, s_hi = _span(start_bound)
    e_lo, e_hi = _span(end_bound)
    if None in (s_lo, s_hi, e_lo, e_hi):
        return None
    return max(0.0, (e_lo - s_hi).total_seconds() / 60), (e_hi - s_lo).total_seconds() / 60


def _occurrence_end(o: dict, r: dict):
    """When the occurrence was last observable: a one-shot event ends with its onset; a
    ``continuing`` condition ends at the EARLIEST of its own mitigation / verified recovery and the
    installs of the fixes of ITS links (``links``, default every link), and is treated as still ongoing
    (None) while none of those is known. (Self-review 2026-09-29: the latest of every fix in the record
    let a report written long after a mitigation count while an unrelated later fix existed.)"""
    if not o.get("continuing"):
        return _span(o.get("onset"))[1]
    ends = []
    for key in ("mitigation", "restored_verification"):
        hi = _span(o.get(key))[1]
        if hi is not None:
            ends.append(hi)
    links = set(o.get("links") or [f["link"] for f in r["fix"]])
    for f in r["fix"]:
        d = f["deployed"]
        if f["link"] in links and isinstance(d, dict) and d.get("live_by"):
            ends.append(_span({"at": d["live_by"]})[1])
    return min(ends) if ends else None


def _fix_steps(r: dict):
    """Yield (span, evidence ids) for every dated fix step: a mitigation, an install or a verification.
    An operator report may describe one of these rather than an occurrence (e.g. "verified live on the
    box"); it is then judged against that step."""
    for f in r["fix"]:
        d = f["deployed"]
        if isinstance(d, dict) and d.get("live_by") and d.get("evidence"):
            lo = _parse(d["not_live_before"]) if d.get("not_live_before") else None
            yield (lo, _span({"at": d["live_by"]})[1]), d["evidence"]
        for key in ("mitigated", "verified_recovered"):
            step = f[key]
            if isinstance(step, dict) and isinstance(step["at"], dict):
                yield _span(step["at"]), step["at"].get("evidence", [])


def _latency_errs(rid: str, o: dict) -> list[str]:
    errs = []
    det = o["first_machine_detection"]
    det_at = det["at"] if isinstance(det, dict) else det
    pairs = {"detection": (o["onset"], det_at), "notification": (det_at, o["alert"]["confirmed"]),
             "recovery": (o["onset"], o["restored_verification"])}
    for name, (a, b) in pairs.items():
        lat = o["latencies"][name]
        if not isinstance(lat, dict):
            continue
        if lat["min_minutes"] > lat["max_minutes"]:
            errs.append(f"{rid}: {name} latency min > max")
        feasible = _feasible_minutes(a, b)
        if feasible is None:
            errs.append(f"{rid}: numeric {name} latency needs both endpoint times bounded")
        elif lat["min_minutes"] > feasible[0] + 0.01 or lat["max_minutes"] < feasible[1] - 0.01:
            errs.append(f"{rid}: {name} latency [{lat['min_minutes']}, {lat['max_minutes']}] is tighter than its "
                        f"endpoint bounds allow [{feasible[0]:.2f}, {feasible[1]:.2f}]")
    return errs


def _binding_errs(rid: str, kind: str, entry: dict, root: Path) -> list[str]:
    path = root / entry["acceptance"]
    if not path.is_file():
        return [f"{rid}: certified {kind} acceptance file {entry['acceptance']} not found"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["acceptance_sha256"]:
        return [f"{rid}: certified {kind} acceptance file hash mismatch"]
    acc = json.loads(data)
    errs = []
    if acc.get("verdict") != "accepted":
        errs.append(f"{rid}: certified {kind} whose runner verdict is {acc.get('verdict')!r}")
    if kind == "defence":
        if acc.get("patch_sha256") != entry["patch_sha256"]:
            errs.append(f"{rid}: certified defence patch hash differs from its acceptance")
        missing = set(entry["killing_nodes"]) - set(acc.get("certificates", {}))
        if missing:
            errs.append(f"{rid}: killing nodes without a certificate in the acceptance: {sorted(missing)[:3]}")
    else:
        spec = acc.get("spec", {})
        if [s["node"] for s in spec.get("symptom_nodes", [])] != entry["symptom_nodes"]:
            errs.append(f"{rid}: certified replay symptom nodes differ from its acceptance")
        if spec.get("fix_set") != entry["fix_set"]:
            errs.append(f"{rid}: certified replay fix set differs from its acceptance")
    return errs


def _semantic(r: dict, ids: set[str], publish: bool = True, evidence_root: Path | None = None) -> list[str]:
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
    for e in r["evidence"]:
        if e["kind"] != "contemporaneous_operator_report":
            continue
        if "written_at" not in e:
            errs.append(f"{rid}: {e['id']} operator report needs written_at")
            continue
        cites = [o for o in r.get("occurrences", []) if e["id"] in set(_ev_refs(o))]
        steps = [span for span, refs in _fix_steps(r) if e["id"] in refs]
        if not cites and not steps:
            errs.append(f"{rid}: {e['id']} operator report is not cited by any occurrence or fix step it describes")
            continue
        w_lo, w_hi = _span({"at": e["written_at"]})     # a date-only time covers its whole ET day
        window = timedelta(hours=48)
        ok = False
        for o in cites:
            onset_lo, _ = _span(o["onset"])
            if onset_lo is not None and w_hi <= onset_lo:
                continue                                  # certainly written before the occurrence began
            end = _occurrence_end(o, r)
            if end is None or w_lo <= end + window:
                ok = True
        if cites and not ok:
            errs.append(f"{rid}: {e['id']} is not contemporaneous with the occurrence it describes "
                        "(written before it began, or more than 48 h after its last known time)")
        # each role is judged on its own: a report cited by a fix step must be contemporaneous with one
        if steps and not any((lo is None or w_hi > lo) and (hi is None or w_lo <= hi + window) for lo, hi in steps):
            errs.append(f"{rid}: {e['id']} is not contemporaneous with the fix step it describes "
                        "(written before the step, or more than 48 h after it)")
    if disp == "unresolved_candidate" and not r.get("missing_evidence"):
        errs.append(f"{rid}: unresolved_candidate must name its missing_evidence")
    if (disp == "pre_ship_exclusion") == r["counted"]:
        errs.append(f"{rid}: counted must be false exactly for pre_ship_exclusion")

    links = {m["n"] for m in r["mechanism"]}
    for f in r["fix"]:
        if f["link"] not in links:
            errs.append(f"{rid}: fix names link {f['link']}, which the mechanism does not have")
    for o in r.get("occurrences", []):
        for n in o.get("links", []):
            if n not in links:
                errs.append(f"{rid}: an occurrence names link {n}, which the mechanism does not have")
    fx = r["fixtures"]
    fixed_links = [f["link"] for f in r["fix"] if isinstance(f["implemented"], dict)]
    if publish and disp == "observed_incident" and r["tier"] == "A" and r["counted"] and fixed_links:
        if not fx["historical_replay"]:
            errs.append(f"{rid}: fixed Tier-A incident needs a historical_replay entry (certified or unavailable)")
        covered = {d["link"] for d in fx["current_defence"]}
        for n in fixed_links:
            if n not in covered:
                errs.append(f"{rid}: current_defence has no entry for link {n}")
    has_fixture = (any(h["status"] == "certified" for h in fx["historical_replay"])
                   or any(d["status"] == "certified" for d in fx["current_defence"])
                   or fx["expected_failure"] or fx["characterization"] or fx.get("config"))
    if publish and r["plan_named"] and not has_fixture and not fx.get("deferred"):
        errs.append(f"{rid}: plan-named record needs a fixture or an explicit deferral")
    unfixed = any(f["implemented"] == "unfixed" for f in r["fix"])
    if publish and unfixed and "§10" in r["contract"]["source"] and not (fx["expected_failure"] or fx["characterization"]):
        errs.append(f"{rid}: unfixed defect with a fixed (§10) contract needs an expected-failure fixture")
    if publish and evidence_root is not None:
        for h in fx["historical_replay"]:
            if h["status"] == "certified":
                errs += _binding_errs(rid, "replay", h, evidence_root)
        for d in fx["current_defence"]:
            if d["status"] == "certified":
                errs += _binding_errs(rid, "defence", d, evidence_root)

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
        errs += _latency_errs(rid, o)
    return errs


def validate(records: list[dict], *, schema: dict | None = None, publish: bool = True,
             evidence_root: Path | str | None = None) -> list[str]:
    """``publish=False`` is draft mode: the fixture-completeness rules (filled in from acceptance
    objects at build time) are skipped; everything else applies. Publication passes ``evidence_root``
    (the repository root) so certified entries are bound to their acceptance artifacts."""
    import jsonschema

    schema = schema or json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    root = Path(evidence_root) if evidence_root is not None else None
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
            errors.extend(_semantic(r, ids, publish, root))
    return errors
