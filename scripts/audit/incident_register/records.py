"""Register record validation (design v3.1 §2, §3, §7; plan Task 8; Codex phase-1 r3 #6).

``validate(records)`` returns a list of error strings (empty = publishable): the JSON schema in
``record_schema.json`` plus the cross-field rules a schema cannot state. A record with any error is
not published. With ``evidence_root`` (publication), every CERTIFIED replay/defence entry is bound to
its acceptance artifact: the file exists under the root, its sha256 matches, the runner verdict is
``accepted``, and it covers the claimed nodes / patch / fix set. Needs ``jsonschema``
(``uv run --with jsonschema==4.23.0 ...``); it is deliberately not in the project lock.

Contemporaneity (design §3; plan ruling 6 as replaced after Codex phase-1 r4 #5): every CITATION of an
operator report by a dated claim — an occurrence bound (onset, observed times, detection, alert,
awareness, action, mitigation, restoration) or a fix step (mitigation, install, verification) — is
qualified on its own: the report was written no earlier than the claim's earliest instant and within
48 h after its latest; an open-ended claim anchors nothing. One timely citation never qualifies another.
A continuing condition is witnessed only at its supported ``observed`` times (absence of a known end is
not evidence of continuity). An observed incident needs a QUALIFIED PRIMARY WITNESS to the deviation: an
onset, observed-time or machine-detection claim of an occurrence citing a primary machine observation or
a qualifying primary report; a fix-step report establishes only its step. A date-only time covers its
whole ET day.

Expected-failure fixtures are bound like certificates: publication with ``evidence_root`` requires the
pair's acceptance artifact (hash, verdict ``accepted``, the nodes accepted with a reviewed connection)
and the registry it accepted (hash, the record's exception); publication without an evidence root is
refused whenever such claims exist (Codex phase-1 r4 #6).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

SCHEMA_PATH = Path(__file__).with_name("record_schema.json")
REGISTRY_PATH = "docs/audit/2026-09-29-incident-register-evidence/expected_failures.json"
ET = ZoneInfo("America/New_York")
WINDOW = timedelta(hours=48)
_WITNESS_ROLES = ("onset", "observed", "first_machine_detection")


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


def _claims(obj, path: str = ""):
    """Yield ``(path, (earliest, latest), evidence ids)`` for every dated claim inside ``obj``: a bound
    (at / not_before / not_after) or a dated install (``live_by`` with its ``not_live_before``)."""
    if isinstance(obj, dict):
        refs = obj.get("evidence")
        if isinstance(refs, list):
            if obj.get("live_by"):
                lo = _parse(obj["not_live_before"]) if obj.get("not_live_before") else None
                yield path, (lo, _span({"at": obj["live_by"]})[1]), refs
            elif any(k in obj for k in ("at", "not_before", "not_after")):
                yield path, _span(obj), refs
        for k, v in obj.items():
            if k != "evidence":
                yield from _claims(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _claims(v, f"{path}[{i}]")


def _qualified(report: dict, span) -> bool:
    """A report qualifies for ONE dated claim: written no earlier than the claim's earliest instant and
    within 48 h after its latest. An open-ended claim (no latest instant) anchors nothing."""
    lo, hi = span
    w_lo, w_hi = _span({"at": report["written_at"]})
    if hi is None or (lo is not None and w_hi < lo):           # certainly written before the claim's time
        return False
    return w_lo <= hi + WINDOW


def _witnessed(r: dict, ev: dict) -> bool:
    """Some occurrence has an onset / observed-time / machine-detection claim citing a primary machine
    observation or a primary operator report that qualifies for that very claim."""
    for o in r.get("occurrences", []):
        for role in _WITNESS_ROLES:
            for _path, span, refs in _claims(o.get(role), role):
                for ref in refs:
                    e = ev.get(ref)
                    if e is None or e["strength"] != "primary":
                        continue
                    if e["kind"] == "machine_observation":
                        return True
                    if e["kind"] == "contemporaneous_operator_report" and "written_at" in e and _qualified(e, span):
                        return True
    return False


def _latency_errs(rid: str, o: dict) -> list[str]:
    errs = []
    det = o["first_machine_detection"]
    det_at = det["at"] if isinstance(det, dict) else det
    pairs = {"detection": (o["onset"], det_at), "notification": (det_at, o["alert"]["confirmed"]),
             "recovery": (o["onset"], o["restored_verification"])}
    for name, (a, b) in pairs.items():
        s_lo, _ = _span(a)
        _, e_hi = _span(b)
        if s_lo is not None and e_hi is not None and e_hi < s_lo:          # Codex phase-1 r4 #6
            errs.append(f"{rid}: impossible chronology: the {name} interval ends before it starts")
            continue
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


def _ef_binding_errs(rid: str, kind: str, entry: dict, root: Path) -> list[str]:
    """An expected-failure or characterization claim (one schema, one pair) is bound to the accepted pair
    that reproduced it and to the registry that pair accepted (Codex phase-1 r4 #6: a bare path to a
    missing file used to satisfy publication). Its controls must be nodes that pair passed, and its
    exception the registered ``module.qualname`` exactly."""
    path = root / entry["acceptance"]
    if not path.is_file():
        return [f"{rid}: {kind} acceptance file {entry['acceptance']} not found"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.get("acceptance_sha256"):
        return [f"{rid}: {kind} acceptance file hash mismatch"]
    acc = json.loads(data)
    errs = []
    if acc.get("verdict") != "accepted":
        errs.append(f"{rid}: {kind} pair whose verdict is {acc.get('verdict')!r}")
    missing = sorted(set(entry["nodes"]) - set(acc.get("accepted_nodes", [])))
    if missing:
        errs.append(f"{rid}: {kind} nodes the pair did not accept: {missing[:3]}")
    for n in entry["nodes"]:
        status = acc.get("connections", {}).get(n)
        if status not in ("value_match", "exception_shape"):
            errs.append(f"{rid}: {n} has no reviewed connection in the pair ({status})")
    for c in entry.get("controls", []):
        if c not in acc.get("passed_nodes", []):
            errs.append(f"{rid}: control {c} did not pass in the pair")
    reg = root / REGISTRY_PATH
    if not reg.is_file():
        return errs + [f"{rid}: the expected-failure registry {REGISTRY_PATH} is missing"]
    entries = json.loads(reg.read_text())["entries"]
    if hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest() != acc.get("registry_sha256"):
        errs.append(f"{rid}: the registry differs from the one the pair accepted")
    by_node = {x["node"]: x for x in entries}
    for n in entry["nodes"]:
        x = by_node.get(n)
        if x is None or x["exception"] != entry["exception"]:
            errs.append(f"{rid}: {n} is not registered with the record's exception {entry['exception']}")
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
    ev = {e["id"]: e for e in r.get("evidence", [])}
    claims = list(_claims({k: v for k, v in r.items() if k != "evidence"}))
    for e in r["evidence"]:
        if e["kind"] != "contemporaneous_operator_report":
            continue
        if "written_at" not in e:
            errs.append(f"{rid}: {e['id']} operator report needs written_at")
            continue
        citing = [(path, span) for path, span, refs in claims if e["id"] in refs]
        if not citing:
            errs.append(f"{rid}: {e['id']} operator report is not cited by any dated claim it describes")
        for path, span in citing:                        # every citation on its own (Codex phase-1 r4 #5)
            if not _qualified(e, span):
                errs.append(f"{rid}: {e['id']} is not contemporaneous with the claim at {path} (a report must be "
                            "written after the claim's earliest time and within 48 h of its latest)")
    if disp == "observed_incident" and not _witnessed(r, ev):
        errs.append(f"{rid}: observed_incident needs a qualified primary witness to the deviation (an onset, "
                    "observed-time or machine-detection claim citing a primary machine observation or a "
                    "qualifying primary operator report)")
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
    bound_claims = ([h for h in fx["historical_replay"] if h["status"] == "certified"]
                    + [d for d in fx["current_defence"] if d["status"] == "certified"] + fx["expected_failure"]
                    + fx["characterization"])
    if publish and bound_claims and evidence_root is None:
        errs.append(f"{rid}: publication needs an evidence root to bind its certified and expected-failure claims")
    if publish and evidence_root is not None:
        for h in fx["historical_replay"]:
            if h["status"] == "certified":
                errs += _binding_errs(rid, "replay", h, evidence_root)
        for d in fx["current_defence"]:
            if d["status"] == "certified":
                errs += _binding_errs(rid, "defence", d, evidence_root)
        for x in fx["expected_failure"]:
            errs += _ef_binding_errs(rid, "expected-failure", x, evidence_root)
        for x in fx["characterization"]:
            errs += _ef_binding_errs(rid, "characterization", x, evidence_root)

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
