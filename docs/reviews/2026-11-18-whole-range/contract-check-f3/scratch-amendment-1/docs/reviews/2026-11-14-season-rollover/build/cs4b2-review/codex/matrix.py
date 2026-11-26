"""Independent base/branch Cartesian review of real on-disk loaders.

Run without pytest. Records interned exact outcomes, audit counts and separately
checks the contract oracle, so unexpected differences remain inspectable.
"""
import datetime as dt
import itertools
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from zoneinfo import ZoneInfo

import review_support as audit

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ET = ZoneInfo("America/New_York")
NOW = dt.datetime(2027, 7, 15, 12, tzinfo=ET)
ID = "a" * 32
AUTO = "bts_contest_streak_auto_v2"
MANUAL = "bts_contest_streak_manual_v3"


def contract(**kw):
    return dict(schema_version="bts_season_contract_v1", season=2027, status="automatic",
                activation_id=ID, switched_at=NOW.isoformat(), revoked_at=None, note=None) | kw


def automatic(**kw):
    return dict(schema_version=AUTO, season=2027, activation_id=ID,
                active_streak=12, best_streak=14, source="mlb_bts_profile",
                source_date="2027-07-14", recorded_at=NOW.isoformat()) | kw


def dated(day="2027-07-15", **kw):
    date = dt.date.fromisoformat(day)
    return dict(schema_version=MANUAL, season=date.year, effective_date=day,
                expires_at=dt.datetime.combine(date, dt.time.max, ET).isoformat(),
                active_streak=12, best_streak=14, saver_state="used", source="manual_cli",
                recorded_at=NOW.isoformat()) | kw


SWITCHES = {"absent": None, "automatic": contract(), "revoked": contract(status="revoked", revoked_at=NOW.isoformat()),
            "other_season": contract(season=2026)}
for n, value in enumerate([[], contract(schema_version=None), contract(season=True),
                          contract(status=[]), contract(activation_id="bad"),
                          contract(switched_at=None), contract(revoked_at="bad"), contract(note=[])], 1):
    SWITCHES[f"corrupt{n}"] = value
MANUALS = {
    "absent": None,
    "legacy": dict(active_streak=5, best_streak=9, override_expires_at="2027-07-16T16:00:00Z"),
    "expired": dict(active_streak=5, best_streak=9, override_expires_at="2027-07-14T16:00:00Z"),
    "no_expiry": dict(active_streak=5, best_streak=9),
    "claimed_auto_v1": automatic(schema_version="bts_contest_streak_auto_v1", source="mlb_bts_profile", override_expires_at="2027-07-16T16:00:00Z"),
    "claimed_auto_v2": automatic(source="mlb_bts_profile", override_expires_at="2027-07-16T16:00:00Z"),
    "misleading_source": dict(active_streak=5, source="contest_streak", override_expires_at="2027-07-16T16:00:00Z"),
    "today": dated(), "yesterday": dated("2027-07-14"), "tomorrow": dated("2027-07-16"),
    "invalid": dated(saver_state="bad"), "corrupt": b"{", "unreadable": "PERMISSION",
}
AUTOS = {
    "absent": None, "covered": automatic(), "v1": automatic(schema_version="bts_contest_streak_auto_v1"),
    "missing_schema": {k:v for k,v in automatic().items() if k != "schema_version"},
    "null_schema": automatic(schema_version=None), "number_schema": automatic(schema_version=3),
    "list_schema": automatic(schema_version=[]), "object_schema": automatic(schema_version={}),
    "unknown_schema": automatic(schema_version="unknown"), "other_id": automatic(activation_id="b"*32),
    "other_season": automatic(season=2026), "both_mismatch": automatic(season=2026, activation_id="b"*32),
    "season_bool": automatic(season=True), "season_string": automatic(season="2027"),
    "id_number": automatic(activation_id=3), "id_list": automatic(activation_id=[]),
    "id_upper": automatic(activation_id="A"*32), "refused": automatic(active_streak=58, best_streak=58),
    "corrupt": b"{", "undated": automatic(source_date=None, recorded_at=None),
    "recorded_only": automatic(source_date=None), "future": automatic(source_date="2027-07-20"),
    "as_of_only": automatic(source_date=None, as_of="2027-07-14", recorded_at=None),
}
for schema in ["bts_contest_streak_auto_v1", None, 3, "unknown"]:
    AUTOS[f"undated_{schema}"] = automatic(schema_version=schema, source_date=None, recorded_at=None)
MODELS = {"present": dict(season=2027, streak=9, saver_available=True), "absent": None, "corrupt": b"{"}
SAVERS = {"absent": None, "not_earned": dict(season=2027, state="not_earned"),
          "active": dict(season=2027, state="active"), "used": dict(season=2027, state="used"),
          "stale": dict(season=2026, state="active"), "invalid": dict(season=2027, state=[]),
          "corrupt": b"{", "huge_integer": b'{"n":' + b"9"*5000 + b"}"}


def install(path, value):
    if path.exists():
        path.chmod(0o600)
    path.unlink(missing_ok=True)
    if value is not None:
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        if value == "PERMISSION":
            path.chmod(0)


def worker(label):
    sys.path.insert(0, str(ROOT / "src" if label == "branch" else OUT / "base/src"))
    import bts.contest_state as cs
    from bts.health.contest_state import check
    logging.disable(logging.CRITICAL)
    picks = OUT / "matrix-files/picks"
    directory = picks / "account_state"
    directory.mkdir(parents=True, exist_ok=True)
    for day, result in [("2027-07-15", "hit"), ("2027-07-16", "miss")]:
        (picks / (day + ".json")).write_text(json.dumps(dict(result=result)))
    intern, values, rows = {}, [], []
    errors = []
    def call(fn, *a, **kw):
        audit.begin("matrix")
        try:
            value = fn(*a, **kw)
            if isinstance(value, list):
                result = ["ok", [asdict(x) for x in value]]
            else:
                result = ["ok", asdict(value) if value is not None else None]
        except Exception as exc:
            result = [type(exc).__name__, str(exc)]
        events = audit.end()
        reads = sum(e == "open" and "season_contract.json" in a[0] for e,a in events)
        mutations = [(e,a) for e,a in events if e != "open" or (len(a)>2 and a[2].isdigit() and int(a[2]) & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))]
        if mutations:
            errors.append(["mutation", mutations])
        key = json.dumps(result, default=str, sort_keys=True)
        if key not in intern:
            intern[key] = len(values)
            values.append(json.loads(key))
        return [intern[key], reads]
    for sw, manual, auto in itertools.product(SWITCHES, MANUALS, AUTOS):
        install(directory / "season_contract.json", SWITCHES[sw])
        install(directory / "contest_streak.manual.json", MANUALS[manual])
        install(directory / "contest_streak.json", AUTOS[auto])
        selected = [call(cs.load_contest_streak_state, picks, now=NOW, require_dated_auto=r) for r in [False, True]]
        for model, saver in itertools.product(MODELS, SAVERS):
            install(picks / "streak.json", MODELS[model])
            install(directory / "saver_state.json", SAVERS[saver])
            answers = selected + [call(cs.load_decision_streak_state, picks, now=NOW, require_contest_state=r) for r in [False, True]]
            answers += [call(check, picks, now=NOW, expected=r) for r in [False, True]]
            rows.append(answers)
    output = dict(source=cs.__file__, axes=[list(x) for x in [SWITCHES, MANUALS, AUTOS, MODELS, SAVERS]],
                  values=values, rows=rows, audit_errors=errors)
    (OUT / f"matrix-{label}.json").write_text(json.dumps(output, separators=(",",":")))
    print(label, len(rows), "cells", len(values), "distinct outcomes", len(errors), "audit mutations", flush=True)


def compare():
    base = json.loads((OUT / "matrix-base.json").read_text())
    branch = json.loads((OUT / "matrix-branch.json").read_text())
    assert base["axes"] == branch["axes"]
    summary, health_classes, errors = Counter(), Counter(), []
    def expected_reason(sw, selected):
        if sw.startswith("corrupt"):
            return "corrupt season switch, check " + sw[len("corrupt"):]
        if selected is not None and selected["path"].endswith("contest_streak.manual.json"):
            return "legacy manual override"
        if sw in ["absent", "revoked", "other_season"]:
            return "switch is " + sw
        if selected is None:
            return "no automatic observation"
        if selected["schema_version"] != AUTO:
            return "schema_version"
        if selected["season"] != 2027:
            return "automatic observation season"
        if selected["activation_id"] != ID:
            return "automatic observation activation_id"
        return None
    for names, b, c in zip(itertools.product(*base["axes"]), base["rows"], branch["rows"]):
        bv, cv = [[data["values"][item[0]] for item in row] for data,row in [(base,b), (branch,c)]]
        def require(ok, kind):
            if not ok:
                errors.append(dict(names=names, kind=kind, base=bv, branch=cv, reads=[b,c]))
        for i in [0,1,2,4]:
            require(bv[i] == cv[i], f"preserved_{i}")
            require(c[i][1] == 0, f"no_switch_read_{i}")
        selected = cv[1][1]
        bypass = cv[1][0] == "ok" and selected is not None and selected["effective_date"] == "2027-07-15"
        reason = None if cv[1][0] != "ok" or bypass else expected_reason(names[0], selected)
        require(c[3][1] == (0 if cv[1][0] != "ok" or bypass else 1), "decision_switch_count")
        if cv[1][0] != "ok":
            require(cv[3] == cv[1], "loader_error_priority")
            require(cv[5] == ["ok", [dict(level="CRITICAL",source="contest_state",message=cv[1][1])]], "health_loader_error")
            require(c[5][1] == 0, "health_loader_switch_count")
        elif reason:
            require(cv[3][0] == "ContestStateError" and reason in cv[3][1].split("\n")[0], "refusal_reason")
            if bv[5][0] != "ok":
                require(cv[5] == bv[5], "health_existing_exception")
            else:
                extra = dict(level="CRITICAL", source="contest_state", message=cv[3][1])
                require(cv[5] == ["ok", bv[5][1] + [extra]], "health_exact_append_last")
                require(c[5][1] == 1, "health_switch_once")
        else:
            require(cv[3] == bv[3], "accepted_exact_decision")
            require(cv[5] == bv[5], "accepted_exact_health")
            require(c[5][1] == (0 if bypass or bv[5][0] != "ok" else 1), "accepted_health_switch")
        summary["decision_changed" if bv[3] != cv[3] else "decision_identical"] += 1
        summary["health_changed" if bv[5] != cv[5] else "health_identical"] += 1
        summary["live_accepted" if cv[3][0] == "ok" else "live_raises"] += 1
        if cv[3][0] == "ok":
            require(bypass or reason is None, "uncovered_accepted")
        if cv[5][0] == "ok" and cv[3][0] == "ok" and any(x["level"] == "CRITICAL" for x in cv[5][1]):
            health_classes["accepted_but_existing_health_critical"] += 1
        if cv[5][0] != "ok" and reason:
            health_classes["existing_postloader_exception_before_append"] += 1
    report = dict(cells=len(base["rows"]), axes=base["axes"], sources=[base["source"],branch["source"]],
                  summary=summary, health_classes=health_classes, errors=len(errors), examples=errors[:20])
    (OUT / "matrix-summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert not errors, (len(errors), "see matrix-summary.json")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        worker(sys.argv[1])
    else:
        for label in ["base", "branch"]:
            subprocess.run([sys.executable, __file__, label], check=True)
        compare()
