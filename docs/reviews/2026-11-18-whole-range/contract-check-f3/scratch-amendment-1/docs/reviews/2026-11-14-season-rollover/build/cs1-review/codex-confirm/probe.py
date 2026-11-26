"""Narrow confirmation probes; every production call uses real on-disk files."""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone

from bts import contest_state as head

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bts.contest_state_before", ROOT / "scratch-acea53f8/bts/contest_state.py")
before = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = before
spec.loader.exec_module(before)
NOW = datetime(2026, 6, 18, 16, tzinfo=timezone.utc)
SOURCE = date(2026, 6, 16)
LIMIT = sys.get_int_max_str_digits()
assert LIMIT == 4300
BAD = {
    "integer": b"9" * (LIMIT + 1),
    "object_integer": b'{"result":' + b"9" * (LIMIT + 1) + b"}",
    "deep": b"[" * 200_000 + b"]" * 200_000,
}
RESULTS = {"python": sys.version, "head": str(Path(head.__file__).resolve()), "before": before.__file__, "digit_limit": LIMIT, "recursion_limit": sys.getrecursionlimit()}


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


capture = Capture()
head.logger.addHandler(capture)
head.logger.setLevel(logging.WARNING)
before.logger.addHandler(capture)
before.logger.setLevel(logging.WARNING)


def observation(p, kind, source="2026-06-18", expiry="2026-06-18T23:59:59Z"):
    a = p / "account_state"
    a.mkdir(exist_ok=True)
    value = {"active_streak": 7, "best_streak": 20, "source_date": source}
    if kind == "manual":
        value["override_expires_at"] = expiry
    target = a / ("contest_streak.manual.json" if kind == "manual" else "contest_streak.json")
    target.write_text(json.dumps(value), encoding="utf-8")
    return target


def fingerprint(p):
    return {str(f.relative_to(p)): hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob("*") if f.is_file()}


def error(call, expected=None):
    try:
        result = call()
    except Exception as exc:
        if expected:
            assert isinstance(exc, expected), (type(exc), str(exc))
        return {"exception": type(exc).__name__, "message": str(exc), "cause": type(exc.__cause__).__name__ if exc.__cause__ else None}
    assert expected is None, result
    if dataclasses.is_dataclass(result):
        result = dataclasses.asdict(result)
    return {"result": result}


def normalized(call):
    return json.dumps(error(call), sort_keys=True, default=str)


def model_matrix():
    results = []
    for corruption, raw in BAD.items():
        for kind in (None, "manual", "auto"):
            for required in (False, True):
                with tempfile.TemporaryDirectory(dir=ROOT, prefix="model-") as tmp:
                    p = Path(tmp)
                    if kind:
                        observation(p, kind)
                        (p / "streak.json").write_text('{"streak":4,"saver_available":false}')
                        baseline = dataclasses.asdict(head.load_decision_streak_state(p, require_contest_state=required, now=NOW))
                    (p / "streak.json").write_bytes(raw)
                    disk_before = fingerprint(p)
                    capture.records.clear()
                    call = lambda: head.load_decision_streak_state(p, require_contest_state=required, now=NOW)
                    if kind:
                        actual = dataclasses.asdict(call())
                        assert actual["model_streak"] is actual["model_saver_available"] is None
                        assert actual["model_state_error"]
                        for key in ("model_streak", "model_saver_available", "model_state_error"):
                            actual.pop(key)
                            baseline.pop(key)
                        assert actual == baseline, (corruption, kind, required)
                        outcome = "contest observation identical; diagnostics None"
                    else:
                        outcome = error(call, head.ContestStateError)
                        expected = "required but missing" if required else "model state is corrupt"
                        assert expected in outcome["message"]
                    assert len(capture.records) == 1
                    assert fingerprint(p) == disk_before
                    old = error(lambda: before.load_decision_streak_state(p, require_contest_state=required, now=NOW), (ValueError, RecursionError))
                    results.append({"input": corruption, "observation": kind, "required": required, "head": outcome, "before": old["exception"], "warnings": 1, "disk_unchanged": True})
    return results


def pick_matrix():
    results = []
    for corruption, raw in BAD.items():
        for kind in ("manual", "auto"):
            for required in (False, True):
                with tempfile.TemporaryDirectory(dir=ROOT, prefix="pick-") as tmp:
                    p = Path(tmp)
                    observation(p, kind, source=SOURCE.isoformat())
                    (p / "2026-06-18.json").write_bytes(raw)
                    (p / "2026-06-17.json").write_text('{"result":"hit"}')
                    snap = fingerprint(p)
                    assert head._resolved_pick_dates(p) == [date(2026, 6, 17)]
                    assert not head._has_unconfirmed_miss(p, SOURCE)
                    state = head.load_decision_streak_state(p, require_contest_state=required, now=NOW)
                    assert state.streak == 7 and state.status == "lagged"
                    assert snap == fingerprint(p)
                    # Prove each old scan independently fails, rather than short-circuiting on a miss.
                    for fn in (lambda: before._resolved_pick_dates(p), lambda: before._has_unconfirmed_miss(p, SOURCE)):
                        error(fn, (ValueError, RecursionError))
                    (p / "2026-06-17.json").write_text('{"result":"miss"}')
                    assert head._has_unconfirmed_miss(p, SOURCE)
                    state = head.load_decision_streak_state(p, require_contest_state=required, now=NOW)
                    assert state.status == "stale"
                    results.append({"input": corruption, "observation": kind, "required": required, "resolved": ["2026-06-17"], "miss_before_after": [False, True], "status_before_after": ["lagged", "stale"]})
    return results


def contest_matrix():
    results = []
    for corruption, raw in BAD.items():
        exception = RecursionError if corruption == "deep" else ValueError
        for kind in ("manual", "auto"):
            companions = (None, "auto") if kind == "manual" else (None, "active_manual", "expired_manual", "legacy_manual")
            for companion in companions:
                for required in (False, True):
                    with tempfile.TemporaryDirectory(dir=ROOT, prefix="contest-") as tmp:
                        p = Path(tmp)
                        path = observation(p, kind)
                        path.write_bytes(raw)
                        if companion == "auto":
                            observation(p, "auto")
                        elif companion:
                            m = observation(p, "manual", expiry="2026-06-18T23:59:59Z" if companion == "active_manual" else "2026-06-17T23:59:59Z")
                            if companion == "legacy_manual":
                                data = json.loads(m.read_text())
                                data.pop("override_expires_at")
                                m.write_text(json.dumps(data))
                        direct = error(lambda: head._parse_state_file(path, require_dated=required), head.ContestStateError)
                        assert direct["cause"] == exception.__name__ and str(path) in direct["message"]
                        snap = fingerprint(p)
                        call = lambda: head.load_decision_streak_state(p, require_contest_state=required, now=NOW)
                        if companion == "active_manual":
                            state = call()
                            assert state.streak == 7
                            assert before.load_decision_streak_state(p, require_contest_state=required, now=NOW).streak == 7
                            outcome = "manual wins before bad automatic read"
                        else:
                            outcome = error(call, head.ContestStateError)
                            assert outcome["cause"] == exception.__name__ and path.name in outcome["message"]
                            error(lambda: before.load_decision_streak_state(p, require_contest_state=required, now=NOW), exception)
                        assert snap == fingerprint(p)
                        results.append({"input": corruption, "target": kind, "companion": companion, "required": required, "outcome": outcome})
    return results


def recovery():
    decoder = json._default_decoder
    limit = sys.getrecursionlimit()
    direct = error(lambda: json.loads(BAD["deep"].decode()), RecursionError)
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="recovery-") as tmp:
        p = Path(tmp)
        path = observation(p, "manual")
        good = path.read_bytes()
        for i in range(100):
            path.write_bytes(BAD["deep"])
            out = error(lambda: head._parse_state_file(path), head.ContestStateError)
            assert out["cause"] == "RecursionError"
            path.write_bytes(good)
            assert head.load_decision_streak_state(p, require_contest_state=True, now=NOW).streak == 7
            assert json.loads('{"second_read": [1, 2, 3]}') == {"second_read": [1, 2, 3]}
        assert decoder is json._default_decoder and limit == sys.getrecursionlimit()
    return {"exact_DEEP_direct_exception": direct, "corrupt_then_healthy_cycles": 100, "decoder_and_recursion_limit_unchanged": True}


def differential():
    # Ordinary, edge and schema-invalid JSON: compare full results/errors, not only success.
    atoms = [None, True, False, 0, -1, 7, 57, 58, 1.0, float("inf"), float("nan"), "", "7", "miss", [], {}, [1], {"x": 1}]
    raws = [json.dumps(v).encode() for v in atoms]
    raws += [b"", b"{", b"\xff", b"\x00", b"\xef\xbb\xbf{}", b"1e999999", b"1e-999999", b"NaN", b"Infinity", b"-Infinity"]
    for key in ("streak", "saver_available", "result", "active_streak", "best_streak", "source_date", "as_of", "recorded_at", "override_expires_at", "source", "schema_version"):
        for atom in atoms:
            value = {"active_streak": 7, "best_streak": 20, "source_date": "2026-06-18", key: atom}
            raws.append(json.dumps(value).encode())
    counts = {"model": 0, "manual": 0, "auto": 0, "resolved": 0, "miss": 0}
    differences = []
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="differential-") as tmp:
        p = Path(tmp)
        a = p / "account_state"
        a.mkdir()
        targets = {"model": p / "streak.json", "manual": a / "contest_streak.manual.json", "auto": a / "contest_streak.json", "resolved": p / "2026-06-18.json", "miss": p / "2026-06-18.json"}
        for label, target in targets.items():
            for raw in raws:
                target.write_bytes(raw)
                for required in (False, True):
                    def invoke(mod):
                        if label == "resolved":
                            return mod._resolved_pick_dates(p)
                        if label == "miss":
                            return mod._has_unconfirmed_miss(p, SOURCE)
                        return mod.load_decision_streak_state(p, require_contest_state=required, now=NOW)
                    old = normalized(lambda: invoke(before))
                    new = normalized(lambda: invoke(head))
                    counts[label] += 1
                    if old != new:
                        differences.append({"site": label, "raw": raw.decode(errors="replace"), "required": required, "before": old, "head": new})
            target.unlink()
    assert not differences, differences[:3]
    return {"counts": counts, "total": sum(counts.values()), "differences": differences}


def try_audit():
    tree = ast.parse(Path(head.__file__).read_text())
    functions = ("_parse_state_file", "_resolved_pick_dates", "_has_unconfirmed_miss", "_load_model_diagnostics")
    result = {}
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in functions:
            blocks = []
            for node in ast.walk(fn):
                if isinstance(node, ast.Try) and any("_UNREADABLE" in ast.unparse(h.type) for h in node.handlers):
                    blocks.append({"line": node.lineno, "body": [ast.unparse(stmt) for stmt in node.body]})
            result[fn.name] = blocks
    return result


def fs_probes():
    results = []
    for kind in ("directory", "loop", "permission"):
        for site in ("model", "manual", "auto", "pick"):
            with tempfile.TemporaryDirectory(dir=ROOT, prefix="fs-") as tmp:
                p = Path(tmp)
                (p / "account_state").mkdir()
                target = p / {"model": "streak.json", "manual": "account_state/contest_streak.manual.json", "auto": "account_state/contest_streak.json", "pick": "2026-06-18.json"}[site]
                if kind == "directory":
                    target.mkdir()
                elif kind == "loop":
                    target.symlink_to(target.name)
                else:
                    target.write_text('{}')
                    target.chmod(0)
                raw_read = error(lambda: target.read_text(encoding="utf-8"), OSError)
                def invoke(mod):
                    if site == "model":
                        return mod._load_model_diagnostics(p)
                    if site == "pick":
                        return [mod._resolved_pick_dates(p), mod._has_unconfirmed_miss(p, SOURCE)]
                    return mod.load_decision_streak_state(p, require_contest_state=True, now=NOW)
                new = normalized(lambda: invoke(head))
                old = normalized(lambda: invoke(before))
                assert old == new
                direct = None
                if site in ("manual", "auto"):
                    direct = error(lambda: head._parse_state_file(target), head.ContestStateError)
                results.append({"kind": kind, "site": site, "raw_read": raw_read, "exists": target.exists(), "decision_or_scan": json.loads(new), "direct_parse": direct, "unchanged_from_before": True})
                if kind == "permission":
                    target.chmod(0o600)
    return results


def special_inputs():
    # Large finite exponent text does not invoke int conversion; float infinity is accepted.
    raws = {
        "positive_overflow": b"1e999999",
        "negative_overflow": b"-1e999999",
        "underflow": b"1e-999999",
        "long_exponent": b"1e" + b"9" * 100_000,
        "long_fraction": b"0." + b"9" * 100_000,
        "long_float_coefficient": b"9" * (LIMIT + 1) + b".0",
        "nul": b'{"x":"\x00"}',
        "escaped_nul": b'{"x":"\\u0000"}',
        "bom": b"\xef\xbb\xbf{}",
        "huge_string": b'"' + b"x" * (8 * 1024 * 1024) + b'"',
        "huge_string_object": b'{"active_streak":7,"source_date":"2026-06-18","x":"' + b"x" * (8 * 1024 * 1024) + b'"}',
    }
    results = []
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="special-") as tmp:
        p = Path(tmp)
        for label, raw in raws.items():
            target = p / "streak.json"
            target.write_bytes(raw)
            try:
                decoded = json.loads(target.read_text(encoding="utf-8"))
                direct = {"type": type(decoded).__name__}
                if not isinstance(decoded, (str, dict)):
                    direct["value"] = repr(decoded)
            except Exception as exc:
                direct = {"exception": type(exc).__name__, "message": str(exc)}
            sites = {}
            sites["model"] = error(lambda: head._load_model_diagnostics(p))
            pick = p / "2026-06-18.json"
            pick.write_bytes(raw)
            sites["resolved"] = error(lambda: head._resolved_pick_dates(p))
            sites["miss"] = error(lambda: head._has_unconfirmed_miss(p, SOURCE))
            account = p / "account_state"
            account.mkdir(exist_ok=True)
            for name in ("contest_streak.manual.json", "contest_streak.json"):
                state = account / name
                state.write_bytes(raw)
                sites[name] = error(lambda: head._parse_state_file(state))
            results.append({"input": label, "bytes": len(raw), "direct": direct, "sites": sites})
    return results


if __name__ == "__main__":
    for name, fn in (("try_audit", try_audit), ("model", model_matrix), ("pick", pick_matrix), ("contest", contest_matrix), ("recovery", recovery), ("differential", differential), ("filesystem", fs_probes), ("special_inputs", special_inputs)):
        RESULTS[name] = fn()
        print(name, len(RESULTS[name]) if isinstance(RESULTS[name], list) else "OK", flush=True)
        (ROOT / "probe-results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    print("ALL PROBES PASSED", flush=True)
