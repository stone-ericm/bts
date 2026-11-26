"""Adversarial probe for CS6b-2 items 1, 2, 3, and 5:
- Row 5 AST and raw-byte identity of untouched definitions in src/bts/health/alert.py
- Item 1: Conforming status file parity (base 59efa825 vs branch 38d707bd) across 3,000+ cases
- Items 2 & 3: Malformed JSON contents, Rows 2 & 3 exactness, 50 MB / 200k-deep / 100k-key inputs,
  special filesystem objects, write guard parity, real denied reads, and stat->read TOCTOU races.
"""

from __future__ import annotations

import ast
import contextlib
import json
import logging
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TZ"] = "America/New_York"
sys.dont_write_bytecode = True
if hasattr(time, "tzset"):
    time.tzset()

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_ALERT_PATH = REVIEW_DIR / "scratch_base" / "src" / "bts" / "health" / "alert.py"
BRANCH_ALERT_PATH = REPO_ROOT / "src" / "bts" / "health" / "alert.py"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Strict offline network guard (allow loopback only)
_ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost"}
_real_getaddrinfo = socket.getaddrinfo
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _guarded_getaddrinfo(host, *args, **kwargs):
    if host not in _ALLOWED_HOSTS:
        raise AssertionError(f"Forbidden DNS lookup: {host!r}")
    return _real_getaddrinfo(host, *args, **kwargs)


def _check_addr(addr):
    if isinstance(addr, tuple) and addr:
        if addr[0] not in _ALLOWED_HOSTS:
            raise AssertionError(f"Forbidden outbound connect: {addr!r}")


def _guarded_connect(self, addr):
    _check_addr(addr)
    return _real_connect(self, addr)


def _guarded_connect_ex(self, addr):
    _check_addr(addr)
    return _real_connect_ex(self, addr)


socket.getaddrinfo = _guarded_getaddrinfo
socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex

import bts.dm as bts_dm
import bts.posting as bts_posting
from bts.health import alert as branch_alert

bts_dm.send_dm = Mock(side_effect=AssertionError("unmocked bts.dm.send_dm"))
bts_posting.post_to_bluesky = Mock(side_effect=AssertionError("unmocked post_to_bluesky"))


def load_base_alert() -> ModuleType:
    code = compile(BASE_ALERT_PATH.read_bytes(), str(BASE_ALERT_PATH), "exec")
    mod = ModuleType("_probe_base_alert")
    sys.modules[mod.__name__] = mod
    exec(code, mod.__dict__)
    return mod


base_alert = load_base_alert()


class ProbeTimeoutError(BaseException):
    pass


@contextlib.contextmanager
def bounded(seconds: float, label: str):
    def _on_alarm(*_):
        raise ProbeTimeoutError(f"Probe timed out after {seconds}s: {label}")

    old_handler = signal.signal(signal.SIGALRM, _on_alarm)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        signal.setitimer(signal.ITIMER_REAL, *old_timer)


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[tuple[int, str, str | None]] = []

    def emit(self, record: logging.LogRecord) -> None:
        exc_txt = type(record.exc_info[1]).__name__ if record.exc_info and record.exc_info[1] else None
        self.records.append((record.levelno, record.getMessage(), exc_txt))


@contextlib.contextmanager
def capture_logs(logger_name: str = "bts.health.alert"):
    logger = logging.getLogger(logger_name)
    old_level = logger.level
    old_propagate = logger.propagate
    handler = ListHandler()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def make_clock(year: int, month: int, day: int, hour: int = 12):
    class PinnedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            base_dt = datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)
            if tz is not None:
                return base_dt.astimezone(tz)
            return base_dt.replace(tzinfo=None)

    return PinnedClock


def check_row5_identity() -> dict:
    diff_files = subprocess.check_output(
        ["git", "diff", "--name-only", "59efa825", "HEAD", "--", "src"],
        cwd=REPO_ROOT,
        text=True,
    ).splitlines()
    assert diff_files == ["src/bts/health/alert.py"], f"Unexpected src diff: {diff_files}"

    base_bytes = BASE_ALERT_PATH.read_bytes()
    branch_bytes = BRANCH_ALERT_PATH.read_bytes()
    base_lines = base_bytes.splitlines(keepends=True)
    branch_lines = branch_bytes.splitlines(keepends=True)
    base_tree = ast.parse(base_bytes)
    branch_tree = ast.parse(branch_bytes)

    def top_slices(tree: ast.Module, lines: list[bytes]) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
                out[node.name] = b"".join(lines[start : node.end_lineno])
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                out[node.targets[0].id] = b"".join(lines[node.lineno - 1 : node.end_lineno])
        return out

    b_slices = top_slices(base_tree, base_lines)
    h_slices = top_slices(branch_tree, branch_lines)
    unchanged_names = [
        "HEALTH_DM_STATUS_SCHEMA_VERSION",
        "Alert",
        "log_alerts",
        "_write_health_dm_delivery_status",
        "dispatch_dm_for_critical",
        "format_health_dm_body",
    ]
    for name in unchanged_names:
        assert b_slices[name] == h_slices[name], f"Row 5 mismatch in {name}"
    # Module docstring check
    assert ast.get_docstring(base_tree) == ast.get_docstring(branch_tree)
    return {
        "src_files_changed": diff_files,
        "byte_identical_symbols": unchanged_names,
        "added_helpers": sorted(set(h_slices) - set(b_slices)),
    }


def check_conforming_parity() -> dict:
    d_same = date(2027, 7, 14)
    d_next = date(2027, 7, 15)
    clock_same = make_clock(2027, 7, 14, 16)  # 12:00 EDT
    clock_next = make_clock(2027, 7, 15, 16)  # 12:00 EDT

    def build_priors(tmp_dir: Path) -> dict[str, bytes | None]:
        base_alert.datetime = clock_same
        priors: dict[str, bytes | None] = {}
        p = tmp_dir / "seed.json"

        base_alert._write_health_dm_delivery_status(
            p, status="sent", dm_recipient="offline.test",
            critical_count=1, warn_attention_count=0,
            body="BTS health CRITICAL alert(s):\n- [a] msg a",
            sent_day=d_same.isoformat(), sent_sources=["CRITICAL:a"],
        )
        priors["sent_one"] = p.read_bytes()

        base_alert._write_health_dm_delivery_status(
            p, status="sent", dm_recipient="offline.test",
            critical_count=2, warn_attention_count=1,
            body="BTS health CRITICAL alert(s):\n- [a] msg a\n- [b] msg b\n\nWARN attention:\n- [w1] warn 1",
            sent_day=d_same.isoformat(),
            sent_sources=["CRITICAL:a", "CRITICAL:b:inc", "WARN_ATTENTION:w1"],
        )
        priors["sent_several"] = p.read_bytes()

        base_alert._write_health_dm_delivery_status(
            p, status="failed", dm_recipient="offline.test",
            critical_count=1, warn_attention_count=0,
            body="BTS health CRITICAL alert(s):\n- [a] msg a",
            error="RuntimeError: boom",
        )
        priors["failed_crit1"] = p.read_bytes()

        base_alert._write_health_dm_delivery_status(
            p, status="failed", dm_recipient="offline.test",
            critical_count=3, warn_attention_count=1,
            body="BTS health CRITICAL alert(s):\n- [a] 1",
            error="TimeoutError: timed out",
        )
        priors["failed_crit3"] = p.read_bytes()

        base_alert._write_health_dm_delivery_status(
            p, status="failed", dm_recipient="offline.test",
            critical_count=0, warn_attention_count=1,
            body="BTS health WARN attention:\n- [w1] warn 1",
            error="RuntimeError: warn fail",
        )
        priors["failed_no_crit"] = p.read_bytes()

        base_alert._write_health_dm_delivery_status(
            p, status="skipped_no_recipient", dm_recipient=None,
            critical_count=1, warn_attention_count=0,
            body="BTS health CRITICAL alert(s):\n- [a] msg a",
            error="dm_recipient is not configured",
        )
        priors["skipped_crit"] = p.read_bytes()

        base_alert._write_health_dm_delivery_status(
            p, status="skipped_no_recipient", dm_recipient="",
            critical_count=0, warn_attention_count=1,
            body="BTS health WARN attention:\n- [w1] warn 1",
            error="dm_recipient is not configured",
        )
        priors["skipped_warn"] = p.read_bytes()

        priors["legacy_no_dedup"] = json.dumps({
            "schema_version": "bts_health_dm_delivery_status_v1",
            "status": "sent",
            "critical_count": 1,
            "warn_attention_count": 0,
            "body_sha256": "abc123",
            "body_first_line": "BTS health CRITICAL alert(s):",
        }, indent=2, sort_keys=True).encode("utf-8") + b"\n"

        priors["no_file"] = None
        return priors

    def mk_alerts(mod, shape: str):
        A = mod.Alert
        if shape == "one_crit_covered":
            return [A("CRITICAL", "a", "msg a")], []
        if shape == "one_crit_uncovered":
            return [A("CRITICAL", "u", "msg u")], []
        if shape == "several_crit_covered":
            return [A("CRITICAL", "a", "m1"), A("CRITICAL", "b", "m2", incident_key="b:inc")], []
        if shape == "several_crit_partial":
            return [
                A("CRITICAL", "a", "m1"),
                A("CRITICAL", "b", "m2", incident_key="b:inc"),
                A("CRITICAL", "c", "m3"),
            ], []
        if shape == "warn_covered":
            return [], [A("WARN", "w1", "warn 1")]
        if shape == "warn_uncovered":
            return [], [A("WARN", "w2", "warn 2")]
        if shape == "both_covered":
            return [A("CRITICAL", "a", "m1")], [A("WARN", "w1", "warn 1")]
        if shape == "both_uncovered":
            return [A("CRITICAL", "u", "m1")], [A("WARN", "w2", "warn 2")]
        if shape == "empty":
            return [], []
        if shape == "info_and_plain_warn_only":
            return [A("INFO", "inf", "i"), A("WARN", "plain_w", "pw")], []
        raise ValueError(shape)

    shapes = [
        "one_crit_covered", "one_crit_uncovered", "several_crit_covered",
        "several_crit_partial", "warn_covered", "warn_uncovered",
        "both_covered", "both_uncovered", "empty", "info_and_plain_warn_only",
    ]
    recipients = ["offline.test", None, ""]
    days = [("same", d_same, clock_same), ("next", d_next, clock_next)]
    pass_date_modes = [True, False]
    use_path_modes = [True, False]
    transport_modes = ["ok", "fail"]

    total = 0
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as td:
        tmp = Path(td)
        priors = build_priors(tmp)

        def run_one(mod, logger_name: str, prior_bytes: bytes | None, shape: str,
                    recipient, target_day: date, clock_cls, pass_date: bool,
                    use_path: bool, transport_mode: str):
            p = tmp / f"run_{mod.__name__}.json"
            p.unlink(missing_ok=True)
            if use_path and prior_bytes is not None:
                p.write_bytes(prior_bytes)
            mod.datetime = clock_cls
            send_mock = Mock(
                side_effect=RuntimeError("transport down") if transport_mode == "fail" else None,
                return_value="dm-ok",
            )
            mod.send_dm = send_mock
            alerts, warns = mk_alerts(mod, shape)
            with capture_logs(logger_name) as logs:
                ret = mod.dispatch_dm_for_health_alerts(
                    alerts,
                    recipient,
                    warn_attention=warns,
                    status_path=p if use_path else None,
                    now_et_date=target_day if pass_date else None,
                )
            disk = p.read_bytes() if (use_path and p.exists()) else None
            return ret, send_mock.call_args_list, logs, disk

        # Ensure base_alert logs to its own logger name or capture base_alert.log.name
        for prior_name, prior_bytes in priors.items():
            for use_path in use_path_modes:
                if not use_path and prior_name != "no_file":
                    continue
                for shape in shapes:
                    for recipient in recipients:
                        for day_label, target_day, clock_cls in days:
                            for pass_date in pass_date_modes:
                                for t_mode in transport_modes:
                                    b_out = run_one(
                                        base_alert, base_alert.log.name, prior_bytes,
                                        shape, recipient, target_day, clock_cls,
                                        pass_date, use_path, t_mode,
                                    )
                                    h_out = run_one(
                                        branch_alert, branch_alert.log.name, prior_bytes,
                                        shape, recipient, target_day, clock_cls,
                                        pass_date, use_path, t_mode,
                                    )
                                    assert b_out == h_out, (
                                        f"Parity mismatch for prior={prior_name}, use_path={use_path}, "
                                        f"shape={shape}, recipient={recipient!r}, day={day_label}, "
                                        f"pass_date={pass_date}, transport={t_mode}:\n"
                                        f"BASE={b_out}\nBRANCH={h_out}"
                                    )
                                    total += 1

        # Exhaustive search over synthetic conforming dicts to test whether removing
        # `resurface is None` can ever alter behaviour on any valid-typed JSON object.
        synthetic_checked = 0
        for st in ["sent", "failed", "skipped_no_recipient", "other", None]:
            for cc in [0, 1, 2, None]:
                for sday in [d_same.isoformat(), d_next.isoformat(), None]:
                    for ssrc in [None, [], ["CRITICAL:a"], ["CRITICAL:b"]]:
                        obj = {}
                        if st is not None:
                            obj["status"] = st
                        if cc is not None:
                            obj["critical_count"] = cc
                        if sday is not None:
                            obj["sent_day"] = sday
                        if ssrc is not None:
                            obj["sent_sources"] = ssrc
                        raw = json.dumps(obj).encode("utf-8")
                        for shape in ["one_crit_covered", "one_crit_uncovered", "empty"]:
                            b_out = run_one(
                                base_alert, base_alert.log.name, raw, shape,
                                "offline.test", d_same, clock_same, True, True, "ok",
                            )
                            h_out = run_one(
                                branch_alert, branch_alert.log.name, raw, shape,
                                "offline.test", d_same, clock_same, True, True, "ok",
                            )
                            assert b_out == h_out, f"Synthetic object diff on {obj}: {b_out} vs {h_out}"
                            synthetic_checked += 1

        # Explicit gate-order verification:
        # 1. Empty-body gate precedes recipient gate & write (unless resurface banner exists)
        p_gate = tmp / "gate.json"
        p_gate.write_bytes(priors["sent_one"])
        branch_alert.send_dm = Mock()
        assert branch_alert.dispatch_dm_for_health_alerts(
            [], None, status_path=p_gate, now_et_date=d_same
        ) is False
        assert p_gate.read_bytes() == priors["sent_one"], "Empty-body gate did not return before recipient write"

        # 2. Banner-only send passes empty-body gate, skips dedup, hits recipient gate or sends
        p_gate.write_bytes(priors["failed_crit1"])
        assert branch_alert.dispatch_dm_for_health_alerts(
            [], None, status_path=p_gate, now_et_date=d_same
        ) is False
        assert json.loads(p_gate.read_bytes())["status"] == "skipped_no_recipient"

        p_gate.write_bytes(priors["failed_crit1"])
        assert branch_alert.dispatch_dm_for_health_alerts(
            [], "offline.test", status_path=p_gate, now_et_date=d_same
        ) is True
        assert branch_alert.send_dm.call_count == 1
        saved_banner = json.loads(p_gate.read_bytes())
        assert saved_banner["status"] == "sent" and saved_banner["sent_sources"] == []

        # 3. Dedup precedes recipient gate (covered key + dm_recipient=None suppresses without overwriting!)
        p_gate.write_bytes(priors["sent_one"])
        assert branch_alert.dispatch_dm_for_health_alerts(
            [branch_alert.Alert("CRITICAL", "a", "msg a")], None,
            status_path=p_gate, now_et_date=d_same,
        ) is False
        assert p_gate.read_bytes() == priors["sent_one"], "Dedup did not precede recipient gate"

    return {
        "writer_parity_combinations": total,
        "synthetic_object_combinations": synthetic_checked,
        "conforming_differences_found": 0,
        "gate_order_preserved": True,
    }


def check_malformed_and_special_paths() -> dict:
    d_same = date(2027, 7, 14)
    d_stale = date(2027, 7, 13)
    clock_same = make_clock(2027, 7, 14, 16)
    branch_alert.datetime = clock_same

    A = branch_alert.Alert
    crit = A("CRITICAL", "probe", "critical probe")
    warn = A("WARN", "probe", "warn probe")
    key = "CRITICAL:probe"
    banner_1 = "⚠️ Previous health alert DM FAILED to send (1 CRITICAL) — it may have been missed."

    lanes = [
        "health_dm_delivery_status.json",
        "contest_stale_reading_dm_delivery_status.json",
        "contest_morning_check_dm_delivery_status.json",
    ]
    modes = [
        ("critical", [crit], [], "offline.test", True),
        ("warn", [], [warn], "offline.test", True),
        ("empty", [], [], "offline.test", False),
        ("no_recipient", [crit], [], None, False),
    ]

    assert sys.getrecursionlimit() == 1000, f"Unexpected default recursion limit: {sys.getrecursionlimit()}"

    raw_contents: dict[str, bytes] = {
        "array": b"[]",
        "null": b"null",
        "string": b'"hello"',
        "int": b"42",
        "float": b"-3.14",
        "true": b"true",
        "false": b"false",
        "nested_arr_20k": b"[" * 20_000 + b"0" + b"]" * 20_000,
        "nested_obj_20k": b'{"a":' * 20_000 + b"0" + b"}" * 20_000,
        "nested_arr_200k": b"[" * 200_000 + b"0" + b"]" * 200_000,
        "nested_obj_200k": b'{"a":' * 200_000 + b"0" + b"}" * 200_000,
        "invalid_json": b"{not valid json",
        "non_utf8": b"\xff\xfe\x80",
        "empty_file": b"",
        "whitespace": b"   \n\t ",
        "top_nan": b"NaN",
        "top_inf": b"Infinity",
        "huge_int_5000d": b"9" * 5000,
        "obj_huge_int_5000d": b'{"status":"failed","critical_count":' + b"9" * 5000 + b"}",
        "wrong_types_all": json.dumps({
            "schema_version": [], "status": {}, "updated_at": 123,
            "recipient_configured": "yes", "recipient": [], "critical_count": {},
            "warn_attention_count": [], "body_sha256": 0, "body_first_line": None,
            "error": [], "sent_day": {}, "sent_sources": 99,
        }).encode(),
        "status_unhashable_list": b'{"status": []}',
        "status_unhashable_dict": b'{"status": {}}',
        "sent_day_int": b'{"status":"sent","sent_day":20270714,"sent_sources":["CRITICAL:probe"]}',
        "sent_day_bool": b'{"status":"sent","sent_day":true,"sent_sources":["CRITICAL:probe"]}',
        "sent_day_list": b'{"status":"sent","sent_day":[],"sent_sources":["CRITICAL:probe"]}',
        "sent_day_null": b'{"status":"sent","sent_day":null,"sent_sources":["CRITICAL:probe"]}',
        "sources_str": json.dumps({"status": "sent", "sent_day": d_same.isoformat(), "sent_sources": key}).encode(),
        "sources_dict": json.dumps({"status": "sent", "sent_day": d_same.isoformat(), "sent_sources": {key: True}}).encode(),
        "sources_int_dict": json.dumps({"status": "sent", "sent_day": d_same.isoformat(), "sent_sources": [1, {}]}).encode(),
        "sources_int_only": json.dumps({"status": "sent", "sent_day": d_same.isoformat(), "sent_sources": [1]}).encode(),
        "sources_surrogate": b'{"status":"sent","sent_day":"2027-07-14","sent_sources":["\\ud800"]}',
    }

    content_runs = 0
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as td:
        tmp = Path(td)

        # 1. Test all raw_contents across all 3 lanes x 4 modes
        for lane in lanes:
            for cname, raw in raw_contents.items():
                for mname, c_list, w_list, recip, expected_ret in modes:
                    p = tmp / lane
                    p.write_bytes(raw)
                    send_mock = Mock(return_value="ok")
                    branch_alert.send_dm = send_mock
                    with bounded(2.0, f"{lane}:{cname}:{mname}"):
                        ret = branch_alert.dispatch_dm_for_health_alerts(
                            c_list, recip, warn_attention=w_list,
                            status_path=p, now_et_date=d_same,
                        )
                    assert ret is expected_ret, f"{cname}/{mname}: got {ret}"
                    assert send_mock.call_count == int(expected_ret)
                    if expected_ret:
                        expected_body = branch_alert.format_health_dm_body(c_list, w_list)
                        assert send_mock.call_args.args == (recip, expected_body)
                        persisted = json.loads(p.read_bytes())
                        assert persisted["status"] == "sent"
                        if cname == "sources_surrogate" and mname == "critical":
                            assert persisted["sent_sources"] == ["CRITICAL:probe", "\ud800"]
                        elif mname == "critical":
                            assert persisted["sent_sources"] == ["CRITICAL:probe"]
                    elif mname == "no_recipient":
                        assert json.loads(p.read_bytes())["status"] == "skipped_no_recipient"
                    else:
                        assert p.read_bytes() == raw
                    content_runs += 1

        # 2. 50 MB file tests (valid covering, valid non-covering, invalid JSON)
        p50 = tmp / "big50mb.json"
        pad = "x" * (50 * 1024 * 1024)
        for kind50 in ("covering", "non_covering", "invalid"):
            if kind50 == "covering":
                p50.write_text(json.dumps({
                    "status": "sent", "sent_day": d_same.isoformat(),
                    "sent_sources": [key], "pad": pad,
                }))
            elif kind50 == "non_covering":
                p50.write_text(json.dumps({
                    "status": "sent", "sent_day": d_same.isoformat(),
                    "sent_sources": ["CRITICAL:other"], "pad": pad,
                }))
            else:
                p50.write_text("{" + pad)
            assert p50.stat().st_size >= 50 * 1024 * 1024
            send_mock = Mock(return_value="ok")
            branch_alert.send_dm = send_mock
            with bounded(5.0, f"50MB:{kind50}"):
                ret = branch_alert.dispatch_dm_for_health_alerts(
                    [crit], "offline.test", status_path=p50, now_et_date=d_same,
                )
            if kind50 == "covering":
                assert ret is False and send_mock.call_count == 0
            else:
                assert ret is True and send_mock.call_count == 1
                saved50 = json.loads(p50.read_bytes())
                assert saved50["status"] == "sent"
                expected_srcs = ["CRITICAL:other", key] if kind50 == "non_covering" else [key]
                assert saved50["sent_sources"] == expected_srcs
            p50.unlink(missing_ok=True)

        # 3. 100,000 strings in sent_sources, duplicate keys, mixed junk
        many_other = [f"CRITICAL:src_{i:06d}" for i in range(100_000)]
        p_many = tmp / "many.json"
        # Non-covering 100k
        p_many.write_text(json.dumps({
            "status": "sent", "sent_day": d_same.isoformat(), "sent_sources": many_other,
        }))
        send_mock = Mock(return_value="ok")
        branch_alert.send_dm = send_mock
        with bounded(3.0, "100k_non_covering"):
            assert branch_alert.dispatch_dm_for_health_alerts(
                [crit], "offline.test", status_path=p_many, now_et_date=d_same,
            ) is True
        saved_many = json.loads(p_many.read_bytes())
        assert saved_many["status"] == "sent" and len(saved_many["sent_sources"]) == 100_001
        assert key in saved_many["sent_sources"]
        # Covering 100k+1 (second call suppressed)
        with bounded(3.0, "100k_covering"):
            assert branch_alert.dispatch_dm_for_health_alerts(
                [crit], "offline.test", status_path=p_many, now_et_date=d_same,
            ) is False
        assert send_mock.call_count == 1
        p_many.unlink(missing_ok=True)

        # Duplicate strings in sent_sources & mixed junk
        p_mix = tmp / "mix.json"
        p_mix.write_text(
            '{"status":"sent","sent_day":"2027-07-14",'
            '"sent_sources":["CRITICAL:other","CRITICAL:other",1,{},null,true,NaN,Infinity]}'
        )
        branch_alert.send_dm = Mock(return_value="ok")
        assert branch_alert.dispatch_dm_for_health_alerts(
            [crit], "offline.test", status_path=p_mix, now_et_date=d_same,
        ) is True
        assert json.loads(p_mix.read_bytes())["sent_sources"] == ["CRITICAL:other", key]

        # Covering with mixed junk -> suppresses and leaves file untouched
        raw_cov = (
            b'{"status":"sent","sent_day":"2027-07-14",'
            b'"sent_sources":["CRITICAL:probe",1,{},null,true,NaN]}'
        )
        p_mix.write_bytes(raw_cov)
        branch_alert.send_dm = Mock(return_value="ok")
        assert branch_alert.dispatch_dm_for_health_alerts(
            [crit], "offline.test", status_path=p_mix, now_et_date=d_same,
        ) is False
        assert branch_alert.send_dm.call_count == 0
        assert p_mix.read_bytes() == raw_cov

        # Duplicate JSON object keys: last key wins in Python json.loads
        p_mix.write_text(
            '{"status":"failed","critical_count":2,"status":"sent",'
            '"sent_day":"2027-07-14","sent_sources":["CRITICAL:probe"]}'
        )
        assert branch_alert.dispatch_dm_for_health_alerts(
            [crit], "offline.test", status_path=p_mix, now_et_date=d_same,
        ) is False

        # Stale sent_day and non-sent status ignore prior sent_sources
        for bad_prior in (
            {"status": "sent", "sent_day": d_stale.isoformat(), "sent_sources": [key, "CRITICAL:old"]},
            {"status": "failed", "critical_count": 0, "sent_day": d_same.isoformat(), "sent_sources": [key, "CRITICAL:old"]},
            {"status": "skipped_no_recipient", "sent_day": d_same.isoformat(), "sent_sources": [key, "CRITICAL:old"]},
        ):
            p_mix.write_text(json.dumps(bad_prior))
            branch_alert.send_dm = Mock(return_value="ok")
            assert branch_alert.dispatch_dm_for_health_alerts(
                [crit], "offline.test", status_path=p_mix, now_et_date=d_same,
            ) is True
            assert json.loads(p_mix.read_bytes())["sent_sources"] == [key]

        # 4. Row 2: critical_count matrix (including huge ints, NaN, Infinity, bool, float)
        huge_valid = 10**100
        count_cases = [
            (json.dumps({"status": "failed", "critical_count": 1}), True, 1),
            (json.dumps({"status": "failed", "critical_count": 3}), True, 3),
            (json.dumps({"status": "failed", "critical_count": huge_valid}), True, huge_valid),
            ('{"status": "failed", "critical_count": ' + "9" * 5000 + "}", False, None),
            (json.dumps({"status": "failed", "critical_count": 0}), False, None),
            (json.dumps({"status": "failed", "critical_count": -1}), False, None),
            (json.dumps({"status": "failed", "critical_count": -(10**100)}), False, None),
            (json.dumps({"status": "failed", "critical_count": None}), False, None),
            (json.dumps({"status": "failed", "critical_count": "1"}), False, None),
            (json.dumps({"status": "failed", "critical_count": 1.5}), False, None),
            (json.dumps({"status": "failed", "critical_count": True}), False, None),
            (json.dumps({"status": "failed", "critical_count": False}), False, None),
            ('{"status": "failed", "critical_count": NaN}', False, None),
            ('{"status": "failed", "critical_count": Infinity}', False, None),
            ('{"status": "failed", "critical_count": -Infinity}', False, None),
            (json.dumps({"status": "sent", "critical_count": 2}), False, None),
        ]
        p_cc = tmp / "cc.json"
        for raw_txt, want_banner, n_val in count_cases:
            for with_alert in (False, True):
                p_cc.write_text(raw_txt)
                send_mock = Mock(return_value="ok")
                branch_alert.send_dm = send_mock
                alerts_in = [crit] if with_alert else []
                with bounded(2.0, f"cc:{want_banner}:{with_alert}"):
                    ret = branch_alert.dispatch_dm_for_health_alerts(
                        alerts_in, "offline.test", status_path=p_cc, now_et_date=d_same,
                    )
                expected_banner = (
                    f"⚠️ Previous health alert DM FAILED to send ({n_val} CRITICAL) — it may have been missed."
                    if want_banner else None
                )
                alert_body = branch_alert.format_health_dm_body(alerts_in, [])
                if expected_banner:
                    expected_full = expected_banner + ("\n\n" + alert_body if alert_body else "")
                else:
                    expected_full = alert_body
                assert ret is (expected_full is not None)
                assert send_mock.call_count == int(expected_full is not None)
                if expected_full is not None:
                    assert send_mock.call_args.args[1] == expected_full
                    assert json.loads(p_cc.read_bytes())["status"] == "sent"
                else:
                    assert p_cc.read_text() == raw_txt

        # 5. Special filesystem objects & permissions & write guard parity
        special_results = {}

        def run_modes_on_path(target_path: Path, label: str, writable: bool, expect_not_reg_log: bool = False):
            for mname, c_list, w_list, recip, expected_ret in modes:
                send_mock = Mock(return_value="ok")
                branch_alert.send_dm = send_mock
                with capture_logs() as logs, bounded(2.0, f"special:{label}:{mname}"):
                    ret = branch_alert.dispatch_dm_for_health_alerts(
                        c_list, recip, warn_attention=w_list,
                        status_path=target_path, now_et_date=d_same,
                    )
                assert ret is expected_ret
                assert send_mock.call_count == int(expected_ret)
                if mname in ("critical", "warn", "no_recipient"):
                    if writable:
                        want_st = "sent" if expected_ret else "skipped_no_recipient"
                        assert json.loads(target_path.read_bytes())["status"] == want_st
                    else:
                        warn_msgs = [msg for lvl, msg, _ in logs if lvl == logging.WARNING]
                        assert any("could not write health DM delivery status" in m for m in warn_msgs), (
                            f"Missing write warning for {label}/{mname}: {logs}"
                        )
                        if expect_not_reg_log:
                            assert any("not a regular file" in m for m in warn_msgs), (
                                f"Expected 'not a regular file' for {label}/{mname}: {warn_msgs}"
                            )

        # 5a. Directory
        d_dir = tmp / "a_dir"
        d_dir.mkdir()
        run_modes_on_path(d_dir, "directory", writable=False, expect_not_reg_log=True)
        special_results["directory"] = "ok"

        # 5b. FIFO without writer
        fifo_plain = tmp / "plain.fifo"
        os.mkfifo(fifo_plain)
        run_modes_on_path(fifo_plain, "fifo_no_writer", writable=False, expect_not_reg_log=True)
        special_results["fifo_no_writer"] = "ok"

        # 5c. FIFO WITH a writer (pre-loaded with covering 'sent' payload so an unguarded read wouldn't block,
        # it would consume the pipe and falsely suppress!)
        fifo_writer = tmp / "writer.fifo"
        os.mkfifo(fifo_writer)
        rw_fd = os.open(fifo_writer, os.O_RDWR | os.O_NONBLOCK)
        pipe_payload = json.dumps({
            "status": "sent", "sent_day": d_same.isoformat(), "sent_sources": [key],
        }).encode() + b"\n"
        os.write(rw_fd, pipe_payload)
        try:
            run_modes_on_path(fifo_writer, "fifo_with_writer", writable=False, expect_not_reg_log=True)
            # Verify our pipe payload was NEVER read or written by the dispatcher!
            still_in_pipe = os.read(rw_fd, 4096)
            assert still_in_pipe == pipe_payload, f"FIFO bytes were disturbed: {still_in_pipe!r}"
        finally:
            os.close(rw_fd)
        special_results["fifo_with_writer"] = "ok_untouched"

        # 5d. Socket (AF_UNIX if permitted by sandbox; otherwise test via stat S_IFSOCK mock + document Seatbelt)
        sock_path = tmp / "s.sock"
        unix_sock = None
        try:
            unix_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            unix_sock.bind(str(sock_path))
            has_real_sock = True
        except PermissionError as exc:
            has_real_sock = False
            special_results["unix_socket_bind_in_sandbox"] = f"Seatbelt blocked bind ({exc})"
        try:
            if has_real_sock:
                run_modes_on_path(sock_path, "unix_socket", writable=False, expect_not_reg_log=True)
                special_results["unix_socket"] = "real_sock_ok"
            else:
                # Verify S_IFSOCK handling directly via stat_result with S_IFSOCK mode
                fake_sock = tmp / "fake_sock_target"
                fake_sock.write_text("{}")
                real_stat = Path.stat
                sock_stat = os.stat_result((stat.S_IFSOCK | 0o600, 0, 0, 1, 0, 0, 0, 0, 0, 0))
                try:
                    Path.stat = lambda self, *a, **kw: sock_stat if self == fake_sock else real_stat(self, *a, **kw)
                    run_modes_on_path(fake_sock, "simulated_socket_mode", writable=False, expect_not_reg_log=True)
                finally:
                    Path.stat = real_stat
                special_results["unix_socket"] = "mode_verified_ok"
        finally:
            if unix_sock is not None:
                unix_sock.close()

        # 5e. Device (/dev/null via symlink)
        devnull_link = tmp / "devnull.json"
        devnull_link.symlink_to("/dev/null")
        run_modes_on_path(devnull_link, "symlink_devnull", writable=False, expect_not_reg_log=True)
        special_results["symlink_to_devnull"] = "ok_logged_not_regular"

        # 5f. Looping symlink
        loop_link = tmp / "loop.json"
        loop_link.symlink_to(loop_link.name)
        run_modes_on_path(loop_link, "loop_symlink", writable=False, expect_not_reg_log=False)
        special_results["looping_symlink"] = "ok"

        # 5g. Dangling symlink in existing dir (base writes through it and creates target; branch must too!)
        for mname, c_list, w_list, recip, expected_ret in modes:
            dangling = tmp / f"dangling_{mname}.json"
            dangling_target = tmp / f"dangling_{mname}.target"
            dangling.symlink_to(dangling_target.name)
            send_mock = Mock(return_value="ok")
            branch_alert.send_dm = send_mock
            ret = branch_alert.dispatch_dm_for_health_alerts(
                c_list, recip, warn_attention=w_list, status_path=dangling, now_et_date=d_same,
            )
            assert ret is expected_ret
            if mname == "empty":
                assert not dangling_target.exists()
            else:
                assert dangling_target.is_file()
                assert json.loads(dangling.read_bytes())["status"] == (
                    "sent" if expected_ret else "skipped_no_recipient"
                )
        # Also verify base writes through dangling symlink identically
        b_dang = tmp / "base_dang.json"
        b_dang_t = tmp / "base_dang.target"
        b_dang.symlink_to(b_dang_t.name)
        base_alert.send_dm = Mock(return_value="ok")
        base_alert.datetime = clock_same
        base_alert.dispatch_dm_for_health_alerts([base_alert.Alert("CRITICAL", "probe", "critical probe")], "offline.test", status_path=b_dang, now_et_date=d_same)
        h_dang = tmp / "branch_dang.json"
        h_dang_t = tmp / "branch_dang.target"
        h_dang.symlink_to(h_dang_t.name)
        branch_alert.send_dm = Mock(return_value="ok")
        branch_alert.datetime = clock_same
        branch_alert.dispatch_dm_for_health_alerts([crit], "offline.test", status_path=h_dang, now_et_date=d_same)
        assert b_dang_t.read_bytes() == h_dang_t.read_bytes()
        special_results["dangling_symlink_writable_target"] = "writes_through_matching_base"

        # 5h. Dangling symlink to missing parent dir
        dang_bad = tmp / "dang_bad.json"
        dang_bad.symlink_to(tmp / "no_such_dir" / "target.json")
        run_modes_on_path(dang_bad, "dangling_missing_parent", writable=False, expect_not_reg_log=False)
        special_results["dangling_symlink_missing_parent"] = "ok"

        # 5i. Symlink to regular file (base and branch parity)
        for mname, c_list, w_list, recip, expected_ret in modes:
            reg_t = tmp / f"reg_{mname}.target"
            reg_t.write_text("[]")
            reg_l = tmp / f"reg_{mname}.json"
            reg_l.symlink_to(reg_t.name)
            branch_alert.send_dm = Mock(return_value="ok")
            ret = branch_alert.dispatch_dm_for_health_alerts(
                c_list, recip, warn_attention=w_list, status_path=reg_l, now_et_date=d_same,
            )
            assert ret is expected_ret
            if mname != "empty":
                assert reg_l.is_symlink()
                assert json.loads(reg_t.read_bytes())["status"] == (
                    "sent" if expected_ret else "skipped_no_recipient"
                )
        special_results["symlink_to_regular"] = "ok"

        # 5j. Real denied read via chmod 0o000 (unreadable & unwritable) and chmod 0o200 (write-only!)
        assert os.geteuid() != 0, "Expected non-root user for chmod permission checks"
        perm_000 = tmp / "perm000.json"
        perm_000.write_text(json.dumps({"status": "failed", "critical_count": 2}))
        perm_000.chmod(0o000)
        try:
            with contextlib.suppress(Exception):
                perm_000.read_bytes()
                raise AssertionError("chmod 0o000 did not block read")
            run_modes_on_path(perm_000, "real_chmod_000", writable=False, expect_not_reg_log=False)
        finally:
            perm_000.chmod(0o600)
        special_results["real_denied_read_000"] = "ok"

        perm_200 = tmp / "perm200.json"
        perm_200.write_text(json.dumps({"status": "sent", "sent_day": d_same.isoformat(), "sent_sources": [key]}))
        perm_200.chmod(0o200)
        try:
            try:
                perm_200.read_text()
                raise AssertionError("chmod 0o200 did not block read")
            except PermissionError:
                pass
            branch_alert.send_dm = Mock(return_value="ok")
            assert branch_alert.dispatch_dm_for_health_alerts(
                [crit], "offline.test", status_path=perm_200, now_et_date=d_same,
            ) is True
            perm_200.chmod(0o600)
            assert json.loads(perm_200.read_bytes())["status"] == "sent"
        finally:
            perm_200.chmod(0o600)
        special_results["real_denied_read_200_writable"] = "ok_persisted_sent"

        # 5k. Parent is a regular file
        parent_file = tmp / "parent_is_a_file"
        parent_file.write_text("not a dir")
        child_under_file = parent_file / "status.json"
        run_modes_on_path(child_under_file, "parent_is_file", writable=False, expect_not_reg_log=False)
        special_results["parent_is_file"] = "ok"

        # 5l. Path changes type between reader's stat() and read_text()
        real_stat = Path.stat
        for swap_kind in ("to_dir", "to_missing", "to_loop"):
            swap_p = tmp / f"swap_{swap_kind}.json"
            swap_p.write_text('{"status":"sent","sent_day":"2027-07-14","sent_sources":["CRITICAL:probe"]}')
            swapped = False

            def stat_then_swap(self, *a, **kw):
                nonlocal swapped
                res = real_stat(self, *a, **kw)
                if self == swap_p and not swapped:
                    swapped = True
                    swap_p.unlink()
                    if swap_kind == "to_dir":
                        swap_p.mkdir()
                    elif swap_kind == "to_loop":
                        swap_p.symlink_to(swap_p.name)
                return res

            Path.stat = stat_then_swap
            try:
                branch_alert.send_dm = Mock(return_value="ok")
                with capture_logs() as logs, bounded(2.0, f"swap:{swap_kind}"):
                    ret = branch_alert.dispatch_dm_for_health_alerts(
                        [crit], "offline.test", status_path=swap_p, now_et_date=d_same,
                    )
                assert swapped and ret is True and branch_alert.send_dm.call_count == 1
                if swap_kind == "to_missing":
                    assert json.loads(swap_p.read_bytes())["status"] == "sent"
                else:
                    assert any("could not write health DM delivery status" in m for _, m, _ in logs)
            finally:
                Path.stat = real_stat
            special_results[f"race_stat_then_{swap_kind}"] = "ok"

        # 5m. Subprocess test of TOCTOU swap: regular -> writerless FIFO between stat() and read_text()
        fifo_race_code = f"""
import os, signal, sys
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0, {str(REPO_ROOT / 'src')!r})
from bts.health import alert
alert.send_dm = Mock(return_value='ok')
p = Path({str(tmp / 'race_fifo.json')!r})
p.write_text('{{}}')
real_stat = Path.stat
done = False
def swap_fifo(self, *a, **kw):
    global done
    res = real_stat(self, *a, **kw)
    if self == p and not done:
        done = True
        p.unlink()
        os.mkfifo(p)
    return res
Path.stat = swap_fifo
def on_alarm(*_):
    print('BLOCKED_ON_SWAP_FIFO')
    sys.exit(42)
signal.signal(signal.SIGALRM, on_alarm)
signal.setitimer(signal.ITIMER_REAL, 0.5)
alert.dispatch_dm_for_health_alerts([alert.Alert('CRITICAL', 'x', 'm')], 'me', status_path=p)
print('DID_NOT_BLOCK')
"""
        proc = subprocess.run(
            [sys.executable, "-c", fifo_race_code],
            capture_output=True, text=True, timeout=5,
        )
        special_results["race_stat_then_fifo"] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
        }

        # 5n. Subprocess test of 200,000-deep nesting (ensure clean process without C stack crash)
        p200k = tmp / "sub200k.json"
        p200k.write_bytes(b"[" * 200_000 + b"0" + b"]" * 200_000)
        sub200k_code = f"""
import sys
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0, {str(REPO_ROOT / 'src')!r})
from bts.health import alert
alert.send_dm = Mock(return_value='ok')
ret = alert.dispatch_dm_for_health_alerts(
    [alert.Alert('CRITICAL', 'x', 'm')], 'me', status_path=Path({str(p200k)!r})
)
assert ret is True and alert.send_dm.call_count == 1
print('SUB200K_OK')
"""
        proc200k = subprocess.run(
            [sys.executable, "-c", sub200k_code],
            capture_output=True, text=True, timeout=10,
        )
        assert proc200k.returncode == 0 and proc200k.stdout.strip() == "SUB200K_OK", proc200k.stderr
        special_results["subprocess_200k_nesting"] = "SUB200K_OK"

    return {
        "malformed_content_runs": content_runs,
        "special_results": special_results,
    }


def main() -> None:
    row5 = check_row5_identity()
    parity = check_conforming_parity()
    malformed = check_malformed_and_special_paths()
    summary = {
        "row5": row5,
        "conforming_parity": parity,
        "malformed_and_special": malformed,
    }
    out_path = REVIEW_DIR / "probe_dispatcher_and_parity.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
