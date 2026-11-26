"""Run the archived base and review branch with captured alert transport."""

import ast
import itertools
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from zoneinfo import ZoneInfo

from bts.health import alert


HERE = Path(__file__).resolve().parent
SOURCE = Path("src/bts/health/alert.py")
ROOT = HERE.parents[1]
DAY = date(2027, 7, 14)
NEXT = date(2027, 7, 15)
ET = ZoneInfo("America/New_York")
CRIT = alert.Alert("CRITICAL", "alpha", "first critical")
OTHER = alert.Alert("CRITICAL", "other", "another critical", "by-incident")
WARN = alert.Alert("WARN", "omega", "warning")
KEY = "CRITICAL:alpha"


class Timeout(BaseException):
    pass


@contextmanager
def bounded(seconds=3):
    def expire(*_):
        raise Timeout(f"blocked for {seconds} seconds")

    handler = signal.signal(signal.SIGALRM, expire)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


def archived_base(folder):
    with subprocess.Popen(["git", "archive", "59efa825", "src"], cwd=ROOT, stdout=subprocess.PIPE) as git:
        with tarfile.open(fileobj=git.stdout, mode="r|") as archive:
            archive.extractall(folder, filter="data")
        assert git.wait() == 0
    module = ModuleType("cs6b2_archived_base_alert")
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile((folder / SOURCE).read_bytes(), str(folder / SOURCE), "exec"), module.__dict__)
    return module


def run(module, path, critical=(), warn=(), *, day=DAY, explicit=True, recipient=True, fail=False):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            at = cls.combine(day, datetime.min.time(), ET).replace(hour=23, minute=59)
            return at.astimezone(tz) if tz else at.replace(tzinfo=None)

    lines, calls = [], []

    class Capture(logging.Handler):
        def emit(self, record):
            lines.append((record.levelname, record.getMessage(),
                          record.exc_info[0].__name__ if record.exc_info else None))

    def send(to, body):
        calls.append((to, body))
        if fail:
            raise RuntimeError("captured DM transport failure")

    logger, capture = module.log, Capture()
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(capture)
    try:
        with patch.object(module, "send_dm", send), patch.object(module, "datetime", Clock), bounded():
            result = module.dispatch_dm_for_health_alerts(
                list(critical), "offline.test" if recipient else None, list(warn),
                status_path=path, now_et_date=day if explicit else None,
            )
    finally:
        logger.removeHandler(capture)
        logger.setLevel(old_level)
    return result, calls, lines


def conforming(base, path):
    fixtures = {}
    for name, critical, warn, recipient, fail in [
        ("sent-one", [CRIT], [], True, False),
        ("sent-many", [CRIT, OTHER], [WARN], True, False),
        ("sent-warn", [], [WARN], True, False),
        ("failed-critical", [CRIT], [], True, True),
        ("failed-no-criticals", [], [WARN], True, True),
        ("skipped", [CRIT], [WARN], False, False),
    ]:
        path.unlink(missing_ok=True)
        run(base, path, critical, warn, recipient=recipient, fail=fail)
        fixtures[name] = path.read_bytes()
    path.write_bytes(fixtures["failed-critical"])
    run(base, path)
    fixtures["sent-banner-only"] = path.read_bytes()
    legacy = json.loads(fixtures["sent-many"])
    del legacy["sent_day"], legacy["sent_sources"]
    fixtures["legacy"] = json.dumps(legacy).encode()
    fixtures["missing"] = None
    counts = {}
    for name, initial in fixtures.items():
        for (critical, warn), day, explicit, supplied, recipient, fail in itertools.product(
            [([], []), ([CRIT], []), ([OTHER], []), ([CRIT, OTHER], []),
             ([], [WARN]), ([CRIT], [WARN]), ([CRIT, OTHER], [WARN])],
            [DAY, NEXT], [False, True], [False, True], [False, True], [False, True],
        ):
            results = []
            for module in (base, alert):
                path.unlink(missing_ok=True)
                if initial is not None:
                    path.write_bytes(initial)
                outcome = run(module, path if supplied else None, critical, warn, day=day,
                              explicit=explicit, recipient=recipient, fail=fail)
                results.append((outcome, path.read_bytes() if path.exists() else None))
            assert results[0] == results[1], (name, critical, warn, day, explicit, supplied, recipient, fail, results)
            counts[name] = counts.get(name, 0) + 1
    print(f"Conforming parity (return, attempts/body, log level/text/exception class, exact bytes): {counts}", flush=True)


def change_scope(folder):
    def definitions(source):
        lines = source.splitlines(keepends=True)
        return {node.name: "".join(lines[node.lineno - 1:node.end_lineno])
                for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}

    paths = subprocess.check_output(["git", "diff", "--name-only", "59efa825", "HEAD", "--", "src"], cwd=ROOT)
    assert paths.decode().splitlines() == [str(SOURCE)]
    left = definitions((folder / SOURCE).read_text())
    right = definitions((ROOT / SOURCE).read_text())
    changed = {name for name in left if left[name] != right[name]}
    assert changed == {"_resurface_note", "dispatch_dm_for_health_alerts"}
    print(f"Existing top-level definitions changed: {sorted(changed)}; added: {sorted(right.keys() - left.keys())}")


MODES = [("critical", [CRIT], True), ("empty", [], True), ("no-recipient", [CRIT], False)]


def stable(path, initial, label, *, writable=True, expect_warning=False):
    for name, critical, recipient in MODES:
        with bounded(), initial() as actual:
            absent = run(alert, path / "absent.json", critical, recipient=recipient)
            (path / "absent.json").unlink(missing_ok=True)
            result = run(alert, actual, critical, recipient=recipient)
            assert result[:2] == absent[:2], (label, name, result, absent)
            if critical and writable:
                payload = json.loads(actual.read_bytes())
                assert payload["status"] == ("sent" if recipient else "skipped_no_recipient"), (label, name, payload)
                assert payload["updated_at"] == "2027-07-15T03:59:00+00:00", (label, name, payload)
            warnings = [entry for entry in result[2] if "could not write health DM delivery status" in entry[1]]
            assert len(warnings) == int(bool(critical) and expect_warning), (label, name, result)
    print(f"Stable {label}: all three missing-file comparisons passed", flush=True)


def malformed(folder):
    raw = {"array": b"[]", "null": b"null", "string": b'"text"', "number": b"42", "true": b"true",
           "nested-20000": b"[" * 20_000 + b"0" + b"]" * 20_000,
           "nested-200000": b"[" * 200_000 + b"0" + b"]" * 200_000,
           "syntax": b"{invalid", "UTF-8": b"\xff", "empty": b"",
           "50-MB": b'"' + b"x" * (50_000_000 - 2) + b'"',
           "every-field": json.dumps(dict(schema_version=[], status={}, updated_at=[], recipient_configured=[],
                                          recipient={}, critical_count=[], warn_attention_count=[], body_sha256=[],
                                          body_first_line=[], error={}, sent_day=[], sent_sources={})).encode(),
           "unhashable-status": b'{"status":[]}',
           "non-string-day": b'{"status":"sent","sent_day":{},"sent_sources":["CRITICAL:alpha"]}',
           "duplicate-last-failed-invalid": b'{"status":"sent","status":"failed","critical_count":1,"critical_count":false}',
           "over-integer-conversion-cap": b'{"status":"failed","critical_count":' + b"9" * 5_000 + b"}"}
    for value in (KEY, {KEY: 1}, [1, {}], [1], [True, "valid", {}], [f"k{i}" for i in range(100_000)]):
        label = f"sources-{str(value)[:40]}" if len(value) < 100_000 else "sources-100000-strings"
        raw[label] = json.dumps(dict(status="sent", sent_day=DAY.isoformat(), sent_sources=value)).encode()
    for value in (0, -1, None, "1", 1.5, True):
        raw[f"count-{value!r}"] = json.dumps(dict(status="failed", critical_count=value)).encode()

    for label, content in raw.items():
        @contextmanager
        def fresh():
            path = folder / "subject.json"
            path.write_bytes(content)
            try:
                yield path
                payload = json.loads(path.read_bytes()) if label == "sources-100000-strings" else {}
                if payload.get("updated_at") and payload.get("status") == "sent":
                    keys = payload["sent_sources"]
                    assert keys == sorted([KEY, *(f"k{i}" for i in range(100_000))])
            finally:
                path.unlink()

        stable(folder, fresh, label)

    for label, payload, suppress, prior_keys, banner in [
        ("mixed covering", dict(status="sent", sent_day=DAY.isoformat(), sent_sources=[KEY, [], 5]), True, [], None),
        ("mixed noncovering", dict(status="sent", sent_day=DAY.isoformat(), sent_sources=["prior", {}, "prior", "\ud800"]), False, ["prior", "\ud800"], None),
        ("stale", dict(status="sent", sent_day=NEXT.isoformat(), sent_sources=[KEY, "prior"]), False, [], None),
        ("not sent", dict(status="other", sent_day=DAY.isoformat(), sent_sources=[KEY, "prior"]), False, [], None),
        ("positive", dict(status="failed", critical_count=1), False, [], "1"),
        ("huge exact int", dict(status="failed", critical_count=int("9" * 4300)), False, [], "9" * 4300),
        ("duplicate last accepted", '{"status":"junk","status":"sent","sent_day":"2027-07-14","sent_sources":[],"sent_sources":["CRITICAL:alpha"]}', True, [], None),
    ]:
        for mode, critical, recipient in MODES:
            path = folder / "subject.json"
            initial = payload if isinstance(payload, str) else json.dumps(payload)
            path.write_text(initial)
            result, calls, lines = run(alert, path, critical, recipient=recipient)
            expected = not suppress and recipient and (bool(critical) or banner is not None)
            assert result is expected and len(calls) == int(expected), (label, mode)
            if calls:
                body = alert.format_health_dm_body(critical) or ""
                if banner is not None:
                    body = f"⚠️ Previous health alert DM FAILED to send ({banner} CRITICAL) — it may have been missed." + ("\n\n" + body if body else "")
                assert calls == [("offline.test", body)], (label, mode)
                data = json.loads(path.read_bytes())
                assert data["status"] == "sent" and data["sent_sources"] == sorted(prior_keys + ([KEY] if critical else []))
            elif (suppress or not critical) and banner is None:
                assert path.read_text() == initial
            else:
                assert json.loads(path.read_bytes())["status"] == "skipped_no_recipient"
            path.unlink()
        print(f"Row 2/3 {label}: passed", flush=True)


def special_paths(base, folder):
    @contextmanager
    def special(kind):
        with tempfile.TemporaryDirectory(prefix="obj-", dir=folder) as temp:
            local = Path(temp)
            path = local / "status"
            target = local / "target"
            fd = sock = None
            if kind == "directory":
                path.mkdir()
            elif kind in ("fifo", "fifo-with-writer"):
                os.mkfifo(path)
                if kind == "fifo-with-writer":
                    reader = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                    fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
                    os.close(reader)
            elif kind == "socket":
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.bind(os.path.relpath(path))
            elif kind == "device":
                path.symlink_to("/dev/null")
            elif kind == "loop":
                path.symlink_to(path.name)
            elif kind in ("dangling", "regular-link"):
                if kind == "regular-link":
                    target.write_bytes(b"{}")
                path.symlink_to(target.name)
            elif kind == "parent-file":
                path.write_text("parent")
                path = path / "child"
            elif kind == "denied":
                path.write_bytes(b'{"status":"failed","critical_count":1}')
                path.chmod(0)
                try:
                    path.read_bytes()
                except PermissionError:
                    pass
                else:
                    raise AssertionError("chmod did not deny the actual read")
            elif kind == "missing-parent":
                path = path / "child"
            try:
                yield path
            finally:
                if fd is not None:
                    os.close(fd)
                if sock is not None:
                    sock.close()

    for kind in ("directory", "fifo", "fifo-with-writer", "socket", "device", "loop", "dangling",
                 "regular-link", "parent-file", "denied", "missing-parent"):
        writable = kind in ("dangling", "regular-link", "missing-parent")
        stable(folder, lambda: special(kind), kind, writable=writable, expect_warning=not writable)
        if writable:
            samples = []
            for module in (base, alert):
                with special(kind) as path:
                    outcome = run(module, path, [CRIT])
                    samples.append((outcome, path.read_bytes(), path.is_symlink()))
            assert samples[0] == samples[1], (kind, samples)


def races(folder):
    original = Path.stat
    for stage, stat_number in [("before read open", 1), ("before write open", 2)]:
        for name, critical, recipient in MODES:
            with tempfile.TemporaryDirectory(dir=folder) as temp:
                path = Path(temp) / "status"
                replacement = Path(temp) / "replacement"
                path.write_text("{}")
                os.mkfifo(replacement)
                stats, calls, timed_out = 0, [], False

                def swap(self, *args, **kwargs):
                    nonlocal stats
                    result = original(self, *args, **kwargs)
                    if self == path:
                        stats += 1
                        if stats == stat_number:
                            os.replace(replacement, path)
                    return result

                try:
                    with patch.object(Path, "stat", swap), patch.object(alert, "send_dm", side_effect=lambda *a: calls.append(a)), bounded(1):
                        result = alert.dispatch_dm_for_health_alerts(
                            critical, "offline.test" if recipient else None, status_path=path, now_et_date=DAY,
                        )
                except Timeout:
                    timed_out = True
                    result = None
                print(f"Race {stage} {name}: timeout={timed_out} sends={len(calls)} result={result} stats={stats}", flush=True)
                assert timed_out is (stat_number == 1 or bool(critical)), (stage, name, result)


def main():
    assert os.environ.get("CS6B2_OFFLINE") == "1", "launch with review sitecustomize on PYTHONPATH"
    # Keep recursion and integer conversion limits explicit for reproducible acceptance tests.
    sys.setrecursionlimit(2_000)
    sys.set_int_max_str_digits(4300)
    with tempfile.TemporaryDirectory(prefix="dispatch-", dir=HERE) as root:
        folder = Path(root)
        base = archived_base(folder)
        change_scope(folder)
        conforming(base, folder / "status.json")
        malformed(folder)
        races(folder)
        # Darwin Unix socket pathnames have a short limit. Bind through a relative pathname.
        previous = Path.cwd()
        os.chdir(folder)
        try:
            special_paths(base, Path("."))
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    main()
