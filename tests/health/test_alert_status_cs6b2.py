"""Offline coverage for malformed health DM status snapshots."""

import io
import json
import os
import signal
import subprocess
import sys
import tarfile
from datetime import date, datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

from bts.health import alert


DAY = date(2027, 7, 14)
KEY = "CRITICAL:probe"
CRITICAL = alert.Alert("CRITICAL", "probe", "critical probe")
WARN = alert.Alert("WARN", "probe", "warning probe")
BANNER = "⚠️ Previous health alert DM FAILED to send (1 CRITICAL) — it may have been missed."


@pytest.fixture(params=["health", "contest_stale_reading", "contest_morning_check"])
def status_path(tmp_path, request):
    return tmp_path / f"{request.param}_dm_delivery_status.json"


@pytest.fixture
def transport(monkeypatch):
    send = Mock()
    monkeypatch.setattr(alert, "send_dm", send)
    return send


def dispatch(path, critical=(), warn=(), recipient="offline.test", day=DAY):
    return alert.dispatch_dm_for_health_alerts(
        list(critical), recipient, warn_attention=list(warn),
        status_path=path, now_et_date=day,
    )


def saved(path):
    return json.loads(path.read_bytes())


@pytest.fixture
def bounded_fifo():
    def timeout(*_):
        pytest.fail("dispatcher blocked on a FIFO")

    prior_handler = signal.signal(signal.SIGALRM, timeout)
    prior_timer = signal.setitimer(signal.ITIMER_REAL, 2)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        signal.setitimer(signal.ITIMER_REAL, *prior_timer)


@pytest.mark.parametrize("kind", [
    "array", "null", "string", "number", "bool", "nested",
    "green_json", "green_utf8", "green_permission", "green_directory",
    "fifo", "dangling", "loop", "green_regular_symlink", "green_missing",
])
@pytest.mark.parametrize("mode", ["critical", "warn", "empty", "no_recipient"])
def test_status_shapes(kind, mode, status_path, transport, monkeypatch, request):
    content = {
        "array": b"[]", "null": b"null", "string": b'"str"', "number": b"42",
        "bool": b"true", "nested": b"[" * 20_000 + b"0" + b"]" * 20_000,
        "green_json": b"{invalid", "green_utf8": b"\xff",
        "green_permission": b'{"status":"failed","critical_count":1}',
    }
    target = status_path.with_suffix(".target")
    if kind in content:
        status_path.write_bytes(content[kind])
    elif kind == "green_directory":
        status_path.mkdir()
    elif kind == "fifo":
        os.mkfifo(status_path)
        request.getfixturevalue("bounded_fifo")
    elif kind == "loop":
        status_path.symlink_to(status_path.name)
    elif kind in ("dangling", "green_regular_symlink"):
        if kind == "green_regular_symlink":
            target.write_text("{}")
        status_path.symlink_to(target.name)

    if kind == "green_permission":
        real_open = Path.open

        def denied_read(path, mode="r", *args, **kwargs):
            if path == status_path and mode.startswith("r"):
                raise PermissionError("injected status read denial")
            return real_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", denied_read)
    if kind == "nested":
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(2000)
    try:
        result = dispatch(
            status_path,
            critical=[CRITICAL] if mode in ("critical", "no_recipient") else [],
            warn=[WARN] if mode == "warn" else [],
            recipient=None if mode == "no_recipient" else "offline.test",
        )
    finally:
        if kind == "nested":
            sys.setrecursionlimit(old_limit)
    attempted = mode in ("critical", "warn")
    assert transport.call_count == int(attempted)
    assert result is attempted
    if attempted:
        assert transport.call_args.args[1] == alert.format_health_dm_body(
            [CRITICAL] if mode == "critical" else [], [WARN] if mode == "warn" else [],
        )
    if kind == "green_directory":
        assert status_path.is_dir()
    elif kind == "fifo":
        assert status_path.exists() and not status_path.is_file()
    elif kind == "loop":
        assert status_path.readlink() == Path(status_path.name)
    elif mode != "empty":
        with monkeypatch.context() as read:
            if kind == "green_permission":
                read.setattr(Path, "open", real_open)
            assert saved(status_path)["status"] == ("sent" if attempted else "skipped_no_recipient")
    elif kind in content:
        if kind == "green_permission":
            with real_open(status_path, "rb") as file:
                assert file.read() == content[kind]
        else:
            assert status_path.read_bytes() == content[kind]
    elif kind == "green_regular_symlink":
        assert target.read_text() == "{}"
    else:
        assert not status_path.exists()


@pytest.mark.parametrize("count", [1, 0, -1, None, "1", 1.5, True])
@pytest.mark.parametrize("critical", [[], [CRITICAL]], ids=["banner_only", "with_alert"])
def test_failed_count(count, critical, status_path, transport):
    original = {"status": "failed", "critical_count": count}
    status_path.write_text(json.dumps(original))
    banner = type(count) is int and count > 0
    body = alert.format_health_dm_body(critical)
    expected = BANNER + ("\n\n" + body if body else "") if banner else body
    result = dispatch(status_path, critical)
    assert result is (expected is not None)
    assert transport.call_count == int(expected is not None)
    if expected is None:
        assert saved(status_path) == original
    else:
        assert transport.call_args.args[1] == expected
        assert saved(status_path)["status"] == "sent"
        assert saved(status_path)["critical_count"] == len(critical)


@pytest.mark.parametrize("sources", [
    [1, {}], [1], [KEY, 1, {}], ["CRITICAL:other", {}, 1], KEY, {KEY: True}, None, [],
])
@pytest.mark.parametrize("prior_status,day", [
    ("sent", DAY), ("sent", date(2027, 7, 13)), ("skipped_no_recipient", DAY),
])
def test_prior_sources(sources, prior_status, day, status_path, transport):
    original = {"status": prior_status, "sent_day": day.isoformat(), "sent_sources": sources}
    status_path.write_text(json.dumps(original))
    keys = (
        {key for key in sources if isinstance(key, str)}
        if isinstance(sources, list) and prior_status == "sent" and day == DAY else set()
    )
    result = dispatch(status_path, [CRITICAL])
    assert result is (KEY not in keys)
    assert transport.call_count == int(KEY not in keys)
    if KEY in keys:
        assert saved(status_path) == original
    else:
        assert saved(status_path)["status"] == "sent"
        assert saved(status_path)["sent_sources"] == sorted(keys | {KEY})


@pytest.mark.parametrize("persistent", [False, True])
def test_dedup_after_corruption(persistent, status_path, transport, caplog):
    if persistent:
        status_path.mkdir()
    else:
        status_path.write_text("[]")
    assert dispatch(status_path, [CRITICAL]) is True
    assert dispatch(status_path, [CRITICAL]) is persistent
    assert transport.call_count == (2 if persistent else 1)
    if persistent:
        assert caplog.text.count("could not write health DM delivery status") == 2
    else:
        assert saved(status_path)["status"] == "sent"
        assert dispatch(status_path, [CRITICAL], day=date(2027, 7, 15)) is True
        assert transport.call_count == 2


@pytest.fixture(scope="module")
def base_alert():
    source = "src/bts/health/alert.py"
    archive = subprocess.check_output(["git", "archive", "59efa825", source])
    with tarfile.open(fileobj=io.BytesIO(archive)) as base:
        code = compile(base.extractfile(source).read(), source, "exec")
    module = ModuleType("_cs6b2_base_alert")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, module.__name__, module)
        exec(code, module.__dict__)
    return module


@pytest.mark.parametrize("prior", [
    "sent_one", "sent_many", "failed_critical", "failed_warn", "skipped", "legacy",
])
@pytest.mark.parametrize("mode", ["covered", "new", "warn", "empty", "no_recipient", "failure"])
def test_green_writer_parity(base_alert, prior, mode, tmp_path, monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2027, 7, 14, 12, tzinfo=tz)

    path = tmp_path / "status.json"
    monkeypatch.setattr(base_alert, "datetime", Clock)
    monkeypatch.setattr(alert, "datetime", Clock)
    if prior == "legacy":
        path.write_text('{"schema_version":"bts_health_dm_delivery_status_v1","status":"sent"}')
    else:
        status = prior.split("_")[0]
        count = 0 if prior == "failed_warn" else 1
        keys = {
            "sent_one": [KEY],
            "sent_many": [KEY, "WARN_ATTENTION:probe"],
        }.get(prior)
        base_alert._write_health_dm_delivery_status(
            path, status="skipped_no_recipient" if status == "skipped" else status,
            dm_recipient=None if status == "skipped" else "offline.test",
            critical_count=count, warn_attention_count=int(prior in ("failed_warn", "sent_many")),
            body="prior DM", error="prior error" if status != "sent" else None,
            sent_day=DAY.isoformat() if keys else None, sent_sources=keys,
        )
    original = path.read_bytes()
    critical = {
        "covered": [CRITICAL], "new": [alert.Alert("CRITICAL", "new", "new probe")],
        "no_recipient": [CRITICAL], "failure": [alert.Alert("CRITICAL", "new", "new probe")],
    }.get(mode, [])

    def run(module):
        path.write_bytes(original)
        send = Mock(side_effect=RuntimeError("offline transport failure") if mode == "failure" else None)
        monkeypatch.setattr(module, "send_dm", send)
        result = module.dispatch_dm_for_health_alerts(
            critical, None if mode == "no_recipient" else "offline.test",
            warn_attention=[WARN] if mode == "warn" else [], status_path=path, now_et_date=DAY,
        )
        return result, send.call_args_list, path.read_bytes()

    assert run(alert) == run(base_alert)


def test_one_status_snapshot(tmp_path, transport, monkeypatch):
    path = tmp_path / "status.json"
    path.write_text("{}")
    original = Path.open
    reads = []

    def count_reads(self, mode="r", *args, **kwargs):
        if self == path and mode == "r":
            reads.append(self)
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", count_reads)
    assert dispatch(path, [CRITICAL]) is True
    assert reads == [path]
    transport.assert_called_once()
