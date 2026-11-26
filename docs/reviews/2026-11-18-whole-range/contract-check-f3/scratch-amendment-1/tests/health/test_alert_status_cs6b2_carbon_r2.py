"""Round-2 supplemental regression tests for CS6b-2:
- Symlink-to-FIFO replacement race on both read and write (no peer, writer with bytes, active reader).
- Missing-to-FIFO race on write (path absent at `_persist_health_dm_delivery_status`'s `stat()`,
  FIFO placed before `_write_regular_file_text`'s `os.open()`).
- Write `os.open` omits `O_TRUNC` (truncation happens only via `ftruncate` after `fstat` confirms `S_ISREG`).
- Descriptor leak check when exceptions occur *inside* `TextIOWrapper` (`UnicodeDecodeError` on read,
  `ENOSPC` on write) after `os.open` and `os.fstat` have succeeded.
- Existing regular file preserves `st_ino` and custom mode (`0o600`), creation honors `umask`,
  universal-newline statuses (`\\r\\n`, `\\r`) dedup, and non-ASCII DM first lines persist ASCII-safe JSON.
"""

from __future__ import annotations

import builtins
import contextlib
import errno
import json
import logging
import os
import signal
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pytest

from bts.health import alert

DAY = date(2027, 7, 14)
CRIT = alert.Alert("CRITICAL", "probe", "critical — 東京")
KEY = "CRITICAL:probe"


@contextlib.contextmanager
def bounded(seconds: float = 2.0):
    def _on_timeout(*_):
        pytest.fail(f"dispatcher blocked >= {seconds}s")

    prior_handler = signal.signal(signal.SIGALRM, _on_timeout)
    prior_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        signal.setitimer(signal.ITIMER_REAL, *prior_timer)


@pytest.mark.parametrize("phase", ["read", "write"])
@pytest.mark.parametrize("peer", ["none", "reader", "writer_bytes"])
def test_symlink_to_fifo_swapped_after_stat_cannot_block_or_transfer_bytes(
    tmp_path, monkeypatch, caplog, phase, peer,
):
    status_path = tmp_path / "health_dm_delivery_status.json"
    covering = (json.dumps({"status": "sent", "sent_day": DAY.isoformat(), "sent_sources": [KEY]}) + "\n").encode()
    status_path.write_bytes(covering if phase == "read" else b"{}")

    fifo = tmp_path / "backing.fifo"
    os.mkfifo(fifo)
    staged_link = tmp_path / "staged.link"
    staged_link.symlink_to(fifo)

    fds: list[int] = []
    if peer == "reader":
        fds.append(os.open(fifo, os.O_RDONLY | os.O_NONBLOCK))
    elif peer == "writer_bytes":
        r_fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        w_fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        os.write(w_fd, covering)
        fds.extend([r_fd, w_fd])

    real_stat = Path.stat
    target_hit = 1 if phase == "read" else 2
    hits = 0

    def _stat_then_swap_symlink(self: Path, *a, **kw):
        nonlocal hits
        st = real_stat(self, *a, **kw)
        if self == status_path:
            hits += 1
            if hits == target_hit:
                os.replace(staged_link, status_path)
        return st

    send = Mock(return_value="ok")
    monkeypatch.setattr(alert, "send_dm", send)
    monkeypatch.setattr(Path, "stat", _stat_then_swap_symlink)
    try:
        with caplog.at_level(logging.WARNING, logger="bts.health.alert"), bounded(2.0):
            res = alert.dispatch_dm_for_health_alerts(
                [CRIT], "offline.test", status_path=status_path, now_et_date=DAY,
            )
        assert hits >= target_hit
        assert res is True
        send.assert_called_once()
        assert "could not write health DM delivery status" in caplog.text
        if peer == "writer_bytes":
            assert os.read(fds[0], 4096) == covering
        elif peer == "reader":
            assert os.read(fds[0], 4096) == b""
    finally:
        for fd in fds:
            os.close(fd)


@pytest.mark.parametrize("with_reader", [False, True])
def test_missing_path_turning_into_fifo_before_writer_open_cannot_block(
    tmp_path, monkeypatch, caplog, with_reader,
):
    status_path = tmp_path / "sub" / "health_dm_delivery_status.json"
    fifo = tmp_path / "prepared.fifo"
    os.mkfifo(fifo)
    reader_fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK) if with_reader else None

    real_mkdir = Path.mkdir

    def _mkdir_then_place_fifo(self: Path, *a, **kw):
        res = real_mkdir(self, *a, **kw)
        if self == status_path.parent and not status_path.exists():
            os.replace(fifo, status_path)
        return res

    send = Mock(return_value="ok")
    monkeypatch.setattr(alert, "send_dm", send)
    monkeypatch.setattr(Path, "mkdir", _mkdir_then_place_fifo)
    try:
        with caplog.at_level(logging.WARNING, logger="bts.health.alert"), bounded(2.0):
            assert alert.dispatch_dm_for_health_alerts(
                [CRIT], "offline.test", status_path=status_path, now_et_date=DAY,
            ) is True
        send.assert_called_once()
        assert "could not write health DM delivery status" in caplog.text
        if reader_fd is not None:
            assert os.read(reader_fd, 4096) == b""
    finally:
        if reader_fd is not None:
            os.close(reader_fd)


def test_writer_open_omits_o_trunc_so_type_check_strictly_precedes_truncation(tmp_path, monkeypatch):
    status_path = tmp_path / "status.json"
    seen_write_flags: list[int] = []
    real_os_open = os.open

    def _spy_os_open(path, flags, *a, **kw):
        if Path(path) == status_path and (flags & (os.O_WRONLY | os.O_RDWR)):
            seen_write_flags.append(flags)
        return real_os_open(path, flags, *a, **kw)

    monkeypatch.setattr(alert, "send_dm", Mock(return_value="ok"))
    monkeypatch.setattr(os, "open", _spy_os_open)
    assert alert.dispatch_dm_for_health_alerts(
        [CRIT], "offline.test", status_path=status_path, now_et_date=DAY,
    ) is True
    assert len(seen_write_flags) == 1
    assert bool(seen_write_flags[0] & os.O_NONBLOCK) is True
    assert (seen_write_flags[0] & os.O_TRUNC) == 0


def test_no_descriptor_leaks_when_stream_read_or_write_raises(tmp_path, monkeypatch):
    bad_utf8 = tmp_path / "bad_utf8.json"
    bad_utf8.write_bytes(b"\xff\xfe\x80")
    write_target = tmp_path / "write_enospc.json"
    write_target.write_text("{}")

    monkeypatch.setattr(alert, "send_dm", Mock(return_value="ok"))
    orig_open = builtins.open

    class _EnospcStream:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def write(self, _s: str) -> int:
            raise OSError(errno.ENOSPC, "No space left on device")

        def __exit__(self, *exc_info):
            return self._inner.__exit__(*exc_info)

    def _failing_write_open(file, mode="r", *a, **kw):
        h = orig_open(file, mode, *a, **kw)
        if isinstance(file, int) and mode == "w":
            return _EnospcStream(h)
        return h

    # Warm up codecs once before measuring `/dev/fd`
    alert._read_health_dm_delivery_status(bad_utf8)
    before = frozenset(os.listdir("/dev/fd"))
    for _ in range(100):
        assert alert._read_health_dm_delivery_status(bad_utf8) is None
    with monkeypatch.context() as mp:
        mp.setattr(builtins, "open", _failing_write_open)
        for _ in range(100):
            alert._persist_health_dm_delivery_status(
                write_target,
                status="sent",
                dm_recipient="offline.test",
                critical_count=1,
                warn_attention_count=0,
                body="body",
                sent_day=DAY.isoformat(),
                sent_sources=[KEY],
            )
    assert frozenset(os.listdir("/dev/fd")) == before


def test_inplace_inode_mode_umask_and_universal_newlines(tmp_path, monkeypatch):
    monkeypatch.setattr(alert, "send_dm", Mock(return_value="ok"))

    # Umask creation parity
    prior_umask = os.umask(0o077)
    try:
        created = tmp_path / "umask077.json"
        assert alert.dispatch_dm_for_health_alerts(
            [CRIT], "offline.test", status_path=created, now_et_date=DAY,
        ) is True
        assert (created.stat().st_mode & 0o777) == 0o600
    finally:
        os.umask(prior_umask)

    # Existing inode and custom mode preserved across overwrite, non-ASCII body stored as ASCII JSON
    ino_before = created.stat().st_ino
    created.write_text('{"status":"failed","critical_count":1,"pad":"' + ("x" * 4096) + '"}')
    assert alert.dispatch_dm_for_health_alerts(
        [CRIT], "offline.test", status_path=created, now_et_date=DAY,
    ) is True
    assert created.stat().st_ino == ino_before
    assert (created.stat().st_mode & 0o777) == 0o600
    raw = created.read_bytes()
    raw.decode("ascii")  # pure ASCII JSON even with ⚠️, —, and 東京 in the DM body

    # Universal newlines (\r\n and \r) on read still dedup cleanly
    for sep in (b"\r\n", b"\r"):
        nl_path = tmp_path / f"nl_{len(sep)}.json"
        nl_path.write_bytes(
            sep.join([
                b"{",
                b'  "status": "sent",',
                f'  "sent_day": "{DAY.isoformat()}",'.encode("ascii"),
                f'  "sent_sources": ["{KEY}"]'.encode("ascii"),
                b"}",
            ])
        )
        send = Mock(return_value="ok")
        monkeypatch.setattr(alert, "send_dm", send)
        assert alert.dispatch_dm_for_health_alerts(
            [CRIT], "offline.test", status_path=nl_path, now_et_date=DAY,
        ) is False
        send.assert_not_called()
