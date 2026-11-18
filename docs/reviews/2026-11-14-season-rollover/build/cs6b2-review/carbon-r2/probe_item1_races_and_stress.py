"""Round-2 independent probe for Item 1:
1. Reproduce Round-1 BLOCK findings 1 (read race) and 2 (write race) on 38d707bd
   (`git archive 38d707bd src`) and verify both are closed on HEAD (`a50d439f`):
   - Regular file -> FIFO swap between reader's `stat()` and open, across:
     (no writer, writer open holding covering bytes, active writer closing after 30ms,
      reader only) x (CRITICAL, empty list, no recipient).
   - Regular file -> FIFO swap between writer's `stat()` and open, across:
     (no peer, with reader, with writer holding bytes) x (CRITICAL, empty list, no recipient),
     plus missing -> FIFO race before writer open.
2. Stress HEAD (`a50d439f`):
   - Path replaced with FIFO between `os.open` and `os.fstat` (read and write).
   - Path is or becomes a symlink (and 2-hop symlink chain) to a FIFO or `/dev/null`.
   - `O_NONBLOCK` left set on regular file: verify flag on fd, multi-MB reads/writes,
     and exact CPython `io` behavior if `EAGAIN` occurs.
   - Descriptor leak check: exact `/dev/fd` set before and after 10,000 calls of each
     of 14 failure/execution kinds (140,000 calls total).
   - Regular file held open for writing with `flock(LOCK_EX)` + POSIX `fcntl(F_SETLK, F_WRLCK)`
     + macOS `O_EXLOCK` by another process, plus BSD `UF_IMMUTABLE` / `UF_APPEND` flags.
"""

from __future__ import annotations

import builtins
import contextlib
import errno
import fcntl
import io
import json
import logging
import os
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
LOG_PATH = REVIEW_DIR / "probe_item1.log"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# --- Outward network & DNS guard (loopback and AF_UNIX allowed) ---
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_GETADDRINFO = socket.getaddrinfo
_REAL_GETHOSTBYNAME = socket.gethostbyname
_REAL_GETHOSTBYADDR = socket.gethostbyaddr


def _is_allowed_host(host: object) -> bool:
    if host is None:
        return True
    h = host.decode("ascii", "ignore") if isinstance(host, bytes) else str(host)
    if h in _LOOPBACK_HOSTS or h.startswith("127."):
        return True
    try:
        local_names = {socket.gethostname(), socket.getfqdn()}
    except Exception:
        local_names = set()
    return h in local_names


def _guarded_connect(self: socket.socket, address: object) -> object:
    if self.family == socket.AF_UNIX:
        return _REAL_CONNECT(self, address)
    host = address[0] if isinstance(address, tuple) and address else None
    if not _is_allowed_host(host):
        raise RuntimeError(f"forbidden outward socket connect: {address!r}")
    return _REAL_CONNECT(self, address)


def _guarded_connect_ex(self: socket.socket, address: object) -> int:
    if self.family == socket.AF_UNIX:
        return _REAL_CONNECT_EX(self, address)
    host = address[0] if isinstance(address, tuple) and address else None
    if not _is_allowed_host(host):
        raise RuntimeError(f"forbidden outward socket connect_ex: {address!r}")
    return _REAL_CONNECT_EX(self, address)


def _guarded_getaddrinfo(host: object, port: object, *args: object, **kwargs: object) -> object:
    if not _is_allowed_host(host):
        raise RuntimeError(f"forbidden DNS lookup: {host!r}")
    return _REAL_GETADDRINFO(host, port, *args, **kwargs)


def _guarded_gethostbyname(hostname: str) -> str:
    if not _is_allowed_host(hostname):
        raise RuntimeError(f"forbidden gethostbyname: {hostname!r}")
    return _REAL_GETHOSTBYNAME(hostname)


def _guarded_gethostbyaddr(ip: str) -> object:
    if not _is_allowed_host(ip):
        raise RuntimeError(f"forbidden gethostbyaddr: {ip!r}")
    return _REAL_GETHOSTBYADDR(ip)


socket.socket.connect = _guarded_connect  # type: ignore[assignment]
socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[assignment]
socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]
socket.gethostbyname = _guarded_gethostbyname  # type: ignore[assignment]
socket.gethostbyaddr = _guarded_gethostbyaddr  # type: ignore[assignment]


from bts.health import alert as head_alert  # noqa: E402


DAY = date(2027, 7, 14)
KEY = "CRITICAL:probe"


class PinnedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        dt = datetime(2027, 7, 14, 16, 0, 0, tzinfo=timezone.utc)
        return dt.astimezone(tz) if tz is not None else dt.replace(tzinfo=None)


class ProbeBlockedError(BaseException):
    """BaseException so `except Exception` in persistence cannot swallow timeouts."""


@contextlib.contextmanager
def bounded_alarm(seconds: float):
    def _on_alarm(*_):
        raise ProbeBlockedError(f"blocked >= {seconds:.2f}s")

    prior_handler = signal.signal(signal.SIGALRM, _on_alarm)
    prior_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        signal.setitimer(signal.ITIMER_REAL, *prior_timer)


@contextlib.contextmanager
def capture_alert_logs():
    logger = logging.getLogger("bts.health.alert")
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.DEBUG)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def load_archived_alert(rev: str, mod_name: str) -> ModuleType:
    rel = "src/bts/health/alert.py"
    raw = subprocess.check_output(["git", "archive", rev, rel], cwd=REPO_ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        member = tf.extractfile(rel)
        assert member is not None
        source_bytes = member.read()
    code = compile(source_bytes, f"{rev}:{rel}", "exec")
    mod = ModuleType(mod_name)
    mod.__file__ = f"{rev}:{rel}"
    sys.modules[mod_name] = mod
    exec(code, mod.__dict__)
    mod.log = logging.getLogger("bts.health.alert")
    return mod


def run_dispatch_case(
    mod: ModuleType,
    status_path: Path,
    mode: str,
    timeout_s: float,
) -> dict[str, object]:
    crit_alert = mod.Alert("CRITICAL", "probe", "critical probe")
    alerts = [] if mode == "empty" else [crit_alert]
    recipient = None if mode == "no_recipient" else "offline.test"
    send_mock = Mock(return_value="ok")
    orig_send = mod.send_dm
    orig_dt = mod.datetime
    mod.send_dm = send_mock
    mod.datetime = PinnedClock
    t0 = time.monotonic()
    blocked = False
    ret = None
    err: BaseException | None = None
    try:
        with capture_alert_logs() as records:
            try:
                with bounded_alarm(timeout_s):
                    ret = mod.dispatch_dm_for_health_alerts(
                        alerts,
                        recipient,
                        status_path=status_path,
                        now_et_date=DAY,
                    )
            except ProbeBlockedError as exc:
                blocked = True
                err = exc
            except BaseException as exc:
                err = exc
    finally:
        mod.send_dm = orig_send
        mod.datetime = orig_dt
    elapsed_ms = round((time.monotonic() - t0) * 1000.0, 1)
    return {
        "blocked": blocked,
        "ret": ret,
        "err": repr(err) if err is not None else None,
        "dm_calls": send_mock.call_count,
        "warnings": [r.getMessage() for r in records if r.levelno == logging.WARNING],
        "errors": [r.getMessage() for r in records if r.levelno >= logging.ERROR],
        "infos": [r.getMessage() for r in records if r.levelno == logging.INFO],
        "elapsed_ms": elapsed_ms,
    }


def prepare_fifo(
    fifo_path: Path, fifo_state: str, covering_payload: bytes
) -> tuple[list[int], threading.Thread | None]:
    """Create a FIFO at `fifo_path` in the requested state; return (open_fds, optional_thread)."""
    os.mkfifo(fifo_path)
    fds: list[int] = []
    bg_thread: threading.Thread | None = None
    if fifo_state == "no_peer":
        pass
    elif fifo_state == "writer_open_bytes":
        keep_r = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        fds.append(keep_r)
        w_fd = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
        fds.append(w_fd)
        os.write(w_fd, covering_payload)
    elif fifo_state == "writer_eof_bytes":
        keep_r = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        fds.append(keep_r)
        w_fd = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
        os.write(w_fd, covering_payload)

        def _close_later() -> None:
            time.sleep(0.04)
            os.close(w_fd)

        bg_thread = threading.Thread(target=_close_later, daemon=True)
        bg_thread.start()
    elif fifo_state == "reader_only":
        r_fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        fds.append(r_fd)
    else:
        raise ValueError(f"unknown fifo_state: {fifo_state}")
    return fds, bg_thread


def main() -> None:
    lines: list[str] = []

    def emit(msg: str) -> None:
        print(msg)
        lines.append(msg)

    r1_alert = load_archived_alert("38d707bd", "_cs6b2_r1_alert")
    assert not hasattr(r1_alert, "_read_regular_file_text")
    assert hasattr(head_alert, "_read_regular_file_text")
    assert hasattr(head_alert, "_write_regular_file_text")

    covering_bytes = (
        json.dumps({"status": "sent", "sent_day": DAY.isoformat(), "sent_sources": [KEY]}) + "\n"
    ).encode("utf-8")

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_item1_") as tmp_str:
        tmp = Path(tmp_str)

        # =========================================================================
        # PART 1A: Finding 1 — Reader stat() -> open() swap to FIFO
        # =========================================================================
        emit("=== PART 1A: Finding 1 (Reader stat -> open swap to FIFO) ===")
        fifo_states_read = ["no_peer", "writer_open_bytes", "writer_eof_bytes", "reader_only"]
        call_modes = ["critical", "empty", "no_recipient"]

        for fifo_state in fifo_states_read:
            for mode in call_modes:
                for label, mod, t_sec in [("38d707bd", r1_alert, 0.35), ("HEAD", head_alert, 2.0)]:
                    case_dir = tmp / f"r1a_{fifo_state}_{mode}_{label}"
                    case_dir.mkdir()
                    status_path = case_dir / "health_dm_delivery_status.json"
                    status_path.write_bytes(covering_bytes)
                    fifo_path = case_dir / "swap.fifo"
                    fds, bg_thread = prepare_fifo(fifo_path, fifo_state, covering_bytes)

                    real_stat = Path.stat
                    stat_hits = 0

                    def _swap_on_first_stat(self: Path, *a: object, **kw: object):
                        nonlocal stat_hits
                        res = real_stat(self, *a, **kw)
                        if self == status_path:
                            stat_hits += 1
                            if stat_hits == 1:
                                os.replace(fifo_path, status_path)
                        return res

                    Path.stat = _swap_on_first_stat  # type: ignore[method-assign]
                    try:
                        out = run_dispatch_case(mod, status_path, mode, t_sec)
                    finally:
                        Path.stat = real_stat  # type: ignore[method-assign]
                        if bg_thread is not None:
                            bg_thread.join(timeout=1.0)

                    remaining_pipe_bytes = b""
                    if fifo_state in ("writer_open_bytes", "writer_eof_bytes"):
                        try:
                            remaining_pipe_bytes = os.read(fds[0], 8192)
                        except BlockingIOError:
                            remaining_pipe_bytes = b""
                    for fd in fds:
                        os.close(fd)

                    emit(
                        f"READ_RACE state={fifo_state} mode={mode} rev={label}: "
                        f"blocked={out['blocked']} ret={out['ret']} dm={out['dm_calls']} "
                        f"warns={len(out['warnings'])} pipe_left={len(remaining_pipe_bytes)}B "
                        f"ms={out['elapsed_ms']}"
                    )

                    assert stat_hits >= 1
                    if label == "38d707bd":
                        if fifo_state in ("no_peer", "writer_open_bytes", "reader_only"):
                            assert out["blocked"] is True, f"expected 38d707bd block: {out}"
                            assert out["dm_calls"] == 0
                        elif fifo_state == "writer_eof_bytes":
                            # 38d707bd consumed the FIFO's bytes as a valid covering status!
                            assert out["blocked"] is False
                            assert remaining_pipe_bytes == b""
                            if mode == "critical":
                                assert out["ret"] is False and out["dm_calls"] == 0
                    else:
                        assert out["blocked"] is False, f"HEAD blocked! {out}"
                        assert out["err"] is None, f"HEAD raised! {out}"
                        expected_send = mode == "critical"
                        assert out["ret"] is expected_send
                        assert out["dm_calls"] == (1 if expected_send else 0)
                        if fifo_state in ("writer_open_bytes", "writer_eof_bytes"):
                            assert remaining_pipe_bytes == covering_bytes
                        if mode in ("critical", "no_recipient"):
                            assert len(out["warnings"]) == 1
                            assert "not a regular file" in out["warnings"][0]
                        else:
                            assert len(out["warnings"]) == 0

        # =========================================================================
        # PART 1B: Finding 2 — Writer stat() -> open() swap to FIFO (+ missing->FIFO)
        # =========================================================================
        emit("\n=== PART 1B: Finding 2 (Writer stat -> open swap to FIFO) ===")
        fifo_states_write = ["no_peer", "reader_only", "writer_open_bytes"]
        for fifo_state in fifo_states_write:
            for mode in call_modes:
                for label, mod, t_sec in [("38d707bd", r1_alert, 0.35), ("HEAD", head_alert, 2.0)]:
                    case_dir = tmp / f"r1b_{fifo_state}_{mode}_{label}"
                    case_dir.mkdir()
                    status_path = case_dir / "health_dm_delivery_status.json"
                    status_path.write_text("{}")
                    fifo_path = case_dir / "swap_w.fifo"
                    fds, _ = prepare_fifo(fifo_path, fifo_state, covering_bytes)

                    real_stat = Path.stat
                    stat_hits = 0

                    def _swap_on_second_stat(self: Path, *a: object, **kw: object):
                        nonlocal stat_hits
                        res = real_stat(self, *a, **kw)
                        if self == status_path:
                            stat_hits += 1
                            if stat_hits == 2:
                                os.replace(fifo_path, status_path)
                        return res

                    Path.stat = _swap_on_second_stat  # type: ignore[method-assign]
                    try:
                        out = run_dispatch_case(mod, status_path, mode, t_sec)
                    finally:
                        Path.stat = real_stat  # type: ignore[method-assign]

                    pipe_bytes = b""
                    if fifo_state in ("reader_only", "writer_open_bytes"):
                        try:
                            pipe_bytes = os.read(fds[0], 16384)
                        except BlockingIOError:
                            pipe_bytes = b""
                    for fd in fds:
                        os.close(fd)

                    emit(
                        f"WRITE_RACE state={fifo_state} mode={mode} rev={label}: "
                        f"stats={stat_hits} blocked={out['blocked']} ret={out['ret']} "
                        f"dm={out['dm_calls']} warns={len(out['warnings'])} "
                        f"pipe_bytes={len(pipe_bytes)}B ms={out['elapsed_ms']}"
                    )

                    if mode == "empty":
                        assert stat_hits == 1
                        assert out["blocked"] is False and out["ret"] is False and out["dm_calls"] == 0
                        continue

                    assert stat_hits == 2
                    if label == "38d707bd":
                        if fifo_state == "no_peer":
                            assert out["blocked"] is True
                            assert out["dm_calls"] == (1 if mode == "critical" else 0)
                            assert len(out["warnings"]) == 0
                        elif fifo_state == "reader_only":
                            assert out["blocked"] is False
                            assert len(pipe_bytes) > 0 and b"bts_health_dm_delivery_status_v1" in pipe_bytes
                            assert len(out["warnings"]) == 0
                        elif fifo_state == "writer_open_bytes":
                            assert out["blocked"] is False
                            assert pipe_bytes.startswith(covering_bytes)
                            assert len(pipe_bytes) > len(covering_bytes)
                            assert len(out["warnings"]) == 0
                    else:
                        assert out["blocked"] is False, f"HEAD blocked on write! {out}"
                        assert out["err"] is None
                        assert out["ret"] is (mode == "critical")
                        assert out["dm_calls"] == (1 if mode == "critical" else 0)
                        assert len(out["warnings"]) == 1
                        assert "could not write health DM delivery status" in out["warnings"][0]
                        if fifo_state == "reader_only":
                            assert pipe_bytes == b"", f"HEAD wrote into FIFO! {pipe_bytes!r}"
                        elif fifo_state == "writer_open_bytes":
                            assert pipe_bytes == covering_bytes, "HEAD altered FIFO buffer!"

        # Missing status_path at stat(), FIFO created before os.open in writer
        emit("\n--- Missing -> FIFO race before writer open ---")
        for fifo_state in ["no_peer", "reader_only"]:
            for label, mod, t_sec in [("38d707bd", r1_alert, 0.35), ("HEAD", head_alert, 2.0)]:
                case_dir = tmp / f"r1b_missing_to_fifo_{fifo_state}_{label}"
                case_dir.mkdir()
                status_path = case_dir / "health_dm_delivery_status.json"
                fifo_path = case_dir / "appear.fifo"
                fds, _ = prepare_fifo(fifo_path, fifo_state, covering_bytes)
                real_mkdir = Path.mkdir

                def _mkdir_then_place_fifo(self: Path, *a: object, **kw: object):
                    res = real_mkdir(self, *a, **kw)
                    if self == status_path.parent and not status_path.exists():
                        os.replace(fifo_path, status_path)
                    return res

                Path.mkdir = _mkdir_then_place_fifo  # type: ignore[method-assign]
                try:
                    out = run_dispatch_case(mod, status_path, "critical", t_sec)
                finally:
                    Path.mkdir = real_mkdir  # type: ignore[method-assign]

                pipe_bytes = b""
                if fifo_state == "reader_only":
                    try:
                        pipe_bytes = os.read(fds[0], 8192)
                    except BlockingIOError:
                        pipe_bytes = b""
                for fd in fds:
                    os.close(fd)

                emit(
                    f"MISSING_TO_FIFO state={fifo_state} rev={label}: "
                    f"blocked={out['blocked']} ret={out['ret']} warns={len(out['warnings'])} "
                    f"pipe_bytes={len(pipe_bytes)}B"
                )
                if label == "38d707bd":
                    if fifo_state == "no_peer":
                        assert out["blocked"] is True
                    else:
                        assert len(pipe_bytes) > 0 and len(out["warnings"]) == 0
                else:
                    assert out["blocked"] is False and out["ret"] is True
                    assert len(out["warnings"]) == 1 and pipe_bytes == b""

        # =========================================================================
        # PART 1C: Stressing HEAD (5 stress vectors)
        # =========================================================================
        emit("\n=== PART 1C-1: Swap between os.open and os.fstat ===")
        for side in ("read", "write"):
            sdir = tmp / f"open_to_fstat_{side}"
            sdir.mkdir()
            status_path = sdir / "status.json"
            status_path.write_bytes(covering_bytes if side == "read" else b"{}")
            hardlink = sdir / "orig_inode.link"
            os.link(status_path, hardlink)
            fifo_path = sdir / "late.fifo"
            os.mkfifo(fifo_path)

            real_os_open = os.open
            real_os_fstat = os.fstat
            target_fd: list[int] = []
            swapped = False

            def _track_open(p: object, flags: int, *a: object, **kw: object) -> int:
                fd = real_os_open(p, flags, *a, **kw)
                is_write = bool(flags & (os.O_WRONLY | os.O_RDWR))
                if Path(str(p)) == status_path and is_write == (side == "write"):
                    target_fd.append(fd)
                return fd

            def _swap_before_fstat(fd: int):
                nonlocal swapped
                if fd in target_fd and not swapped:
                    swapped = True
                    os.replace(fifo_path, status_path)
                return real_os_fstat(fd)

            os.open = _track_open  # type: ignore[assignment]
            os.fstat = _swap_before_fstat  # type: ignore[assignment]
            try:
                out = run_dispatch_case(head_alert, status_path, "critical", 2.0)
            finally:
                os.open = real_os_open  # type: ignore[assignment]
                os.fstat = real_os_fstat  # type: ignore[assignment]

            assert swapped is True
            assert out["blocked"] is False and out["err"] is None
            if side == "read":
                assert out["ret"] is False and out["dm_calls"] == 0
                assert stat.S_ISFIFO(status_path.lstat().st_mode)
            else:
                assert out["ret"] is True and out["dm_calls"] == 1
                assert len(out["warnings"]) == 0
                assert json.loads(hardlink.read_bytes())["status"] == "sent"
                assert stat.S_ISFIFO(status_path.lstat().st_mode)
            emit(f"OPEN_TO_FSTAT side={side}: swapped={swapped} ret={out['ret']} dm={out['dm_calls']} OK")

        emit("\n=== PART 1C-2: Symlinks (1-hop & 2-hop) to FIFO and /dev/null (stable + race) ===")
        for hop_kind in ("symlink_fifo", "chain2_fifo", "symlink_devnull"):
            for phase in ("stable", "race_read", "race_write"):
                for fifo_state in (["no_peer"] if hop_kind == "symlink_devnull" else ["no_peer", "writer_open_bytes", "reader_only"]):
                    sdir = tmp / f"sym_{hop_kind}_{phase}_{fifo_state}"
                    sdir.mkdir()
                    status_path = sdir / "status.json"
                    link_stage = sdir / "staged.link"
                    fds = []
                    if hop_kind == "symlink_devnull":
                        link_stage.symlink_to("/dev/null")
                    else:
                        fifo_target = sdir / "backing.fifo"
                        fds, _ = prepare_fifo(fifo_target, fifo_state, covering_bytes)
                        if hop_kind == "symlink_fifo":
                            link_stage.symlink_to(fifo_target)
                        else:
                            mid = sdir / "mid.link"
                            mid.symlink_to(fifo_target)
                            link_stage.symlink_to(mid)

                    real_stat = Path.stat
                    hits = 0
                    if phase == "stable":
                        os.replace(link_stage, status_path)
                    else:
                        status_path.write_bytes(covering_bytes if phase == "race_read" else b"{}")
                        target_hit = 1 if phase == "race_read" else 2

                        def _swap_sym(self: Path, *a: object, **kw: object):
                            nonlocal hits
                            st = real_stat(self, *a, **kw)
                            if self == status_path:
                                hits += 1
                                if hits == target_hit:
                                    os.replace(link_stage, status_path)
                            return st

                        Path.stat = _swap_sym  # type: ignore[method-assign]

                    try:
                        out = run_dispatch_case(head_alert, status_path, "critical", 2.0)
                    finally:
                        Path.stat = real_stat  # type: ignore[method-assign]

                    pipe_left = b""
                    if fifo_state in ("writer_open_bytes", "reader_only") and fds:
                        try:
                            pipe_left = os.read(fds[0], 8192)
                        except BlockingIOError:
                            pipe_left = b""
                    for fd in fds:
                        os.close(fd)

                    assert out["blocked"] is False and out["err"] is None
                    assert out["ret"] is True and out["dm_calls"] == 1
                    assert len(out["warnings"]) == 1
                    assert "could not write health DM delivery status" in out["warnings"][0]
                    if fifo_state == "writer_open_bytes":
                        assert pipe_left == covering_bytes
                    elif fifo_state == "reader_only":
                        assert pipe_left == b""
        emit("SYMLINK_STRESS: all 21 stable + race symlink-to-FIFO / chain / devnull cases OK")

        emit("\n=== PART 1C-3: O_NONBLOCK on regular file (flags, 8MB I/O, EAGAIN semantics) ===")
        nb_dir = tmp / "nonblock_check"
        nb_dir.mkdir()
        nb_file = nb_dir / "nb.json"
        observed_flags: dict[str, int] = {}
        orig_builtin_open = builtins.open

        def _spy_open(file: object, mode: str = "r", *a: object, **kw: object):
            if isinstance(file, int):
                fl = fcntl.fcntl(file, fcntl.F_GETFL)
                observed_flags[mode] = fl
            return orig_builtin_open(file, mode, *a, **kw)

        builtins.open = _spy_open  # type: ignore[assignment]
        try:
            head_alert._write_regular_file_text(nb_file, '{"ok": true}\n')
            read_back = head_alert._read_regular_file_text(nb_file)
        finally:
            builtins.open = orig_builtin_open  # type: ignore[assignment]
        assert read_back == '{"ok": true}\n'
        assert bool(observed_flags["w"] & os.O_NONBLOCK) is True
        assert bool(observed_flags["r"] & os.O_NONBLOCK) is True
        emit(
            f"O_NONBLOCK_FLAGS: write_nonblock={bool(observed_flags['w'] & os.O_NONBLOCK)} "
            f"read_nonblock={bool(observed_flags['r'] & os.O_NONBLOCK)}"
        )

        # Multi-megabyte regular file read & write (8 MiB > 1000x io.DEFAULT_BUFFER_SIZE)
        big_file = nb_dir / "big_8mb.json"
        big_payload = json.dumps({
            "pad": "A" * (8 * 1024 * 1024),
            "status": "sent",
            "sent_day": DAY.isoformat(),
            "sent_sources": [KEY],
        }) + "\n"
        head_alert._write_regular_file_text(big_file, big_payload)
        assert big_file.stat().st_size == len(big_payload.encode("utf-8"))
        assert head_alert._read_regular_file_text(big_file) == big_payload
        big_out = run_dispatch_case(head_alert, big_file, "critical", 3.0)
        assert big_out["ret"] is False and big_out["dm_calls"] == 0
        big_file.unlink()
        emit(f"O_NONBLOCK_8MB: wrote & read {len(big_payload)} chars intact; dedup suppressed OK")

        # Observe exact CPython 3.12 TextIOWrapper behavior if an fd actually yields EAGAIN:
        r_pipe, w_pipe = os.pipe()
        os.set_blocking(r_pipe, False)
        os.set_blocking(w_pipe, False)
        try:
            eagain_read_zero_exc: str | None = None
            try:
                with open(r_pipe, "r", closefd=False) as h:
                    h.read()
            except BaseException as exc:
                eagain_read_zero_exc = f"{type(exc).__name__}: {exc}"

            os.write(w_pipe, b'{"partial": 1')
            with open(r_pipe, "r", closefd=False) as h:
                partial_res = h.read()

            eagain_write_exc: str | None = None
            try:
                with open(w_pipe, "w", closefd=False) as h:
                    h.write("X" * (2 * 1024 * 1024))
            except BaseException as exc:
                eagain_write_exc = f"{type(exc).__name__} (isinstance OSError={isinstance(exc, OSError)})"
        finally:
            os.close(r_pipe)
            os.close(w_pipe)
        emit(
            f"CPYTHON_EAGAIN_BEHAVIOR: "
            f"read_0B_eagain={eagain_read_zero_exc!r} | "
            f"read_partial_eagain={partial_res!r} | "
            f"write_eagain={eagain_write_exc!r}"
        )

        # =========================================================================
        # PART 1C-4: Descriptor leak check — 10,000 calls of each of 14 failure kinds
        # =========================================================================
        emit("\n=== PART 1C-4: Open descriptor leak check (10,000 calls x 14 failure kinds) ===")
        leak_dir = tmp / "fd_leaks"
        leak_dir.mkdir()
        N = 10_000

        def snap_fds() -> frozenset[str]:
            return frozenset(os.listdir("/dev/fd"))

        missing_p = leak_dir / "missing.json"
        perm000_p = leak_dir / "perm000.json"
        perm000_p.write_text("{}")
        perm000_p.chmod(0o000)

        reg_ok_p = leak_dir / "reg_ok.json"
        reg_ok_p.write_text("{}")

        fifo_nopeer_p = leak_dir / "nopeer.fifo"
        os.mkfifo(fifo_nopeer_p)

        dir_p = leak_dir / "adir"
        dir_p.mkdir()

        bad_utf8_p = leak_dir / "bad_utf8.json"
        bad_utf8_p.write_bytes(b"\xff\xfe\x80\x81")

        bad_json_p = leak_dir / "bad_json.json"
        bad_json_p.write_bytes(b"{not_json")

        fifo_with_reader_p = leak_dir / "with_reader.fifo"
        os.mkfifo(fifo_with_reader_p)
        keep_reader_fd = os.open(fifo_with_reader_p, os.O_RDONLY | os.O_NONBLOCK)

        devnull_sym_p = leak_dir / "devnull.sym"
        devnull_sym_p.symlink_to("/dev/null")

        scratch_write_p = leak_dir / "scratch_write.json"
        scratch_write_p.write_text("{}")

        def _expect_exc(fn, exc_type):
            try:
                fn()
            except exc_type:
                return
            raise AssertionError(f"{fn} did not raise {exc_type}")

        real_fstat = os.fstat
        real_ftruncate = os.ftruncate

        def _run_fstat_fail_read():
            def _bad_fstat(_fd: int):
                raise OSError(errno.EIO, "simulated fstat EIO")
            os.fstat = _bad_fstat  # type: ignore[assignment]
            try:
                for _ in range(N):
                    assert head_alert._read_health_dm_delivery_status(reg_ok_p) is None
            finally:
                os.fstat = real_fstat  # type: ignore[assignment]

        def _run_fstat_fail_write():
            def _bad_fstat(_fd: int):
                raise OSError(errno.EIO, "simulated fstat EIO")
            os.fstat = _bad_fstat  # type: ignore[assignment]
            try:
                for _ in range(N):
                    _expect_exc(lambda: head_alert._write_regular_file_text(scratch_write_p, "{}\n"), OSError)
            finally:
                os.fstat = real_fstat  # type: ignore[assignment]

        def _run_ftruncate_fail_write():
            def _bad_ftrunc(_fd: int, _sz: int):
                raise OSError(errno.EIO, "simulated ftruncate EIO")
            os.ftruncate = _bad_ftrunc  # type: ignore[assignment]
            try:
                for _ in range(N):
                    _expect_exc(lambda: head_alert._write_regular_file_text(scratch_write_p, "{}\n"), OSError)
            finally:
                os.ftruncate = real_ftruncate  # type: ignore[assignment]

        def _run_enospc_write():
            orig_open = builtins.open

            class _EnospcHandle:
                def __init__(self, inner):
                    self._inner = inner

                def __enter__(self):
                    self._inner.__enter__()
                    return self

                def write(self, _s: str) -> int:
                    raise OSError(errno.ENOSPC, "No space left on device")

                def __exit__(self, *exc_info):
                    return self._inner.__exit__(*exc_info)

            def _enospc_open(f, mode="r", *a, **kw):
                h = orig_open(f, mode, *a, **kw)
                if isinstance(f, int) and mode == "w":
                    return _EnospcHandle(h)
                return h

            builtins.open = _enospc_open  # type: ignore[assignment]
            try:
                for _ in range(N):
                    _expect_exc(lambda: head_alert._write_regular_file_text(scratch_write_p, "{}\n"), OSError)
            finally:
                builtins.open = orig_open  # type: ignore[assignment]

        failure_kinds = [
            ("read_open_enoent", lambda: [_expect_exc(lambda: head_alert._read_regular_file_text(missing_p), FileNotFoundError) for _ in range(N)]),
            ("read_open_eacces", lambda: [_expect_exc(lambda: head_alert._read_regular_file_text(perm000_p), PermissionError) for _ in range(N)]),
            ("read_fstat_oserror", _run_fstat_fail_read),
            ("read_nonreg_fifo", lambda: [_expect_exc(lambda: head_alert._read_regular_file_text(fifo_nopeer_p), OSError) for _ in range(N)]),
            ("read_nonreg_dir", lambda: [_expect_exc(lambda: head_alert._read_regular_file_text(dir_p), OSError) for _ in range(N)]),
            ("read_utf8_decode_err", lambda: [_expect_exc(lambda: head_alert._read_regular_file_text(bad_utf8_p), UnicodeDecodeError) for _ in range(N)]),
            ("read_json_syntax_err", lambda: [head_alert._read_health_dm_delivery_status(bad_json_p) for _ in range(N)]),
            ("write_open_enxio_fifo", lambda: [_expect_exc(lambda: head_alert._write_regular_file_text(fifo_nopeer_p, "{}\n"), OSError) for _ in range(N)]),
            ("write_fstat_oserror", _run_fstat_fail_write),
            ("write_nonreg_fifo_reader", lambda: [_expect_exc(lambda: head_alert._write_regular_file_text(fifo_with_reader_p, "{}\n"), OSError) for _ in range(N)]),
            ("write_nonreg_devnull", lambda: [_expect_exc(lambda: head_alert._write_regular_file_text(devnull_sym_p, "{}\n"), OSError) for _ in range(N)]),
            ("write_ftruncate_oserror", _run_ftruncate_fail_write),
            ("write_handle_enospc", _run_enospc_write),
            ("write_surrogate_encode_err", lambda: [_expect_exc(lambda: head_alert._write_regular_file_text(scratch_write_p, "\ud800\n"), UnicodeEncodeError) for _ in range(N)]),
        ]

        try:
            for kind_name, runner_fn in failure_kinds:
                before_set = snap_fds()
                t_start = time.monotonic()
                runner_fn()
                dt_ms = round((time.monotonic() - t_start) * 1000.0, 1)
                after_set = snap_fds()
                assert before_set == after_set, (
                    f"FD LEAK in {kind_name}: before={sorted(before_set)} after={sorted(after_set)}"
                )
                emit(f"FD_LEAK_10K kind={kind_name}: fds_before={len(before_set)} fds_after={len(after_set)} leaked=0 ({dt_ms}ms)")
        finally:
            os.close(keep_reader_fd)
            perm000_p.chmod(0o600)

        # =========================================================================
        # PART 1C-5: Regular file with locks / open writer / BSD flags
        # =========================================================================
        emit("\n=== PART 1C-5: Regular file with locks, concurrent open writer, and BSD flags ===")
        lock_dir = tmp / "locks"
        lock_dir.mkdir()
        locked_file = lock_dir / "locked_status.json"
        locked_file.write_text('{"status":"failed","critical_count":1}\n')

        exlock_file = lock_dir / "exlock_status.json"
        exlock_file.write_text('{"status":"failed","critical_count":1}\n')

        holder_code = (
            "import fcntl, os, struct, sys\n"
            "p1, p2 = sys.argv[1], sys.argv[2]\n"
            "fd1 = os.open(p1, os.O_RDWR)\n"
            "fcntl.flock(fd1, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "lockdata = struct.pack('qqihh', 0, 0, 0, fcntl.F_WRLCK, os.SEEK_SET)\n"
            "fcntl.fcntl(fd1, fcntl.F_SETLK, lockdata)\n"
            "fd2 = os.open(p2, os.O_RDWR | getattr(os, 'O_EXLOCK', 0) | os.O_NONBLOCK)\n"
            "sys.stdout.write('LOCKED\\n')\n"
            "sys.stdout.flush()\n"
            "sys.stdin.read()\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", holder_code, str(locked_file), str(exlock_file)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            ready_line = proc.stdout.readline().strip() if proc.stdout else ""
            assert ready_line == "LOCKED", f"lock holder failed: {proc.stderr.read() if proc.stderr else ''}"

            out_flock_fcntl = run_dispatch_case(head_alert, locked_file, "critical", 2.0)
            out_exlock = run_dispatch_case(head_alert, exlock_file, "critical", 2.0)
            assert out_flock_fcntl["blocked"] is False and out_flock_fcntl["ret"] is True
            assert json.loads(locked_file.read_bytes())["status"] == "sent"
            assert out_exlock["blocked"] is False and out_exlock["ret"] is True
            assert json.loads(exlock_file.read_bytes())["status"] == "sent"
            emit(
                f"CONCURRENT_WRITER_LOCKS: flock+F_SETLK(blocked={out_flock_fcntl['blocked']}, saved=sent) | "
                f"O_EXLOCK(blocked={out_exlock['blocked']}, saved=sent)"
            )
        finally:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=5)

        if hasattr(os, "chflags") and hasattr(stat, "UF_IMMUTABLE"):
            for flag_name in ("UF_IMMUTABLE", "UF_APPEND"):
                flag_val = getattr(stat, flag_name, 0)
                if not flag_val:
                    continue
                uf_file = lock_dir / f"{flag_name.lower()}.json"
                uf_file.write_text('{"status":"failed","critical_count":1}\n')
                try:
                    os.chflags(uf_file, flag_val)
                except OSError as exc:
                    emit(f"BSD_CHFLAGS {flag_name}: skipped ({exc})")
                    continue
                try:
                    out_uf = run_dispatch_case(head_alert, uf_file, "critical", 2.0)
                    assert out_uf["blocked"] is False and out_uf["ret"] is True and out_uf["dm_calls"] == 1
                    assert len(out_uf["warnings"]) == 1
                    assert "could not write health DM delivery status" in out_uf["warnings"][0]
                    emit(f"BSD_CHFLAGS {flag_name}: blocked=False ret=True warn={out_uf['warnings'][0]}")
                finally:
                    os.chflags(uf_file, 0)

    emit("\nITEM 1 PROBE COMPLETE: ALL CHECKS PASSED")
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
