"""Round-2 independent probe for:
- Item 2 completion: simulated EROFS & ENOSPC at write, subprocess locale/encoding
  parity (LC_ALL=C with PYTHONUTF8=0, PYTHONUTF8=0 + PYTHONCOERCECLOCALE=0, PYTHONUTF8=1,
  and en_US.UTF-8) for non-ASCII body first lines and raw UTF-8 files, and universal
  newline translation on read.
- Item 3: `git diff` + AST verification, 2,112-pair conforming-file comparison vs 59efa825,
  240-run malformed-content matrix at default recursion limit 1000, and stable special-path
  matrix.
"""

from __future__ import annotations

import ast
import builtins
import codecs
import contextlib
import errno
import io
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
LOG_PATH = REVIEW_DIR / "probe_encoding_and_item3.log"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# --- Outward network & DNS guard ---
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
        raise RuntimeError(f"forbidden outward connect: {address!r}")
    return _REAL_CONNECT(self, address)


def _guarded_connect_ex(self: socket.socket, address: object) -> int:
    if self.family == socket.AF_UNIX:
        return _REAL_CONNECT_EX(self, address)
    host = address[0] if isinstance(address, tuple) and address else None
    if not _is_allowed_host(host):
        raise RuntimeError(f"forbidden outward connect_ex: {address!r}")
    return _REAL_CONNECT_EX(self, address)


def _guarded_getaddrinfo(host: object, port: object, *a: object, **kw: object) -> object:
    if not _is_allowed_host(host):
        raise RuntimeError(f"forbidden DNS: {host!r}")
    return _REAL_GETADDRINFO(host, port, *a, **kw)


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
NEXT_DAY = date(2027, 7, 15)


class ProbeBlockedError(BaseException):
    pass


@contextlib.contextmanager
def bounded_alarm(seconds: float = 2.0):
    def _on_alarm(*_):
        raise ProbeBlockedError(f"blocked >= {seconds:.2f}s")

    prior_h = signal.signal(signal.SIGALRM, _on_alarm)
    prior_t = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_h)
        signal.setitimer(signal.ITIMER_REAL, *prior_t)


@contextlib.contextmanager
def capture_alert_logs():
    logger = logging.getLogger("bts.health.alert")
    records: list[tuple[int, str]] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append((record.levelno, record.getMessage()))

    h = _H(level=logging.DEBUG)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(h)
    try:
        yield records
    finally:
        logger.removeHandler(h)
        logger.setLevel(old_level)


def make_clock(utc_dt: datetime):
    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return utc_dt.astimezone(tz) if tz is not None else utc_dt.replace(tzinfo=None)

    return _Clock


def load_archived_source_and_module(rev: str, mod_name: str) -> tuple[bytes, ModuleType]:
    rel = "src/bts/health/alert.py"
    raw = subprocess.check_output(["git", "archive", rev, rel], cwd=REPO_ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        member = tf.extractfile(rel)
        assert member is not None
        src_bytes = member.read()
    code = compile(src_bytes, f"{rev}:{rel}", "exec")
    mod = ModuleType(mod_name)
    mod.__file__ = f"{rev}:{rel}"
    sys.modules[mod_name] = mod
    exec(code, mod.__dict__)
    mod.log = logging.getLogger("bts.health.alert")
    return src_bytes, mod


def ast_func_slices(src_bytes: bytes) -> dict[str, bytes]:
    lines = src_bytes.splitlines(keepends=True)
    tree = ast.parse(src_bytes)
    out: dict[str, bytes] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
            end = node.end_lineno
            assert end is not None
            out[node.name] = b"".join(lines[start:end])
    return out


def main() -> None:
    out_lines: list[str] = []

    def emit(msg: str) -> None:
        print(msg)
        out_lines.append(msg)

    base_bytes, base_alert = load_archived_source_and_module("59efa825", "_cs6b2_base_59efa825")
    r1_bytes, _ = load_archived_source_and_module("38d707bd", "_cs6b2_r1_38d707bd")
    head_bytes = (REPO_ROOT / "src" / "bts" / "health" / "alert.py").read_bytes()

    ClockDay1 = make_clock(datetime(2027, 7, 15, 3, 59, 0, tzinfo=timezone.utc))  # 2027-07-14 23:59 EDT
    ClockDay2 = make_clock(datetime(2027, 7, 16, 3, 59, 0, tzinfo=timezone.utc))  # 2027-07-15 23:59 EDT

    sample_kw = dict(
        status="sent",
        dm_recipient="offline.test",
        critical_count=1,
        warn_attention_count=0,
        body="BTS health CRITICAL alert(s):\n- [a] msg",
        error=None,
        sent_day="2027-07-14",
        sent_sources=["CRITICAL:a"],
    )

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_enc_i3_") as tmp_s:
        tmp = Path(tmp_s)

        # =========================================================================
        # 2F: Simulated EROFS (at open) and ENOSPC (at write after truncate)
        # =========================================================================
        emit("=== 2F: Simulated EROFS and ENOSPC at write (59efa825 vs HEAD) ===")
        fs_err_path = tmp / "fs_err_status.json"
        seed_prior = b'{"status":"sent","sent_day":"2027-07-13","sent_sources":["CRITICAL:x"]}\n'

        orig_base_dt, orig_head_dt = base_alert.datetime, head_alert.datetime
        base_alert.datetime = ClockDay1
        head_alert.datetime = ClockDay1
        try:
            # EROFS at open
            real_path_open = Path.open
            real_os_open = os.open

            fs_err_path.write_bytes(seed_prior)

            def _erofs_path_open(self: Path, mode: str = "r", *a: object, **kw: object):
                if self == fs_err_path and mode == "w":
                    raise OSError(errno.EROFS, "Read-only file system", str(self))
                return real_path_open(self, mode, *a, **kw)

            Path.open = _erofs_path_open  # type: ignore[method-assign]
            try:
                with capture_alert_logs() as b_erofs_logs:
                    base_alert._write_health_dm_delivery_status(fs_err_path, **sample_kw)
            finally:
                Path.open = real_path_open  # type: ignore[method-assign]
            b_erofs_bytes = fs_err_path.read_bytes()

            fs_err_path.write_bytes(seed_prior)

            def _erofs_os_open(p: object, flags: int, *a: object, **kw: object) -> int:
                if Path(str(p)) == fs_err_path and (flags & (os.O_WRONLY | os.O_RDWR)):
                    raise OSError(errno.EROFS, "Read-only file system", str(p))
                return real_os_open(p, flags, *a, **kw)

            os.open = _erofs_os_open  # type: ignore[assignment]
            try:
                with capture_alert_logs() as h_erofs_logs:
                    head_alert._persist_health_dm_delivery_status(fs_err_path, **sample_kw)
            finally:
                os.open = real_os_open  # type: ignore[assignment]
            h_erofs_bytes = fs_err_path.read_bytes()
            assert b_erofs_bytes == h_erofs_bytes == seed_prior
            assert b_erofs_logs == h_erofs_logs and len(h_erofs_logs) == 1

            # ENOSPC during write() (strictly mode == "w" so binary .write_bytes/.read_bytes unaffected)
            orig_io_open = io.open
            orig_builtin_open = builtins.open

            class _EnospcWriter:
                def __init__(self, inner):
                    self._inner = inner

                def __enter__(self):
                    self._inner.__enter__()
                    return self

                def write(self, _text: str) -> int:
                    raise OSError(errno.ENOSPC, "No space left on device")

                def __exit__(self, *exc_info):
                    return self._inner.__exit__(*exc_info)

            def _enospc_open(file: object, mode: str = "r", *a: object, **kw: object):
                h = orig_io_open(file, mode, *a, **kw)  # type: ignore[arg-type]
                if mode == "w" and (isinstance(file, int) or Path(str(file)) == fs_err_path):
                    return _EnospcWriter(h)
                return h

            fds_before = frozenset(os.listdir("/dev/fd"))
            fs_err_path.write_bytes(seed_prior)
            io.open = _enospc_open  # type: ignore[assignment]
            builtins.open = _enospc_open  # type: ignore[assignment]
            try:
                with capture_alert_logs() as b_enospc_logs:
                    base_alert._write_health_dm_delivery_status(fs_err_path, **sample_kw)
            finally:
                io.open = orig_io_open  # type: ignore[assignment]
                builtins.open = orig_builtin_open  # type: ignore[assignment]
            b_enospc_bytes = fs_err_path.read_bytes()

            fs_err_path.write_bytes(seed_prior)
            io.open = _enospc_open  # type: ignore[assignment]
            builtins.open = _enospc_open  # type: ignore[assignment]
            try:
                with capture_alert_logs() as h_enospc_logs:
                    head_alert._persist_health_dm_delivery_status(fs_err_path, **sample_kw)
            finally:
                io.open = orig_io_open  # type: ignore[assignment]
                builtins.open = orig_builtin_open  # type: ignore[assignment]
            h_enospc_bytes = fs_err_path.read_bytes()
            fds_after = frozenset(os.listdir("/dev/fd"))

            assert b_enospc_bytes == h_enospc_bytes == b""
            assert b_enospc_logs == h_enospc_logs and len(h_enospc_logs) == 1
            assert fds_before == fds_after
            emit(
                f"FS_ERRORS: EROFS(untouched={b_erofs_bytes == seed_prior}, same_warn=True) | "
                f"ENOSPC(base_len={len(b_enospc_bytes)}, head_len={len(h_enospc_bytes)}, "
                f"same_warn=True, leaked_fds=0)"
            )
        finally:
            base_alert.datetime = orig_base_dt
            head_alert.datetime = orig_head_dt

        # =========================================================================
        # 2G: Subprocess locale & encoding checks
        # =========================================================================
        emit("\n=== 2G: Subprocess locale & encoding checks (LC_ALL=C, PYTHONUTF8=0/1, non-ASCII line 1) ===")
        locale_child = """
import codecs, io, json, locale, os, subprocess, sys, tarfile
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

repo = Path(sys.argv[1])
work = Path(sys.argv[2])
sys.path.insert(0, str(repo / "src"))
from bts.health import alert as head_alert

raw = subprocess.check_output(["git", "archive", "59efa825", "src/bts/health/alert.py"], cwd=repo)
with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
    code = compile(tf.extractfile("src/bts/health/alert.py").read(), "base", "exec")
base_alert = ModuleType("_base")
sys.modules["_base"] = base_alert
exec(code, base_alert.__dict__)

class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        dt = datetime(2027, 7, 15, 3, 59, 0, tzinfo=timezone.utc)
        return dt.astimezone(tz) if tz is not None else dt.replace(tzinfo=None)

base_alert.datetime = Clock
head_alert.datetime = Clock

# 1) Verify open(fd) and Path.open() resolve to the exact same codec
probe_f = work / "enc_probe.txt"
probe_f.write_bytes(b"ascii\\n")
with probe_f.open("r") as ph:
    p_read_enc = codecs.lookup(ph.encoding).name
fd = os.open(probe_f, os.O_RDONLY | os.O_NONBLOCK)
try:
    with open(fd, "r", closefd=False) as fh:
        f_read_enc = codecs.lookup(fh.encoding).name
finally:
    os.close(fd)
assert p_read_enc == f_read_enc, (p_read_enc, f_read_enc)

# 2) Direct helper vs Path.write_text / Path.read_text on non-ASCII string
non_ascii_sample = "⚠️ café — 東京\\nsecond line\\n"
d_base = work / "direct_p.txt"
d_head = work / "direct_h.txt"
base_exc = None
head_exc = None
try:
    d_base.write_text(non_ascii_sample)
    b_rt = d_base.read_text()
except UnicodeEncodeError as e:
    base_exc = type(e).__name__
    b_rt = None
try:
    head_alert._write_regular_file_text(d_head, non_ascii_sample)
    h_rt = head_alert._read_regular_file_text(d_head)
except UnicodeEncodeError as e:
    head_exc = type(e).__name__
    h_rt = None
assert base_exc == head_exc and b_rt == h_rt

# 3) Dispatcher with non-ASCII first line (resurface banner ⚠️...—...) + non-ASCII alert message
bp = work / "b_status.json"
hp = work / "h_status.json"
seed = json.dumps({"status": "failed", "critical_count": 2}).encode("ascii")
bp.write_bytes(seed)
hp.write_bytes(seed)
bm, hm = Mock(return_value="ok"), Mock(return_value="ok")
base_alert.send_dm = bm
head_alert.send_dm = hm
b_ret = base_alert.dispatch_dm_for_health_alerts(
    [base_alert.Alert("CRITICAL", "café—東京", "🚨 Crítico — München 東京")],
    "offline.test", status_path=bp, now_et_date=date(2027, 7, 14),
)
h_ret = head_alert.dispatch_dm_for_health_alerts(
    [head_alert.Alert("CRITICAL", "café—東京", "🚨 Crítico — München 東京")],
    "offline.test", status_path=hp, now_et_date=date(2027, 7, 14),
)
assert b_ret == h_ret is True
assert bm.call_args_list == hm.call_args_list
assert bp.read_bytes() == hp.read_bytes()

# 4) Writer directly with non-ASCII first line in `body`
bp2 = work / "b_direct.json"
hp2 = work / "h_direct.json"
for mod, p in [(base_alert, bp2), (head_alert, hp2)]:
    mod._write_health_dm_delivery_status(
        p, status="sent", dm_recipient="offline.test", critical_count=1, warn_attention_count=0,
        body="🚨 Primera línea no-ASCII — 東京\\nsecond line",
        sent_day="2027-07-14", sent_sources=["CRITICAL:café"],
    )
assert bp2.read_bytes() == hp2.read_bytes()

# 5) Raw unescaped UTF-8 bytes in status file on disk
raw_utf8 = b'{"status":"sent","sent_day":"2027-07-14","sent_sources":["CRITICAL:caf\\xc3\\xa9"]}\\n'
bp3 = work / "b_rawutf8.json"
hp3 = work / "h_rawutf8.json"
bp3.write_bytes(raw_utf8)
hp3.write_bytes(raw_utf8)
bm3, hm3 = Mock(return_value="ok"), Mock(return_value="ok")
base_alert.send_dm = bm3
head_alert.send_dm = hm3
r_b3 = base_alert.dispatch_dm_for_health_alerts(
    [base_alert.Alert("CRITICAL", "café", "m")], "offline.test", status_path=bp3, now_et_date=date(2027, 7, 14),
)
r_h3 = head_alert.dispatch_dm_for_health_alerts(
    [head_alert.Alert("CRITICAL", "café", "m")], "offline.test", status_path=hp3, now_et_date=date(2027, 7, 14),
)
assert r_b3 == r_h3
assert bp3.read_bytes() == hp3.read_bytes()

print(json.dumps({
    "codec": p_read_enc,
    "utf8_mode": sys.flags.utf8_mode,
    "direct_nonascii_err": head_exc,
    "raw_utf8_suppressed": (not r_h3),
}))
"""
        env_variants = [
            ("LC_ALL=C_UTF8=0", {"LC_ALL": "C", "PYTHONUTF8": "0"}),
            ("LC_ALL=C_UTF8=0_NOCOERCE", {"LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}),
            ("LC_ALL=C_UTF8=1", {"LC_ALL": "C", "PYTHONUTF8": "1"}),
            ("LC_ALL=en_US.UTF-8_UTF8=0", {"LC_ALL": "en_US.UTF-8", "PYTHONUTF8": "0"}),
        ]
        for env_label, overrides in env_variants:
            sub_work = tmp / f"enc_{env_label}"
            sub_work.mkdir()
            c_env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("LC_", "LANG", "PYTHONUTF8", "PYTHONCOERCE"))
            }
            c_env.update(overrides)
            proc_res = subprocess.run(
                [sys.executable, "-c", locale_child, str(REPO_ROOT), str(sub_work)],
                env=c_env,
                capture_output=True,
                text=True,
                check=True,
            )
            info = json.loads(proc_res.stdout.strip())
            emit(
                f"ENCODING_ENV {env_label}: codec={info['codec']} utf8_mode={info['utf8_mode']} "
                f"direct_err={info['direct_nonascii_err']} raw_utf8_suppressed={info['raw_utf8_suppressed']} "
                f"base==head=True"
            )

        # =========================================================================
        # 2H: Universal-newline translation on read (\r\n, \r, mixed)
        # =========================================================================
        emit("\n=== 2H: Universal-newline translation on read ===")
        nl_dir = tmp / "newlines"
        nl_dir.mkdir()
        for nl_name, raw_nl in [
            ("crlf", b"line1\r\nline2\r\nline3\r\n"),
            ("cr", b"line1\rline2\rline3\r"),
            ("mixed", b"line1\r\nline2\rline3\n"),
        ]:
            nl_f = nl_dir / f"{nl_name}.txt"
            nl_f.write_bytes(raw_nl)
            assert head_alert._read_regular_file_text(nl_f) == nl_f.read_text() == "line1\nline2\nline3\n"

        for nl_name, sep in [("crlf_json", "\r\n"), ("cr_json", "\r")]:
            raw_json = sep.join([
                "{",
                '  "status": "sent",',
                f'  "sent_day": "{DAY.isoformat()}",',
                '  "sent_sources": ["CRITICAL:probe"]',
                "}",
            ]).encode("utf-8")
            bf = nl_dir / f"b_{nl_name}.json"
            hf = nl_dir / f"h_{nl_name}.json"
            bf.write_bytes(raw_json)
            hf.write_bytes(raw_json)
            bm, hm = Mock(), Mock()
            orig_bs, orig_hs = base_alert.send_dm, head_alert.send_dm
            base_alert.send_dm, head_alert.send_dm = bm, hm
            try:
                rb = base_alert.dispatch_dm_for_health_alerts(
                    [base_alert.Alert("CRITICAL", "probe", "m")], "offline.test", status_path=bf, now_et_date=DAY,
                )
                rh = head_alert.dispatch_dm_for_health_alerts(
                    [head_alert.Alert("CRITICAL", "probe", "m")], "offline.test", status_path=hf, now_et_date=DAY,
                )
            finally:
                base_alert.send_dm, head_alert.send_dm = orig_bs, orig_hs
            assert rb is False and rh is False and bm.call_count == 0 and hm.call_count == 0
        emit("UNIVERSAL_NEWLINES: \\r\\n, \\r, and mixed line endings match Path.read_text and base_alert")

        # =========================================================================
        # SECTION 3: Item 3 — Nothing else moved
        # =========================================================================
        emit("\n=== SECTION 3: Nothing else moved (diff, AST, 2,112 conforming pairs, malformed & special paths) ===")

        diff_38_src = subprocess.check_output(
            ["git", "diff", "38d707bd", "HEAD", "--name-only", "--", "src"], cwd=REPO_ROOT, text=True
        ).splitlines()
        diff_59_src = subprocess.check_output(
            ["git", "diff", "59efa825", "HEAD", "--name-only", "--", "src"], cwd=REPO_ROOT, text=True
        ).splitlines()
        assert diff_38_src == ["src/bts/health/alert.py"]
        assert diff_59_src == ["src/bts/health/alert.py"]

        base_defs = ast_func_slices(base_bytes)
        r1_defs = ast_func_slices(r1_bytes)
        head_defs = ast_func_slices(head_bytes)

        added_since_r1 = set(head_defs) - set(r1_defs)
        assert added_since_r1 == {"_write_regular_file_text", "_read_regular_file_text"}
        for unchanged_vs_r1 in (
            "Alert", "log_alerts", "_persist_health_dm_delivery_status",
            "dispatch_dm_for_critical", "format_health_dm_body",
            "_resurface_note", "dispatch_dm_for_health_alerts",
        ):
            assert r1_defs[unchanged_vs_r1] == head_defs[unchanged_vs_r1], f"moved vs 38d707bd: {unchanged_vs_r1}"

        for unchanged_vs_base in ("Alert", "log_alerts", "dispatch_dm_for_critical", "format_health_dm_body"):
            assert base_defs[unchanged_vs_base] == head_defs[unchanged_vs_base], f"moved vs 59efa825: {unchanged_vs_base}"
        emit(
            "SRC_AST_DIFF: only src/bts/health/alert.py touched; added={'_write_regular_file_text', "
            "'_read_regular_file_text'}; all other functions byte-identical to 38d707bd"
        )

        # 3B: Conforming-file parity matrix (2,112 pairs vs 59efa825)
        conf_dir = tmp / "conforming"
        conf_dir.mkdir()
        orig_base_dt, orig_head_dt = base_alert.datetime, head_alert.datetime
        base_alert.datetime = ClockDay1
        head_alert.datetime = ClockDay1
        try:
            priors: dict[str, bytes | None] = {}
            seed_tmp = conf_dir / "seed.json"

            def _gen_prior(status: str, crit_n: int, warn_n: int, body: str, err, s_day, s_srcs) -> bytes:
                base_alert._write_health_dm_delivery_status(
                    seed_tmp, status=status,
                    dm_recipient=None if status == "skipped_no_recipient" else "offline.test",
                    critical_count=crit_n, warn_attention_count=warn_n, body=body,
                    error=err, sent_day=s_day, sent_sources=s_srcs,
                )
                return seed_tmp.read_bytes()

            priors["sent_one"] = _gen_prior("sent", 1, 0, "b1", None, DAY.isoformat(), ["CRITICAL:a"])
            priors["sent_many"] = _gen_prior(
                "sent", 2, 1, "b2", None, DAY.isoformat(),
                ["CRITICAL:a", "CRITICAL:b:inc1", "WARN_ATTENTION:w1"],
            )
            priors["sent_warn_only"] = _gen_prior("sent", 0, 1, "b3", None, DAY.isoformat(), ["WARN_ATTENTION:w1"])
            priors["sent_banner_only"] = _gen_prior("sent", 0, 0, "banner", None, DAY.isoformat(), [])
            priors["failed_crit1"] = _gen_prior("failed", 1, 0, "bf1", "net err", None, None)
            priors["failed_crit3"] = _gen_prior("failed", 3, 1, "bf3", "timeout", None, None)
            priors["failed_warn0"] = _gen_prior("failed", 0, 1, "bf0", "warn fail", None, None)
            priors["skipped"] = _gen_prior("skipped_no_recipient", 1, 0, "bs", "dm_recipient is not configured", None, None)
            priors["legacy"] = b'{"schema_version":"bts_health_dm_delivery_status_v1","status":"sent"}\n'
            priors["missing"] = None
            priors["none_path"] = None

            def _make_mix(mod: ModuleType, mix_name: str):
                A = mod.Alert
                if mix_name == "covered_crit":
                    return [A("CRITICAL", "a", "m_a")], []
                if mix_name == "uncovered_crit":
                    return [A("CRITICAL", "new_src", "m_new")], []
                if mix_name == "partial_inc":
                    return [A("CRITICAL", "a", "m_a"), A("CRITICAL", "b", "m_b2", incident_key="b:inc2")], []
                if mix_name == "covered_warn":
                    return [], [A("WARN", "w1", "mw1")]
                if mix_name == "uncovered_warn":
                    return [], [A("WARN", "w2", "mw2")]
                if mix_name == "crit_and_warn":
                    return [A("CRITICAL", "a", "m_a")], [A("WARN", "w2", "mw2")]
                if mix_name == "empty":
                    return [], []
                if mix_name == "info_and_plain_warn":
                    return [A("INFO", "i", "mi"), A("WARN", "pw", "mpw")], []
                raise ValueError(mix_name)

            mix_names = [
                "covered_crit", "uncovered_crit", "partial_inc", "covered_warn",
                "uncovered_warn", "crit_and_warn", "empty", "info_and_plain_warn",
            ]
            shared_status_file = conf_dir / "pair_status.json"
            pairs_checked = 0

            for prior_name, prior_bytes in priors.items():
                for mix_name in mix_names:
                    for recipient in ("offline.test", None, ""):
                        for day_kind in ("same", "next"):
                            clk = ClockDay1 if day_kind == "same" else ClockDay2
                            target_day = DAY if day_kind == "same" else NEXT_DAY
                            for explicit_date in (True, False):
                                for send_fails in (False, True):
                                    pair_out = {}
                                    for tag, mod in (("base", base_alert), ("head", head_alert)):
                                        mod.datetime = clk
                                        if prior_name == "none_path":
                                            s_arg = None
                                        else:
                                            s_arg = shared_status_file
                                            if prior_bytes is None:
                                                shared_status_file.unlink(missing_ok=True)
                                            else:
                                                shared_status_file.write_bytes(prior_bytes)
                                        send_m = Mock(
                                            side_effect=RuntimeError("transport down") if send_fails else None,
                                            return_value="ok",
                                        )
                                        orig_s = mod.send_dm
                                        mod.send_dm = send_m
                                        c_alerts, w_alerts = _make_mix(mod, mix_name)
                                        try:
                                            with capture_alert_logs() as lrecs:
                                                ret = mod.dispatch_dm_for_health_alerts(
                                                    c_alerts,
                                                    recipient,
                                                    warn_attention=w_alerts,
                                                    status_path=s_arg,
                                                    now_et_date=target_day if explicit_date else None,
                                                )
                                        finally:
                                            mod.send_dm = orig_s
                                        disk_after = (
                                            shared_status_file.read_bytes()
                                            if s_arg is not None and shared_status_file.exists()
                                            else None
                                        )
                                        pair_out[tag] = (ret, send_m.call_args_list, lrecs, disk_after)
                                    assert pair_out["base"] == pair_out["head"], (
                                        f"MISMATCH prior={prior_name} mix={mix_name} rec={recipient!r} "
                                        f"day={day_kind} exp={explicit_date} fail={send_fails}"
                                    )
                                    pairs_checked += 1
            emit(f"CONFORMING_MATRIX: {pairs_checked} base-vs-HEAD pairs matched 100% (body, ret, logs, exact bytes)")
        finally:
            base_alert.datetime = orig_base_dt
            head_alert.datetime = orig_head_dt

        # 3C: Malformed content & stable special paths across all 3 lanes x 4 modes
        mal_dir = tmp / "malformed"
        mal_dir.mkdir()
        assert sys.getrecursionlimit() == 1000
        malformed_payloads = {
            "empty_bytes": b"",
            "array": b"[]",
            "null": b"null",
            "str": b'"hello"',
            "int": b"42",
            "bool": b"true",
            "nested_20k_arr": b"[" * 20_000 + b"0" + b"]" * 20_000,
            "nested_20k_obj": b'{"a":' * 20_000 + b"0" + b"}" * 20_000,
            "nested_200k_arr": b"[" * 200_000 + b"0" + b"]" * 200_000,
            "big_5000_digits": b"9" * 5000,
            "nan": b"NaN",
            "inf": b"Infinity",
            "bad_json": b"{invalid",
            "bad_utf8": b"\xff\xfe\x80",
            "unhashable_status": b'{"status":[]}',
            "bad_crit_bool": b'{"status":"failed","critical_count":true}',
            "bad_crit_float": b'{"status":"failed","critical_count":1.5}',
            "bad_crit_str": b'{"status":"failed","critical_count":"2"}',
            "bad_sources_mixed_unhash": b'{"status":"sent","sent_day":"2027-07-14","sent_sources":[1,{}]}',
            "bad_sources_str": b'{"status":"sent","sent_day":"2027-07-14","sent_sources":"CRITICAL:probe"}',
        }
        lanes = ["health", "contest_stale_reading", "contest_morning_check"]
        modes4 = ["critical", "warn", "empty", "no_recipient"]
        mal_runs = 0
        orig_head_dt = head_alert.datetime
        head_alert.datetime = ClockDay1
        try:
            for lane in lanes:
                for p_name, raw_b in malformed_payloads.items():
                    for m_name in modes4:
                        sp = mal_dir / f"{lane}_dm_delivery_status.json"
                        sp.write_bytes(raw_b)
                        sm = Mock(return_value="ok")
                        orig_s = head_alert.send_dm
                        head_alert.send_dm = sm
                        try:
                            with bounded_alarm(2.0):
                                r = head_alert.dispatch_dm_for_health_alerts(
                                    [head_alert.Alert("CRITICAL", "probe", "c")] if m_name in ("critical", "no_recipient") else [],
                                    None if m_name == "no_recipient" else "offline.test",
                                    warn_attention=[head_alert.Alert("WARN", "probe", "w")] if m_name == "warn" else [],
                                    status_path=sp,
                                    now_et_date=DAY,
                                )
                        finally:
                            head_alert.send_dm = orig_s
                        exp_send = m_name in ("critical", "warn")
                        assert r is exp_send and sm.call_count == int(exp_send)
                        if exp_send:
                            assert "Previous health alert" not in sm.call_args.args[1]
                            assert json.loads(sp.read_bytes())["status"] == "sent"
                        elif m_name == "no_recipient":
                            assert json.loads(sp.read_bytes())["status"] == "skipped_no_recipient"
                        else:
                            assert sp.read_bytes() == raw_b
                        mal_runs += 1

            # Stable special paths across 3 call modes
            spec_dir = tmp / "special_stable"
            spec_dir.mkdir()
            stable_kinds = [
                "directory", "fifo_nopeer", "fifo_writer", "devnull_sym",
                "loop_sym", "parent_is_file", "chmod_000", "chmod_200",
                "dangling_sym", "regular_sym",
            ]
            spec_runs = 0
            for sk in stable_kinds:
                for m_name in ("critical", "empty", "no_recipient"):
                    s_case = spec_dir / f"{sk}_{m_name}"
                    s_case.mkdir()
                    sp = s_case / "status.json"
                    extra_fds: list[int] = []
                    if sk == "directory":
                        sp.mkdir()
                    elif sk == "fifo_nopeer":
                        os.mkfifo(sp)
                    elif sk == "fifo_writer":
                        os.mkfifo(sp)
                        rw = os.open(sp, os.O_RDWR | os.O_NONBLOCK)
                        os.write(rw, b'{"status":"sent","sent_day":"2027-07-14","sent_sources":["CRITICAL:probe"]}\n')
                        extra_fds.append(rw)
                    elif sk == "devnull_sym":
                        sp.symlink_to("/dev/null")
                    elif sk == "loop_sym":
                        sp.symlink_to(sp.name)
                    elif sk == "parent_is_file":
                        pf = s_case / "not_a_dir"
                        pf.write_text("file")
                        sp = pf / "status.json"
                    elif sk == "chmod_000":
                        sp.write_text('{"status":"failed","critical_count":1}')
                        sp.chmod(0o000)
                    elif sk == "chmod_200":
                        sp.write_text('{"status":"failed","critical_count":1}')
                        sp.chmod(0o200)
                    elif sk == "dangling_sym":
                        sp.symlink_to("created_target.json")
                    elif sk == "regular_sym":
                        (s_case / "real.json").write_text("{}")
                        sp.symlink_to("real.json")

                    sm = Mock(return_value="ok")
                    orig_s = head_alert.send_dm
                    head_alert.send_dm = sm
                    try:
                        with capture_alert_logs() as srecs, bounded_alarm(2.0):
                            r = head_alert.dispatch_dm_for_health_alerts(
                                [] if m_name == "empty" else [head_alert.Alert("CRITICAL", "probe", "c")],
                                None if m_name == "no_recipient" else "offline.test",
                                status_path=sp,
                                now_et_date=DAY,
                            )
                    finally:
                        head_alert.send_dm = orig_s
                        for efd in extra_fds:
                            os.close(efd)
                        if sk in ("chmod_000", "chmod_200"):
                            sp.chmod(0o600)

                    assert r is (m_name == "critical")
                    assert sm.call_count == (1 if m_name == "critical" else 0)
                    warns = [msg for lvl, msg in srecs if lvl == logging.WARNING]
                    if m_name == "empty":
                        assert len(warns) == 0
                    elif sk in ("chmod_200", "dangling_sym", "regular_sym"):
                        assert len(warns) == 0
                        assert json.loads(sp.read_bytes())["status"] == (
                            "sent" if m_name == "critical" else "skipped_no_recipient"
                        )
                    else:
                        assert len(warns) == 1 and "could not write health DM delivery status" in warns[0]
                    spec_runs += 1
        finally:
            head_alert.datetime = orig_head_dt
        emit(
            f"MALFORMED_AND_SPECIAL: {mal_runs} malformed runs (at recursionlimit=1000) + "
            f"{spec_runs} stable special-path runs OK"
        )

    emit("\nENCODING & ITEM 3 PROBE COMPLETE: ALL CHECKS PASSED")
    LOG_PATH.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
