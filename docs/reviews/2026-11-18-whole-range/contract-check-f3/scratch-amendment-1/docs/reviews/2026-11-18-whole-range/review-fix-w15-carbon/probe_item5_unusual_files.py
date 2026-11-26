"""Probe family 5: bounded child-process checks for FIFOs, directories, symlinks, sockets, non-UTF-8, oversized files, broken stderr, ².json, file raw root, unlistable season, and FD leaks."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
HEAD_SRC = REPO_ROOT / "src"


VALID = {"away": 1, "home": 2, "away_tid": 51, "home_tid": 52}


def feed_str(entry: dict = VALID) -> str:
    return json.dumps({
        "gameData": {
            "probablePitchers": {"away": {"id": entry["away"]}, "home": {"id": entry["home"]}},
            "teams": {"away": {"id": entry["away_tid"]}, "home": {"id": entry["home_tid"]}},
        }
    })


CHILD_CODE = """
import json, os, sys
from datetime import date, datetime
from pathlib import Path

from bts.features import compute, probable_pitcher_cache as ppc
from bts.health import probable_pitcher_cache as h_ppc

at = datetime.fromisoformat("2027-01-01T04:59:00+00:00")
class Frozen(datetime):
    @classmethod
    def now(cls, tz=None):
        return at.astimezone(tz) if tz is not None else at.replace(tzinfo=None)
ppc.datetime = Frozen

mode = sys.argv[1]
target = Path(sys.argv[2])

if mode == "build":
    raw = Path(sys.argv[3])
    if len(sys.argv) > 4 and sys.argv[4] == "broken_stderr":
        r_fd, w_fd = os.pipe()
        os.close(r_fd)  # closed read end -> real OS EPIPE / BrokenPipeError on write/flush!
        sys.stderr = os.fdopen(w_fd, "w", buffering=1)
    compute._probable_pitcher_cache_path = target
    out = compute._build_probable_pitcher_lookup(str(raw))
    print(json.dumps({str(k): v for k, v in out.items()}))
elif mode == "health":
    alerts = h_ppc.check(target, date(2026, 12, 31))
    print(json.dumps([[a.level, a.source, a.incident_key, a.message] for a in alerts]))
elif mode == "fd_leak":
    # Test open_regular on directory and FIFO 50 times and verify open FD count does not grow!
    from bts.probable_pitcher_status import open_regular
    d = target / "adir"
    d.mkdir()
    f = target / "afifo"
    os.mkfifo(f)
    def open_fds():
        cnt = 0
        for fd in range(3, 256):
            try:
                os.fstat(fd)
                cnt += 1
            except OSError:
                pass
        return cnt
    before = open_fds()
    for _ in range(50):
        for p in (d, f):
            try:
                with open_regular(p):
                    pass
            except OSError:
                pass
    after = open_fds()
    print(json.dumps({"before": before, "after": after}))
"""


def run_bounded(args: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(HEAD_SRC)
    return subprocess.run(
        [sys.executable, "-c", CHILD_CODE, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cache_and_raw_edges(root: Path) -> None:
    print("=== 5(a). Cache path & raw feed unusual files (bounded child, 5s timeout) ===")
    # 1. Cache path shapes: FIFO, directory, symlink->regular, symlink->FIFO, symlink->dir, broken symlink, symlink loop, unix socket, non-UTF-8, oversized 250 KiB valid & invalid
    shapes = [
        "fifo", "directory", "symlink_regular", "symlink_fifo", "symlink_dir",
        "broken_symlink", "symlink_loop", "unix_socket", "non_utf8",
        "oversized_valid", "oversized_invalid",
    ]
    for kind in shapes:
        w = root / f"c_{kind}"
        cache = w / "data/models/probable_pitcher_lookup.json"
        raw = w / "data/raw"
        (raw / "2025").mkdir(parents=True)
        cache.parent.mkdir(parents=True)
        (raw / "2025/100.json").write_text(feed_str(VALID))
        sock = None
        try:
            if kind == "fifo":
                os.mkfifo(cache)
            elif kind == "directory":
                cache.mkdir()
            elif kind == "symlink_regular":
                tgt = w / "real_cache.json"
                tgt.write_text(json.dumps({"50": VALID}))
                cache.symlink_to(tgt)
            elif kind == "symlink_fifo":
                tgt = w / "fifo_tgt"
                os.mkfifo(tgt)
                cache.symlink_to(tgt)
            elif kind == "symlink_dir":
                tgt = w / "dir_tgt"
                tgt.mkdir()
                cache.symlink_to(tgt)
            elif kind == "broken_symlink":
                cache.symlink_to(w / "missing.json")
            elif kind == "symlink_loop":
                cache.symlink_to(cache)
            elif kind == "unix_socket":
                # Keep path short for AF_UNIX sun_path limit (104 bytes on macOS)
                short_dir = Path(tempfile.mkdtemp(dir="/tmp", prefix="w15s_"))
                sock_path = short_dir / "s.sock"
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.bind(str(sock_path))
                cache.symlink_to(sock_path)
            elif kind == "non_utf8":
                cache.write_bytes(b"\xff\xfe\xfa\xfb")
            elif kind == "oversized_valid":
                big = {str(1000 + i): VALID for i in range(4000)}  # ~220 KiB > 64 KiB
                cache.write_text(json.dumps(big))
            elif kind == "oversized_invalid":
                cache.write_bytes(b"{" + b"x" * 200_000)

            res = run_bounded(["build", str(cache), str(raw)])
            assert res.returncode == 0, f"{kind}: {res.stderr}"
            parsed = json.loads(res.stdout)
            assert "100" in parsed, kind

            if kind in ("fifo", "directory", "symlink_fifo", "symlink_dir", "unix_socket"):
                # Special cache: reported unreadable_cache, NEVER preserved, NEVER replaced!
                assert not (cache.parent / "rejected").exists(), f"{kind} should not create rejected/"
                if kind == "fifo":
                    assert cache.is_fifo()
                elif kind == "directory":
                    assert cache.is_dir() and not cache.is_symlink()
                else:
                    assert cache.is_symlink()
                st = json.loads(next((w / "data/health_state/probable_pitcher_cache").glob("*.json")).read_text())
                assert st["reason"] == "unreadable_cache" and st["preserved"] is None
            elif kind in ("broken_symlink", "symlink_loop"):
                # Unknown/unverifiable symlink target: NOT replaced, reported unreadable_cache with preservation failure!
                assert cache.is_symlink(), f"{kind} must not be replaced!"
                st = json.loads(next((w / "data/health_state/probable_pitcher_cache").glob("*.json")).read_text())
                assert st["reason"] == "unreadable_cache" and st["preserved"] is None
                assert len(res.stderr.strip().splitlines()) == 2
            elif kind == "symlink_regular":
                # Verified regular target: followed for read (50 + 100), replaced atomically without mutating target!
                assert "50" in parsed and "100" in parsed
                assert not cache.is_symlink() and cache.is_file()
                assert json.loads((w / "real_cache.json").read_text()) == {"50": VALID}
                assert res.stderr == ""
            elif kind == "oversized_valid":
                assert len(parsed) == 4001 and res.stderr == ""
            elif kind in ("non_utf8", "oversized_invalid"):
                rej = list((cache.parent / "rejected").glob("*.json"))
                assert len(rej) == 1
                assert json.loads(cache.read_text()) == {"100": VALID}
        finally:
            if sock is not None:
                sock.close()

    # 2. Raw feed shapes + ².json + -100.json + regular-file raw root + FIFO raw root + unlistable season + inaccessible raw root + real OS EPIPE broken stderr
    for rkind in (
        "feed_fifo", "feed_dir", "feed_broken_symlink", "feed_symlink_fifo", "feed_non_utf8",
        "feed_oversized_bad", "superscript_two", "signed_names", "file_raw_root", "fifo_raw_root",
        "unlistable_season", "unlistable_raw_root", "broken_stderr_epipe",
    ):
        w = root / f"r_{rkind}"
        cache = w / "data/models/probable_pitcher_lookup.json"
        raw = w / "data/raw"
        cache.parent.mkdir(parents=True)
        cache.write_text(json.dumps({"100": None}))
        if rkind == "file_raw_root":
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text("not a dir")
        elif rkind == "fifo_raw_root":
            raw.parent.mkdir(parents=True, exist_ok=True)
            os.mkfifo(raw)
        else:
            (raw / "2025").mkdir(parents=True)
            f100 = raw / "2025/100.json"
            if rkind == "feed_fifo":
                os.mkfifo(f100)
            elif rkind == "feed_dir":
                f100.mkdir()
            elif rkind == "feed_broken_symlink":
                f100.symlink_to(w / "nope.json")
            elif rkind == "feed_symlink_fifo":
                ff = w / "ff"
                os.mkfifo(ff)
                f100.symlink_to(ff)
            elif rkind == "feed_non_utf8":
                f100.write_bytes(b"\xff\xfe")
            elif rkind == "feed_oversized_bad":
                f100.write_bytes(b"{" + b"a" * 200_000)
            elif rkind == "superscript_two":
                (raw / "2025/².json").write_text(feed_str(VALID))
            elif rkind == "signed_names":
                (raw / "2025/-100.json").write_text(feed_str(VALID))
                (raw / "2025/+100.json").write_text(feed_str(VALID))
            elif rkind == "unlistable_season":
                (raw / "2025").chmod(0o000)
            elif rkind == "unlistable_raw_root":
                raw.chmod(0o000)
            elif rkind == "broken_stderr_epipe":
                f100.write_text(feed_str(VALID))

        try:
            extra = ["broken_stderr"] if rkind == "broken_stderr_epipe" else []
            res = run_bounded(["build", str(cache), str(raw), *extra])
            assert res.returncode == 0, f"{rkind}: {res.stderr}"
            st = json.loads(next((w / "data/health_state/probable_pitcher_cache").glob("*.json")).read_text())
            if rkind in ("feed_fifo", "feed_dir", "feed_broken_symlink", "feed_symlink_fifo", "feed_non_utf8", "feed_oversized_bad", "unlistable_season", "unlistable_raw_root"):
                if (rkind == "unlistable_season" and os.access(raw / "2025", os.R_OK)) or (rkind == "unlistable_raw_root" and os.access(raw, os.R_OK)):
                    pass
                else:
                    assert st["unusable_feed"] == 1 and st["without_feed"] == 0, (rkind, st)
            elif rkind in ("superscript_two", "signed_names", "file_raw_root", "fifo_raw_root"):
                assert st["without_feed"] == 1 and st["unusable_feed"] == 0, (rkind, st)
            elif rkind == "broken_stderr_epipe":
                assert json.loads(res.stdout) == {"100": VALID}
                assert st["rescanned"] == 1
        finally:
            if rkind == "unlistable_season":
                (raw / "2025").chmod(0o700)
            elif rkind == "unlistable_raw_root":
                raw.chmod(0o700)
    print("PASS: All cache and raw feed unusual file types bounded and classified per contract.")


def test_rejected_status_health_and_fd_leak(root: Path) -> None:
    print("=== 5(b). rejected/, status dir, health records (bounded), and FD leak check ===")
    # 1. FD leak check on open_regular(dir) and open_regular(fifo)
    fd_dir = root / "fd_check"
    fd_dir.mkdir()
    res_fd = run_bounded(["fd_leak", str(fd_dir)])
    assert res_fd.returncode == 0, res_fd.stderr
    fd_counts = json.loads(res_fd.stdout)
    assert fd_counts["before"] == fd_counts["after"], f"FD LEAK DETECTED: {fd_counts}"
    print(f"  FD leak check OK: open FDs before={fd_counts['before']}, after={fd_counts['after']} (50 dir + 50 FIFO opens)")

    # 2. rejected/ as FIFO, file, symlink-to-dir; health_state/probable_pitcher_cache as FIFO, file, symlink-to-dir, chmod 000
    for artifact in ("rejected", "status"):
        for kind in ("fifo", "file", "symlink_dir"):
            w = root / f"art_{artifact}_{kind}"
            cache = w / "data/models/probable_pitcher_lookup.json"
            raw = w / "data/raw"
            (raw / "2025").mkdir(parents=True)
            cache.parent.mkdir(parents=True)
            cache.write_text('{"100": null}')
            (raw / "2025/100.json").write_text(feed_str(VALID))
            folder = cache.parent / "rejected" if artifact == "rejected" else w / "data/health_state/probable_pitcher_cache"
            folder.parent.mkdir(parents=True, exist_ok=True)
            if kind == "fifo":
                os.mkfifo(folder)
            elif kind == "file":
                folder.write_text("blocker")
            elif kind == "symlink_dir":
                real_d = w / "real_dir"
                real_d.mkdir()
                folder.symlink_to(real_d)

            res = run_bounded(["build", str(cache), str(raw)])
            assert res.returncode == 0, (artifact, kind, res.stderr)
            assert json.loads(res.stdout) == {"100": VALID}
            assert cache.read_text() == '{"100": null}', "active cache must stay untouched"
            assert len(res.stderr.strip().splitlines()) == 2

    # 3. Health check bounded on FIFO, directory, symlink, broken symlink, chmod 000 record, chmod 000 status dir
    for hkind in ("fifo_rec", "dir_rec", "sym_rec", "broken_sym_rec", "unreadable_rec", "fifo_sdir", "sym_sdir", "unreadable_sdir"):
        w = root / f"h_{hkind}"
        sdir = w / "probable_pitcher_cache"
        fname = "2026-12-31.invalid_entries.0123456789ab.json"
        try:
            if hkind == "fifo_sdir":
                os.mkfifo(sdir)
            elif hkind == "sym_sdir":
                tgt = w / "other"
                tgt.mkdir()
                sdir.symlink_to(tgt)
            elif hkind == "unreadable_sdir":
                sdir.mkdir()
                sdir.chmod(0o000)
            else:
                sdir.mkdir()
                rec = sdir / fname
                if hkind == "fifo_rec":
                    os.mkfifo(rec)
                elif hkind == "dir_rec":
                    rec.mkdir()
                elif hkind == "sym_rec":
                    tgt = w / "tgt.json"
                    tgt.write_text("{}")
                    rec.symlink_to(tgt)
                elif hkind == "broken_sym_rec":
                    rec.symlink_to(w / "missing.json")
                elif hkind == "unreadable_rec":
                    rec.write_text("{}")
                    rec.chmod(0o000)
            res = run_bounded(["health", str(sdir)])
            assert res.returncode == 0, (hkind, res.stderr)
            alerts = json.loads(res.stdout)
            assert len(alerts) == 1
            assert alerts[0][0] == "WARN" and "status_unreadable" in alerts[0][2]
        finally:
            if hkind == "unreadable_sdir":
                sdir.chmod(0o700)
            elif hkind == "unreadable_rec":
                (sdir / fname).chmod(0o600)
    print("PASS: All rejected/, status dir, and health check unusual file types bounded and non-blocking.")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="probe5_") as tmp:
        root = Path(tmp)
        test_cache_and_raw_edges(root)
        test_rejected_status_health_and_fd_leak(root)
    print("ALL PROBE 5 CHECKS PASSED")


if __name__ == "__main__":
    main()
