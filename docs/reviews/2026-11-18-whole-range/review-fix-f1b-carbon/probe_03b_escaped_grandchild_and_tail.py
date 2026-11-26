"""Probe 03b: Escaped grandchild lock retention, lock-path edge cases, 23:45/NYE/DST, and malformed markers."""
from __future__ import annotations

from datetime import date, datetime
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
TIMEOUT_PATH = "/usr/bin/timeout"
FLOCK_PATH = "/usr/bin/flock"
LOCK_DEFAULT = "/home/bts/.check-pick-entered.lock"
DAY = "2027-03-24"
PITCH = f"{DAY}T19:40:00-04:00"


def _kill_pids(pids: set[int], sig: int = signal.SIGKILL) -> None:
    for pid in list(pids):
        if pid <= 1 or pid == os.getpid():
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _flock_main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[0] != "-n":
        sys.stderr.write("flock: bad args\n")
        return 64
    lock_path, cmd = argv[1], argv[2:]
    try:
        fd = os.open(lock_path, os.O_RDONLY | os.O_NOCTTY | os.O_CREAT, 0o666)
    except OSError as exc:
        if exc.errno == errno.EISDIR:
            try:
                fd = os.open(lock_path, os.O_RDONLY | os.O_NOCTTY)
            except OSError as exc2:
                sys.stderr.write(f"flock: cannot open lock file {lock_path}: {exc2.strerror}\n")
                return 66 if exc2.errno == errno.ENOENT else 71
        else:
            sys.stderr.write(f"flock: cannot open lock file {lock_path}: {exc.strerror}\n")
            return 66 if exc.errno == errno.ENOENT else 65
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 1
    os.set_inheritable(fd, True)
    os.environ["PROBE03B_LOCK_FD"] = str(fd)
    os.execvp(cmd[0], cmd)


def _child_main(argv: list[str]) -> int:
    mode = argv[0]
    if mode == "spawn_escaped_grandchild":
        pid_file = Path(argv[1])
        ready_file = Path(argv[2])
        fd = int(os.environ["PROBE03B_LOCK_FD"])
        gc = subprocess.Popen(
            [sys.executable, __file__, "--child", "grandchild_hold", str(ready_file)],
            start_new_session=True,
            pass_fds=(fd,),
        )
        pid_file.write_text(str(gc.pid))
        # Monitored parent stays in timeout's group with default SIGTERM disposition, so timeout exits 124!
        time.sleep(10.0)
        return 0

    if mode == "grandchild_hold":
        Path(argv[1]).write_text(str(os.getpid()))
        time.sleep(10.0)
        return 0

    if mode == "hold_until_file":
        ready_file, gate_file = Path(argv[1]), Path(argv[2])
        ready_file.write_text(str(os.getpid()))
        deadline = time.monotonic() + 8.0
        while not gate_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        return 0

    if mode == "simple_ok":
        Path(argv[1]).write_text(str(os.getpid()))
        return 0

    raise AssertionError(f"unknown mode {mode}")


def check_escaped_grandchild_and_lock_edges(pids: set[int]) -> dict:
    if str(OUT_DIR) not in sys.path:
        sys.path.insert(0, str(OUT_DIR))
    from probe_02_k1_to_k5_and_wrapper import generate_enter_cron_line

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        line, _, cron_log = generate_enter_cron_line(root / "inst")
        cron_log.touch()
        shell_tmpl = line.split(maxsplit=5)[5]
        flock_cmd = [sys.executable, __file__, "--flock"]
        base_env = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO}", "TZ": "America/New_York"}

        def make_cmd(lock_p: Path, inner: list[str], dur: str = "0.7") -> str:
            s = (
                shell_tmpl
                .replace(FLOCK_PATH, shlex.join(flock_cmd), 1)
                .replace(LOCK_DEFAULT, shlex.quote(str(lock_p)), 1)
                .replace(f"{TIMEOUT_PATH} --verbose -k 30 840", f"{TIMEOUT_PATH} --verbose -k 0.2 {dur}", 1)
            )
            return re.sub(r"\S+/uv run bts check-pick-entered.*?(?=>>)", lambda _: shlex.join(inner) + " ", s, count=1)

        # 1. Monitored child spawns a setsid() grandchild inheriting lock_fd, then monitored child dies on TERM (exit 124)
        esc_lock = root / "esc.lock"
        gc_pid_file = root / "gc.pid"
        gc_ready_file = root / "gc.ready"
        p_esc = subprocess.Popen(
            ["/bin/sh", "-c", make_cmd(esc_lock, [sys.executable, __file__, "--child", "spawn_escaped_grandchild", str(gc_pid_file), str(gc_ready_file)], dur="0.7")],
            env=base_env,
        )
        pids.add(p_esc.pid)
        rc_esc = p_esc.wait(timeout=6.0)
        if gc_pid_file.exists():
            pids.add(int(gc_pid_file.read_text()))
        assert rc_esc == 124, f"expected 124 when monitored parent died on TERM, got {rc_esc}"
        assert gc_ready_file.exists()
        gc_pid = int(gc_pid_file.read_text())

        # Contender exits 1 without running (so no two writers exist!)
        not_run = root / "not_run"
        p_cont = subprocess.Popen(
            ["/bin/sh", "-c", make_cmd(esc_lock, [sys.executable, __file__, "--child", "simple_ok", str(not_run)], dur="2.0")],
            env=base_env,
        )
        pids.add(p_cont.pid)
        assert p_cont.wait(timeout=4.0) == 1
        assert not not_run.exists()

        # Kill escaped grandchild -> lock freed -> next run succeeds
        _kill_pids({gc_pid})
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with esc_lock.open("r") as lf:
                try:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                    break
                except BlockingIOError:
                    time.sleep(0.02)

        p_rec = subprocess.Popen(
            ["/bin/sh", "-c", make_cmd(esc_lock, [sys.executable, __file__, "--child", "simple_ok", str(not_run)], dur="2.0")],
            env=base_env,
        )
        pids.add(p_rec.pid)
        assert p_rec.wait(timeout=4.0) == 0
        assert not_run.exists()

        # 2. Unlinking the lock inode while a holder is alive allows a second holder on the new inode (Amendment 5 limit)
        unl_lock = root / "unlinked.lock"
        h_ready, h_gate = root / "h_ready", root / "h_gate"
        p_h1 = subprocess.Popen(
            ["/bin/sh", "-c", make_cmd(unl_lock, [sys.executable, __file__, "--child", "hold_until_file", str(h_ready), str(h_gate)], dur="6.0")],
            env=base_env,
        )
        pids.add(p_h1.pid)
        deadline = time.monotonic() + 5.0
        while not h_ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert h_ready.exists()
        pids.add(int(h_ready.read_text()))

        unl_lock.unlink()  # breaks inode continuity!
        second_ran = root / "second_ran_after_unlink"
        rc_h2 = subprocess.run(
            ["/bin/sh", "-c", make_cmd(unl_lock, [sys.executable, __file__, "--child", "simple_ok", str(second_ran)], dur="2.0")],
            env=base_env,
            timeout=5,
        ).returncode
        h_gate.write_text("done")
        p_h1.wait(timeout=5.0)
        assert rc_h2 == 0 and second_ran.exists()

        # 3. Stale regular file, 0444 read-only file, directory lock, missing parent, 0000 unreadable file
        stale_lock = root / "stale.lock"
        stale_lock.write_text("leftover")
        m_stale = root / "ran_stale"
        assert subprocess.run(["/bin/sh", "-c", make_cmd(stale_lock, [sys.executable, __file__, "--child", "simple_ok", str(m_stale)])], env=base_env, timeout=5).returncode == 0
        assert m_stale.exists()

        ro_lock = root / "readonly.lock"
        ro_lock.write_text("")
        ro_lock.chmod(0o444)
        m_ro = root / "ran_ro"
        assert subprocess.run(["/bin/sh", "-c", make_cmd(ro_lock, [sys.executable, __file__, "--child", "simple_ok", str(m_ro)])], env=base_env, timeout=5).returncode == 0
        assert m_ro.exists()

        dir_lock = root / "dir.lock"
        dir_lock.mkdir()
        m_dir = root / "ran_dir"
        assert subprocess.run(["/bin/sh", "-c", make_cmd(dir_lock, [sys.executable, __file__, "--child", "simple_ok", str(m_dir)])], env=base_env, timeout=5).returncode == 0
        assert m_dir.exists()

        cron_log.write_text("")
        bad_lock = root / "missing_parent" / "watchdog.lock"
        m_bad = root / "ran_bad"
        rc_bad = subprocess.run(["/bin/sh", "-c", make_cmd(bad_lock, [sys.executable, __file__, "--child", "simple_ok", str(m_bad)])], env=base_env, timeout=5).returncode
        assert rc_bad != 0 and not m_bad.exists()
        assert "cannot open lock file" in cron_log.read_text()

        if os.geteuid() != 0:
            cron_log.write_text("")
            zero_lock = root / "zero.lock"
            zero_lock.write_text("")
            zero_lock.chmod(0o000)
            try:
                m_zero = root / "ran_zero"
                rc_zero = subprocess.run(["/bin/sh", "-c", make_cmd(zero_lock, [sys.executable, __file__, "--child", "simple_ok", str(m_zero)])], env=base_env, timeout=5).returncode
                assert rc_zero != 0 and not m_zero.exists()
                assert "cannot open lock file" in cron_log.read_text()
            finally:
                zero_lock.chmod(0o600)

    return {
        "escaped_grandchild_holds_lock_with_exit_124_until_reaped": True,
        "unlinked_inode_allows_concurrent_run_matching_amendment5_limit": True,
        "stale_readonly_and_dir_locks_succeed": True,
        "unopenable_locks_log_to_cron_log_and_skip": True,
    }


def check_2345_midnight_nye_dst_and_malformed_markers() -> dict:
    sys.path.insert(0, str(REPO))
    from tests.test_wr_f1_wrong_game import MATCH, WRONG, World, row

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        w_d1 = World(root / "midnight", "2027-03-24")
        w_d1.commit(pitch="2027-03-25T00:30:00-04:00")
        w_d1.capture([WRONG], "20270324T210000Z")
        rounds = {1: date(2027, 3, 24), 2: date(2027, 3, 25)}
        r_2345 = w_d1.check("23:45", MATCH, rounds=rounds, elapsed=960)
        assert r_2345["exit"] == 1 and r_2345["marker"]["date"] == "2027-03-24"
        assert r_2345["marker"]["game_unresolved"] is True

        w_d2 = World(root / "midnight", "2027-03-25")
        w_d2.commit(pitch="2027-03-25T19:40:00-04:00")
        w_d2.capture([{"id": 78, "roundId": 2, "feedId": None}], "20270325T210000Z")
        r_d2 = w_d2.check("18:30", [row(unit=78, rnd=2)], rounds=rounds)
        assert r_d2["exit"] == 0 and r_d2["marker"]["status"] == "confirmed"
        assert "game_unresolved" not in r_d2["marker"]

        w_nye = World(root / "nye", "2027-12-31")
        w_nye.commit(pitch="2028-01-01T00:40:00-05:00")
        w_nye.capture([WRONG], "20280101T040000Z")
        r_nye = w_nye.check("23:45", MATCH, rounds={1: date(2027, 12, 31)})
        assert r_nye["exit"] == 0 and r_nye["marker"]["status"] == "confirmed"

        et = ZoneInfo("America/New_York")
        for dst_day in (date(2027, 3, 14), date(2027, 11, 7)):
            instants = [
                datetime(dst_day.year, dst_day.month, dst_day.day, hr, mn, tzinfo=et).timestamp()
                for hr in range(10, 24)
                for mn in (0, 15, 30, 45)
            ]
            assert len(instants) == 56
            assert all((b - a) == 900.0 for a, b in zip(instants, instants[1:]))

        malformed_outcomes = {}
        for name, raw_bytes in (
            ("json_list", b"[]"),
            ("json_null", b"null"),
            ("bad_utf8", b"\xff\xfe"),
            ("non_iter_escalations", json.dumps({"date": DAY, "status": "alerted", "escalations": 42}).encode()),
        ):
            wm = World(root / f"mal_{name}")
            wm.commit(pitch=PITCH)
            wm.marker.parent.mkdir(parents=True, exist_ok=True)
            wm.marker.write_bytes(raw_bytes)
            res = wm.check("18:30", MATCH)
            assert res["exception"] is not None
            malformed_outcomes[name] = res["exception"]

    return {
        "run_2345_crossing_midnight_isolated_from_next_day": True,
        "nye_utc_year_limit_verified": True,
        "dst_56_slots_all_900s_apart": True,
        "inherited_malformed_marker_exceptions": malformed_outcomes,
    }


def run_all() -> dict:
    pids: set[int] = set()
    try:
        summary = {
            "escaped_and_lock_edges": check_escaped_grandchild_and_lock_edges(pids),
            "midnight_dst_and_markers": check_2345_midnight_nye_dst_and_malformed_markers(),
        }
    finally:
        _kill_pids(pids)
    (OUT_DIR / "log_03b_escaped_grandchild_and_tail.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def test_probe_03b():
    res = run_all()
    assert res["escaped_and_lock_edges"]["escaped_grandchild_holds_lock_with_exit_124_until_reaped"] is True


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--flock":
        raise SystemExit(_flock_main(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        raise SystemExit(_child_main(sys.argv[2:]))
    print(json.dumps(run_all(), indent=2))
