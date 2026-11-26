"""Probe 03: Hunt the two-writers / lost-observation class under serialized cron wrapper + standing rows."""
from __future__ import annotations

from datetime import date, datetime, timedelta
import errno
import fcntl
import gzip
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
    """Util-linux flock(1) faithful model: O_RDONLY|O_NOCTTY|O_CREAT, EISDIR->O_RDONLY fallback."""
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
    os.environ["PROBE03_LOCK_FD"] = str(fd)
    os.execvp(cmd[0], cmd)


def _child_main(argv: list[str]) -> int:
    sys.path.insert(0, str(REPO))
    mode = argv[0]
    if mode == "timeout_stage":
        stage = argv[1]
        w_root = Path(argv[2])
        from unittest.mock import patch
        from bts import receipt_io
        from bts.cli import _atomic_write_json
        from tests.test_wr_f1_wrong_game import MATCH, World

        w = World(w_root)
        real_fdopen, real_replace = os.fdopen, os.replace

        def hang_here(lbl: str):
            if stage == lbl:
                (w_root / f"reached_{lbl}").write_text(str(os.getpid()))
                time.sleep(10.0)

        def send_dm(_, msg):
            hang_here("during_dm")
            (w_root / "dm_accepted.txt").write_text(msg)
            hang_here("between_dm_and_mark")

        class HalfWriter:
            def __init__(self, stream, lbl):
                self.stream, self.lbl = stream, lbl

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self.stream.close()

            def write(self, data):
                self.stream.write(data[: max(1, len(data) // 2)])
                self.stream.flush()
                hang_here(self.lbl)

        def wrapped_atomic(path, obj):
            if stage == "during_mark_tmp":
                with patch("os.fdopen", lambda *a, **k: HalfWriter(real_fdopen(*a, **k), "during_mark_tmp")):
                    return _atomic_write_json(path, obj)
            return _atomic_write_json(path, obj)

        def wrapped_replace(src, dst):
            real_replace(src, dst)
            p = Path(dst)
            if "pick_entry_receipts" in p.parts:
                hang_here("after_receipt_seal" if p.suffix == ".sealed" else "between_receipt_rename_and_seal")

        w.check("18:45", MATCH, more_patches=[
            ("bts.leaderboard.auth.load_session_cookies", lambda: hang_here("before_fetch") or {}),
            ("bts.contest_fetch.fetch_pending_predictions", lambda *_: hang_here("during_fetch") or MATCH),
            ("bts.dm.send_dm", send_dm),
            ("bts.cli._atomic_write_json", wrapped_atomic),
            ("os.replace", wrapped_replace),
        ])
        return 0

    if mode == "hang_fifo_or_sleep":
        w_root = Path(argv[1])
        sub = argv[2]
        (w_root / f"slot_{sub}_pid").write_text(str(os.getpid()))
        from tests.test_wr_f1_wrong_game import MATCH, World
        w = World(w_root)
        res = w.check("18:45", MATCH)
        (w_root / f"slot_{sub}_done").write_text(str(res["exit"]))
        return res["exit"]

    if mode == "escaped_session":
        rec_file = Path(argv[1])
        os.setsid()
        rec_file.write_text(str(os.getpid()))
        time.sleep(10.0)
        return 0

    if mode == "simple_ok":
        Path(argv[1]).write_text(str(os.getpid()))
        return 0

    raise AssertionError(f"unknown mode {mode}")


def _generate_watchdog_shell(tmp_root: Path) -> tuple[str, Path]:
    if str(OUT_DIR) not in sys.path:
        sys.path.insert(0, str(OUT_DIR))
    from probe_02_k1_to_k5_and_wrapper import generate_enter_cron_line
    line, _, cron_log = generate_enter_cron_line(tmp_root)
    cron_log.touch()
    return line.split(maxsplit=5)[5], cron_log


def hunt_timeout_after_recorded_observation(pids: set[int]) -> dict:
    """Once game_unresolved is on disk (18:30), verify a timeout kill at ANY step of 18:45 NEVER loses it."""
    sys.path.insert(0, str(REPO))
    from bts import entry_receipt
    from tests.test_wr_f1_wrong_game import MATCH, RIGHT, WRONG, World

    stages = [
        "before_fetch",
        "during_fetch",
        "during_dm",
        "between_dm_and_mark",
        "during_mark_tmp",
        "between_receipt_rename_and_seal",
        "after_receipt_seal",
    ]
    checked = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shell_tmpl, cron_log = _generate_watchdog_shell(root / "inst")
        lock_file = root / "watchdog.lock"
        flock_cmd = [sys.executable, __file__, "--flock"]
        base_env = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO}", "TZ": "America/New_York"}

        for stage in stages:
            w_root = root / f"stage_{stage}"
            w = World(w_root)
            w.commit(pitch=PITCH)
            w.capture([WRONG])
            # 18:30 records wrong_game (even if DM failed!)
            r_first = w.check("18:30", MATCH, dm="failure")
            assert r_first["marker"]["game_unresolved"] is True
            assert r_first["marker"]["status"] == "dm_failed"
            recs_before = len(entry_receipt.discover(w.picks, DAY))

            inner = [sys.executable, __file__, "--child", "timeout_stage", stage, str(w_root)]
            cmd_str = (
                shell_tmpl
                .replace(FLOCK_PATH, shlex.join(flock_cmd), 1)
                .replace(LOCK_DEFAULT, shlex.quote(str(lock_file)), 1)
                .replace(f"{TIMEOUT_PATH} --verbose -k 30 840", f"{TIMEOUT_PATH} --verbose -k 0.2 0.8", 1)
            )
            cmd_str = re.sub(r"\S+/uv run bts check-pick-entered.*?(?=>>)", lambda _: shlex.join(inner) + " ", cmd_str, count=1)

            proc = subprocess.Popen(["/bin/sh", "-c", cmd_str], env=base_env)
            pids.add(proc.pid)
            rc = proc.wait(timeout=6.0)
            reached_file = w_root / f"reached_{stage}"
            if reached_file.exists():
                pids.add(int(reached_file.read_text()))
            assert rc == 124, f"stage={stage} expected 124, got {rc}"

            # Marker must still be valid JSON and still have game_unresolved == True!
            m_after = w.read_marker()
            assert m_after is not None and m_after.get("game_unresolved") is True, f"lost on {stage}: {m_after}"

            # Discoverable receipts: only after_receipt_seal adds 1 sealed receipt
            recs_after = entry_receipt.discover(w.picks, DAY)
            assert len(recs_after) == recs_before + (1 if stage == "after_receipt_seal" else 0)

            # Now add a conflicting capture and run 19:00: must STILL refuse to confirm!
            w.capture([RIGHT], f"{DAY}T230000Z")
            r_next = w.check("19:00", MATCH)
            assert r_next["exit"] == 1
            assert r_next["marker"]["status"] == "alerted"
            assert r_next["marker"]["game_unresolved"] is True
            checked.append(stage)

    return {"stages_preserving_prior_observation": checked}


def check_stated_first_observation_timeout_loss_and_trickle(pids: set[int]) -> dict:
    """Verify the two contract-stated limit cases from timeout:
    (1) First wrong-game run timed out during DM send before mark() loses unrecorded observation;
    (2) Response trickling past 840s with <15m to cutoff misses last alert window where base sends.
    Also check trickle <840s crossing cutoff in-process."""
    sys.path.insert(0, str(REPO))
    from tests.test_wr_f1_wrong_game import MATCH, RIGHT, WRONG, World

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shell_tmpl, _ = _generate_watchdog_shell(root / "inst")
        lock_file = root / "watchdog.lock"
        flock_cmd = [sys.executable, __file__, "--flock"]
        base_env = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO}", "TZ": "America/New_York"}

        # (1) Stated limit: 18:15 initial no_pick -> 18:30 first wrong_game hangs in DM send -> 18:45 conflicting capture confirms
        w1 = World(root / "loss1")
        w1.commit(pitch=PITCH)
        w1.capture([WRONG])
        assert w1.check("18:25")["marker"]["escalations"] == ["initial"]

        inner = [sys.executable, __file__, "--child", "timeout_stage", "during_dm", str(w1.root)]
        cmd_str = (
            shell_tmpl
            .replace(FLOCK_PATH, shlex.join(flock_cmd), 1)
            .replace(LOCK_DEFAULT, shlex.quote(str(lock_file)), 1)
            .replace(f"{TIMEOUT_PATH} --verbose -k 30 840", f"{TIMEOUT_PATH} --verbose -k 0.2 0.7", 1)
        )
        cmd_str = re.sub(r"\S+/uv run bts check-pick-entered.*?(?=>>)", lambda _: shlex.join(inner) + " ", cmd_str, count=1)
        proc = subprocess.Popen(["/bin/sh", "-c", cmd_str], env=base_env)
        pids.add(proc.pid)
        assert proc.wait(timeout=6.0) == 124
        if (w1.root / "reached_during_dm").exists():
            pids.add(int((w1.root / "reached_during_dm").read_text()))

        assert "game_unresolved" not in w1.read_marker()
        w1.capture([RIGHT], f"{DAY}T230000Z")
        r1_next = w1.check("18:45", MATCH)
        stated_loss_reproduced = (r1_next["exit"] == 0 and r1_next["marker"]["status"] == "confirmed")
        assert stated_loss_reproduced

        # (2) Trickle <840s crossing cutoff during check: preserves game_unresolved, skips DM, prints 'during this check'
        w2 = World(root / "trickle_short")
        w2.commit(pitch=PITCH)  # 19:40 pitch -> 19:35 cutoff
        w2.capture([WRONG])
        r_trickle = w2.check("19:30", MATCH, elapsed=360)  # 6 min elapsed -> 19:36 > 19:35 cutoff
        assert r_trickle["exit"] == 1
        assert r_trickle["dm"] == []
        assert r_trickle["marker"]["game_unresolved"] is True
        assert "during this check" in r_trickle["stdout"]
        assert r_trickle["receipts"][0]["observation"]["before_cutoff"] is False

    return {
        "stated_first_obs_timeout_loss_verified": stated_loss_reproduced,
        "trickle_past_cutoff_under_840s_preserves_wg": True,
    }


def check_two_consecutive_hung_runs_and_lock_edges(pids: set[int]) -> dict:
    """Check:
    - two consecutive hung runs (persistent capture FIFO) bounded by timeout, Slot 3 succeeds once FIFO replaced;
    - escaped session child holding lock blocks next slot cleanly (exit 1, no two writers) until child exits;
    - stale lock file, read-only 0444 lock file, directory lock path, unwritable/unopenable lock path, unlinked lock inode;
    - unmarked vs marked legacy cron lines across install.
    """
    sys.path.insert(0, str(REPO))
    from tests.test_wr_f1_wrong_game import WRONG, World

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shell_tmpl, cron_log = _generate_watchdog_shell(root / "inst")
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

        # 1. Two consecutive hung runs via capture FIFO, then recovery on Slot 3
        w_fifo = World(root / "fifo_world")
        w_fifo.commit(pitch=PITCH)
        fifo_path = w_fifo.units / f"{DAY}T210000Z.json.gz"
        os.mkfifo(fifo_path)
        lock1 = root / "fifo.lock"

        for slot_name in ("s1", "s2"):
            p = subprocess.Popen(
                ["/bin/sh", "-c", make_cmd(lock1, [sys.executable, __file__, "--child", "hang_fifo_or_sleep", str(w_fifo.root), slot_name], dur="0.7")],
                env=base_env,
            )
            pids.add(p.pid)
            rc = p.wait(timeout=6.0)
            if (w_fifo.root / f"slot_{slot_name}_pid").exists():
                pids.add(int((w_fifo.root / f"slot_{slot_name}_pid").read_text()))
            assert rc == 124, f"{slot_name} expected 124, got {rc}"
            assert not (w_fifo.root / f"slot_{slot_name}_done").exists()

        # Replace FIFO with real capture -> Slot 3 acquires lock and detects wrong_game (exit 1)!
        fifo_path.unlink()
        w_fifo.capture([WRONG])
        p3 = subprocess.Popen(
            ["/bin/sh", "-c", make_cmd(lock1, [sys.executable, __file__, "--child", "hang_fifo_or_sleep", str(w_fifo.root), "s3"], dur="5.0")],
            env=base_env,
        )
        pids.add(p3.pid)
        assert p3.wait(timeout=8.0) == 1
        assert (w_fifo.root / "slot_s3_done").read_text() == "1"
        assert w_fifo.read_marker()["game_unresolved"] is True

        # 2. Escaped descendant (setsid) retaining lock fd after wrapper returns 124
        esc_lock = root / "esc.lock"
        esc_pid_file = root / "esc.pid"
        p_esc = subprocess.Popen(
            ["/bin/sh", "-c", make_cmd(esc_lock, [sys.executable, __file__, "--child", "escaped_session", str(esc_pid_file)], dur="0.6")],
            env=base_env,
        )
        pids.add(p_esc.pid)
        assert p_esc.wait(timeout=6.0) == 124
        esc_pid = int(esc_pid_file.read_text())
        pids.add(esc_pid)

        # Next slot sees lock held -> exits 1 without starting child!
        ok_marker = root / "should_not_run"
        p_cont = subprocess.Popen(
            ["/bin/sh", "-c", make_cmd(esc_lock, [sys.executable, __file__, "--child", "simple_ok", str(ok_marker)], dur="2.0")],
            env=base_env,
        )
        pids.add(p_cont.pid)
        assert p_cont.wait(timeout=4.0) == 1
        assert not ok_marker.exists()

        # Kill escaped child -> lock released -> next slot succeeds!
        _kill_pids({esc_pid})
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                os.kill(esc_pid, 0)
                time.sleep(0.02)
            except ProcessLookupError:
                break
        p_rec = subprocess.Popen(
            ["/bin/sh", "-c", make_cmd(esc_lock, [sys.executable, __file__, "--child", "simple_ok", str(ok_marker)], dur="2.0")],
            env=base_env,
        )
        pids.add(p_rec.pid)
        assert p_rec.wait(timeout=4.0) == 0
        assert ok_marker.exists()

        # 3. Stale lock file, mode-0444 read-only lock file, directory lock path, missing parent, mode-0000 lock file
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

        # Missing parent directory -> open error logged to cron.log, command does not start
        cron_log.write_text("")
        bad_lock = root / "missing_parent" / "watchdog.lock"
        m_bad = root / "ran_bad"
        rc_bad = subprocess.run(["/bin/sh", "-c", make_cmd(bad_lock, [sys.executable, __file__, "--child", "simple_ok", str(m_bad)])], env=base_env, timeout=5).returncode
        assert rc_bad != 0
        assert not m_bad.exists()
        assert "cannot open lock file" in cron_log.read_text()

        # Mode-0000 unreadable lock file -> open error logged, command does not start (when non-root)
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
        "two_consecutive_hung_runs_recover_on_slot_3": True,
        "escaped_child_prevents_second_writer_until_reaped": True,
        "stale_readonly_and_dir_locks_succeed_under_util_linux_flags": True,
        "unopenable_lock_logs_to_cron_log_and_skips_cli": True,
    }


def check_2345_midnight_nye_dst_and_malformed_markers() -> dict:
    """Check 23:45 run crossing midnight, NYE UTC capture year limit, DST slot spacing, and malformed markers."""
    sys.path.insert(0, str(REPO))
    from tests.test_wr_f1_wrong_game import MATCH, RIGHT, WRONG, World, row

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # 1. 23:45 run on Day 1 (for a 00:30 ET nightcap on Day 2's clock) crossing midnight before finishing:
        # Day 1 marker has date=2027-03-24; Day 2 check at 18:30 ignores Day 1's game_unresolved!
        w_d1 = World(root / "midnight", "2027-03-24")
        w_d1.commit(pitch="2027-03-25T00:30:00-04:00")
        w_d1.capture([WRONG], "20270324T210000Z")
        rounds = {1: date(2027, 3, 24), 2: date(2027, 3, 25)}
        # Run starts at 23:45, takes 960s (delayed/long) so wall clock is 00:01 next day, still < 00:25 cutoff
        r_2345 = w_d1.check("23:45", MATCH, rounds=rounds, elapsed=960)
        assert r_2345["exit"] == 1
        assert r_2345["marker"]["date"] == "2027-03-24"
        assert r_2345["marker"]["game_unresolved"] is True

        # Next day (2027-03-25) on same root: unbound unit confirms normally because yesterday's marker is ignored!
        w_d2 = World(root / "midnight", "2027-03-25")
        w_d2.commit(pitch="2027-03-25T19:40:00-04:00")
        w_d2.capture([{"id": 78, "roundId": 2, "feedId": None}], "20270325T210000Z")
        r_d2 = w_d2.check("18:30", [row(unit=78, rnd=2)], rounds=rounds)
        assert r_d2["exit"] == 0
        assert r_d2["marker"]["date"] == "2027-03-25"
        assert r_d2["marker"]["status"] == "confirmed"
        assert "game_unresolved" not in r_d2["marker"]

        # 2. Dec 31 ET 23:45 vs Jan 1 UTC capture filename (stated limit A6.6):
        w_nye = World(root / "nye", "2027-12-31")
        w_nye.commit(pitch="2028-01-01T00:40:00-05:00")
        # Only capture is named 20280101T040000Z.json.gz (UTC year 2028), while target.year is 2027!
        w_nye.capture([WRONG], "20280101T040000Z")
        r_nye = w_nye.check("23:45", MATCH, rounds={1: date(2027, 12, 31)})
        assert r_nye["exit"] == 0 and r_nye["marker"]["status"] == "confirmed"  # stated limit A6.6

        # 3. DST spring-forward (2027-03-14) and fall-back (2027-11-07) in America/New_York for */15 10-23:
        et = ZoneInfo("America/New_York")
        for dst_day in (date(2027, 3, 14), date(2027, 11, 7)):
            instants = []
            for hr in range(10, 24):
                for mn in (0, 15, 30, 45):
                    dt_et = datetime(dst_day.year, dst_day.month, dst_day.day, hr, mn, tzinfo=et)
                    instants.append(dt_et.timestamp())
            assert len(instants) == 56
            gaps = [b - a for a, b in zip(instants, instants[1:])]
            assert all(g == 900.0 for g in gaps), gaps

        # 4. Inherited malformed marker shapes (A5.4): [], null, invalid UTF-8, non-iterable escalations
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
            assert res["exception"] is not None  # raises as on base (A5.4)
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
            "timeout_after_prior_observation": hunt_timeout_after_recorded_observation(pids),
            "first_observation_loss_and_trickle": check_stated_first_observation_timeout_loss_and_trickle(pids),
            "consecutive_hangs_and_lock_edges": check_two_consecutive_hung_runs_and_lock_edges(pids),
            "midnight_dst_and_markers": check_2345_midnight_nye_dst_and_malformed_markers(),
        }
    finally:
        _kill_pids(pids)
    (OUT_DIR / "log_03_class_hunt_and_standing.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def test_probe_03():
    res = run_all()
    assert len(res["timeout_after_prior_observation"]["stages_preserving_prior_observation"]) == 7
    assert res["consecutive_hangs_and_lock_edges"]["two_consecutive_hung_runs_recover_on_slot_3"] is True


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--flock":
        raise SystemExit(_flock_main(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        raise SystemExit(_child_main(sys.argv[2:]))
    print(json.dumps(run_all(), indent=2))
