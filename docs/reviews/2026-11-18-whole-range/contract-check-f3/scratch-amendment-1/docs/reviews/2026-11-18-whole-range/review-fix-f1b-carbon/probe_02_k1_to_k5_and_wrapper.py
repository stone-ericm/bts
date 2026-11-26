"""Probe 02: K1-K5 end-to-end on real CLI, real installer output, real /usr/bin/timeout, and real uv."""
from __future__ import annotations

from datetime import date
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FLOCK_PATH = "/usr/bin/flock"
TIMEOUT_PATH = "/usr/bin/timeout"
LOCK_DEFAULT = "/home/bts/.check-pick-entered.lock"
DAY = "2027-03-24"
PITCH = f"{DAY}T19:40:00-04:00"


def _kill_recorded(pids: set[int], sig: int = signal.SIGKILL) -> None:
    for pid in list(pids):
        if pid <= 1 or pid == os.getpid():
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _util_linux_flock_main(argv: list[str]) -> int:
    """Faithful model of util-linux flock(1) open + flock(2) + execvp semantics."""
    if len(argv) < 3 or argv[0] != "-n":
        sys.stderr.write("flock: usage error\n")
        return 64
    lock_path = argv[1]
    cmd = argv[2:]
    flags = os.O_RDONLY | os.O_NOCTTY | os.O_CREAT
    try:
        fd = os.open(lock_path, flags, 0o666)
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
    os.environ["PROBE_LOCK_FD"] = str(fd)
    os.execvp(cmd[0], cmd)


def _child_main(argv: list[str]) -> int:
    mode = argv[0]
    trace = Path(os.environ["PROBE_TRACE"])
    with trace.open("a") as f:
        f.write(f"child-start {os.getpid()}\n")

    if mode == "real_cli_hold_or_wrong":
        root = Path(argv[1])
        submode = argv[2]
        sys.path.insert(0, str(REPO))
        from tests.test_wr_f1_wrong_game import MATCH, World
        w = World(root)
        if submode == "hold_no_pick":
            gate = root / "release_a"
            ready = root / "a_in_fetch"

            def slow_pending(*_a, **_kw):
                ready.write_text(str(os.getpid()))
                deadline = time.monotonic() + 10.0
                while not gate.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                return []

            res = w.check("18:30", more_patches=[("bts.contest_fetch.fetch_pending_predictions", slow_pending)])
            (root / "a_result.json").write_text(json.dumps({"exit": res["exit"], "marker": res["marker"]}))
            return res["exit"]
        elif submode == "wrong_game_dm_fail":
            res = w.check("18:45", MATCH, dm="failure")
            (root / "b_result.json").write_text(json.dumps({"exit": res["exit"], "marker": res["marker"]}))
            return res["exit"]
        raise AssertionError(submode)

    if mode == "term_sleep":
        def on_term(*_):
            with trace.open("a") as f:
                f.write(f"child-term {os.getpid()}\n")
            raise SystemExit(143)

        signal.signal(signal.SIGTERM, on_term)
        with trace.open("a") as f:
            f.write(f"child-ready {os.getpid()}\n")
        print("probe child term_sleep", flush=True)
        time.sleep(10.0)
        return 0

    if mode == "ignore_term":
        def on_term_ignore(*_):
            with trace.open("a") as f:
                f.write(f"child-ignored-term {os.getpid()}\n")

        signal.signal(signal.SIGTERM, on_term_ignore)
        with trace.open("a") as f:
            f.write(f"child-ready {os.getpid()}\n")
        print("probe child ignore_term", flush=True)
        time.sleep(10.0)
        return 0

    if mode == "quick":
        with trace.open("a") as f:
            f.write(f"child-ready {os.getpid()}\n")
        print("probe child quick", flush=True)
        return 0

    raise AssertionError(f"unknown child mode {mode}")


def _ignore_term_wrapper_main(argv: list[str]) -> int:
    trace = Path(os.environ["PROBE_TRACE"])
    signal.signal(signal.SIGTERM, lambda *_: trace.open("a").write(f"wrapper-ignored-term {os.getpid()}\n"))
    fd = os.environ.get("PROBE_LOCK_FD")
    pass_fds = (int(fd),) if fd is not None else ()
    child = subprocess.Popen([sys.executable, __file__, "--child", *argv], pass_fds=pass_fds)
    with trace.open("a") as f:
        f.write(f"wrapper-spawned {child.pid}\n")
    try:
        rc = child.wait(timeout=10.0)
        return rc if rc >= 0 else 128 - rc
    finally:
        if child.poll() is None:
            _kill_recorded({child.pid})
            child.wait(timeout=3.0)


def generate_enter_cron_line(tmp_root: Path) -> tuple[str, Path, Path]:
    home = tmp_root / "home"
    (home / "projects" / "bts").mkdir(parents=True)
    (home / "projects" / "bts" / ".env").write_text("")
    (home / ".local" / "bin").mkdir(parents=True)
    (home / "logs").mkdir(parents=True)
    uv_shim = home / ".local" / "bin" / "uv"
    uv_shim.write_text(
        '#!/bin/sh\n[ "$1 $2 $3" = "run bts entry-intent" ] || exit 97\n'
        f'shift 2\nexec "{sys.executable}" -c "from bts.cli import cli; cli()" "$@"\n'
    )
    uv_shim.chmod(uv_shim.stat().st_mode | stat.S_IXUSR)
    shims = tmp_root / "shims"
    shims.mkdir()
    cron_file = tmp_root / "crontab.txt"
    cron_shim = shims / "crontab"
    cron_shim.write_text(
        f'#!/bin/sh\nf="{cron_file}"\ncase "$1" in\n'
        '  -l) [ -f "$f" ] && cat "$f" || { echo "no crontab for user" >&2; exit 1; } ;;\n'
        '  -) cat > "$f.new" && mv "$f.new" "$f" ;;\n'
        '  *) exit 98 ;;\nesac\n'
    )
    cron_shim.chmod(cron_shim.stat().st_mode | stat.S_IXUSR)
    (home / ".bts-orchestrator.toml").write_text(
        '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'
    )
    env = {
        "HOME": str(home),
        "PATH": f"{shims}:/usr/bin:/bin",
        "HEALTHCHECKS_PING_URL": "https://hc-ping.invalid/probe02",
        "TZ": "America/New_York",
    }
    script = REPO / "scripts" / "cron-setup-hetzner.sh"
    res = subprocess.run(["bash", str(script), "install"], env=env, capture_output=True, text=True, check=True, timeout=15)
    assert res.returncode == 0
    lines = [ln for ln in cron_file.read_text().splitlines() if "bts check-pick-entered" in ln]
    assert len(lines) == 1
    return lines[0], home, home / "logs" / "cron.log"


def _parse_trace(path: Path) -> list[tuple[str, int]]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        parts = ln.split()
        if len(parts) == 2:
            out.append((parts[0], int(parts[1])))
    return out


def check_k1() -> dict:
    sys.path.insert(0, str(REPO))
    from tests.test_wr_f1_wrong_game import FIRST, LEG, MATCH, RIGHT, WRONG, World, row

    all_clear = f"✅ BTS pick entry confirmed for {DAY} (match)."
    checked = 0
    with tempfile.TemporaryDirectory() as td:
        for prior_initial in (False, True):
            for bridge in ("crosswalk", "no_pick", "mismatch"):
                for unproved in ("conflict", "null_feed", "wrong_round", "ambiguous"):
                    w = World(Path(td) / f"k1_{prior_initial}_{bridge}_{unproved}")
                    w.commit(pitch=PITCH)
                    w.capture([WRONG])
                    if prior_initial:
                        r0 = w.check("18:25")
                        assert r0["marker"]["escalations"] == ["initial"]
                    r1 = w.check("18:35", MATCH, dm="failure")
                    assert r1["exit"] == 1
                    assert r1["marker"]["game_unresolved"] is True
                    assert r1["marker"]["status"] == ("alerted" if prior_initial else "dm_failed")

                    if bridge == "crosswalk":
                        r2 = w.check("18:45", MATCH, crosswalk={})
                    elif bridge == "no_pick":
                        r2 = w.check("18:45", [])
                    else:
                        r2 = w.check("18:45", [row(player=3)])
                    assert r2["marker"]["game_unresolved"] is True

                    rows_final = MATCH
                    if unproved == "conflict":
                        w.capture([RIGHT], f"{DAY}T230000Z")
                    elif unproved == "null_feed":
                        w.units.joinpath(f"{DAY}T210000Z.json.gz").unlink()
                        w.capture([{"id": 77, "roundId": 1, "feedId": None}])
                    elif unproved == "wrong_round":
                        w.units.joinpath(f"{DAY}T210000Z.json.gz").unlink()
                        w.capture([{"id": 77, "roundId": 2, "feedId": 102}])
                    elif unproved == "ambiguous":
                        rows_final = [row(77, 1, 1), row(78, 1, 2)]

                    r3 = w.check("19:00", rows_final)
                    assert r3["exit"] == 1
                    assert r3["marker"]["status"] == "alerted"
                    assert r3["marker"]["reason"] == "wrong_game"
                    assert r3["marker"]["game_unresolved"] is True
                    assert all_clear not in r3["attempted_dm"]
                    assert r3["dm"] == [FIRST.format(legs=LEG.split(", the entry")[0])]
                    checked += 1
    return {"k1_combinations_passed": checked}


def check_k2_and_k3() -> dict:
    sys.path.insert(0, str(REPO))
    from bts import entry_receipt
    from tests.test_wr_f1_wrong_game import MATCH, RIGHT, WRONG, World

    uv_bin = shutil.which("uv")
    assert uv_bin is not None
    assert Path(TIMEOUT_PATH).is_file()

    recorded_pids: set[int] = set()
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            line, home, cron_log = generate_enter_cron_line(root)
            cron_log.touch()

            # Verify schedule & arithmetic from real generated line
            parts = line.split(maxsplit=5)
            assert parts[:5] == ["*/15", "10-23", "*", "*", "*"]
            shell_template = parts[5]
            m = re.search(
                r"/usr/bin/flock -n /home/bts/\.check-pick-entered\.lock /usr/bin/timeout --verbose -k (\d+) (\d+) \S+/uv run bts check-pick-entered",
                shell_template,
            )
            assert m is not None, shell_template
            grace_s, limit_s = int(m.group(1)), int(m.group(2))
            interval_s = int(parts[0].split("/")[1]) * 60
            assert (grace_s, limit_s, interval_s) == (30, 840, 900)
            assert limit_s + grace_s < interval_s

            lock_file = root / "live.lock"
            flock_cmd = [sys.executable, __file__, "--flock"]

            def build_shell(inner_cmd: list[str], *, dur: str = "1.0", grace: str = "0.25") -> str:
                s = shell_template
                s = s.replace(FLOCK_PATH, shlex.join(flock_cmd), 1)
                s = s.replace(LOCK_DEFAULT, shlex.quote(str(lock_file)), 1)
                s = s.replace(
                    f"{TIMEOUT_PATH} --verbose -k 30 840",
                    f"{TIMEOUT_PATH} --verbose -k {grace} {dur}",
                    1,
                )
                s = re.sub(
                    r"\S+/uv run bts check-pick-entered.*?(?=>>)",
                    lambda _: shlex.join(inner_cmd) + " ",
                    s,
                    count=1,
                )
                return s

            # --- K2: Real CLI Run A holds lock mid-fetch; Run B contends and does NOT run; after A ends, next slot runs ---
            w_root = root / "k2_world"
            w = World(w_root)
            w.commit(pitch=PITCH)
            assert w.check("18:25")["marker"]["escalations"] == ["initial"]
            w.capture([WRONG])
            receipts_before = len(entry_receipt.discover(w.picks, DAY))

            trace_a = root / "trace_a.txt"
            trace_b = root / "trace_b.txt"
            trace_c = root / "trace_c.txt"
            base_env = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO}", "TZ": "America/New_York"}

            proc_a = subprocess.Popen(
                ["/bin/sh", "-c", build_shell([sys.executable, __file__, "--child", "real_cli_hold_or_wrong", str(w_root), "hold_no_pick"], dur="8.0")],
                env={**base_env, "PROBE_TRACE": str(trace_a)},
            )
            recorded_pids.add(proc_a.pid)

            deadline = time.monotonic() + 8.0
            while not (w_root / "a_in_fetch").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert (w_root / "a_in_fetch").exists()
            recorded_pids.add(int((w_root / "a_in_fetch").read_text()))

            log_before_b = cron_log.read_bytes()
            t0 = time.monotonic()
            proc_b = subprocess.Popen(
                ["/bin/sh", "-c", build_shell([sys.executable, __file__, "--child", "real_cli_hold_or_wrong", str(w_root), "wrong_game_dm_fail"], dur="8.0")],
                env={**base_env, "PROBE_TRACE": str(trace_b)},
            )
            recorded_pids.add(proc_b.pid)
            rc_b = proc_b.wait(timeout=5.0)
            b_ms = round((time.monotonic() - t0) * 1000, 1)
            assert rc_b == 1, f"expected contender exit 1, got {rc_b}"
            assert not trace_b.exists()
            assert not (w_root / "b_result.json").exists()
            assert cron_log.read_bytes() == log_before_b

            # Release A; verify A finishes cleanly, then next slot C acquires lock and records wrong_game
            (w_root / "release_a").write_text("go")
            rc_a = proc_a.wait(timeout=8.0)
            assert rc_a == 1  # no_pick exits 1

            proc_c = subprocess.Popen(
                ["/bin/sh", "-c", build_shell([sys.executable, __file__, "--child", "real_cli_hold_or_wrong", str(w_root), "wrong_game_dm_fail"], dur="8.0")],
                env={**base_env, "PROBE_TRACE": str(trace_c)},
            )
            recorded_pids.add(proc_c.pid)
            rc_c = proc_c.wait(timeout=8.0)
            assert rc_c == 1
            assert w.read_marker()["game_unresolved"] is True

            # Add conflicting capture: next check at 19:00 must NOT confirm or send all-clear!
            w.capture([RIGHT], f"{DAY}T230000Z")
            r_1900 = w.check("19:00", MATCH)
            assert r_1900["exit"] == 1
            assert r_1900["marker"]["status"] == "alerted"
            assert r_1900["marker"]["game_unresolved"] is True
            assert len(entry_receipt.discover(w.picks, DAY)) == receipts_before + 3

            # --- K3a: Real /usr/bin/timeout + real uv + Python child receiving TERM -> exit 124, lock freed ---
            cron_log.write_text("")
            trace_k3_term = root / "trace_k3_term.txt"
            uv_cmd_term = [uv_bin, "run", "--offline", "--no-project", "--no-sync", sys.executable, __file__, "--child", "term_sleep"]
            proc_term = subprocess.Popen(
                ["/bin/sh", "-c", build_shell(uv_cmd_term, dur="1.0", grace="0.4")],
                env={**base_env, "PROBE_TRACE": str(trace_k3_term)},
            )
            recorded_pids.add(proc_term.pid)
            rc_term = proc_term.wait(timeout=8.0)
            for _, pid in _parse_trace(trace_k3_term):
                recorded_pids.add(pid)
            term_labels = {lbl for lbl, _ in _parse_trace(trace_k3_term)}
            term_log = cron_log.read_text()
            assert rc_term == 124, f"expected 124, got {rc_term}, log={term_log!r}"
            assert {"child-ready", "child-term"} <= term_labels, term_labels

            # Verify lock is immediately acquirable after TERM
            trace_after_term = root / "trace_after_term.txt"
            proc_after_term = subprocess.Popen(
                ["/bin/sh", "-c", build_shell([sys.executable, __file__, "--child", "quick"], dur="4.0")],
                env={**base_env, "PROBE_TRACE": str(trace_after_term)},
            )
            recorded_pids.add(proc_after_term.pid)
            assert proc_after_term.wait(timeout=5.0) == 0

            # --- K3b: Real /usr/bin/timeout + TERM-ignoring wrapper & child -> KILL after grace, exit 137, lock freed ---
            cron_log.write_text("")
            trace_k3_kill = root / "trace_k3_kill.txt"
            kill_cmd = [sys.executable, __file__, "--ignore-term-wrapper", "ignore_term"]
            proc_kill = subprocess.Popen(
                ["/bin/sh", "-c", build_shell(kill_cmd, dur="0.8", grace="0.3")],
                env={**base_env, "PROBE_TRACE": str(trace_k3_kill)},
            )
            recorded_pids.add(proc_kill.pid)
            rc_kill = proc_kill.wait(timeout=8.0)
            for _, pid in _parse_trace(trace_k3_kill):
                recorded_pids.add(pid)
            kill_labels = {lbl for lbl, _ in _parse_trace(trace_k3_kill)}
            kill_log = cron_log.read_text()
            assert rc_kill == 137, f"expected 137, got {rc_kill}, log={kill_log!r}"
            assert {"wrapper-ignored-term", "child-ready", "child-ignored-term"} <= kill_labels, kill_labels

            trace_after_kill = root / "trace_after_kill.txt"
            proc_after_kill = subprocess.Popen(
                ["/bin/sh", "-c", build_shell([sys.executable, __file__, "--child", "quick"], dur="4.0")],
                env={**base_env, "PROBE_TRACE": str(trace_after_kill)},
            )
            recorded_pids.add(proc_after_kill.pid)
            assert proc_after_kill.wait(timeout=5.0) == 0

            # --- K3c: Real /usr/bin/timeout + REAL uv + TERM-ignoring Python child! ---
            # Does real `uv run` stay alive forwarding/waiting until KILL, or exit early on TERM?
            cron_log.write_text("")
            trace_uv_ign = root / "trace_uv_ign.txt"
            uv_ign_cmd = [uv_bin, "run", "--offline", "--no-project", "--no-sync", sys.executable, __file__, "--child", "ignore_term"]
            proc_uv_ign = subprocess.Popen(
                ["/bin/sh", "-c", build_shell(uv_ign_cmd, dur="1.0", grace="0.3")],
                env={**base_env, "PROBE_TRACE": str(trace_uv_ign)},
            )
            recorded_pids.add(proc_uv_ign.pid)
            rc_uv_ign = proc_uv_ign.wait(timeout=8.0)
            uv_ign_records = _parse_trace(trace_uv_ign)
            for _, pid in uv_ign_records:
                recorded_pids.add(pid)
            uv_ign_log = cron_log.read_text()

            # Check whether lock is already free right after proc_uv_ign returns!
            lock_free_immediately = False
            with lock_file.open("r") as lf:
                try:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                    lock_free_immediately = True
                except BlockingIOError:
                    lock_free_immediately = False
            # Clean up any surviving child if it outlived uv
            _kill_recorded({pid for _, pid in uv_ign_records})

            return {
                "k2_contender_exit": rc_b,
                "k2_contender_ms": b_ms,
                "k2_serialized_preserves_wrong_game": True,
                "k3_arithmetic": {"limit": limit_s, "grace": grace_s, "sum": limit_s + grace_s, "interval": interval_s},
                "k3_real_uv_term_exit": rc_term,
                "k3_real_uv_term_log": term_log.strip(),
                "k3_kill_exit": rc_kill,
                "k3_kill_log": kill_log.strip(),
                "k3_real_uv_with_term_ignoring_child": {
                    "exit_code": rc_uv_ign,
                    "lock_free_immediately": lock_free_immediately,
                    "trace_labels": [lbl for lbl, _ in uv_ign_records],
                    "log": uv_ign_log.strip(),
                },
            }
    finally:
        _kill_recorded(recorded_pids)


def check_k4_unwrapped_overlap() -> dict:
    """Verify K4: without the cron lock, two concurrent CLI runs overwrite the newer wrong-game marker."""
    sys.path.insert(0, str(REPO))
    from click.testing import CliRunner
    from tests.test_wr_f1_wrong_game import MATCH, RIGHT, WRONG, World

    with tempfile.TemporaryDirectory() as td:
        w = World(Path(td))
        w.commit(pitch=PITCH)
        assert w.check("18:25")["marker"]["escalations"] == ["initial"]
        w.capture([WRONG])
        orig_runner = w.runner

        def nested_b(*_a, **_kw):
            w.runner = CliRunner()
            try:
                b = w.check("18:45", MATCH, dm="failure")
                assert b["marker"]["game_unresolved"] is True
            finally:
                w.runner = orig_runner
            return []

        a = w.check("18:30", more_patches=[("bts.contest_fetch.fetch_pending_predictions", nested_b)])
        assert "game_unresolved" not in a["marker"]
        w.capture([RIGHT], f"{DAY}T230000Z")
        final = w.check("19:00", MATCH)
        all_clear = f"✅ BTS pick entry confirmed for {DAY} (match)."
        reproduced = (final["marker"]["status"] == "confirmed" and final["dm"] == [all_clear])
        assert reproduced
        return {"k4_unwrapped_reproduces_stale_overwrite": reproduced}


def run_all() -> dict:
    summary = {
        "k1": check_k1(),
        "k2_k3": check_k2_and_k3(),
        "k4": check_k4_unwrapped_overlap(),
    }
    (OUT_DIR / "log_02_k1_to_k5_and_wrapper.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def test_probe_02():
    res = run_all()
    assert res["k1"]["k1_combinations_passed"] == 24
    assert res["k2_k3"]["k2_serialized_preserves_wrong_game"] is True
    assert res["k4"]["k4_unwrapped_reproduces_stale_overwrite"] is True


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--flock":
        raise SystemExit(_util_linux_flock_main(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        raise SystemExit(_child_main(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "--ignore-term-wrapper":
        raise SystemExit(_ignore_term_wrapper_main(sys.argv[2:]))
    print(json.dumps(run_all(), indent=2))
