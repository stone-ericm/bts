"""Run all pytest families beneath a single yield_run.sh job, serially."""

import os
import re
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = Path("src/bts/health/alert.py")
NEW = ["tests/health/test_alert_status_cs6b2.py", "tests/test_alert_status_callers_cs6b2.py"]
NEIGHBORS = ["tests/test_morning_state_cs6b1.py", "tests/test_saved_pick_paths_cs5a.py"]
COMBINED = ["tests/test_lightgbm_optional.py", "tests/test_web_saver.py", *NEIGHBORS, "tests/health/test_alert.py"]
CALLERS = str((HERE / "test_caller_replay.py").relative_to(ROOT))


def revised(text, old, new):
    assert old in text, old
    return text.replace(old, new, 1)


def variants(source):
    snapshot = "    resurface = _resurface_note(prior)"
    dedup = '    if prior is not None and prior.get("status") == "sent" and prior.get("sent_day") == now_day:'
    return [
        ("non-object", "test_status_shapes and array and critical", "return data if isinstance(data, dict) else None", "return data"),
        ("read-regular", "test_status_shapes and fifo and empty", "        if not stat.S_ISREG(path.stat().st_mode):\n            return None\n", ""),
        ("write-regular", "test_status_shapes and fifo and critical", "if not stat.S_ISREG(mode):", "if False:"),
        ("recursion", "test_status_shapes and nested and critical", "(OSError, ValueError, RecursionError)", "(OSError, ValueError)"),
        ("bool-count", "test_failed_count", "type(n) is int and n > 0", "isinstance(n, int) and n > 0"),
        ("zero-count", "test_failed_count", "type(n) is int and n > 0", "type(n) is int and n >= 0"),
        ("sources-truthiness", "test_prior_sources", "if isinstance(sources, list):", "if sources:"),
        ("keep-non-strings", "test_prior_sources", "[source for source in sources if isinstance(source, str)]", "list(sources)"),
        ("second-read", "test_one_status_snapshot", snapshot, "    prior = _read_health_dm_delivery_status(status_path)\n" + snapshot),
        ("other-status", "test_prior_sources", dedup, '    if prior is not None and prior.get("sent_day") == now_day:'),
        ("other-day", "test_prior_sources", dedup, '    if prior is not None and prior.get("status") == "sent":'),
        ("drop-union", "test_prior_sources", "sorted(set(prior_sent_sources) | set(alert_key))", "alert_key"),
        ("swallow-send", "test_status_shapes and green_missing and critical", "        send_dm(dm_recipient, body)\n        _persist", "        _persist"),
    ]


def archive(source, target):
    target.mkdir()
    with subprocess.Popen(["git", "archive", source, "src"], cwd=ROOT, stdout=subprocess.PIPE) as git:
        with tarfile.open(fileobj=git.stdout, mode="r|") as bundle:
            bundle.extractall(target, filter="data")
        assert git.wait() == 0


def run(name, args, *, source=None, base=False, year=None, expected=0, state_check=False, verbose=False):
    env = dict(os.environ, CS6B2_OFFLINE="1", TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1",
               CS6B2_SOURCE="base" if base else "branch")
    import_paths = [HERE, ROOT]
    if source is not None:
        import_paths.insert(1, source / "src")
    env["PYTHONPATH"] = os.pathsep.join(map(str, import_paths))
    if year:
        env["DYLD_INSERT_LIBRARIES"] = str(HERE / "realtime.dylib")
        instant = datetime(year, 7, 15, 12, tzinfo=ZoneInfo("America/New_York"))
        env["CS6B2_WALL_EPOCH"] = str(int(instant.timestamp()))
    log = HERE / f"{name}.log"
    with log.open("w") as out:
        check = "import datetime, socket, time, bts; print(bts.__file__, datetime.datetime.now(), time.time(), time.monotonic(), socket.getaddrinfo.__module__)"
        info = subprocess.check_output([sys.executable, "-c", check], env=env, cwd=ROOT, text=True, timeout=15)
        out.write(f"source / wall time / epoch / monotonic / guard: {info}")
        out.flush()
        assert "sitecustomize" in info
        if source:
            assert str(source) in info
        if year:
            assert f"{year}-07-15 12:00:" in info
        command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--tb=short"]
        if state_check:
            command.extend(["-p", "review_state"])
        if verbose:
            command.append("-rA")
        exit_code = subprocess.run(command + args, env=env, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                   timeout=180).returncode
    lines = log.read_text().splitlines()
    summary = next((line for line in reversed(lines) if re.search(r" in \d+\.\d+s", line)), "no summary")
    failures = [line for line in lines if line.startswith("FAILED ")]
    print(f"{name}: rc={exit_code} (expected {expected}); {summary}", flush=True)
    if failures and name.startswith("mutant"):
        print(f"  {failures[0]}", flush=True)
    return exit_code == expected


def main():
    ok = []
    with tempfile.TemporaryDirectory(prefix="test-sources-", dir=HERE) as cache:
        folder = Path(cache)
        base = folder / "base"
        mutant = folder / "mutant"
        archive("59efa825", base)
        archive("HEAD", mutant)
        source = (mutant / SOURCE).read_text()
        ok.append(run("caller-branch", [CALLERS], verbose=True))
        ok.append(run("caller-base", [CALLERS], source=base, base=True, verbose=True))
        ok.append(run("new-before-neighbors", [*NEW, *NEIGHBORS], state_check=True))
        ok.append(run("neighbors-before-new", [*NEIGHBORS, *NEW], state_check=True))
        ok.append(run("combined", [*NEW, *COMBINED], state_check=True))
        for year in (2027, 2028):
            ok.append(run(f"wall-{year}", [*NEW, NEIGHBORS[1]], year=year, state_check=True))
        ok.append(run("base-authored", [*NEW, NEIGHBORS[1] + "::test_refusal_survives_non_object_health_status"],
                      source=base, base=True, expected=1, verbose=True))
        for name, selection, old, new in variants(source):
            (mutant / SOURCE).write_text(revised(source, old, new))
            ok.append(run(f"mutant-{name}", [NEW[0], "-x", "-k", selection], source=mutant, expected=1))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
