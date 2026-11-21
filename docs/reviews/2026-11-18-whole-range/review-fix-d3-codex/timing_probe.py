"""Test the wait's actual elapsed time and close output during an active run."""

import argparse
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from shell_probe import DEPLOY, HERE, tests


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def timing_shim():
    source = replace_once(tests.SHIM, 'elif command == "timeout":\n    dispatch(args[3:])', '''
elif command == "timeout":
    if args[3:] == ["sleep", "30"]:
        executable = root / "shims/sleep"
        os.execv("/usr/bin/timeout", ["timeout", *args[:3], str(executable), "30"])
    dispatch(args[3:])''')
    return replace_once(source, 'elif command in {"sleep", "uv"}:', '''
elif command == "sleep" and scenario in {"wait-real", "wait-timeout"}:
    (root / "actual-wait-start").write_text(str(time.monotonic()))
    time.sleep(30 if scenario == "wait-real" else 44)
    (root / "actual-wait-end").write_text(str(time.monotonic()))
elif command in {"sleep", "uv"}:''')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("wait-real", "wait-timeout", "closed-success", "closed-failure"))
    case = parser.parse_args().case
    original = inspect.getsource(tests.run_box_script)
    shims = timing_shim()
    helper = replace_once(original, 'stdout, stderr = process.communicate(timeout=15)',
                          'stdout, stderr = process.communicate(timeout=49)')
    if case.startswith("closed-"):
        helper = replace_once(helper, '        os.close(reader)\n', '')
        helper = replace_once(helper, '        try:\n            stdout, stderr =', '''
        end = time.monotonic() + 10
        while not (box / "reached-reset").exists() and time.monotonic() < end:
            time.sleep(0.01)
        assert (box / "reached-reset").exists()
        os.close(reader)
        (box / "reset-can-resume").touch()
        try:
            stdout, stderr =''')
        shims = replace_once(shims, '    elif args[0] == "reset":', '''
    elif args[0] == "reset":
        if scenario.startswith("closed-"):
            (root / "reached-reset").touch()
            end = time.monotonic() + 4
            while not (root / "reset-can-resume").exists() and time.monotonic() < end:
                time.sleep(0.01)
            if not (root / "reset-can-resume").exists():
                raise SystemExit(90)
            branch.write_text(args[-1])
            if scenario == "closed-failure":
                raise SystemExit(17)''')
    env = vars(tests).copy()
    exec(compile(helper, "<test harness with bounded timing adapter>", "exec"), env)
    with TemporaryDirectory(prefix="wrd3-timing-", dir=HERE) as scratch:
        box = tests.box.__wrapped__(Path(scratch))
        for shim in (box / "shims").iterdir():
            shim.write_text(shim.read_text().split("\n", 1)[0] + "\n" + shims)
        started = time.monotonic()
        result, calls = env["run_box_script"](box, DEPLOY, case, closed_output=case.startswith("closed-"))
        duration = time.monotonic() - started
        print(json.dumps(dict(case=case, status=result.returncode, time=duration, output=result.stdout, errors=result.stderr),
                         ensure_ascii=False), flush=True)
        if case.startswith("wait-"):
            assert ["timeout", "-k", "2", "33", "sleep", "30"] in calls
        if case == "wait-real":
            elapsed = float((box / "actual-wait-end").read_text()) - float((box / "actual-wait-start").read_text())
            assert elapsed >= 30 and result.returncode == 0 and "CANARY PASSED" in result.stdout
            print(f"sleep-shim elapsed {elapsed:.3f}s", flush=True)
        elif case == "wait-timeout":
            assert 32 <= duration < 49 and not (box / "actual-wait-end").exists()
            assert result.returncode == 1 and "Check wait failed (timeout)." in result.stdout
            assert (box / "stopped").exists()
        else:
            contained = case == "closed-failure"
            assert (box / "head").read_text() == tests.TARGET
            record = (box / "run/bts-deploy/attempt").read_text().split()[5]
            assert record == ("contained" if contained else "committed")
            assert (box / "stopped").exists() == contained and result.returncode == int(contained)
            assert not result.stderr


run()
