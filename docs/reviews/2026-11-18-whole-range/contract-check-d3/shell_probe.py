"""Exact base workflow and small Bash models of v7's ordering, offline."""

import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time

import yaml


HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / ".github/workflows/deploy.yml"
OLD = "a" * 40
TARGET = "b" * 40
BASH = "/bin/bash"
PHASE = (
    "disable", "reset", "sync", "legacy-disable", "legacy-rm", "legacy-reload",
    "legacy-pkill", "restart", "head", "wait", "scheduler-state",
    "dashboard-state", "tailnet", "dashboard-http", "contest-state",
    "stamp-stage", "enable", "stamp",
)

BASE_SHIMS = r'''
cd() { builtin cd "$BOX"; }
record() { printf '%s\n' "$*" >>calls; }
id() { record "id $*"; printf '1000\n'; }
sleep() { record "wait $*"; [[ $FAIL != wait ]]; }
curl() {
    record "curl $*"
    case $FAIL in
        curl-exit) printf 200; return 28 ;;
        curl-http) printf 503 ;;
        *) printf 200 ;;
    esac
}
sudo() {
    record "sudo $*"
    case " $* " in
        *" git rev-parse "*) cat sha ;;
        *" git fetch "*) [[ $FAIL != fetch ]] ;;
        *" git reset "*) printf '%s\n' "${!#}" >sha; [[ $FAIL != reset ]] ;;
        *" sync "*) [[ $FAIL != sync ]] ;;
        *" restart "*) [[ $FAIL != restart ]] ;;
        *" is-active --quiet "*) [[ $FAIL != dashboard ]] ;;
        *" is-active "*) printf '%s\n' "${ACTIVE:-active}" ;;
        *" -p ActiveState "*) printf 'activating\n' ;;
        *" -p SubState "*) printf 'auto-restart\n' ;;
        *" -p Result "*) printf 'success\n' ;;
        *" -p ExecMainStatus "*) printf '0\n' ;;
        *" bash -c "*) [[ $FAIL != stamp ]] ;;
        *) return 0 ;;
    esac
}
'''

PHASE_MODEL = r'''
record() { printf '%s\n' "$*" >>calls; }
contain() {
    trap '' HUP INT TERM PIPE
    record "contain $*"
    if [[ $AGAIN ]]; then kill -TERM "$$"; fi
    record stop
    record marker
    record page
    exit 1
}
shim() {
    record "$1"
    if [[ $INJECT == "$1" && $SIGNAL ]]; then
        kill -"$SIGNAL" "$BOX_SHELL"
    fi
    [[ $INJECT != "$1" || $SIGNAL ]] || return 17
}
export BOX_SHELL=$$
trap 'contain interrupted 130' HUP INT TERM PIPE
for step in disable reset sync legacy-disable legacy-rm legacy-reload legacy-pkill restart head wait; do
    shim "$step" || contain "$step" "$?"
done
first=
for step in scheduler-state dashboard-state tailnet dashboard-http contest-state; do
    shim "$step" || {
        status=$?
        record "check $step $status"
        [[ $first || $step == tailnet ]] || first=$step
    }
done
[[ ! $first ]] || contain "$first" "$status"
for step in stamp-stage enable stamp; do
    shim "$step" || contain "$step" "$?"
done
if [[ $LATE ]]; then
    trap - HUP INT TERM PIPE
    kill -TERM "$$"
fi
record pass
'''

RECOVERY_MODEL = r'''
if [[ $MARKER == matching ]]; then
    printf 'stop-only\n'
elif [[ $ENABLED == enabled && $LOCK != timeout ]]; then
    printf 'skip\n'
else
    printf 'contain\n'
fi
'''


def run(root, body, **settings):
    env = dict(os.environ, BOX=str(root), TARGET_SHA=TARGET, **settings)
    return subprocess.run([BASH, "-c", body], cwd=root, env=env, text=True,
                          capture_output=True, timeout=3)


def base(root):
    workflow = yaml.safe_load(BASE.read_text())
    script, = [step["with"]["script"] for step in workflow["jobs"]["deploy"]["steps"]
               if "script" in step.get("with", {})]
    literal = BASE.read_text().split("          script: |\n", 1)[1]
    assert script == re.sub(r"(?m)^ {12}", "", literal)
    script = script.replace("/etc/systemd/system/bts-scheduler.service",
                            str(root / "absent-legacy"))
    for case in ("pass", "auto-restart", "fetch", "reset", "sync", "restart",
                 "wait", "dashboard", "curl-exit", "curl-http", "stamp"):
        (root / "calls").write_text("")
        (root / "sha").write_text(OLD + "\n")
        result = run(root, BASE_SHIMS + script, FAIL=case,
                     ACTIVE="activating" if case == "auto-restart" else "active")
        calls = (root / "calls").read_text().splitlines()
        outcome = (result.returncode, "CANARY PASSED" in result.stdout,
                   any(f"reset --hard {OLD}" in line for line in calls))
        print(f"base {case}: rc/pass-line/reset-old = {outcome}")
        if case in {"pass", "auto-restart"}:
            print("  " + " | ".join(calls))


def phase(root):
    cases = 0
    for step in PHASE:
        for signum in ("", "HUP", "INT", "TERM", "PIPE"):
            (root / "calls").write_text("")
            result = run(root, PHASE_MODEL, INJECT=step, SIGNAL=signum, AGAIN="yes")
            calls = (root / "calls").read_text().splitlines()
            if step == "tailnet" and not signum:
                assert result.returncode == 0 and calls[-1] == "pass"
            else:
                assert result.returncode == 1, (step, signum, result)
                assert calls[-3:] == ["stop", "marker", "page"]
                assert sum(line.startswith("contain ") for line in calls) == 1
                if signum:
                    assert "contain interrupted 130" in calls
            cases += 1
    (root / "calls").write_text("")
    result = run(root, PHASE_MODEL, LATE="yes")
    assert result.returncode == -signal.SIGTERM
    assert "contain " not in (root / "calls").read_text()
    print(f"phase model: {cases} status/signal cases, including second TERM in handler")
    print("after traps reset: TERM exits failed, no handler, stamp and enable reached")


def recovery(root):
    for case, enabled, marker, lock, expected in [
        ("preflight on previously disabled box", "disabled", "other", "free", "contain"),
        ("failed disable, or loss after enable before stamp", "enabled", "other", "free", "skip"),
        ("older failure on this same ref", "enabled", "matching", "free", "stop-only"),
        ("lock timeout with owner still active", "disabled", "other", "timeout", "contain"),
        ("marker claim with lock timeout", "enabled", "matching", "timeout", "stop-only"),
    ]:
        result = run(root, RECOVERY_MODEL, ENABLED=enabled, MARKER=marker, LOCK=lock)
        assert result.stdout.strip() == expected
        print(f"recovery model {case}: {expected}")

    lock_path = root / "mutex"
    with lock_path.open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        child = subprocess.Popen(
            [BASH, "-c", "printf held >&3; /bin/sleep 1"], pass_fds=(guard.fileno(),),
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        )
        try:
            guard.close()
            with lock_path.open("r") as other:
                try:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    print("mutex: inherited descriptor keeps lock after owner's close")
                else:
                    raise AssertionError("inherited lock was released")
        finally:
            child.wait(timeout=2)


def boundaries(root):
    (root / "stamp-target").mkdir()
    (root / "stamp-source").write_text("staged")
    result = run(root, 'mv -f stamp-source stamp-target')
    assert result.returncode == 0
    assert (root / "stamp-target/stamp-source").read_text() == "staged"
    result = run(root, r"""set -o pipefail
producer() { printf 'usable\n'; return 7; }
x=$(producer | head -1); printf '%s/%s\n' "$?" "$x"
""")
    assert result.stdout.strip() == "7/usable"
    result = run(root, r"""set +o pipefail
producer() { printf 'usable\n'; return 7; }
x=$(producer | head -1); printf '%s/%s\n' "$?" "$x"
""")
    assert result.stdout.strip() == "0/usable"
    print("stamp: mv -f reports success by moving under a directory")
    print("tailnet pipeline: producer failure with output is hidden without pipefail")

    args = [BASH, "-c", r"""
trap 'printf trapped >>pipe-result; exit 1' PIPE
printf down
printf missed >>pipe-result
"""]
    reader, writer = os.pipe()
    os.close(reader)
    with os.fdopen(writer, "wb") as closed_pipe:
        result = subprocess.run(args, cwd=root, stdout=closed_pipe, stderr=subprocess.PIPE,
                                timeout=2)
    assert result.returncode == 1 and (root / "pipe-result").read_text() == "trapped"
    print("closed output pipe: Bash can trap PIPE; a closed SSH session can also be silent")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        box = Path(directory)
        start = time.monotonic()
        base(box)
        phase(box)
        recovery(box)
        boundaries(box)
        print(f"elapsed: {time.monotonic() - start:.2f}s")
