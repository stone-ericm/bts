"""Launch one eligible wrapped review job, after checking the shared machine."""

from datetime import datetime, time
import os
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WRAPPER = Path.home() / "bts-runs" / "yield_run.sh"


def wrappers_running():
    result = subprocess.run(["pgrep", "-f", "yield_run.sh"], capture_output=True,
                            text=True, timeout=3, check=False)
    if result.returncode == 1:
        return []
    if result.returncode != 0:
        raise RuntimeError(f"pgrep failed ({result.returncode}): {result.stderr.strip()}")
    active = []
    for pid in result.stdout.splitlines():
        info = subprocess.run(["ps", "-ww", "-p", pid, "-o", "command="], capture_output=True,
                              text=True, timeout=3, check=False)
        if info.returncode == 1:
            continue
        if info.returncode != 0:
            raise RuntimeError(f"ps could not inspect wrapper candidate {pid}: {info.stderr.strip()}")
        try:
            args = shlex.split(info.stdout)
        except ValueError:
            args = info.stdout.split()
        if any(Path(arg).name == WRAPPER.name for arg in args):
            active.append(pid)
    return active


def main(mode):
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.time() < time(9, 45):
        print(f"Review job ineligible before 09:45 ET ({now.isoformat()})", flush=True)
        return 75
    active = wrappers_running()
    if active:
        print(f"Shared test wrapper active (PID {', '.join(active)})", flush=True)
        return 75
    os.chdir(ROOT)
    matplotlib = HERE / "matplotlib"
    matplotlib.mkdir(exist_ok=True)
    env = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1", "TZ": "America/New_York",
        "MPLCONFIGDIR": str(matplotlib),
        "PYTHONPATH": os.pathsep.join([str(HERE / "offline_python"), str(HERE / "patched" / "src")]),
    }
    env.pop("NOTIFY_SOCKET", None)
    env.pop("INVOCATION_ID", None)
    python = ROOT / ".venv" / "bin" / "python"
    script = "run_focused.py" if mode == "focused" else "run_mutants.py"
    args = [] if mode == "focused" else ["run"]
    limit = "360s" if mode == "focused" else "900s"
    logfile = Path(".codex-review") / "wr-w3" / f"{mode}.log"
    cmd = [str(WRAPPER), str(logfile), "/usr/bin/timeout", limit,
           str(python), str(HERE / script), *args]
    print(f"Starting wrapped {mode} review job at {now.isoformat()}", flush=True)
    os.execvpe(str(WRAPPER), cmd, env)


if __name__ == "__main__":
    assert sys.argv[1] in {"focused", "mutations"}
    raise SystemExit(main(sys.argv[1]))
