"""Check the shared gate before launching one round-two wrapper job."""

from datetime import datetime, time
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from launch_review import WRAPPER, wrappers_running

mode = sys.argv[1]
assert mode in {"focused", "mutations"}
now = datetime.now(ZoneInfo("America/New_York"))
assert now.time() >= time(9, 45)
active = wrappers_running()
if active:
    print("Shared wrapper is occupied by", ", ".join(active), flush=True)
    raise SystemExit(75)

os.chdir(ROOT)
mpl = HERE / "matplotlib"
mpl.mkdir(exist_ok=True)
env = os.environ | {"PYTHONDONTWRITEBYTECODE": "1", "TZ": "America/New_York", "MPLCONFIGDIR": str(mpl),
                    "PYTHONPATH": os.pathsep.join([str(HERE / "offline_python"), str(ROOT / "src")])}
env.pop("NOTIFY_SOCKET", None)
env.pop("INVOCATION_ID", None)
job = [str(HERE / "run_review_pytest.py")] if mode == "focused" else [str(HERE / "run_mutations.py"), "run"]
limit = "360s" if mode == "focused" else "900s"
log = f".codex-review/wr-w3/r2/{mode}.log"
cmd = [str(WRAPPER), log, "/usr/bin/timeout", limit, str(ROOT / ".venv/bin/python"), *job]
print(f"Starting round-two {mode} through shared wrapper at {now.isoformat()}", flush=True)
os.execvpe(str(WRAPPER), cmd, env)
