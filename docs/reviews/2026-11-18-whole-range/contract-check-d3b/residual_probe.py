"""Check the exact residual pattern against a scheduler outside the observed units."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("workflow_tests", HERE.parents[1] / "tests/test_deploy_workflow.py")
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
_, recovery = tests.scripts_and_steps()[0]

# This models the registered console script callable invoked directly by an
# operator. No scheduler or real pgrep is started.
detached_command = "python3 -c 'from bts.cli import cli; cli()' schedule --config scheduler.toml"
pattern_branch = '    if command == "pgrep" and scenario == "residual":'
assert tests.SHIM.count(pattern_branch) == 1
shim = tests.SHIM.replace(pattern_branch, f'''    if command == "pgrep":
        import re
        matched = bool(re.search(args[-1], {detached_command!r}))
        with (root / "calls").open("a") as log:
            print(json.dumps(["residual-match", matched]), file=log)
        raise SystemExit(not matched)
''' + pattern_branch)

with TemporaryDirectory(prefix="residual-", dir=HERE) as name:
    box = tests.box.__wrapped__(Path(name))
    for file in (box / "shims").iterdir():
        file.write_text(file.read_text().split("\n", 1)[0] + "\n" + shim)
    runtime = box / "run/bts-deploy"
    runtime.mkdir(mode=0o700, parents=True)
    (runtime / "attempt").write_text(
        f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun - - - -\n"
    )
    (box / "enabled").write_text("enabled")
    (box / "home/projects/bts/data/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/token\n")
    result, calls = tests.run_box_script(box, recovery, "active")
    assert result.returncode == 1 and "Scheduler stopped: verified." in result.stdout
    assert ["pgrep", "-u", "bts", "-f", "bts schedule"] in calls
    assert ["residual-match", False] in calls
    assert ["systemctl", "stop", "bts-scheduler.service"] not in calls
    print(json.dumps({"scope": "modeled manual bts process", "matched": False, "printed": "verified"}))
