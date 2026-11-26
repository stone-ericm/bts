"""Show what each authorized handler does after one failed clock sample."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from shell_probe import DEPLOY, HERE, RECOVER, tests


def change(source, old, new):
    assert source.count(old) == 1
    return source.replace(old, new)


shim = change(tests.SHIM, '        if op == "clock" and scenario == "canary-budget"', '''
        if op == "clock" and (root / "fail-one-clock").exists():
            (root / "fail-one-clock").unlink()
            with (root / "calls").open("a") as log:
                print(json.dumps(["clock-timed-out-once"]), file=log)
            raise SystemExit(124)
        if op == "read" and scenario == "recovery-clock-timeout" and not (root / "armed-recovery").exists():
            (root / "armed-recovery").touch()
            (root / "fail-one-clock").touch()
        if op == "clock" and scenario == "canary-budget"''')
shim = change(shim, '    if args[0] == "rev-parse":\n', '''
    if at == "head" and scenario in {"deploy-clock-timeout", "deploy-clock-control"}:
        if scenario == "deploy-clock-timeout":
            (root / "fail-one-clock").touch()
        raise SystemExit(17)
    if args[0] == "rev-parse":
''')


def new_box(scratch):
    box = tests.box.__wrapped__(Path(scratch))
    for file in (box / "shims").iterdir():
        file.write_text(file.read_text().split("\n", 1)[0] + "\n" + shim)
    (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/a\n")
    return box


def invoke(box, program, scenario):
    (box / "calls").unlink(missing_ok=True)
    outcome, calls = tests.run_box_script(box, program, scenario)
    attempt = (box / "run/bts-deploy/attempt").read_text().split()[5]
    marker = (box / "home/projects/bts/data/.heartbeat").read_text()
    evidence = dict(scenario=scenario, status=outcome.returncode, attempt=attempt, marker=marker,
                    stopped=(box / "stopped").exists(), head=(box / "head").read_text(),
                    stdout=outcome.stdout, stderr=outcome.stderr, calls=calls)
    print(json.dumps(evidence, ensure_ascii=False), flush=True)
    assert outcome.returncode == 1 and evidence["head"] == tests.TARGET
    if scenario.endswith("timeout"):
        assert calls.count(["clock-timed-out-once"]) == 1
        assert not evidence["stopped"] and attempt == "begun" and json.loads(marker)["state"] == "running"
        assert not any(call[:3] == ["systemctl", "--user", "stop"] or call[0] == "curl" for call in calls)
        assert "Scheduler stopped: unverified." in outcome.stdout
    else:
        assert evidence["stopped"] and attempt == "contained" and json.loads(marker)["state"] == "deploy_failed"
    return evidence


with TemporaryDirectory(prefix="wrd3-containment-", dir=HERE) as scratch:
    box = new_box(scratch)
    invoke(box, DEPLOY, "deploy-clock-timeout")
    invoke(box, RECOVER, "recovery-clock-timeout")
    invoke(box, RECOVER, "recovery-clock-control")

with TemporaryDirectory(prefix="wrd3-containment-", dir=HERE) as scratch:
    invoke(new_box(scratch), DEPLOY, "deploy-clock-control")
