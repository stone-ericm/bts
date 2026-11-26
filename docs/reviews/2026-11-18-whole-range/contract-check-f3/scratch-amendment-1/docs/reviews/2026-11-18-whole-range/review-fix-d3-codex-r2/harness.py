"""Load workflow literals and emit short evidence from temporary shim boxes."""

import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WORKFLOW = REPO / ".github/workflows/deploy.yml"
SPEC = importlib.util.spec_from_file_location("workflow_tests", REPO / "tests/test_deploy_workflow.py")
tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tests)
DEPLOY, RECOVER = tests.scripts_and_steps()[0]


def source_at(revision, path, cwd=REPO):
    return subprocess.run(["git", "show", f"{revision}:{path}"], cwd=cwd, capture_output=True,
                          check=True, text=True, timeout=5).stdout


def replace_one(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def make_box(path, shim, harness=tests):
    box = harness.box.__wrapped__(path)
    for file in (box / "shims").iterdir():
        file.write_text(file.read_text().split("\n", 1)[0] + "\n" + shim)
    return box


def invoke(box, label, script, scenario="active", *, runner=tests.run_box_script, **kwargs):
    (box / "calls").unlink(missing_ok=True)
    result, calls = runner(box, script, scenario, **kwargs)
    short_calls = []
    for call in calls:
        if call[0] in {"git", "systemctl", "flock", "pgrep", "curl", "sleep", "mv", "probe"}:
            short_calls.append(call)
        elif call[0] == "python3" and len(call) > 3 and call[1] == "-c":
            short_calls.append(["python3", *call[3:]])
    def read(path):
        entry = box / path
        return entry.read_text() if entry.is_file() else None
    evidence = dict(case=label, rc=result.returncode, stdout=result.stdout.splitlines(), stderr=result.stderr,
                    head=read("head"), enabled=read("enabled"), stopped=(box / "stopped").exists(),
                    stamp=read("home/projects/bts/data/.last_deploy_iso"),
                    heartbeat=read("home/projects/bts/data/.heartbeat"),
                    attempt=read("run/bts-deploy/attempt"), calls=short_calls)
    print(json.dumps(evidence, ensure_ascii=False), flush=True)
    return evidence


def posts(evidence):
    return [call for call in evidence["calls"] if call[0] == "curl" and "--data-raw" in call]
