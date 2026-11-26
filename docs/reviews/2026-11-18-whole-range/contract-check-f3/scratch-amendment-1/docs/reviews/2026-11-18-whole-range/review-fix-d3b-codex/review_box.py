"""Isolate workflow-shell review cases on disposable shim boxes."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location("workflow_tests", ROOT / "tests/test_deploy_workflow.py")
tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tests)
DEPLOY, RECOVER = tests.scripts_and_steps()[0]
OLD, TARGET = tests.OLD, tests.TARGET
SIGNALS = ("HUP", "INT", "TERM", "PIPE")
PRIORS = ("committed", "contained", "begun")
ENV = "BTS_SCHEDULER_HEARTBEAT_PING_URL"


def record(state="begun", owner="123", attempt="1", target=TARGET, old=OLD, tail=None):
    if tail is None:
        tail = "sync verified verified verified" if state == "contained" else "- - - -"
    return f"bts_deploy_attempt_v1 {owner} {attempt} {target} {old} {state} {tail}\n"


def replace(source, old, new, times=1):
    assert source.count(old) == times, old
    return source.replace(old, new)


def inject_shim(fragment):
    return replace(tests.SHIM, "if command == \"sudo\":", fragment + '\nif command == "sudo":')


@contextmanager
def box(initial=None, shim=None, url="https://checks.invalid/review"):
    with TemporaryDirectory(prefix="review-box-", dir=HERE) as directory:
        path = tests.box.__wrapped__(Path(directory))
        if shim is not None:
            for executable in (path / "shims").iterdir():
                executable.write_text(executable.read_text().split("\n", 1)[0] + "\n" + shim)
        runtime = path / "run/bts-deploy"
        runtime.mkdir(mode=0o700, parents=True)
        if initial is not None:
            (runtime / "attempt").write_text(initial)
        (path / "enabled").write_text("enabled")
        (path / "home/projects/bts/data/.heartbeat").write_text('{"state":"running"}')
        (path / "home/projects/bts/.env").write_text(f"{ENV}={url}\n")
        yield path


def group(root, path, population):
    location = root / "sys/fs/cgroup" / path.lstrip("/")
    location.mkdir()
    (location / "cgroup.events").write_text(f"populated {population}\n")


def cgroups(calls):
    return [c[-1] for c in calls if c[0] == "python3" and c[3:4] == ["cgroup"]]


def writes(calls):
    return [c for c in calls if c[0] == "systemctl" and set(c[1:]) &
            {"disable", "enable", "stop", "restart", "daemon-reload"}]


def posts(calls):
    return [c for c in calls if c[0] == "curl" and "--data-raw" in c]


def invoke(root, label, script=DEPLOY, scenario="active", **kwargs):
    (root / "calls").unlink(missing_ok=True)
    result, calls = tests.run_box_script(root, script, scenario, **kwargs)
    def read(file):
        path = root / file
        return path.read_text() if path.is_file() else None
    state = dict(
        rc=result.returncode, out=result.stdout, err=result.stderr,
        head=read("head"), link=read("enabled"),
        attempt=read("run/bts-deploy/attempt"),
        marker=read("home/projects/bts/data/.heartbeat"),
        stamp=read("home/projects/bts/data/.last_deploy_iso"),
        stopped=(root / "stopped").exists(), posts=len(posts(calls)), groups=cgroups(calls),
        writes=writes(calls),
    )
    print(json.dumps(dict(case=label, **state), ensure_ascii=False), flush=True)
    return state, calls


def contained(state, step=None, *, paged=True, fresh_marker=True):
    assert state["rc"] == 1 and state["stopped"] and "CANARY PASSED" not in state["out"], state
    assert "DEPLOY NOT STARTED" not in state["out"] and state["link"] == "disabled", state
    assert state["posts"] == int(paged), state
    if fresh_marker:
        assert json.loads(state["marker"])["state"] == "deploy_failed", state
    if step:
        assert f"contained {step} " in state["attempt"], state


def shell_function(script, name):
    return re.search(r"^" + name + r"\(\) \{\n.*?^\}", script, re.M | re.S)[0]
