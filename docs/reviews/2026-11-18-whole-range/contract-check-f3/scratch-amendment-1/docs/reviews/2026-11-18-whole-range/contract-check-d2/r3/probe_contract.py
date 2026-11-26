"""Test V3 record-state sequences and the starting readers using local fixtures."""

import ast
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from io import BytesIO
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


def child(args, *, env=None):
    return subprocess.run(
        args, stdin=subprocess.DEVNULL, capture_output=True, timeout=3,
        env=env, check=True,
    ).stdout


def git(revision, path):
    return child(["/usr/bin/git", "show", f"{revision}:{path}"])


def digest(value):
    return hashlib.sha256(value).digest()


def unit(fragment, needs_reload):
    cat = b"# /home/bts/.config/systemd/user/example.service\n" + fragment
    show = (
        b"LoadState=loaded\nFragmentPath=/home/bts/.config/systemd/user/example.service\n"
        b"DropInPaths=\nNeedDaemonReload=" + needs_reload + b"\n"
    )
    return digest(cat), digest(show)


@dataclass(frozen=True)
class Record:
    state: str
    before: tuple
    after: tuple = ()


def verifies(record, observed, hold):
    if record is None:
        return False
    if record.state == "switching":
        return hold == "failed" and all(
            actual in (before, after)
            for actual, before, after in zip(observed, record.before, record.after, strict=True)
        )
    return observed == record.before


def switch_probe():
    old = (None, unit(b"old scheduler\n", b"no"), unit(b"old dashboard\n", b"no"))
    new = (digest(b"new cron\n"), unit(b"new scheduler\n", b"no"), unit(b"new dashboard\n", b"no"))
    provisioned = Record("provisioned", old)
    switching = Record("switching", old, new)
    installed = Record("installed", new)
    assert not verifies(None, old, "provisioned")
    assert verifies(provisioned, old, "provisioned")
    assert not verifies(provisioned, replace_cron(old), "provisioned")
    for flags in product((False, True), repeat=3):
        mix = tuple((new if changed else old)[i] for i, changed in enumerate(flags))
        assert verifies(switching, mix, "failed")
        assert not verifies(switching, mix, "pending")
    assert verifies(switching, old, "failed")  # reboot before first artifact
    assert not verifies(switching, old, "pending")  # same job evidence, after re-arming
    dirty = new[0], unit(b"new scheduler\n", b"yes"), old[2]
    assert not verifies(switching, dirty, "failed")
    assert all(dirty[1][i] not in (old[1][i], new[1][i]) for i in (1,))
    assert verifies(installed, new, "failed")  # canary failure leaves proof
    assert verifies(installed, new, "pending")  # a later owner before switch
    assert not verifies(installed, (old[0], new[1], new[2]), "failed")  # lost cron rename
    assert verifies(replace(installed), new, None)  # compatible root refresh leaves jobs alone
    print("8 before/after mixtures pass failed admission and fail pending recheck; "
          "expected reload-dirty cut fails; missing/provisioned/installed controls pass.")


def replace_cron(observed):
    return digest(b"foreign cron\n"), *observed[1:]


def readers(directory):
    source = git("fix/wr-deploy", "scripts/install-deploy-guard-hetzner.sh")
    code = source.split(b"<<'BTS_DEPLOY_HOLD_PYTHON'\n", 1)[1].split(
        b"\nBTS_DEPLOY_HOLD_PYTHON\n", 1,
    )[0]
    tool = {"__name__": "fixture"}
    exec(compile(code, "<guard>", "exec"), tool)
    tool.update(ROOT=directory, UID=os.getuid())
    monitor = {"__name__": "fixture"}
    exec(compile(git("fix/wr-deploy", "scripts/check_heartbeat.py"), "<monitor>", "exec"), monitor)
    hold = {
        "schema": "bts_deploy_hold_v1", "deploy_id": "1-test",
        "owner_unit": "bts-deploy-1-test.service", "state": "failed",
        "step": "abandoned", "target": "a" * 40, "previous": "unknown",
        "armed_at": 1, "updated_at": 1,
    }
    hold_path = directory / "hold"
    monitor["HOLD_PATH"] = hold_path
    now = datetime.fromtimestamp(2, timezone.utc)
    with tool["directory"]() as root:
        cases = [
            (hold, True, True),
            (dict(hold, step="install-jobs"), False, True),
            (dict(hold, step="provisioned", deploy_id="provision-1", owner_unit=None), False, False),
        ]
        for record, guard_valid, monitor_valid in cases:
            tool["publish_json"](root, "hold", record)
            try:
                accepted = tool["read_hold"](root) == record
            except ValueError:
                accepted = False
            assert accepted is guard_valid
            # Model the updated guard scope in front of the still-unmodified
            # production monitor. It validates the returned JSON a second time.
            result = subprocess.CompletedProcess([], 1, json.dumps(record).encode())
            with patch.object(subprocess, "run", return_value=result):
                decision, reason = monitor["_hold_decision"](hold_path, directory / "heartbeat", now)
            assert decision == "fail"
            assert ("unreadable" not in reason) is monitor_valid
    print("Starting guard and monitor both reject the proposed owner-less record as data.")


def response(revision, directory):
    source = ast.parse(git(revision, "src/bts/web.py"))
    post = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == "do_POST")
    calls = []

    def saver(*args, **kwargs):
        calls.append((args, kwargs))
        return 200, "fixture saver write"

    namespace = {
        "saver_transition_response": saver, "PICKS_DIR": directory,
        "_same_origin": lambda *args: True, "datetime": datetime, "timezone": timezone,
    }
    exec(compile(ast.Module(body=[post], type_ignores=[]), "<web handler>", "exec"), namespace)
    body = b"expected_prior=available&new_state=used"
    statuses = []
    handler = SimpleNamespace(
        path="/saver/transition", headers={"Content-Length": str(len(body))},
        rfile=BytesIO(body), wfile=BytesIO(), client_address=("127.0.0.1", 0),
        send_response=statuses.append, send_header=lambda *args: None, end_headers=lambda: None,
    )
    with patch.dict(os.environ, {"BTS_TEST_DEPLOY_GUARD_DIR": str(directory)}):
        namespace["do_POST"](handler)
    return statuses, len(calls)


def environment_probe(directory, guard_directory):
    uv = "/opt/homebrew/bin/uv"
    project = directory / "checkout"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "d2-fixture"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n',
    )
    current, older = project / ".venv", directory / "older-venv"
    for env_path, label in ((current, "guarded"), (older, "october")):
        child([sys.executable, "-m", "venv", "--without-pip", str(env_path)])
        site = next(env_path.glob("lib/python*/site-packages"))
        package = site / "bts"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "web.py").write_text(f"print({label!r})\n")

    fixed_env = {
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin", "HOME": str(directory),
        "UV_CACHE_DIR": str(directory / "uv-cache"), "PYTHONDONTWRITEBYTECODE": "1",
    }
    command = [uv, "run", "--offline", "--no-sync", "--no-python-downloads",
               "--directory", str(project), "python3", "-m", "bts.web"]
    assert child(command, env=fixed_env) == b"guarded\n"
    assert child(command, env=dict(fixed_env, UV_PROJECT_ENVIRONMENT=str(older))) == b"october\n"
    (guard_directory / "writers.lock").touch()
    (guard_directory / "writers.lock").chmod(0o644)
    assert (guard_directory / "hold").exists()
    assert response("fix/wr-deploy", guard_directory) == ([409], 0)
    assert response("619abda7", guard_directory) == ([303], 1)
    print("Offline uv picked alternate venv; with the same hold, guarded POST returned 409 "
          "and October POST reached the saver stub and returned 303.")


def main():
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    pgrep = subprocess.run(
        ["/usr/bin/pgrep", "-x", "victoria3"], stdin=subprocess.DEVNULL,
        capture_output=True, timeout=2,
    )
    assert pgrep.returncode in (0, 1), pgrep.stderr
    print(f"victoria3 present: {pgrep.returncode == 0}; nice 19")
    switch_probe()
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
        directory = Path(temp)
        guard = directory / "guard"
        guard.mkdir(mode=0o755)
        readers(guard)
        environment_probe(directory, guard)


if __name__ == "__main__":
    main()
