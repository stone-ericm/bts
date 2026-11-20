"""Extract the two reviewed trees and run the independent offline review harness."""

from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLD, HEAD, LEGACY = "46ce572d", "0645be4b", "b128854d"
FILES = ("src", "scripts", "tests", "config", "data", ".github", "docs/ops", "CLAUDE.md",
         "INCIDENT.md", "README.md", "pyproject.toml", "uv.lock")
GUARD = "scripts/install-deploy-guard-hetzner.sh"
MONITOR = "scripts/check_heartbeat.py"
CLI = "src/bts/cli.py"
WEB = "src/bts/web.py"
INCIDENT = "INCIDENT.md"
OPS = "docs/ops/2027-season-start.md"
RELEASE = '                release(root, record, "deploy", authorized_at=authorized_at, certified_date=certified_date)\n'
ACTIVATE = '                activate_scheduler(uid, authorized_at)\n'
FOCUSED = (
    "test_deploy_amendment3", "test_deploy_canary", "test_deploy_hold_monitor", "test_deploy_hold_tool",
    "test_deploy_workflow", "test_entry_intent", "test_install_deploy_guard", "test_web_saver",
)
COMPATIBILITY = ("test_heartbeat_staleness", "test_heartbeat_churn", "test_lightgbm_optional", "test_scheduler")

# Round one's fault set, with the release reorder ported to the amended signature.
ROUND1 = (
    ("r1-pending-scheduler", GUARD, (('            return 0 if clear(root) else 1\n',
     '            record = read_hold(root)\n'
     '            return 0 if (record and record["state"] == "pending") or clear(root) else 1\n'),)),
    ("r1-arm-without-writer-lock", GUARD, (
     ('from contextlib import ExitStack, contextmanager\n', 'from contextlib import ExitStack, contextmanager, nullcontext\n'),
     ('                with lock(root, "writers.lock", fcntl.LOCK_EX):\n', '                with nullcontext():\n'))),
    ("r1-release-before-verify", GUARD, ((RELEASE, ''), (ACTIVATE, RELEASE + ACTIVATE))),
    ("r1-skip-guard-stale", GUARD, (('    if hashlib.sha256(script).hexdigest().encode() + b"\\n" != fingerprint:',
     '    if False and hashlib.sha256(script).hexdigest().encode() + b"\\n" != fingerprint:'),)),
    ("r1-ping-success-with-hold", MONITOR, (('    with _hold_snapshot(args.deploy_hold) as verified:\n',
     '    if args.deploy_hold.exists():\n        return _legacy_check(args)\n'
     '    with _hold_snapshot(args.deploy_hold) as verified:\n'),)),
    ("r1-dm-skips-contest-state", CLI, ((
     '    if intent.intent == "research":\n        report("research; contest state skipped", ok=True)',
     '    if intent.intent == "research" or intent.delivery == "dm":\n'
     '        report("research; contest state skipped", ok=True)'),)),
    ("r1-failed-hook-noop", GUARD, (('        if record is None or record["deploy_id"] != run_id:\n',
     '        if record is None or record["deploy_id"] != run_id or record["state"] == "failed":\n'),)),
    ("r1-saver-guard-fails-open", WEB, (('            except OSError:\n                clear = False\n            if clear:',
     '            except OSError:\n                clear = True\n            if clear:'),)),
)

AMENDMENT = (
    ("f01", GUARD, (('if not gated or gated[1] not in CRON_WRITERS',
                    'if not gated:\n            continue\n        if gated[1] not in CRON_WRITERS'),)),
    ("f02", GUARD, (('fields.get("ActiveState") != "active"',
                     'fields.get("ActiveState") not in ("active", "deactivating")'),)),
    ("f03", GUARD, (('or fields["WorkingDirectory"] != PROJECT',
                     'or False and fields["WorkingDirectory"] != PROJECT'),)),
    ("f04", GUARD, (('                    if not verified_stop_dashboard(uid):',
                     '                    if False and not verified_stop_dashboard(uid):'),)),
    ("f05", CLI, (('if validate_runtime_config(config) or not intent.ok:', 'if not intent.ok:'),)),
    ("f06", GUARD, (('if release_date != certified_date:', 'if False and release_date != certified_date:'),)),
    ("f07", INCIDENT, (('bts set-contest-streak --picks-dir <the configured picks_dir> --date',
                        'bts set-contest-streak --date'),)),
    ("f08", GUARD, (('"released_at": int(time.time()) if authorized_at is None else authorized_at,',
                     '"released_at": int(time.time()),'),)),
    ("f09", GUARD, (('window(target, target, time.time() + 30, minimum=30)',
                     'window("", target, time.time() + 30)'),)),
    ("f10", MONITOR, (('            except FileNotFoundError:\n                return _legacy_check(args)',
                       '            except FileNotFoundError:\n                pass'),)),
    ("f11", GUARD, (('if claim in ("not-attempted", "claimed") and', 'if claim == "not-attempted" and'),)),
    ("f12", GUARD, (('return record is None or (record["state"] == "activating" and owner_alive(root, record))',
                     'return record is None or (record["state"] in ("activating", "pending") and owner_alive(root, record))'),)),
    ("f13", MONITOR, (('        record = _bounded_record(hold)\n', '        record = _bounded_record(HOLD_PATH)\n'),)),
    ("f14", GUARD, (('    print(line, flush=True)\n', ''),)),
    ("f15", GUARD, (('    except Refused:\n        address = ""\n', '    except Refused:\n        raise\n'),)),
    ("f16", OPS, (('Confirm it passes on the real configuration and record the classified result.',
                   'Record the classified result.'),)),
)


def setup():
    for rev, files in ((OLD, FILES), (HEAD, FILES), (LEGACY, (MONITOR, WEB))):
        folder = HERE / "trees" / rev
        if folder.is_dir():
            continue
        archive = subprocess.run(["git", "archive", rev, *files], cwd=REPO, capture_output=True,
                                 stdin=subprocess.DEVNULL, timeout=30, check=True).stdout
        folder.mkdir(parents=True)
        with tarfile.open(fileobj=BytesIO(archive)) as bundle:
            bundle.extractall(folder, filter="data")
        print(f"extracted {rev}", flush=True)


def environment(root):
    return {
        **os.environ,
        "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1", "UV_OFFLINE": "1",
        "UV_PYTHON_DOWNLOADS": "never", "GIT_ALLOW_PROTOCOL": "file", "GIT_TERMINAL_PROMPT": "0",
        "MPLCONFIGDIR": str(HERE / "matplotlib"), "UV_CACHE_DIR": str(HERE / "uv-cache"),
        "XDG_CACHE_HOME": str(HERE / "xdg-cache"),
        "PYTHONPATH": os.pathsep.join(str(entry) for entry in (HERE, root / "src", root)),
    }


def run(root, log, command, *, limit):
    with (HERE / log).open("w") as stream:
        print(command, "cwd", root, file=stream, flush=True)
        result = subprocess.run(command, cwd=root, env=environment(root), stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, timeout=limit)
    return result.returncode


def pytest_at(root, label, names):
    log = label + ".log"
    temp = HERE / "pytest-tmp"
    temp.mkdir(exist_ok=True)
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--tb=short", "-ra",
               "-k", "not test_do_post_saver_transition_socket_wiring",
               "--basetemp", str(temp / label), *(f"tests/{name}.py" for name in names)]
    try:
        code = run(root, log, command, limit=150)
    except subprocess.TimeoutExpired:
        return 124, "pytest timeout", []
    content = (HERE / log).read_text()
    summary = content.splitlines()[-1]
    failed = re.findall(r"^FAILED (.+)$", content, flags=re.M)
    return code, summary, failed


def mutate(which):
    with tempfile.TemporaryDirectory(prefix=f"mutations-{which}-", dir=HERE) as directory:
        root = Path(directory)
        shutil.copytree(HERE / "trees" / HEAD, root, dirs_exist_ok=True)
        return mutations_in(root, which)


def mutations_in(root, which):
    changes = ROUND1 if which == "round1" else AMENDMENT
    code, summary, _ = pytest_at(root, f"{which}-control", FOCUSED + (COMPATIBILITY if which == "round1" else ()))
    print(which, "control", code, summary, flush=True)
    if code:
        return code
    invalid = False
    for label, location, edits in changes:
        path = root / location
        original = path.read_bytes()
        text = original.decode()
        for before, after in edits:
            if text.count(before) != 1:
                raise ValueError(f"{label}: replacement anchor appears {text.count(before)} times: {before!r}")
            text = text.replace(before, after, 1)
        changed = text.encode()
        print(label, location, sha256(changed).hexdigest(), flush=True)
        try:
            path.write_bytes(changed)
            code, summary, failed = pytest_at(root, f"mutation-{label}", FOCUSED)
        finally:
            path.write_bytes(original)
        outcome = "SURVIVED" if code == 0 else "CAUGHT" if code == 1 and failed else "INVALID"
        print(label, outcome, summary, failed, flush=True)
        invalid |= outcome == "INVALID"
    return int(invalid)


def compare():
    result = 0
    for rev in (OLD, HEAD):
        root = HERE / "trees" / rev
        status = run(root, f"comparison-{rev}.log", [sys.executable, str(HERE / "compare.py"), rev], limit=75)
        print(rev, "comparison", status, flush=True)
        result |= bool(status)
    return result


def test_and_mutate():
    names = tuple(name for name in FOCUSED if name != "test_deploy_amendment3")
    code, summary, _ = pytest_at(HERE / "trees" / OLD, "base-focused", names)
    print(OLD, "focused", code, summary, flush=True)
    round1 = mutate("round1")
    amendment = mutate("amendment")
    return int(bool(code or round1 or amendment))


if __name__ == "__main__":
    assert os.geteuid() != 0
    action, *args = sys.argv[1:]
    if action == "setup":
        setup()
    elif action == "compare":
        raise SystemExit(compare())
    elif action == "mutate":
        raise SystemExit(mutate(*args))
    elif action == "pytest-and-mutate":
        raise SystemExit(test_and_mutate())
    else:
        raise SystemExit(f"unknown review action {action}")
