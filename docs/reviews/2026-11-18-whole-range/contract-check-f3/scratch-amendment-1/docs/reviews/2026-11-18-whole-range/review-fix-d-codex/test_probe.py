from hashlib import sha256
import difflib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHANGED = ('test_deploy_canary.py', 'test_deploy_hold_monitor.py', 'test_deploy_hold_tool.py',
           'test_deploy_workflow.py', 'test_entry_intent.py', 'test_install_deploy_guard.py', 'test_web_saver.py')
COMPATIBILITY = ('test_heartbeat_staleness.py', 'test_heartbeat_churn.py', 'test_lightgbm_optional.py',
                 'test_scheduler.py')
GUARD = 'scripts/install-deploy-guard-hetzner.sh'
MONITOR = 'scripts/check_heartbeat.py'
CLI = 'src/bts/cli.py'
WEB = 'src/bts/web.py'
RELEASE = '                release(root, record, "deploy")\n'
ACTIVATE = '                activate_scheduler(uid, authorized_at)\n'

MUTANTS = (
    ('pending-scheduler', GUARD, ((
        '            return 0 if clear(root) else 1\n',
        '            record = read_hold(root)\n'
        '            return 0 if (record and record["state"] == "pending") or clear(root) else 1\n'),)),
    ('arm-without-writer-lock', GUARD, (
        ('from contextlib import contextmanager\n', 'from contextlib import contextmanager, nullcontext\n'),
        ('                with lock(root, "writers.lock", fcntl.LOCK_EX):\n', '                with nullcontext():\n'))),
    ('release-before-verify', GUARD, ((RELEASE, ''), (ACTIVATE, RELEASE + ACTIVATE))),
    ('skip-guard-stale', GUARD, ((
        '    if hashlib.sha256(script).hexdigest().encode() + b"\\n" != fingerprint:',
        '    if False and hashlib.sha256(script).hexdigest().encode() + b"\\n" != fingerprint:'),)),
    ('ping-success-with-hold', MONITOR, ((
        '    with _hold_snapshot(args.deploy_hold) as verified:\n',
        '    if args.deploy_hold.exists():\n        return _legacy_check(args)\n'
        '    with _hold_snapshot(args.deploy_hold) as verified:\n'),)),
    ('dm-skips-contest-state', CLI, ((
        '    if intent.intent == "research":\n        report("research; contest state skipped", ok=True)',
        '    if intent.intent == "research" or intent.delivery == "dm":\n'
        '        report("research; contest state skipped", ok=True)'),)),
    ('failed-hook-noop', GUARD, ((
        '        if record is None or record["deploy_id"] != run_id:\n',
        '        if record is None or record["deploy_id"] != run_id or record["state"] == "failed":\n'),)),
    ('saver-guard-fails-open', WEB, ((
        '            except OSError:\n                clear = False\n            if clear:',
        '            except OSError:\n                clear = True\n            if clear:'),)),
)


def run(root, names, output):
    env = {**os.environ, 'TZ': 'America/New_York', 'PYTHONDONTWRITEBYTECODE': '1', 'UV_OFFLINE': '1',
           'UV_CACHE_DIR': str(HERE / 'uv-local-cache'),
           'UV_PYTHON_DOWNLOADS': 'never', 'GIT_TERMINAL_PROMPT': '0',
           'MPLCONFIGDIR': str(HERE / 'matplotlib'),
           'PYTHONPATH': os.pathsep.join(map(str, (HERE / 'local_only', root / 'src', root)))}
    command = [sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '-q', '--tb=short', '-ra',
               *(f'tests/{name}' for name in names)]
    with output.open('w') as stream:
        print('Command', command, 'cwd', root, file=stream, flush=True)
        result = subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL, timeout=720,
                                stdout=stream, stderr=subprocess.STDOUT)
    detail = output.read_text()
    selected_failures = re.findall(r'^FAILED .*', detail, re.M)
    print(output.name, 'rc', result.returncode, *detail.splitlines()[-3:], sep='\n', flush=True)
    return result.returncode, selected_failures


def mutate():
    with tempfile.TemporaryDirectory(prefix='mutation-scratch-', dir=HERE) as directory:
        root = Path(directory)
        for folder in ('src', 'scripts', 'tests', '.github'):
            shutil.copytree(REPO / folder, root / folder, ignore=shutil.ignore_patterns('__pycache__', '.DS_Store'))
        shutil.copy2(REPO / 'pyproject.toml', root / 'pyproject.toml')
        baseline_code, _ = run(root, CHANGED, HERE / 'mutation-control.log')
        if baseline_code:
            print('Scratch control failed. Mutation results would be invalid.', flush=True)
            return 1
        invalid = False
        for identifier, fname, replacements in MUTANTS:
            path = root / fname
            original = path.read_text()
            altered = original
            for before, after in replacements:
                assert altered.count(before) == 1, (identifier, before)
                altered = altered.replace(before, after)
            print('mutating', identifier, fname, sha256(altered.encode()).hexdigest(), flush=True)
            print(''.join(difflib.unified_diff(original.splitlines(keepends=True), altered.splitlines(keepends=True),
                                               fromfile=fname, tofile=fname)), end='', flush=True)
            try:
                path.write_text(altered)
                code, failures = run(root, CHANGED, HERE / f'mutation-{identifier}.log')
            finally:
                path.write_text(original)
            outcome = 'SURVIVED' if code == 0 else ('CAUGHT' if code == 1 and failures else 'INVALID')
            invalid |= outcome == 'INVALID'
            print(identifier, outcome, '; '.join(failures), flush=True)
        return int(invalid)


if __name__ == '__main__':
    if sys.argv[1:] == ['baseline']:
        raise SystemExit(run(REPO, CHANGED + COMPATIBILITY, HERE / 'focused-pytest.log')[0])
    if sys.argv[1:] == ['all']:
        baseline = run(REPO, CHANGED + COMPATIBILITY, HERE / 'focused-pytest.log')[0]
        mutation_status = mutate()
        raise SystemExit(int(bool(baseline or mutation_status)))
    assert sys.argv[1:] == ['mutations']
    raise SystemExit(mutate())
