from datetime import datetime
from io import BytesIO
import os
import shutil
import subprocess
import sys
import tarfile

from review_env import HEAD, HERE, OLD, PY, R1, REPO, baseline_env, tree


SOURCES = ('src', 'scripts', 'tests', '.github', 'config', 'data', 'docs/ops', 'pyproject.toml',
           'uv.lock', 'CLAUDE.md', 'INCIDENT.md')
COMMANDS = {
    'canary': ('canary_probe.py',), 'cron': ('cron_probe.py',),
    'dashboard': ('dashboard_probe.py',), 'operator': ('operator_probe.py',),
    'preflight': ('preflight_probe.py',), 'monitor': ('monitor_probe.py',),
    'owner-command': ('owner_probe.py', '--command-probe'),
    **{f'owner-{mode}-{recovery}': ('owner_probe.py', mode, recovery)
       for mode in ('fail', 'TERM', 'KILL', 'lost-ssh', 'RuntimeMaxSec')
       for recovery in ('hook', 'preflight')},
}


def setup():
    for revision in (OLD, HEAD):
        destination = tree(revision)
        destination.mkdir(parents=True, exist_ok=False)
        result = subprocess.run(['/usr/bin/git', 'archive', revision, *SOURCES], cwd=REPO,
                                capture_output=True, stdin=subprocess.DEVNULL, timeout=15, check=True)
        with tarfile.open(fileobj=BytesIO(result.stdout), mode='r:') as bundle:
            bundle.extractall(destination, filter='data')
        fixture = destination / '.codex-review/wr-d'
        fixture.mkdir(parents=True)
        for file in (*R1.glob('*_probe.py'), R1 / 'brief.md'):
            copy = fixture / file.name
            shutil.copy2(file, copy)
            assert copy.read_bytes() == file.read_bytes()
        shutil.copytree(R1 / 'base', fixture / 'base', ignore=shutil.ignore_patterns('__pycache__'))
        print(revision, 'scratch sources and byte-identical round-1 scripts ready', flush=True)


def replay(revision, name):
    root = tree(revision)
    script, *args = COMMANDS[name]
    path = root / '.codex-review/wr-d' / script
    assert path.read_bytes() == (R1 / script).read_bytes()
    interpreter = '/usr/bin/python3' if name == 'monitor' else str(PY)
    env = {**baseline_env(revision), 'GIT_DIR': str(REPO / '.git'), 'GIT_WORK_TREE': str(root)}
    command = [interpreter, str(path), *args]
    out = HERE / f'unchanged-{revision}-{name}.log'
    with out.open('w') as stream:
        print('Replay', name, revision, datetime.now().isoformat(), file=stream, flush=True)
        try:
            proc = subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                  stdout=stream, stderr=subprocess.STDOUT, timeout=24)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = 124
        print('exit', rc, file=stream, flush=True)
    print(revision, name, rc, out.name, flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['setup']:
        setup()
    else:
        revision, *names = sys.argv[1:]
        for name in names:
            replay(revision, name)
