from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType

HERE = Path(__file__).resolve().parent
R1 = HERE.parent
REPO = HERE.parents[2]
OLD = '46ce572d'
HEAD = '0645be4b'
TARGET = 'b' * 40
PREVIOUS = 'a' * 40
RUN = '1800000000-review'
GOOD = '2027-07-15'
PY = REPO / '.venv/bin/python'


def tree(revision):
    return HERE / 'trees' / revision


def guard_source(revision):
    installer = tree(revision) / 'scripts/install-deploy-guard-hetzner.sh'
    raw = installer.read_text()
    marker = 'BTS_DEPLOY_HOLD_PYTHON'
    start = raw.index(f"<<'{marker}'\n") + len(f"<<'{marker}'\n")
    end = raw.index(f'\n{marker}\n', start)
    return installer, '\n' * raw[:start].count('\n') + raw[start:end] + '\n'


@contextmanager
def fixture(revision):
    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        path = Path(temporary)
        tool = ModuleType('guard_' + revision)
        installer, code = guard_source(revision)
        exec(compile(code, str(installer), 'exec'), vars(tool))
        assert os.geteuid() != 0
        tool.ROOT = path / 'guard'
        tool.ROOT.mkdir(mode=0o755)
        tool.UID = os.geteuid()
        tool.PROJECT = str(path / 'project')
        (Path(tool.PROJECT) / 'data').mkdir(parents=True)
        (tool.ROOT / 'guard-install.sha256').write_text(hashlib.sha256(installer.read_bytes()).hexdigest() + '\n')
        for filename in ('transaction.lock', 'state.lock', 'writers.lock'):
            (tool.ROOT / filename).touch(mode=0o644)
        tool.PROC_LOCKS = path / 'proc-locks'
        tool.PROC_LOCKS.write_text('')
        yield tool, path


def baseline_env(revision):
    source = tree(revision)
    return {**os.environ, 'TZ': 'America/New_York', 'PYTHONDONTWRITEBYTECODE': '1',
            'UV_OFFLINE': '1', 'UV_PYTHON_DOWNLOADS': 'never', 'GIT_TERMINAL_PROMPT': '0',
            'PYTHONPATH': os.pathsep.join(map(str, (R1 / 'local_only', source / 'src', source))),
            'MPLCONFIGDIR': str(HERE / 'matplotlib'), 'UV_CACHE_DIR': str(HERE / 'uv-cache')}


@dataclass
class Captured:
    rc: int
    stdout: str
    stderr: str


def capture(invoke):
    output, errors = StringIO(), StringIO()
    with redirect_stdout(output), redirect_stderr(errors):
        try:
            result = invoke()
        except SystemExit as exc:
            result = exc.code
    return Captured(result, output.getvalue(), errors.getvalue())
