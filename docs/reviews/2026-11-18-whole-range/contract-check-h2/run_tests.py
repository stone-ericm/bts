import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest


def offline(*args, **kwargs):
    raise OSError('WR H contract check: network disabled')


for name in ('connect', 'connect_ex', 'sendto', 'sendmsg'):
    if hasattr(socket.socket, name):
        setattr(socket.socket, name, offline)
socket.getaddrinfo = socket.gethostbyname = socket.gethostbyname_ex = offline

check = Path(__file__).resolve().parent
mode, batch, *selected = sys.argv[1:]
root = check / 'scratch'
git_dir = subprocess.check_output(['git', 'rev-parse', '--absolute-git-dir'],
                                  cwd=check.parents[1], text=True, timeout=4).strip()
os.environ.update(GIT_DIR=git_dir, GIT_WORK_TREE=str(root), PYTHONDONTWRITEBYTECODE='1')
os.chdir(root)
source = check / 'base' / 'src' if mode == 'base' else root / 'src'
sys.path[:0] = [str(source), str(root)]
os.environ['PYTHONPATH'] = str(source)
import bts

if batch.endswith('-http404'):
    from urllib.error import HTTPError
    from bts import util

    def missing(req, **kwargs):
        raise HTTPError(getattr(req, 'full_url', str(req)), 404, 'offline test fixture', {}, None)

    util.urlopen = missing

if not selected:
    lo, hi = map(int, batch.split('-'))
    selected = (check / 'test-files.txt').read_text().splitlines()[lo - 1:hi]
print(f'bts={bts.__file__}\nfiles={selected}', flush=True)
raise SystemExit(pytest.main(['-q', '-m', 'not slow', '-p', 'no:cacheprovider', '--tb=short', *selected]))
