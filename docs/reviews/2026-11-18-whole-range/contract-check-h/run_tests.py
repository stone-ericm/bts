import os
from pathlib import Path
import socket
import sys

import pytest


def offline(*args, **kwargs):
    raise OSError('WR H contract check: network disabled')


for name in ('connect', 'connect_ex', 'sendto', 'sendmsg'):
    if hasattr(socket.socket, name):
        setattr(socket.socket, name, offline)
socket.getaddrinfo = offline
socket.gethostbyname = offline
socket.gethostbyname_ex = offline

check = Path(__file__).resolve().parent
mode, batch, *args = sys.argv[1:]
root = check / 'scratch'
os.chdir(root)
sys.path.insert(0, str(root))
sys.path.insert(0, str(check / 'base' / 'src' if mode == 'base' else root / 'src'))
import bts

files = (check / 'test-files.txt').read_text().splitlines()
if args:
    selected = args
else:
    lo, hi = map(int, batch.split('-'))
    selected = files[lo - 1:hi]
log = check / f'pytest-{mode}-{batch}.log'
with log.open('w') as out:
    real_out = sys.stdout
    sys.stdout = sys.stderr = out
    try:
        print(f'bts={bts.__file__}\nfiles={selected}', flush=True)
        code = pytest.main(['-q', '-m', 'not slow', '-p', 'no:cacheprovider', '--tb=short', *selected])
    finally:
        sys.stdout = sys.stderr = real_out
print(f'{log.name}: {log.read_text().splitlines()[-1]}')
raise SystemExit(code)
