"""Interrupt the real installer after its pipe has filled the crontab shim's staging file."""
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

from installer_probe import ARTIFACTS, ENTER, FOREIGN, OLD, installer, normalized


def interrupt(root, script):
    with patch.object(installer, 'SCRIPT', script):
        box = installer.box.__wrapped__(root)
        box.config.write_text(ENTER)
        old = (FOREIGN + OLD + '\n').encode()
        box.cron.write_bytes(old)
        shim = root / 'shims/crontab'
        normal = shim.read_bytes()
        installer._exe(shim, '#!/bin/sh\n'
                       f'f="{box.cron}"\n'
                       'case "$1" in\n'
                       ' -l) cat "$f" ;;\n'
                       ' -) cat > "$f.new" || exit 1\n'
                       '    touch "$f.ready"\n'
                       '    exec sleep 10 ;;\n'
                       ' *) exit 98 ;;\n'
                       'esac\n')
        env = {'HOME': str(root / 'home'), 'PATH': f'{root}/shims:/usr/bin:/bin',
               'HEALTHCHECKS_PING_URL': 'https://hc-ping.invalid/test-only'}
        proc = subprocess.Popen(['bash', str(script), 'install'], env=env, start_new_session=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            expiry = time.monotonic() + 6
            while not Path(f'{box.cron}.ready').is_file() and time.monotonic() < expiry and proc.poll() is None:
                time.sleep(0.02)
            assert Path(f'{box.cron}.ready').is_file()
            staged = Path(f'{box.cron}.new').read_bytes()
            assert box.cron.read_bytes() == old and staged != old
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            proc.communicate(timeout=4)
        assert box.cron.read_bytes() == old and proc.returncode != 0
        shim.write_bytes(normal)
        assert box.run('install').returncode == 0
        assert box.cron.read_bytes() == staged
        return staged


if __name__ == '__main__':
    base_script = ARTIFACTS / 'base/scripts/cron-setup-hetzner.sh'
    with TemporaryDirectory() as directory:
        parent = Path(directory)
        scratch = parent / 'state'
        base = interrupt(scratch, base_script)
        scratch.rename(parent / 'base-state')
        changed = interrupt(scratch, installer.SCRIPT)
    assert base == normalized(changed)
    print('Base and F5 left the old crontab intact before shim publication; reinstall published the full staged rows.')
