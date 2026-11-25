from pathlib import Path
import runpy
import subprocess
from unittest.mock import patch


original_run = subprocess.run


def run_with_path(*args, **kwargs):
    command = args[0]
    if command[1] == '-c' and 'saved = picks.load_pick' in command[2]:
        script = command[2].replace('saved = picks.load_pick(day, folder)',
                                    'folder = Path(folder)\nsaved = picks.load_pick(day, folder)')
        command = [*command[:2], script, *command[3:]]
        args = (command, *args[1:])
    return original_run(*args, **kwargs)


if __name__ == '__main__':
    with patch.object(subprocess, 'run', run_with_path):
        runpy.run_path(str(Path(__file__).with_name('probe_rollback.py')), run_name='__main__')
