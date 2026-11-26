"""Show the real alert a green 10:15-start mutation would suppress."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from mutation_probe import CASES, ROOT, scratch_copy


def replay(mode):
    import bts
    from probe_support import Replay, load_line
    from tests.test_wr_f1_wrong_game import DAY, WRONG, row

    print(mode, bts.__file__)
    assert '--cron' in load_line('f5')
    with TemporaryDirectory() as directory:
        world = Replay(Path(directory))
        world.commit_at('10:20', at='09:40')
        early = world.invoke('09:45', pending=[row()])
        world.capture([WRONG], f'{DAY}T140000Z')
        ten = world.invoke('10:00', pending=[row()])
        cutoff = world.invoke('10:15', pending=[row()])
        assert cutoff['receipt']['outcome'] == 'outside_window'
        if mode == 'candidate':
            assert early['receipt']['outcome'] == 'base_slot_available' and not early['calls']
            assert ten['calls'] and 'entry is recorded in game 102' in ten['sent'][0]
        else:
            assert early['receipt']['outcome'] == 'observed' and early['marker']['status'] == 'confirmed'
            assert ten['receipt']['outcome'] == 'already_confirmed' and not (ten['calls'] or ten['sent'])
        print('09:45', early['receipt']['outcome'], '10:00', ten['receipt']['outcome'], ten['sent'])


if __name__ == '__main__':
    if len(sys.argv) > 1:
        replay(sys.argv[1])
    else:
        with TemporaryDirectory() as directory:
            copy = Path(directory)
            scratch_copy(copy)
            name, before, after = CASES['start-1015']
            target = copy / name
            current = target.read_text()
            assert current.count(before) == 1
            target.write_text(current.replace(before, after))
            for mode, tree in (('candidate', ROOT), ('mutation', copy)):
                subprocess.run([sys.executable, str(Path(__file__).resolve()), mode], cwd=tree, check=True,
                               env=os.environ | {'PYTHONPATH': str(tree / 'src')}, timeout=10)
