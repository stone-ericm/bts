from datetime import date
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import daily_decision
from probe_lifecycle import check_record, deliver, load_pick_after_check, slate


ROOT = Path(__file__).resolve().parents[2]
DAY = date(2027, 7, 15)


def rollback(root, picks_dir):
    archive = subprocess.run(['git', 'archive', '0a2bec10', 'src'], cwd=ROOT, capture_output=True,
                             check=True, timeout=5).stdout
    subprocess.run(['tar', '-xf', '-', '-C', str(root)], input=archive, check=True, timeout=5)
    src = root / 'src'
    env = os.environ | {'PYTHONPATH': os.pathsep.join((str(src), str(ROOT)))}
    script = '''from pathlib import Path
import sys
import bts
from bts import daily_decision as dd, picks
folder, day, imported = sys.argv[1:]
assert Path(bts.__file__).resolve() == Path(imported)
saved = picks.load_pick(day, folder)
record = dd.load_decision(day, folder)
assert record['objective'] == saved.policy_decision['objective'] == 'longest_streak'
assert dd.decision_objective(record) == 'unknown'
assert dd.is_scoreable_commit(day, folder, saved) and saved.double_down is None
print(bts.__file__, 'parsed the actual delivery, normalized unknown, single remains scoreable')
'''
    subprocess.run([sys.executable, '-c', script, str(picks_dir), str(DAY), str(src / 'bts/__init__.py')],
                   cwd=ROOT, env=env, check=True, timeout=7)


if __name__ == '__main__':
    with TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        folder = root / 'picks'
        with slate(folder, DAY) as (cfg, _, statuses):
            saved, selection = load_pick_after_check(folder, DAY, cfg, statuses)
            with patch('bts.dm.send_dm', return_value='rollback-capture'):
                assert deliver(saved, cfg, folder, selection)
            path = daily_decision.decision_path(str(DAY), folder)
            raw = path.read_bytes()
            check_record(folder, saved, 'delivered')
        rollback(root, folder)
        assert path.read_bytes() == raw
