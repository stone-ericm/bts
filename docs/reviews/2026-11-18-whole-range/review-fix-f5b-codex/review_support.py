from contextlib import contextmanager
from datetime import date, datetime
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTRACT = ROOT / 'docs/reviews/2026-11-18-whole-range/contract-check-f5b'
ARCHIVE = HERE.parent / 'f5-records/docs/reviews/2026-11-18-whole-range'
LEGACY = ARCHIVE / 'review-fix-f5-codex'
ROUND2 = ARCHIVE / 'review-fix-f5-codex-r2'
sys.path.extend((str(LEGACY), str(ROUND2)))
sys.path.insert(0, str(ROOT))

import bts
import probe_support
from tests import test_entry_intent
from tests.test_wr_f1_wrong_game import RIGHT, WRONG, row

DAY = probe_support.DAY
ET = probe_support.ET


def assert_source(src):
    actual = Path(bts.__file__).resolve()
    print('bts.__file__:', actual, flush=True)
    assert actual.is_relative_to(Path(src).resolve()), (actual, src)


def load_lines(version):
    return json.loads((HERE / 'generated.json').read_text())[version]


def load_line(version):
    line, = [line for line in load_lines(version)
             if test_entry_intent.ENTRY_CMD in line and not line.startswith('#')]
    return line


probe_support.load_line = load_line


class World(probe_support.Replay):
    @property
    def early(self):
        return self.marker.with_name('pick_entry_check.early') / f'{self.day}.json'

    def early_state(self):
        return json.loads(self.early.read_bytes()) if self.early.is_file() else None

    def edit_pitch(self, daily, pitch, leg='pick'):
        from bts.picks import save_pick

        getattr(daily, leg).game_time = datetime.fromisoformat(
            f'{self.day}T{pitch}').replace(tzinfo=ET).isoformat()
        save_pick(daily, self.picks)

    def cron(self, at, **kwargs):
        if at in {slot.strftime('%H:%M') for slot in probe_support.clock_slots(load_line('f5'), self.day)}:
            return self.invoke(at, **kwargs)
        return None


def installed(root, script, intent, seed):
    root.mkdir(exist_ok=True)
    box = test_entry_intent.box.__wrapped__(root)
    box.config.write_text(f'[scheduler]\nentry_intent = "{intent}"\n'
                          f'pick_delivery = "{"dm" if intent == "enter" else "private"}"\n'
                          + ('contest_state_required = true\n' if intent == 'enter' else ''))
    box.cron.write_text(seed)

    def run(*args, **kwargs):
        return subprocess.run(*args, **(kwargs | {'timeout': 5}))

    with (patch.object(test_entry_intent, 'SCRIPT', script),
          patch.object(test_entry_intent, 'subprocess', SimpleNamespace(run=run))):
        shown = box.run('show')
        assert shown.returncode == 0, shown
        assert box.cron.read_text() == seed
        added = box.run('install')
        assert added.returncode == 0, added
        lines = box.lines()
        assert shown.stdout.split('Would install:\n')[1].splitlines() == [
            line for line in lines if line.endswith('# BTS-HETZNER')]
        removed = box.run('remove')
        assert removed.returncode == 0, removed
        assert box.lines() == [line for line in lines if not line.endswith('# BTS-HETZNER')]
        return lines


class ReadTimedOut(RuntimeError):
    pass


@contextmanager
def deadline(seconds=0.3):
    def fail(*_):
        raise ReadTimedOut('reader exceeded deadline')

    old = signal.signal(signal.SIGALRM, fail)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def health(world, at='23:59'):
    from bts.health import pick_entry

    now = datetime.fromisoformat(f'{world.day}T{at}').replace(tzinfo=ET)
    with deadline():
        return [(a.level, a.message) for a in pick_entry.check(world.picks, date.fromisoformat(world.day), now)]
