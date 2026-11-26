"""Run the real window branch for every installed slot and minute of first pitch."""
from collections import Counter
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from probe_support import Replay, clock_slots, load_line, no_network
from bts.cli import _check_pick_entered


class Run:
    outcome = None

    def selection(self, *_):
        pass

    def window(self, *_):
        pass

    def set_outcome(self, outcome):
        self.outcome = outcome

    def failed(self, _):
        self.outcome = 'read'


def offline_account():
    raise ValueError('recorded the attempt to read the account')


def run_grid(root):
    f5_line, base_line = load_line('f5'), load_line('base')
    assert '--cron' in f5_line and '--window-min' not in f5_line
    f5_slots, base_slots = list(clock_slots(f5_line)), set(clock_slots(base_line))
    origin = f5_slots[0]
    selection = SimpleNamespace(pick=SimpleNamespace(game_time=''), double_down=None)
    seen, added = Counter(), []
    one_day = [origin + timedelta(minutes=offset) for offset in range(1440)]
    with no_network() as stack:
        for target, substitute in {
            'bts.picks.load_pick_bytes': lambda *_: (selection, b'pick'),
            'bts.daily_decision.load_decision_bytes': lambda *_: ({'scoreable': True}, b'decision'),
            'bts.leaderboard.auth.load_session_cookies': offline_account,
            'click.echo': lambda *_args, **_kwargs: None,
        }.items():
            stack.enter_context(patch(target, substitute))

        def one_pitch(pitch, legs, width, moments):
            selection.pick.game_time = legs[0].isoformat()
            selection.double_down = SimpleNamespace(game_time=legs[1].isoformat()) if len(legs) == 2 else None
            opens, cutoff = pitch - timedelta(minutes=width), pitch - timedelta(minutes=5)
            historic = {slot for slot in base_slots if opens <= slot < cutoff}
            for now in moments:
                run = Run()
                _check_pick_entered(root / 'picks', 'fixture', 'captured.invalid', width, now, run, cron=True)
                expected = ('outside_window' if not opens <= now < cutoff else
                            'base_slot_available' if now.hour < 10 and historic else 'read')
                assert run.outcome == expected, (now, legs, width, run.outcome, expected)
                seen[expected] += 1
                if expected == 'read' and now.hour < 10 and len(legs) == 1 and width == 75:
                    added.append(pitch)

        for pitch in one_day:
            later = min(pitch + timedelta(minutes=125), one_day[-1] + timedelta(minutes=1))
            for legs in ((pitch,), (pitch, later), (later, pitch)):
                one_pitch(pitch, legs, 75, f5_slots)
        for width in (-1, 5, 6, 20, 40, 90, 1000):
            for pitch in (one_day[365], one_day[605], one_day[606], one_day[650], one_day[690], one_day[1439]):
                one_pitch(pitch, (pitch,), width, f5_slots)
        one_pitch(origin + timedelta(hours=10, minutes=5, seconds=1),
                  (origin + timedelta(hours=10, minutes=5, seconds=1),), 75, f5_slots)

    distinct = sorted(set(added))
    assert distinct == [dt for dt in one_day if one_day[6] <= dt <= one_day[605]]
    print('Real window outcomes across minute x installed slot, both double leg orders, and custom widths:', seen)
    print(f'Extra reads at default width only for earliest pitch {distinct[0]:%H:%M}–{distinct[-1]:%H:%M}')


def cli_boundaries(root):
    cases = [('10:05', '09:45', None, 'observed'), ('10:05', '10:00', None, 'outside_window'),
             ('10:05:01', '09:45', None, 'base_slot_available'), ('10:05:01', '10:00:02', None, 'outside_window'),
             ('10:06', '10:00', None, 'observed'), ('10:50', '09:35', None, 'base_slot_available'),
             ('10:50', '09:34:59', None, 'outside_window'), ('10:25', '09:45', 40, 'base_slot_available'),
             ('10:25', '09:45', 39, 'outside_window')]
    for i, (pitch, at, width, outcome) in enumerate(cases):
        world = Replay(root / f'cli-{i}')
        world.commit_at(pitch)
        got = world.invoke(at, window=width)
        assert got['receipt']['outcome'] == outcome, (pitch, at, width, got)
        assert bool(got['calls']) == (outcome == 'observed')
        print('CLI', pitch, at, width, outcome)


if __name__ == '__main__':
    with TemporaryDirectory() as scratch:
        root = Path(scratch)
        run_grid(root)
        cli_boundaries(root)
