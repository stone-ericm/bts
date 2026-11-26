"""Exercise old and new readers and the operational readers on actual pick files."""
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import json
import os
import subprocess
import sys

from click.testing import CliRunner
import bts
from bts import daily_decision, picks, scheduler, skip_policy_shadow, strategy
from bts.cli import cli
from scripts.audit.boundary_shadow_census import run_census
from scripts.audit.season_walk import walk
from commit_paths import GUARD, state

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURES = [('2026-03-25', 50, .96, 'single'), ('2026-03-26', 50, .83, 'single'),
            ('2026-03-27', 56, .99, 'skip')]


def fixture(root, day, streak, p):
    frame = walk._predictions(date.fromisoformat(day), p)
    now = walk.REAL_DATETIME.fromisoformat(day + 'T12:00:00').replace(tzinfo=walk.ET)
    statuses = {pk: {'abstract': 'P', 'detailed': 'Scheduled'} for pk in frame.game_pk}
    with walk.offline(now):
        sel = strategy.select_pick(frame, day, root, streak=streak, saver_available=False,
                                   game_statuses_detailed=statuses, require_detailed_statuses=True)
        current = state()
        current.date = day
        if sel.pick_result is not None:
            daily = sel.pick_result.daily
            picks.save_pick(daily, root)
            scheduler._commit_decision_for_pick(root, day, daily, selection=sel, state=current,
                                                delivery_status='private_locked')
        else:
            preview = frame.iloc[0]
            picks.save_pick(picks.DailyPick(date=day, run_time=now.isoformat(),
                                          pick=picks.pick_from_row(preview)), root)
            current.final_skip_candidate = dict(primary=sel.primary_candidate, source='mdp', streak=streak,
                                                saver_available=False, objective='reach57')
            scheduler._write_endofday_skip(root, day, current)
    return now


def archived(side, root):
    print(side, bts.__file__)
    assert Path(bts.__file__).resolve().is_relative_to(HERE / f'{side}/src')
    for day, _, _, action in FIXTURES:
        decision = daily_decision.load_decision(day, root)
        daily = picks.load_pick(day, root)
        assert decision['action'] == action and decision['scoreable'] == (action != 'skip')
        assert daily.pick.batter_id == 100 and daily.double_down is None
        expected_guard = GUARD if day == FIXTURES[0][0] else None
        assert decision['policy_guard'] == expected_guard
        assert (daily.policy_decision or {}).get('guard') == expected_guard
    if side == 'base':
        home = root / 'written-by-base'
        home.mkdir()
        fixture(home, *FIXTURES[1][:3])


def head(root):
    day_times = {day: fixture(root, day, streak, p) for day, streak, p, _ in FIXTURES}
    files = [daily_decision.decision_path(day, root) for day, *_ in FIXTURES]
    old_bytes = [path.read_bytes() for path in files]
    for side in ('base', 'rollback'):
        subprocess.run([sys.executable, __file__, side, str(root)], check=True, cwd=ROOT, timeout=12,
                       env=os.environ | {'PYTHONPATH': f'{HERE / side / "src"}:{ROOT}'})
    assert [path.read_bytes() for path in files] == old_bytes
    legacy = root / 'written-by-base'
    day = FIXTURES[1][0]
    assert picks.load_pick(day, legacy).policy_decision.get('guard') is None
    assert daily_decision.load_decision(day, legacy).get('policy_guard') is None

    first, last = FIXTURES[0][0], FIXTURES[-1][0]
    ledger = root / 'ledger.jsonl'
    ledger.write_text('\n'.join(json.dumps({'recorded_at': f'{day}T00:00:00Z', 'active_streak': s})
                                for day, s, *_ in FIXTURES))
    census = run_census(picks_dir=root, ledger_path=ledger, policy_path=Path('data/models/mdp_policy.npz'),
                        saver_active=False, start=first, end=last, as_of=last, output=root / 'census.json')
    assert [(row['raw_deployed'], row['parity']) for row in census['rows']] == [
        ('skip', 'guarded_play_excluded'), ('single', 'parity'), ('skip', 'parity')]
    config = root / 'config.toml'
    config.write_text('[scheduler]\npick_delivery="private"\nentry_intent="research"\n')
    for day, _, _, action in FIXTURES:
        now = day_times[day]
        with walk.offline(now), ExitStack() as stack:
            stack.enter_context(patch.object(scheduler, 'fetch_schedule', lambda *_: [{'gamePk': 70000}]))
            slots = stack.enter_context(patch.object(picks, 'resolve_daily_slot_results', Mock(return_value={'pick': 'hit'})))
            result = CliRunner().invoke(cli, ['day-outcome', '--date', day, '--config', str(config), '--picks-dir', str(root)])
            assert result.exit_code == 0, (result.output, result.exception)
            expected = 'deliberate_skip' if action == 'skip' else 'research_commit'
            assert json.loads(result.output)['state'] == expected
            result = CliRunner().invoke(cli, ['check-results', '--date', day, '--picks-dir', str(root), '--allow-stale-scoring'])
            assert result.exit_code == 0, (result.output, result.exception)
            assert slots.call_count == int(action != 'skip')
            if action == 'skip':
                assert 'not scoring' in result.output
            else:
                assert picks.load_pick(day, root).result == 'hit'
        dec = daily_decision.load_decision(day, root)
        assert daily_decision.is_reach57_mdp_skip(dec) == (action == 'skip')
        shadow = skip_policy_shadow.record_skip_from_decision(day, root, now=now)
        assert (shadow is not None) == (action == 'skip')
        print(day, 'day-outcome', expected, 'check-results grading:', slots.call_count, 'shadow member:', bool(shadow))
    assert [path.read_bytes() for path in files] == old_bytes
    print('archived readers, legacy writer, current CLI readers and census verified')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        archived(sys.argv[1], Path(sys.argv[2]))
    else:
        with TemporaryDirectory() as tmp:
            head(Path(tmp))
