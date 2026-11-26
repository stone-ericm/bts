"""Prepare independent source mutants and run the requested focused pytest jobs."""

from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
WRAPPER = Path('/Users/eric/bts-runs/yield_run.sh')
NEW = ['tests/test_wr_a_calendar.py', 'tests/test_wr_a_pick.py', 'tests/test_wr_a_schedule.py']
EDITED = ['tests/' + name for name in (
    'test_alert_status_cs6b2_carbon.py', 'test_saved_pick_paths_cs5a.py', 'test_saver_cs3.py',
    'test_saver_cs3_review.py', 'test_season_end_date.py', 'test_stale_delivery_cs4c.py',
    'test_stale_delivery_cs4c_carbon.py', 'test_stale_delivery_cs4c_review.py',
)]
CONTROLS = ['tests/test_lightgbm_optional.py', 'tests/test_web_saver.py']

SCHED_GUARD = '    if required:\n        refusal = saved_pick_refusal(daily, decision_state)\n'
SCHED_FRESH = '''    if required:
        fresh_choice = (
            selection is not None and selection.decision is not None
            and selection.pick_result is not None and selection.pick_result.daily is daily
        )
        refusal = (None if fresh_choice and {path} else saved_pick_refusal(daily, decision_state))
'''

MUTATIONS = {
    f'fresh-{mode}': ('scheduler.py', SCHED_GUARD,
                      SCHED_FRESH.format(path=f'label == "lineup" and mode == "{mode}"'), NEW[1])
    for mode in ('dm', 'public')
}
MUTATIONS.update({
    'fresh-fallback': ('scheduler.py', SCHED_GUARD,
                       SCHED_FRESH.format(path='label == "fallback"'), NEW[1]),
    'fresh-final-fallback': ('scheduler.py', SCHED_GUARD,
                             SCHED_FRESH.format(path='label == "final fallback"'), NEW[1]),
    'fresh-run': ('cli.py', '\n        refusal = saved_pick_refusal(daily, decision_state)\n',
                  '\n        refusal = saved_pick_refusal(daily, decision_state) if sel.decision is None else None\n', NEW[1]),
    'fresh-run-catchup': ('cli.py', 'refusal = saved_pick_refusal(result.daily, decision_state)',
                          'refusal = saved_pick_refusal(result.daily, decision_state) if sel.decision is None else None', NEW[1]),
    'fresh-orchestrate': ('orchestrator.py', '        if required:\n            refusal = saved_pick_refusal(daily, decision_state)',
                          '        if required and sel.decision is None:\n            refusal = saved_pick_refusal(daily, decision_state)', NEW[1]),
    'fresh-orchestrate-catchup': ('orchestrator.py', '            if required:\n                refusal = saved_pick_refusal(result.daily, decision_state)',
                                  '            if required and sel.decision is None:\n                refusal = saved_pick_refusal(result.daily, decision_state)', NEW[1]),
    'only-streak': ('picks.py', 'record["dated"] == live_dated',
                    'record["streak"] == decision_state.streak', NEW[1]),
    'refuse-auto-auto': ('picks.py', 'live_kind == "automatic" or (live_kind == "dated"',
                         '(live_kind == "dated"', NEW[1]),
    'old-2027': ('strategy.py', '2027: "2027-09-27"', '2027: "2027-09-26"', NEW[0]),
    'inclusive-2026': ('strategy.py', 'return max(0, (end - today).days)',
                       'return max(0, (end - today).days + int(today.year == 2026))', NEW[0]),
    'empty-nonlist-dates': ('scheduler.py', 'raise ValueError("MLB schedule dates must be a list")',
                            'return []', NEW[2]),
    'skip-invalid-game': ('scheduler.py', '''            if not isinstance(game, dict):
                raise ValueError(f"MLB schedule {location}.games[{game_index}] must be an object")
        games.extend(day_games)''', '''            if not isinstance(game, dict):
                continue
            games.append(game)''', NEW[2]),
})


def prepare():
    for name, (filename, old, new, _) in MUTATIONS.items():
        source = REVIEW / 'mutants' / name / 'src'
        shutil.copytree(ROOT / 'src', source, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        target = source / 'bts' / filename
        raw = target.read_text()
        assert raw.count(old) == 1, (name, raw.count(old))
        target.write_text(raw.replace(old, new))
        compile(target.read_text(), str(target), 'exec')


def pytest_job(name):
    import pytest
    from bts import scheduler

    et = ZoneInfo('America/New_York')
    source = ROOT / 'src'
    if name == 'base':
        source = REVIEW / 'base/src'
        files = NEW
    elif name == 'combined':
        files = [*NEW, *CONTROLS, *EDITED]
    elif name.startswith('wall-'):
        year = int(name[5:])
        assert datetime.now(et).year == year and datetime.fromtimestamp(time.time(), et).year == year
        files = NEW
    else:
        source = REVIEW / 'mutants' / name / 'src'
        files = [MUTATIONS[name][3]]
    assert Path(scheduler.__file__).resolve().is_relative_to(source), scheduler.__file__
    print('pytest source', scheduler.__file__, 'ET clock', datetime.now(et).isoformat(), flush=True)
    return pytest.main(['-q', '-p', 'no:cacheprovider', '--tb=short', *files])


def launch_wall_clock(name):
    year = int(name[5:])
    target = datetime(year, 3, 24, 18, 25, tzinfo=ZoneInfo('America/New_York'))
    env = dict(os.environ, DYLD_INSERT_LIBRARIES=str(REVIEW / 'wallclock.dylib'),
               WR_A_WALL_OFFSET_SECONDS=str(round(target.timestamp() - time.time())))
    os.execve(sys.executable, [sys.executable, __file__, 'pytest', name], env)


def wrapper_jobs(names):
    gate_end = datetime(2026, 11, 19, 1, 14, tzinfo=ZoneInfo('America/New_York'))
    assert datetime.now(gate_end.tzinfo) >= gate_end, 'another agent has the studio until 01:14 ET'
    for name in names:
        source = ROOT / 'src'
        if name == 'base':
            source = REVIEW / 'base/src'
        if name in MUTATIONS:
            source = REVIEW / 'mutants' / name / 'src'
        env = dict(os.environ, TZ='America/New_York', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=str(source), BTS_YIELD_SECONDS='5')
        subcommand = 'wall' if name.startswith('wall-') else 'pytest'
        proc = subprocess.run([str(WRAPPER), str(REVIEW / f'pytest-{name}.log'), sys.executable,
                               __file__, subcommand, name], cwd=ROOT, env=env, check=False)
        print(name, int(proc.returncode), flush=True)


if __name__ == '__main__':
    command, *options = sys.argv[1:]
    if command == 'prepare':
        prepare()
    elif command == 'pytest':
        raise SystemExit(pytest_job(options[0]))
    elif command == 'wall':
        launch_wall_clock(options[0])
    elif command == 'wrapper':
        wrapper_jobs(options or ['combined', 'base', 'wall-2027', 'wall-2028', *MUTATIONS])
    else:
        raise SystemExit(f'unknown command: {command}')
