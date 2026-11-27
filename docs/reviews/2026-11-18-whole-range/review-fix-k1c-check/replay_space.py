"""Replay the B8a directory repair with the installed CLI and scheduler."""

import datetime as dt
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
VENV_BIN = ROOT / '.venv/bin'
DAY = '2027-07-15'
SHELLS = ('/bin/sh', '/bin/zsh')
MANUAL = Path('account_state/contest_streak.manual.json')
REQUESTS = Path('account_state/contest_manual_requests.jsonl')
B8A = next(line for line in (ROOT / 'docs/ops/2027-season-start.md').read_text().splitlines()
           if line.startswith('| B8a |'))
FILING = re.search(r'Run `(bts set-contest-streak [^`]+)` with N, B and S', B8A)[1]
FILING = FILING.replace('<today>', DAY).replace(' N ', ' 57 ').replace(' B ', ' 60 ')
FILING = FILING.replace(' S ', ' used ')
STATUS = re.search(r'Check the same directory with `(bts season-contract status [^`]+)`', B8A)[1]


def pin_cli_clock():
    # Python loads this module as sitecustomize for the real bts executable.
    from bts import cli

    real = dt.datetime
    frozen = real.fromisoformat(os.environ['WR_K1C_CLOCK'])

    class Clock(real):
        @classmethod
        def now(cls, tz=None):
            return frozen.astimezone(tz) if tz else frozen.replace(tzinfo=None)

    def offline(*args, **kwargs):
        raise OSError('WR K1c probe is offline')

    dt.datetime = cli.datetime = Clock
    socket.getaddrinfo = offline
    for method in ('connect', 'connect_ex', 'sendto'):
        setattr(socket.socket, method, offline)


def run_shell(shell, command, operator, hook, clock, stdin=''):
    env = dict(os.environ, PATH=f'{VENV_BIN}:{os.defpath}',
               PYTHONPATH=str(hook), WR_K1C_CLOCK=clock.isoformat())
    env.pop('DIR', None)
    invocation = [shell, '-f', '-c'] if shell.endswith('/zsh') else [shell, '-c']
    result = subprocess.run([*invocation, command], cwd=operator, env=env, input=stdin,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=5)
    assert 'Error in sitecustomize' not in result.stdout, result.stdout
    return result.returncode, result.stdout


def run_pair(run, setup=''):
    return run(f'{setup}\n{FILING}\nprintf "filing exit=%s\\n" "$?"\n{STATUS}')


def literal_assignment(path):
    return f'DIR={shlex.quote(str(path))}'


def replay(shell, label, relative, operator_action, *, expected_delivery=True,
           service_suffix=Path('service')):
    # Keep fixture imports out of the CLI startup hook.
    import json
    from pytest import MonkeyPatch
    from bts import dm, orchestrator, picks, scheduler, strategy
    from bts.daily_decision import load_decision
    from bts.health import alert
    from scripts.audit.season_walk.walk import _set_clock, offline
    from tests.test_pick_chosen_on_cs5a import NOW
    from tests.test_saved_pick_paths_cs5a import prepare_run_day, world

    with TemporaryDirectory(prefix='wrk1c-', dir='/private/tmp') as tmp, offline(NOW), MonkeyPatch.context() as mp:
        root = Path(tmp)
        service = root / service_suffix
        operator = root / 'operator'
        hook = root / 'hook'
        service.mkdir(parents=True)
        operator.mkdir()
        hook.mkdir()
        (hook / 'sitecustomize.py').symlink_to(Path(__file__).resolve())
        cfg, _, now, _ = world.__wrapped__(service, mp)
        prepare_run_day(mp)
        cfg['scheduler']['entry_intent'] = 'enter'
        cfg['orchestrator']['picks_dir'] = str(relative)
        live = service / relative
        mp.chdir(service)
        mp.setattr(scheduler, 'notify_ready', lambda: None)
        mp.setattr(scheduler, 'notify_watchdog', lambda: None)
        mp.setattr(strategy, '_mdp_cache', None)
        frame, tier = orchestrator.run_cascade.return_value
        orchestrator.run_cascade.return_value = (frame.assign(p_game_hit=[0.99, 0.97]), tier)
        refusals = []
        filings = []

        def capture(_, body):
            if f'No pick was sent for {DAY}' in body:
                refusals.append(now[0].strftime('%H:%M'))
            return 'refusal-dm'

        def advance(seconds):
            if refusals and not filings:
                assert not dm.send_dm.called and not (live / MANUAL).exists()
                assert now[0] == NOW and (now[0] + dt.timedelta(seconds=seconds)).strftime('%H:%M') == '18:55'

                def run(command, stdin=''):
                    return run_shell(shell, command, operator, hook, now[0], stdin)

                filings.append(operator_action(run, operator, live))
                if expected_delivery:
                    row = json.loads((live / MANUAL).read_text())
                    assert [row[key] for key in ('effective_date', 'active_streak', 'best_streak', 'saver_state')] == [
                        DAY, 57, 60, 'used',
                    ]
                    assert len((live / REQUESTS).read_text().splitlines()) == 1
                    assert sorted(path.name for path in (live / 'account_state').iterdir()) == sorted(
                        [MANUAL.name, REQUESTS.name]
                    )
                else:
                    assert not (live / MANUAL).exists() and not (live / REQUESTS).exists()
                assert not (live / 'streak.json').exists()
            now[0] += dt.timedelta(seconds=seconds)
            _set_clock(now[0])

        mp.setattr(alert, 'send_dm', capture)
        mp.setattr(scheduler, '_watchdog_ping_sleep', advance)
        journal = StringIO()
        with redirect_stderr(journal), redirect_stdout(StringIO()):
            scheduler.run_day(DAY, cfg)
        assert len(filings) == 1 and refusals[0] == '18:35', journal.getvalue()
        assert '[18:55 ET] Running lineup check' in journal.getvalue(), journal.getvalue()
        assert dm.send_dm.call_count == int(expected_delivery), journal.getvalue()
        decision = load_decision(DAY, live)
        if expected_delivery:
            pick = picks.load_pick(DAY, live)
            assert decision['delivery_status'] == 'delivered' and pick.pick.batter_id == 680757
            assert 'Streak: 57' in dm.send_dm.call_args.args[1]
        else:
            assert decision is None or decision['delivery_status'] != 'delivered', decision
        print(f'{shell} {label}: first refusal 18:35; {filings[0]}; 18:55 pick DMs={dm.send_dm.call_count}; '
              f"decision={decision['delivery_status'] if decision else 'absent'}", flush=True)


def correct_filing(run, operator, live):
    code, output = run_pair(run, literal_assignment(live))
    assert code == 0 and 'filing exit=0' in output and 'live decision now: accepted;' in output, output
    assert not (operator / MANUAL).exists() and not (operator / 'data/picks' / MANUAL).exists()
    if live.name != 'picks':
        assert not (live.parent / 'picks' / MANUAL).exists()
    return 'CLI exits 0/0, status accepted, stored 57/60/used in service picks directory'


if __name__ == 'sitecustomize':
    pin_cli_clock()
elif __name__ == '__main__':
    for real_shell in SHELLS:
        replay(real_shell, 'data/live picks', Path('data/live picks'), correct_filing)
