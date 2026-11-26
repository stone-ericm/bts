"""Rehearse timing, repeat filing, and path handling with the real CLI and scheduler."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import timedelta
from io import StringIO
from pathlib import Path
import shlex
from tempfile import TemporaryDirectory

from pytest import MonkeyPatch

from replay_round_two import COMMAND, file_observation, review
from bts import dm, orchestrator, picks, scheduler, strategy
from bts.daily_decision import load_decision
from bts.health import alert
from scripts.audit.season_walk.walk import _invoke, _set_clock, offline
from tests.test_pick_chosen_on_cs5a import DAY, NOW
from tests.test_saved_pick_paths_cs5a import prepare_run_day, world


def directory_case(case):
    with TemporaryDirectory(prefix='wrk1b-dir-') as tmp, offline(NOW), MonkeyPatch.context() as mp:
        root = Path(tmp)
        service_dir = root / 'service'
        operator_dir = root / 'operator'
        service_dir.mkdir()
        operator_dir.mkdir()
        cfg, _, now, _ = world.__wrapped__(service_dir, mp)
        prepare_run_day(mp)
        cfg['scheduler']['entry_intent'] = 'enter'
        relative = {'default': 'data/picks', 'custom-relative': 'local/live-picks',
                    'space-unquoted': 'data/live picks', 'space-quoted': 'data/live picks'}[case]
        cfg['orchestrator']['picks_dir'] = relative
        absolute = service_dir / relative
        mp.chdir(service_dir)
        mp.setattr(scheduler, 'notify_ready', lambda: None)
        mp.setattr(scheduler, 'notify_watchdog', lambda: None)
        mp.setattr(strategy, '_mdp_cache', None)
        frame, tier = orchestrator.run_cascade.return_value
        orchestrator.run_cascade.return_value = (frame.assign(p_game_hit=[0.99, 0.97]), tier)
        failures = []
        filings = []

        def capture(_, body):
            if 'No pick was sent for' in body:
                failures.append(body)
                assert 'live decision now: refused;' in review.status(absolute)
            return 'refusal-dm'

        def advance(seconds):
            if failures and not filings:
                with MonkeyPatch.context() as operator:
                    operator.chdir(operator_dir)
                    if case == 'custom-relative':
                        # Deliberate control: using the raw config value from another cwd.
                        code, output = _invoke('season-contract', 'status', '--picks-dir', relative)
                        assert code == 0 and 'live decision now: refused;' in output
                        code, output = _invoke('set-contest-streak', '--date', DAY, '--streak', 57,
                                               '--best-streak', 60, '--saver', 'used', '--picks-dir', relative)
                        assert code == 0, output
                        assert review.snapshot(operator_dir / relative)
                        assert 'live decision now: refused;' in review.status(absolute)
                    values = {'<today>': DAY, 'N': '57', 'B': '60', 'S': 'used', 'DIR': str(absolute)}
                    if case == 'space-quoted':
                        values['DIR'] = shlex.quote(values['DIR'])
                    command_line = ' '.join(values.get(word, word) for word in COMMAND)
                    code, output = _invoke(*shlex.split(command_line))
                    filings.append((code, output.strip().splitlines()[-1]))
                    if case == 'space-unquoted':
                        assert code == 2 and 'unexpected extra argument (picks)' in output, output
                        assert 'live decision now: refused;' in review.status(absolute)
                        assert not (absolute / 'account_state/contest_streak.manual.json').exists()
                    else:
                        assert code == 0 and 'live decision now: accepted' in review.status(absolute), output
            now[0] += timedelta(seconds=seconds)
            _set_clock(now[0])

        mp.setattr(alert, 'send_dm', capture)
        mp.setattr(scheduler, '_watchdog_ping_sleep', advance)
        journal = StringIO()
        with redirect_stderr(journal), redirect_stdout(StringIO()):
            scheduler.run_day(DAY, cfg)
        assert failures and len(filings) == 1, journal.getvalue()
        assert dm.send_dm.call_count == (0 if case == 'space-unquoted' else 1), journal.getvalue()
        decision = load_decision(DAY, absolute)
        if case == 'space-unquoted':
            assert decision is None
        else:
            assert decision['delivery_status'] == 'delivered'
        print(case, 'CLI', filings[0], 'official sends', dm.send_dm.call_count, flush=True)


def send_in_progress():
    with TemporaryDirectory(prefix='wrk1b-send-') as tmp, offline(NOW), MonkeyPatch.context() as mp:
        root = Path(tmp)
        mp.chdir(root)
        cfg, directory, now, _ = world.__wrapped__(root, mp)
        prepare_run_day(mp)
        cfg['scheduler']['entry_intent'] = 'enter'
        mp.setattr(scheduler, 'notify_ready', lambda: None)
        mp.setattr(scheduler, 'notify_watchdog', lambda: None)
        mp.setattr(strategy, '_mdp_cache', None)
        frame, tier = orchestrator.run_cascade.return_value
        orchestrator.run_cascade.return_value = (frame.assign(p_game_hit=[0.99, 0.97]), tier)
        refused = []
        observations = []

        def capture(_, body):
            if 'No pick was sent for' in body:
                refused.append(body)
            return 'refusal-dm'

        def advance(seconds):
            if refused and not observations:
                file_observation(DAY, directory)
                observations.append(review.snapshot(directory / 'account_state'))
            now[0] += timedelta(seconds=seconds)
            _set_clock(now[0])

        def transport(_, body):
            # An operator opens the earlier refusal while the recovered pick is in flight.
            saved = picks.load_pick(DAY, directory)
            assert refused and saved.delivery_attempted and not picks.pick_was_delivered(saved)
            assert load_decision(DAY, directory) is None
            assert 'Streak: 57' in body
            assert 'live decision now: accepted' in review.status(directory)
            # The scheduler is sending within its check, so B8a's between-runs condition fails.
            assert review.snapshot(directory / 'account_state') == observations[0]
            return 'pick-dm'

        mp.setattr(alert, 'send_dm', capture)
        mp.setattr(scheduler, '_watchdog_ping_sleep', advance)
        dm.send_dm.side_effect = transport
        journal = StringIO()
        with redirect_stderr(journal), redirect_stdout(StringIO()):
            scheduler.run_day(DAY, cfg)
        assert len(refused) == 1 and dm.send_dm.call_count == 1, journal.getvalue()
        assert load_decision(DAY, directory)['delivery_status'] == 'delivered', journal.getvalue()
        assert review.snapshot(directory / 'account_state') == observations[0]
        print('earlier refusal opened mid-send: no write, one official send and delivered decision', flush=True)


if __name__ == '__main__':
    import sys

    group = sys.argv[1]
    if group == 'timing':
        review.run_day('no-fallback', repair=False)
        review.overnight_pending()
        review.file_observation = file_observation
        review.run_day('two-operators', repair=True)
        import probe_operator
        probe_operator.guarded_delivery()
        send_in_progress()
    elif group == 'directories':
        for case in ('default', 'custom-relative', 'space-unquoted', 'space-quoted'):
            directory_case(case)
    else:
        raise ValueError(group)
