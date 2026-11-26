"""Run main delivery guards, persisted attempts, and cached-pick routes offline."""

from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from probe_main import run
from pytest import MonkeyPatch

from bts import contest_state, dm, orchestrator, picks, posting, scheduler
from bts.daily_decision import load_decision
from bts.health import alert
from bts.simulate import mdp, tail_policy
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, record, set_automatic
from tests.test_saved_pick_paths_cs5a import scheduler_state, world


def exercise(root, mode, scenario):
    with MonkeyPatch.context() as mp:
        config, directory, clock, _ = world.__wrapped__(root, mp)
        config['scheduler']['pick_delivery'] = mode
        set_automatic(directory)
        daily = daily_pick()
        daily.chosen_on = record('automatic')
        state = scheduler_state()
        transport = dm.send_dm if mode == 'dm' else posting.post_to_bluesky
        if scenario == 'cutoff':
            fmt = scheduler._format_pick_delivery_text

            def delayed(*args):
                clock[0] = picks.submission_cutoff_et(daily)
                return fmt(*args)

            mp.setattr(scheduler, '_format_pick_delivery_text', delayed)
            path = picks.save_pick(daily, directory)
            assert not scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, scenario)
            assert not path.exists() and not transport.called
            archive, = (directory / DAY).glob('refused_delivery_*.json')
            assert json.loads(archive.read_text())['delivery_attempted'] is False
            assert 'LATE DELIVERY REFUSED' in alert.send_dm.call_args.args[1]
        elif scenario == 'no_recipient':
            config['bluesky'].pop('dm_recipient')
            path = picks.save_pick(daily, directory)
            before = path.read_bytes()
            assert not scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, scenario)
            assert path.read_bytes() == before and not transport.called and not alert.send_dm.called
        elif scenario == 'fallback':
            clock[0] = NOW - timedelta(hours=3)
            set_automatic(directory, now=clock[0] - timedelta(hours=12))
            picks.save_pick(daily, directory)
            cached = picks.load_pick(DAY, directory)
            clock[0] = NOW
            orchestrator.run_cascade.return_value = (None, None)
            refresh = scheduler._refresh_pick_at_fallback_decision(config, DAY, cached, 0.03)
            assert refresh.daily is cached and refresh.selection is None
            assert scheduler._deliver_and_lock_pick(cached, config, directory, state, DAY, scenario,
                                                    selection=refresh.selection)
            assert transport.call_count == 1 and state.pick_locked
            assert load_decision(DAY, directory)['streak'] is None
        elif scenario == 'after_midnight':
            clock[0] = NOW.replace(hour=0, minute=2) + timedelta(days=1)
            daily.pick = replace(daily.pick, game_time=f'{clock[0].date()}T04:25:00Z')
            assert scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, scenario)
            assert picks.load_pick(DAY, directory).delivered_at.startswith(str(clock[0].date()))
            assert not (directory / f'{clock[0].date()}.json').exists()
            assert scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, scenario)
            assert transport.call_count == 1
        else:
            class SimulatedCrash(BaseException):
                pass

            def crash(*args):
                assert picks.load_pick(DAY, directory).delivery_attempted
                raise SimulatedCrash()

            transport.side_effect = crash
            try:
                scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, scenario)
            except SimulatedCrash:
                pass
            else:
                raise AssertionError('crash was not exercised')
            loaded = picks.load_pick(DAY, directory)
            assert loaded.delivery_attempted and not picks.pick_was_delivered(loaded)
            reboot = scheduler_state()
            assert not scheduler._deliver_and_lock_pick(loaded, config, directory, reboot, DAY, scenario)
            assert reboot.pick_locked and transport.call_count == 1
            assert load_decision(DAY, directory)['delivery_status'] == 'locked_unconfirmed'
            assert 'DELIVERY OUTCOME UNKNOWN' in alert.send_dm.call_args.args[1]
        print('verified', mode, scenario)


if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[2]
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp, MonkeyPatch.context() as mp:
        os.chdir(tmp)
        root = Path(tmp)
        for module, var in [(mdp, 'DEFAULT_POLICY_PATH'), (tail_policy, 'DEFAULT_TAIL_POLICY_PATH')]:
            mp.setattr(module, var, str(repo / getattr(module, var)))
        for mode in ['dm', 'public']:
            for scenario in ['cutoff', 'fallback', 'after_midnight', 'attempt_crash']:
                exercise(root / f'{mode}-{scenario}', mode, scenario)
        exercise(root / 'dm-no-recipient', 'dm', 'no_recipient')
        run(root / 'full', 'dm', ['hit', 'miss'], full_day=True)
