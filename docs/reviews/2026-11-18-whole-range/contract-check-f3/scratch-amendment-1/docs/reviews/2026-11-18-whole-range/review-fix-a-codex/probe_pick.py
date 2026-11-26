"""Drive the live delivery entry points with real reads and synthetic ET observations."""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
DAY = '2027-03-24'


@contextmanager
def offline(day=DAY, hour=18, minute=25):
    spec = spec_from_file_location('review_offline', REVIEW / 'slices/slice-3/offline.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    with TemporaryDirectory(prefix='pick-', dir=REVIEW) as folder:
        module.ROOT, module.HERE = ROOT, Path(folder)
        with module.offline(module.instant(day, hour, minute)) as (world, patch):
            yield world, patch, module.instant


class CapturedSchedulerEdge(BaseException):
    pass


def snapshot(world, state=None):
    from bts import picks, scheduler
    from bts.daily_decision import load_decision

    daily = picks.load_pick(world.day, world.picks)
    decision = load_decision(world.day, world.picks)
    state = state or scheduler.load_state(world.day, world.picks)
    return {
        'out': len(world.sends('dm') + world.sends('post')),
        'attempt': daily.delivery_attempted if daily else None,
        'confirmed': picks.pick_was_delivered(daily) if daily else None,
        'locked': state.pick_locked if state else None,
        'decision': decision['delivery_status'] if decision else None,
        'checks': len(state.runs_completed) if state else None,
        'alerts': sum('contest_state' in body for body in world.sends('health-dm')),
        'saved_stamp': daily.chosen_on if daily else None,
    }


def invoke(world, command, *extra):
    from bts import cli
    from click.testing import CliRunner

    options = (['--picks-dir', str(world.picks), '--models-dir', str(world.models)]
               if command in {'run', 'preview'} else ['--config', str(world.config_path)])
    return CliRunner().invoke(cli.cli, [command, '--date', world.day, *options, *extra])


def matrix_case(path, edge):
    from bts import contest_state, orchestrator, picks, scheduler, posting

    catchup = path in {'run-catchup', 'orchestrate-catchup'}
    with offline() as (world, patch, instant):
        world.automatic(0)
        if edge not in {'auto-dated', 'auto-auto'}:
            world.dated(0)
        paths = world.account / 'contest_streak.manual.json', world.account / 'contest_streak.json'
        manual, automatic = paths
        if catchup:
            _, sel, _ = orchestrator.run_and_pick(world.config, world.day)
            picks.save_pick(sel.pick_result.daily, world.picks)
            for game in world.schedule[world.day]:
                game['status'].update(detailedState='In Progress', abstractGameCode='L', statusCode='I')
            patch.setattr(posting, 'should_post_now', lambda *args: (_ for _ in ()).throw(
                AssertionError('catch-up did not use its locked-pick branch')))

        if path in {'dm', 'public', 'fallback'}:
            world.config['scheduler']['pick_delivery'] = 'dm' if path == 'fallback' else path
            if path == 'fallback':
                def fallback_predictions():
                    return world.frame((.99, .98), (world.now.minute == 25, False))

                world.predictions = fallback_predictions

        if path not in {'fallback'}:
            world.predictions = lambda: world.frame((.99, .98))

        def change_observation():
            if edge == 'streak':
                world.dated(30)
            elif edge == 'best':
                world.dated(0, best=1)
            elif edge == 'saver':
                world.dated(0, saver='not_earned')
            elif edge == 'auto-dated':
                world.dated(0)
            elif edge == 'dated-auto':
                manual.unlink()
            elif edge == 'auto-auto':
                world.automatic(7)
            elif edge in {'midnight', 'midnight-dated'}:
                world.now = instant('2027-03-25', 0, 0)
                if edge == 'midnight-dated':
                    world.dated(0)
            elif edge == 'corrupt':
                manual.write_text('{')
            elif edge == 'missing':
                for file in paths:
                    file.unlink()
            elif edge == 'directory':
                manual.unlink()
                manual.mkdir()

        save_pick = picks.save_pick
        phase = 0
        load_state = contest_state.load_decision_streak_state

        def observe(*args, **kwargs):
            nonlocal phase
            if catchup or path in {'run', 'orchestrate'}:
                phase += 1
                if phase == 2:
                    change_observation()
            return load_state(*args, **kwargs)

        def save_and_arm(daily, directory):
            if (path != 'fallback' or world.now.minute != 25) and edge in {
                'unstamped', 'invalid', 'nonrequired',
            }:
                if edge == 'unstamped':
                    daily.chosen_on = None
                elif edge == 'invalid':
                    daily.chosen_on['schema'] = 'unknown'
                else:
                    daily.chosen_on['required'] = False
            return save_pick(daily, directory)

        if edge in {'unstamped', 'invalid', 'nonrequired'} and catchup:
            save_and_arm(sel.pick_result.daily, world.picks)
        patch.setattr(picks, 'save_pick', save_and_arm)
        patch.setattr(contest_state, 'load_decision_streak_state', observe)

        if edge in {'keyboard', 'exit'}:
            def interrupt(*args):
                if edge == 'keyboard':
                    raise KeyboardInterrupt('review during comparison')
                raise SystemExit('review during comparison')

            patch.setattr(picks, 'saved_pick_refusal', interrupt)

        if path in {'dm', 'public', 'fallback'}:
            deliver = scheduler._deliver_and_lock_pick
            label = None
            result = None

            def capture(daily, config, directory, state, day, at, *, selection=None):
                nonlocal label, result
                label = at
                # Mutate after the earlier cutoff and while the final read is entering.
                if edge.startswith('midnight'):
                    def roll(*args, **kwargs):
                        change_observation()
                        return load_state(*args, **kwargs)

                    patch.setattr(contest_state, 'load_decision_streak_state', roll)
                else:
                    change_observation()
                try:
                    result = deliver(daily, config, directory, state, day, at, selection=selection)
                except (KeyboardInterrupt, SystemExit) as error:
                    result = type(error).__name__
                raise CapturedSchedulerEdge

            patch.setattr(scheduler, '_deliver_and_lock_pick', capture)
            try:
                scheduler.run_day(world.day, world.config)
            except CapturedSchedulerEdge:
                pass
            assert label == ('fallback' if path == 'fallback' else 'lineup'), (path, label)
            return {'return': result, 'branch': label, **snapshot(world)}

        if not catchup and edge.startswith('midnight'):
            patch.setattr(posting, 'should_post_now', lambda time, already: not already)
        result = invoke(world, path.split('-')[0])
        msg = [line for line in result.stderr.splitlines()
               if 'no pick:' in line or 'review during' in line or line == 'Aborted!']
        return {'exit': result.exit_code, 'error': type(result.exception).__name__,
                'stderr': msg, **snapshot(world)}


def cost_case(case):
    from bts import orchestrator, picks, scheduler

    with offline(hour=10 if case == 'cold-scheduler' else 18) as (world, patch, _):
        if case in {'private', 'research', 'optional-orchestrate', 'optional-direct-orchestrate',
                    'model-only-run', 'model-only-direct'}:
            world.config['scheduler'].update(contest_state_required=False, pick_delivery='private',
                                             entry_intent='research')
            world.config['health_checks']['contest_state_expected'] = False
            world.write_config()
            (world.picks / 'streak.json').write_text('{"streak": 0}')
        if case == 'research':
            world.config['scheduler']['contest_state_required'] = True
            world.dated(0)
            world.write_config()
        if case in {'cold-run', 'model-only-run'}:
            result = invoke(world, 'run')
        elif case in {'cold-orchestrate', 'optional-orchestrate'}:
            result = invoke(world, 'orchestrate')
        elif case == 'optional-direct-orchestrate':
            assert orchestrator.orchestrate(world.config_path, world.day, config=world.config)
            result = None
        elif case == 'model-only-direct':
            world.config['scheduler'].update(pick_delivery='public')
            scheduler.run_day(world.day, world.config)
            result = None
        elif case == 'preview-catchup':
            world.dated(0)
            preview = invoke(world, 'preview')
            assert preview.exit_code == 0
            assert picks.load_pick(world.day, world.picks).chosen_on['required'] is False
            for game in world.schedule[world.day]:
                game['status'].update(detailedState='In Progress', abstractGameCode='L', statusCode='I')
            result = invoke(world, 'orchestrate')
        else:
            scheduler.run_day(world.day, world.config)
            result = None
        return {'exit': result.exit_code if result else None, 'text': result.stderr if result else '',
                'other_dm_bodies': world.sends('dm'), 'health_bodies': world.sends('health-dm'),
                **snapshot(world)}


def repeated_refusal():
    from bts import picks, scheduler
    from click.testing import CliRunner
    from bts.cli import cli

    with offline() as (world, patch, instant):
        world.dated(0)
        world.config['scheduler']['cluster_min'] = 0
        world.schedule[world.day][1]['gameDate'] = instant(world.day, 19, 15).isoformat()
        save_pick = picks.save_pick
        refused = []

        def correct_each_fresh(daily, directory):
            saved = save_pick(daily, directory)
            stamp = daily.chosen_on
            if stamp and not daily.delivery_attempted and not picks.pick_was_delivered(daily):
                best = stamp['dated']['best_streak']
                world.dated(0, best=best + 1)
            return saved

        deliver = scheduler._deliver_and_lock_pick

        def record(daily, config, directory, state, day, label, *, selection=None):
            result = deliver(daily, config, directory, state, day, label, selection=selection)
            refused.append((label, result, snapshot(world, state)))
            return result

        patch.setattr(picks, 'save_pick', correct_each_fresh)
        patch.setattr(scheduler, '_deliver_and_lock_pick', record)
        scheduler.run_day(world.day, world.config)
        final = snapshot(world)
        world.now += timedelta(days=1)
        result = CliRunner().invoke(cli, [
            'day-outcome', '--date', world.day, '--config', str(world.config_path),
            '--picks-dir', str(world.picks), '--dm-recipient', 'offline.invalid',
        ])
        return {'attempts': refused, 'final': final, 'watchdog_exit': result.exit_code,
                'watchdog': json.loads(result.stdout.splitlines()[0])['state'],
                'health_bodies': world.sends('health-dm')}


def main():
    from bts import scheduler

    print('Scheduler source:', scheduler.__file__)
    paths = ('dm', 'public', 'fallback', 'run', 'orchestrate', 'run-catchup', 'orchestrate-catchup')
    edges = ('equal', 'streak', 'best', 'saver', 'dated-auto', 'auto-dated', 'auto-auto',
             'midnight', 'midnight-dated', 'corrupt', 'missing', 'directory', 'keyboard', 'exit',
             'unstamped', 'invalid', 'nonrequired')
    cases = ('cold-scheduler', 'cold-run', 'cold-orchestrate', 'model-only-run',
             'model-only-direct', 'private', 'research', 'optional-orchestrate',
             'optional-direct-orchestrate', 'preview-catchup')
    measurements = [(f'{path} {edge}', matrix_case, (path, edge)) for path in paths for edge in edges]
    measurements.extend((f'cost {case}', cost_case, (case,)) for case in cases)
    measurements.append(('repeated', repeated_refusal, ()))
    for label, measure, args in measurements:
        output, errors = StringIO(), StringIO()
        try:
            with redirect_stdout(output), redirect_stderr(errors):
                result = measure(*args)
        except Exception as error:
            print(label, 'ERROR', repr(error), output.getvalue()[-400:], errors.getvalue()[-1200:])
        else:
            print(label, json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
