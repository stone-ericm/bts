import json
import logging
from datetime import timedelta
from unittest.mock import Mock

from probe_decisions import (
    DAY, NOW, alert, attempt, case, daily_pick, dm, emit, orchestrator,
    picks, posting, scheduler, scheduler_state, set_dated,
)
from bts.contest_state import ContestStateError
from bts.health import runner


SHARED = 'health_dm_delivery_status.json'
MORNING = 'contest_morning_check_dm_delivery_status.json'
STALE = 'contest_stale_reading_dm_delivery_status.json'
CRITICAL = alert.Alert('CRITICAL', 'x', 'fixture alert')


def status_file(directory, raw=b'[]', name=SHARED):
    path = directory.parent / 'health_state' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        path.write_bytes(raw)
    return path


def summarize(path):
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_bytes())
    except (UnicodeError, ValueError, RecursionError) as error:
        return type(error).__name__
    if not isinstance(body, dict):
        return type(body).__name__
    return {k: body.get(k) for k in ('status', 'critical_count', 'sent_sources', 'body_first_line')}


def matrix():
    same_day = dict(status='sent', sent_day=DAY)
    forms = {
        'missing': None, 'array': b'[]', 'null': b'null', 'string': b'"s"', 'integer': b'1',
        'true': b'true', 'nested': b'[' * 20000 + b']' * 20000, 'bad-json': b'{not json',
        'encoding': b'\xff', 'legacy-sent': b'{"status":"sent","critical_count":1}',
        'legacy-failed': b'{"status":"failed","critical_count":1}',
        'wrong-status': b'{"status":[]}',
    }
    forms.update({f'failed-{n!r}': json.dumps(dict(status='failed', critical_count=n)).encode()
                  for n in ('x', None, True, 0, 2, 1.5)})
    forms.update({f'sent-{n!r}': json.dumps(dict(same_day, sent_sources=n)).encode()
                  for n in ('abc', 7, [1, {}], [1], ['CRITICAL:x'], ['CRITICAL:x', {}])})
    for name, raw in forms.items():
        for mode in ('critical', 'empty', 'no-recipient'):
            with case() as (_, _, (_, directory, _, _)):
                target = status_file(directory, raw)
                def dispatch():
                    return alert.dispatch_dm_for_health_alerts(
                        [] if mode == 'empty' else [CRITICAL],
                        None if mode == 'no-recipient' else 'offline.test',
                        status_path=target, now_et_date=NOW.date())
                result = []
                outcome, _ = attempt(lambda: result.append(dispatch()))
                emit('dispatcher', form=name, mode=mode, outcome=outcome, result=result,
                     dms=alert.send_dm.call_count, status=summarize(target))


def run_and_capture(name, config, directory, path, job):
    outcome, stderr = attempt(job)
    state = scheduler.load_state(DAY, directory)
    emit('caller', name=name, outcome=outcome, health_dms=alert.send_dm.call_count,
         posts=posting.post_to_bluesky.call_count, pick_dms=dm.send_dm.call_count,
         completed=len(state.runs_completed) if state else None,
         locked=state.pick_locked if state else None, status=summarize(path),
         diagnostics=[line.strip() for line in stderr.splitlines()
                      if any(s in line for s in ('BLOCKED', 'blocked', 'MISSED', 'REFUSED',
                                                 'UNKNOWN', 'suppressed', 'failed'))])


def callers():
    for name in ('lineup contest refusal', 'in-loop fallback contest refusal',
                 'post-loop fallback contest refusal', 'delivery contest read refusal',
                 'delivery saved-ticket refusal', 'missed-pick', 'late', 'void',
                 'unconfirmed', 'end-of-day', 'morning', 'stale'):
        with case() as (_, patch, (config, directory, now, statuses)):
            filename = MORNING if name == 'morning' else STALE if name == 'stale' else SHARED
            path = status_file(directory, name=filename)
            daily = daily_pick()
            state = scheduler_state()
            job = lambda: scheduler.run_day(DAY, config)
            if name == 'lineup contest refusal':
                pass  # Real prediction transport returns a frame. Required state is absent.
            elif name in ('in-loop fallback contest refusal', 'post-loop fallback contest refusal'):
                picks.save_pick(daily, directory)
                error = ContestStateError('probe fallback state was withdrawn')
                if name == 'in-loop fallback contest refusal':
                    set_dated(directory)
                    patch.setattr(scheduler, '_refresh_pick_at_fallback_decision', Mock(side_effect=error))
                else:
                    # No predictions means the real final fallback does the required read.
                    orchestrator.run_cascade.return_value = (None, None)
            elif name == 'missed-pick':
                picks.save_pick(daily, directory)
                job = lambda: scheduler._maybe_alert_missed_pick(config, DAY, directory, 10, None, state)
            elif name in ('delivery contest read refusal', 'delivery saved-ticket refusal',
                          'late', 'void', 'unconfirmed'):
                if name == 'delivery saved-ticket refusal':
                    set_dated(directory)
                if name == 'late':
                    now[0] += timedelta(hours=1)
                if name == 'void':
                    statuses[101] = {'abstract': 'P', 'detailed': 'Cancelled', 'code': 'C'}
                if name == 'unconfirmed':
                    daily.delivery_attempted = True
                picks.save_pick(daily, directory)
                job = lambda: scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, 'probe')
            elif name == 'end-of-day':
                config['health_checks']['enabled'] = True
                orchestrator.run_cascade.return_value = (None, None)
                patch.setattr(runner, 'read_systemd_nrestarts', lambda: None)
                patch.setattr(runner, 'get_self_pid', lambda: None)
                patch.setattr(runner.realized_calibration, '_current_deploy_iso', lambda _: None)
            elif name == 'morning':
                orchestrator.run_cascade.return_value = (None, None)
                now[0] -= timedelta(hours=1)
                patch.setattr(scheduler, '_watchdog_ping_sleep',
                              lambda seconds: now.__setitem__(0, now[0] + timedelta(seconds=seconds)))
            else:
                job = lambda: scheduler._warn_stale_contest_reading(
                    {'stale': True, 'reason': 'unreadable'}, config, directory)
            run_and_capture(name, config, directory, path, job)


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    callers()
    matrix()
