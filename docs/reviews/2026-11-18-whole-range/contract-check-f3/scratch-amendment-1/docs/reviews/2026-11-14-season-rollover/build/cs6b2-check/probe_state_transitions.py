import hashlib
import json
import traceback
from unittest.mock import Mock

from probe_decisions import (
    BASE, DAY, FORMS, NOW, CliRunner, alert, attempt, case, cli, daily_pick, decisions,
    emit, entry_receipt, orchestrator, picks, scheduler, scheduler_state, seed, set_dated,
)
from probe_status import status_file


def entry_evidence():
    for name, raw in FORMS.items():
        with case() as (_, patch, (_, directory, _, _)):
            patch.setattr(picks, '_git_head_sha', lambda *args: 'offline-main')
            seed(directory, raw)
            result = CliRunner().invoke(cli.cli, ['check-pick-entered', '--picks-dir', str(directory),
                                                    '--now-et', NOW.replace(tzinfo=None).isoformat()])
            receipts = entry_receipt.discover(directory, DAY)
            emit('entry-binding', kind=name, exit=result.exit_code,
                 input_sha=hashlib.sha256(raw).hexdigest() if raw is not None else None,
                 receipts=[dict(outcome=r['outcome'], detail=r['detail'],
                                commit=(r.get('selection') or {}).get('commit')) for r in receipts])


def fallback_decisions():
    for first_frame in (True, False):
        with case() as (_, patch, (config, directory, _, _)):
            patch.setattr(picks, '_git_head_sha', lambda *args: 'offline-main')
            daily = daily_pick()
            picks.save_pick(daily, directory)
            set_dated(directory)
            path = status_file(directory)
            frame, tier = orchestrator.run_cascade.return_value
            def predict(*_):
                if orchestrator.run_cascade.call_count == 1:
                    return (frame, tier) if first_frame else (None, None)
                (directory / 'account_state' / 'contest_streak.manual.json').unlink(missing_ok=True)
                return frame, tier
            orchestrator.run_cascade.side_effect = predict
            try:
                scheduler.run_day(DAY, config)
            except Exception as error:
                route = [frame.lineno for frame in traceback.extract_tb(error.__traceback__)
                         if frame.name == 'run_day']
                emit('fallback-branch', in_loop=first_frame, error=type(error).__name__,
                     route=route, frames=orchestrator.run_cascade.call_count,
                     health_dms=alert.send_dm.call_count, status=path.read_text())
            else:
                emit('fallback-branch', in_loop=first_frame, outcome='returned')


def finalization_evidence():
    for name, raw in (('valid-commit', json.dumps(BASE).encode()),
                      ('invalid-JSON', FORMS['invalid-JSON']), ('absent', None),
                      ('bad-action', FORMS['action-list'])):
        with case() as (_, _, (_, directory, _, _)):
            _, path = seed(directory, raw)
            state = scheduler_state()
            state.final_skip_candidate = {'source': 'mdp'}
            outcome, _ = attempt(lambda: scheduler._write_endofday_skip(directory, DAY, state))
            emit('end-of-day-write', kind=name, outcome=outcome,
                 preserved=path.exists() and path.read_bytes() == raw,
                 new_skip=path.exists() and b'"action": "skip"' in path.read_bytes())


if __name__ == '__main__':
    entry_evidence()
    fallback_decisions()
    finalization_evidence()
