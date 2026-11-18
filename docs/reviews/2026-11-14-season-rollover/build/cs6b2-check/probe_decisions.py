import contextlib
import hashlib
import io
import json
import os
from collections import Counter
from datetime import timedelta, timezone
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock


REPO = Path.cwd()
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def block_network(event, args):
    if event.startswith('socket.') or event == 'subprocess.Popen':
        raise AssertionError(f'blocked transport: {event}')


sys.addaudithook(block_network)

from click.testing import CliRunner
from pytest import MonkeyPatch
from bts import cli, daily_decision as decisions, dm, orchestrator, picks, posting, scheduler
from bts import entry_receipt, shadow_eval, skip_policy_shadow as shadow
from bts.health import alert
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, record, set_dated
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world


def emit(event, **data):
    print(event, json.dumps(data, default=str, sort_keys=True))


@contextlib.contextmanager
def case():
    with TemporaryDirectory(dir=SCRATCH) as name, MonkeyPatch.context() as patch:
        root = Path(name)
        fixture = world.__wrapped__(root, patch)
        prepare_run_day(patch)
        patch.setattr(scheduler, 'notify_ready', Mock())
        patch.setattr(scheduler, 'notify_watchdog', Mock())
        patch.setattr(shadow_eval, '_current_git_commit', lambda: 'offline-main')
        patch.chdir(root)
        yield root, patch, fixture


def attempt(fn):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            fn()
        except Exception as error:
            return type(error).__name__, stderr.getvalue()
    return 'returned', stderr.getvalue()


def seed(directory, raw, *, pick=None):
    if pick is None:
        pick = daily_pick()
    pick_path = picks.save_pick(pick, directory)
    decision_path = decisions.decision_path(DAY, directory)
    decision_path.parent.mkdir(exist_ok=True)
    if raw is not None:
        decision_path.write_bytes(raw)
    return pick_path, decision_path


def commands(root, directory):
    cfg = root / 'entry.toml'
    cfg.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
    return {
        'check-results': ['--date', DAY],
        'check-pick-entered': ['--now-et', NOW.replace(tzinfo=None).isoformat(),
                               '--dm-recipient', 'offline.test'],
        'day-outcome': ['--date', DAY, '--config', str(cfg), '--dm-recipient', 'offline.test'],
        'skip-policy-shadow-update': ['--date', DAY, '--no-reconcile',
                                      '--status-output', str(root / 'shadow.json')],
    }


BASE = dict(schema_version=decisions.DECISION_SCHEMA, date=DAY, scoreable=True,
            action='single', source='mdp', delivery_status='delivered', objective='reach57')
FORMS = {
    'action-list': json.dumps(dict(BASE, action=[])).encode(),
    'nested': b'[' * 20000 + b']' * 20000,
    'non-UTF-8': b'\xff',
    'invalid-JSON': b'{not json',
    'absent': None,
}


def reproduce():
    for name, raw in FORMS.items():
        with case() as (root, patch, (config, directory, now, _)):
            daily = daily_pick()
            _, path = seed(directory, raw, pick=daily)
            patch.setattr('bts.leaderboard.auth.load_session_cookies', Mock(
                side_effect=AssertionError('unexpected contest verification')))
            for command, args in commands(root, directory).items():
                before_dm = dm.send_dm.call_count
                result = CliRunner().invoke(cli.cli, [command, '--picks-dir', str(directory), *args])
                emit(name, command=command, exit=result.exit_code,
                     exception=type(result.exception).__name__, output=result.output.strip(),
                     dm=dm.send_dm.call_count - before_dm,
                     receipt=[(r.get('outcome'), r.get('commit'))
                              for r in entry_receipt.discover(directory, DAY)])
            if name in ('invalid-JSON', 'absent'):
                # Keep the control entirely on the preview path, without a cached fallback send.
                orchestrator.run_cascade.return_value = (None, None)
            for start in range(1, 4):
                outcome, stderr = attempt(lambda: scheduler.run_day(DAY, config))
                state = scheduler.load_state(DAY, directory)
                emit(name, start=start, outcome=outcome, checks_started=stderr.count('Running lineup check'),
                     checks_completed=len(state.runs_completed), pick_locked=state.pick_locked,
                     predictions=orchestrator.run_cascade.call_count,
                     schedule_reads=scheduler.fetch_schedule.call_count,
                     pick_dms=dm.send_dm.call_count, health_dms=alert.send_dm.call_count,
                     public_posts=posting.post_to_bluesky.call_count,
                     decision_equal=(path.read_bytes() if path.exists() else None) == raw)
                now[0] += timedelta(seconds=30)


def consequences():
    for label, fields, marker in (
        ('private', dict(delivery_status='private_locked'), 'none'),
        ('unconfirmed', dict(delivery_status='locked_unconfirmed'), 'attempt'),
        ('delivered', {}, 'delivered'),
        ('skip', dict(action='skip', scoreable=False, delivery_status='not_applicable'), 'none'),
        ('skip-with-stale-delivered-marker', dict(action='skip', scoreable=False), 'delivered'),
    ):
        for validity in ('valid', 'invalid-JSON'):
            with case() as (root, patch, (config, directory, now, _)):
                set_dated(directory)
                daily = daily_pick()
                daily.chosen_on = record()
                if marker == 'attempt':
                    daily.delivery_attempted = True
                if marker == 'delivered':
                    daily.notification_sent, daily.notification_id = True, 'prior-send'
                initial = json.dumps(dict(BASE, **fields)).encode() if validity == 'valid' else FORMS[validity]
                seed(directory, initial, pick=daily)
                patch.setattr(picks, 'resolve_daily_slot_results', Mock(return_value={'pick': 'hit'}))
                patch.setattr(picks, 'update_streak', Mock(return_value=picks.load_streak(directory, season=2027)))
                results = CliRunner().invoke(cli.cli, ['check-results', '--date', DAY, '--picks-dir', str(directory)])
                # Run scheduling against the original pick even if the check-results probe graded it.
                picks.save_pick(daily, directory)
                orchestrator.run_cascade.return_value = (None, None)
                outcome, stderr = attempt(lambda: scheduler.run_day(DAY, config))
                decision = decisions.load_decision(DAY, directory)
                emit('consequence', label=label, validity=validity, score_calls=picks.update_streak.call_count,
                     grade_exit=results.exit_code, grade_output=results.output,
                     run=outcome, predictions=orchestrator.run_cascade.call_count,
                     pick_dms=dm.send_dm.call_count, result_polls=scheduler.run_result_polling.call_count,
                     decision=decision and (decision['action'], decision['delivery_status']),
                     fallback='FINAL FALLBACK' in stderr or 'delivering on projected data' in stderr)

    for validity in ('valid', 'invalid-JSON', 'action-list', 'absent'):
        with case() as (_, _, (_, directory, _, _)):
            raw = json.dumps(dict(BASE, action='skip', scoreable=False)).encode() if validity == 'valid' else FORMS[validity]
            seed(directory, raw)
            target = shadow.decision_path(DAY, directory)
            target.write_text('{}')
            date = NOW.astimezone(timezone.utc)
            aged, _ = attempt(lambda: emit('shadow-aged', validity=validity,
                                            contradicted=shadow.find_aged_contradictions(directory, now=date + timedelta(days=30))))
            recent, _ = attempt(lambda: emit('shadow-recent', validity=validity,
                                              pruned=shadow.prune_superseded(directory, now=date)))
            emit('shadow', validity=validity, aged=aged, recent=recent, exists=target.exists())

    with case() as (_, _, (config, directory, now, _)):
        path, _ = seed(directory, None)
        path.write_text('{not json')
        for start in range(1, 4):
            outcome, stderr = attempt(lambda: scheduler.run_day(DAY, config))
            emit('malformed-pick', start=start, outcome=outcome,
                 checks_started=stderr.count('Running lineup check'),
                 runs_completed=len(scheduler.load_state(DAY, directory).runs_completed),
                 pick_dms=dm.send_dm.call_count, health_dms=alert.send_dm.call_count)
            now[0] += timedelta(seconds=30)


def parser_and_history():
    fixtures = {'top-NaN': 'NaN', 'optional-NaN': json.dumps(dict(BASE, optional=float('nan'))),
                'lone-surrogate': json.dumps(dict(BASE, optional='\ud800'), ensure_ascii=False),
                'oversize-int': '1' * 5000, 'wrong-date': json.dumps(dict(BASE, date='2027-07-14')),
                'null-date': json.dumps(dict(BASE, date=None)), 'long-string': json.dumps('x' * 5_000_000)}
    for name, value in fixtures.items():
        outcome, _ = attempt(lambda: emit('parser-result', name=name, accepted=decisions.parse_decision(value) is not None))
        emit('parser', name=name, outcome=outcome)
    frozen = REPO / 'docs/audit/2026-09-28-p05-tail-policy-audit.json'
    grouped = Counter()

    def collect(node):
        if isinstance(node, list):
            for item in node:
                collect(item)
        elif isinstance(node, dict):
            schema = node.get('schema_version')
            if schema in decisions.ACCEPTED_SCHEMAS:
                grouped[(schema, type(node.get('action')).__name__, node.get('action'), node.get('delivery_status'))] += 1
            for value in node.values():
                collect(value)

    collect(json.loads(frozen.read_bytes()))
    emit('frozen', path=frozen.relative_to(REPO), shapes=[(*k, v) for k, v in sorted(grouped.items())])
    emit('invalid-control-sha', sha=hashlib.sha256(FORMS['invalid-JSON']).hexdigest())


if __name__ == '__main__':
    reproduce()
    consequences()
    parser_and_history()
