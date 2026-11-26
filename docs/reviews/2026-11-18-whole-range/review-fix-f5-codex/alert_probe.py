"""Run the closing replay and challenge alert preservation with successive pick snapshots."""
from datetime import date, datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bts
from probe_support import ROOT, Replay, clock_slots, load_line, no_network
from bts.health import pick_entry
from tests.test_wr_f1_wrong_game import DAY, RIGHT, WRONG, row


def outcome(run):
    return run['receipt']['outcome']


def installed_between(version, first, last):
    return [slot.strftime('%H:%M') for slot in clock_slots(load_line(version))
            if first <= slot.strftime('%H:%M') <= last]


def closing_fixture(root, version):
    fixture = ROOT / ('docs/reviews/2026-11-18-whole-range/closing-test/codex-blind/'
                      'evidence/cron-early-entry.json.gz')
    with gzip.open(fixture, 'rt') as stream:
        snapshot = next(event[-1] for event in json.load(stream) if event[1] == 'snapshot')
    world = Replay(root / 'closing', version)
    pick = snapshot[f'data/picks/{DAY}.json']
    decision = snapshot[f'data/picks/{DAY}/decision.json']
    trace = []
    for slot in (x.strftime('%H:%M') for x in clock_slots(load_line(version))):
        if slot > '05:10' and not (world.picks / f'{DAY}.json').is_file():
            (world.picks / f'{DAY}.json').write_text(json.dumps(pick))
            target = world.picks / DAY / 'decision.json'
            target.parent.mkdir()
            target.write_text(json.dumps(decision))
            world.capture([{**RIGHT, 'feedId': pick['pick']['game_pk']}], f'{DAY}T091000Z')
        trace.append((slot, world.invoke(slot, pending=[row()], crosswalk={1: pick['pick']['batter_id']})))
    reads = [slot for slot, run in trace if 'pending' in run['calls']]
    morning = world.morning()
    assert reads == (['05:15'] if version == 'f5' else []), reads
    assert morning[1]['state'] == ('entered' if version == 'f5' else 'entry_unverifiable'), morning
    assert len(morning[2]) == (version == 'base')
    print('closing fixture:', reads, morning)


def shared_window(root, version, changes):
    world = Replay(root / changes, version)
    if changes == 'late-commit':
        world.commit_at('10:50', at='09:30', delivery=None)
    else:
        world.commit_at('10:50', at='09:00')
    runs = {}
    for slot in installed_between(version, '09:45', '10:45'):
        if slot == '10:00':
            if changes == 'late-commit':
                world.commit_at('10:50', at='09:52')
            if changes == 'bind-wrong':
                world.capture([WRONG], f'{DAY}T140000Z')
            else:
                world.capture([RIGHT], f'{DAY}T140000Z')
        runs[slot] = world.invoke(slot, pending=[row()] if slot == '09:45' or changes == 'bind-wrong' else [])
    if version == 'f5':
        early = runs['09:45']
        assert outcome(early) == ('not_committed' if changes == 'late-commit' else 'base_slot_available')
        assert not (early['calls'] or early['attempts'] or early['marker_changed'])
    assert 'pending' in runs['10:00']['calls'] and runs['10:00']['sent']
    assert not runs['10:45']['calls']
    morning = world.morning()
    assert morning[1]['state'] == ('entry_unverifiable' if changes == 'bind-wrong' else 'not_entered')
    print(changes, [(slot, outcome(run), len(run['sent'])) for slot, run in runs.items()], morning[1]['state'])


def revised_window(root, version, previous_alert=False):
    world = Replay(root / f'revised-{previous_alert}', version)
    world.commit_at('10:05', at='09:40')
    trace = []
    for slot in installed_between(version, '09:45', '10:45'):
        if slot == '10:00':
            # Restore/correct the committed selection between launches. No manual watchdog runs.
            world.commit_at('10:50' if not previous_alert else '10:06', at='09:52', game=102)
            world.capture([RIGHT], f'{DAY}T135200Z')
        run = world.invoke(slot, pending=[row()] if not previous_alert else [])
        trace.append((slot, run))
    ten = dict(trace)['10:00']
    morning = world.morning()
    if version == 'base':
        assert ten['calls'] and ten['sent'], ten
        if not previous_alert:
            assert 'entry is recorded in game 101' in ten['sent'][0]
    elif previous_alert:
        assert ten['calls'] and not ten['sent'] and ten['marker']['status'] == 'alerted', ten
    else:
        early = dict(trace)['09:45']['receipt']
        assert early['qualification']['slots'][0]['state'] == 'unit_unverified', early
        assert not ten['calls'] and not ten['sent'] and outcome(ten) == 'already_confirmed', ten
    print('revised window, early alert:', previous_alert,
          [(slot, outcome(run), run['calls'], run['sent']) for slot, run in trace], morning[1]['state'])


def corrected_pitch(root, version):
    from bts.picks import save_pick
    world = Replay(root / 'corrected-pitch', version)
    pick = world.commit_at('10:05', at='09:40')
    trace = {}
    for slot in installed_between(version, '09:45', '10:45'):
        if slot == '10:00':
            # Only the saved first pitch changes, as when restoring corrected game metadata.
            pick.pick.game_time = f'{DAY}T10:50:00-04:00'
            save_pick(pick, world.picks)
            world.capture([WRONG], f'{DAY}T140000Z')
        trace[slot] = world.invoke(slot, pending=[row()])
    morning = world.morning()
    if version == 'base':
        assert 'entry is recorded in game 102' in trace['10:00']['sent'][0]
        assert morning[1]['state'] == 'entry_unverifiable' and morning[2]
    else:
        assert trace['09:45']['receipt']['qualification']['slots'][0]['state'] == 'unit_unverified'
        assert trace['10:00']['receipt']['outcome'] == 'already_confirmed' and not trace['10:00']['calls']
        assert morning[1]['state'] == 'entered_game_unverified' and not morning[2]
    print('same batter and game, corrected saved pitch:',
          [(slot, outcome(run), run['sent']) for slot, run in trace.items()], morning[1]['state'], len(morning[2]))


def previous_date_health(root, version):
    from bts.health import runner as health_runner

    world = Replay(root / 'prior-health', version)
    yesterday = date.fromisoformat(DAY) - timedelta(days=1)
    Replay(world.root, version, yesterday.isoformat()).commit_at('19:10', at='18:05')
    world.set_marker({'date': yesterday.isoformat(), 'status': 'dm_failed', 'reason': 'no_pick'})
    world.commit_at('06:10', at='04:55')
    world.capture([RIGHT], f'{DAY}T085500Z')
    result = None
    for slot in installed_between(version, '05:00', '05:00'):
        result = world.invoke(slot, pending=[row()])
    observed = datetime.fromisoformat(f'{DAY}T09:01:00+00:00')
    sent = []
    source_check = pick_entry.check
    safe_run = health_runner._safe_run
    with no_network() as stack:
        stack.enter_context(patch.object(health_runner, '_safe_run',
                                        lambda key, run: safe_run(key, run) if key == 'pick_entry' else []))
        stack.enter_context(patch.object(pick_entry, 'check', lambda picks, today: source_check(picks, today, observed)))
        stack.enter_context(patch.object(health_runner.realized_calibration, '_current_deploy_iso', lambda _: None))
        stack.enter_context(patch('bts.health.alert.send_dm', lambda recipient, message: sent.append(message)))
        alerts = health_runner.run_all_checks(world.picks, world.root / 'models', today=yesterday,
                                              dm_recipient=world.recipient)
    assert [alert.level for alert in alerts] == (['CRITICAL'] if version == 'base' else [])
    assert len(sent) == (1 if version == 'base' else 0), sent
    if sent:
        assert 'the operator was never reached' in sent[0]
    print('prior-date health after 05:00:', outcome(result) if result else 'no launch', [a.level for a in alerts], sent)


if __name__ == '__main__':
    version = sys.argv[1]
    print('bts.__file__:', bts.__file__)
    assert ('/base/src/' in str(bts.__file__)) == (version == 'base')
    with TemporaryDirectory() as scratch:
        root = Path(scratch)
        closing_fixture(root, version)
        for change in ('bind-wrong', 'remove-entry', 'late-commit'):
            shared_window(root, version, change)
        for previous_alert in (False, True):
            revised_window(root, version, previous_alert)
        corrected_pitch(root, version)
        previous_date_health(root, version)
