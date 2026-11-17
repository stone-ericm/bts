"""Offline, read-only runtime inventory of existing live-decision fixtures."""
import sys
sys.dont_write_bytecode = True

def deny_network(event, args):
    if event in {'socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyaddr',
                 'socket.connect', 'socket.connect_ex', 'socket.sendto'}:
        raise RuntimeError('audit: network forbidden: ' + event)
sys.addaudithook(deny_network)

import datetime as dt
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas  # initialize before datetime substitution

REAL_DATETIME = dt.datetime
NOW = REAL_DATETIME(2027, 7, 15, 12, tzinfo=ZoneInfo('America/New_York'))
class Clock(REAL_DATETIME):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)
dt.datetime = Clock

import bts.contest_state as cs
import bts.season_contract as sc
import bts.dm as dm
dm.send_dm = lambda *a, **kw: 'offline-captured'

ROWS = []
CURRENT = ''
ACTIVE = []
old_loader = cs.load_contest_streak_state
old_decision = cs.load_decision_streak_state

def observed_loader(*args, **kwargs):
    selected = old_loader(*args, **kwargs)
    if ACTIVE:
        ACTIVE[-1]['selected'] = selected
    return selected
cs.load_contest_streak_state = observed_loader

def decision(picks, *args, **kwargs):
    cell = {'nodeid': CURRENT, 'required': kwargs.get('require_contest_state', False)}
    ACTIVE.append(cell)
    try:
        value = old_decision(picks, *args, **kwargs)
        cell['outcome'] = 'accepted'
        cell['streak'] = value.streak
        return value
    except Exception as exc:
        cell['outcome'] = type(exc).__name__
        cell['message'] = str(exc)
        raise
    finally:
        ACTIVE.pop()
        selected = cell.pop('selected', None)
        now = kwargs.get('now') or NOW
        if selected is not None:
            cell.update(path=selected.path.name, schema=selected.schema_version,
                        dated=cs.dated_observation_in_effect(selected, now))
        if cell['required'] and cell['outcome'] == 'accepted' and not cell.get('dated'):
            # No extra application-state read: old-schema/manual fixtures can
            # NEVER be covered. Exact v2 can be audited separately if encountered.
            cell['newly_refused'] = (selected.path.name == 'contest_streak.manual.json'
                                     or selected.schema_version != sc.AUTO_SCHEMA)
            cell['v2_needs_coverage_audit'] = not cell['newly_refused']
        ROWS.append(cell)
cs.load_decision_streak_state = decision

def pytest_runtest_setup(item):
    global CURRENT
    CURRENT = item.nodeid
    if item.name == 'test_heartbeat_path_refreshes_during_sleep':
        # This existing test explicitly measures real elapsed wall time.
        import bts.heartbeat as heartbeat
        heartbeat.datetime = REAL_DATETIME

def pytest_runtest_teardown(item):
    if item.name == 'test_heartbeat_path_refreshes_during_sleep':
        import bts.heartbeat as heartbeat
        heartbeat.datetime = Clock

def pytest_sessionfinish(session, exitstatus):
    root = Path(__file__).resolve().parent
    (root / 'fixture-inventory.json').write_text(json.dumps(ROWS, indent=2) + '\n')
    affected = {row['nodeid'] for row in ROWS if row.get('newly_refused')}
    summary = {'real_decisions': len(ROWS), 'required_calls': sum(r['required'] for r in ROWS),
               'newly_refused_calls': sum(bool(r.get('newly_refused')) for r in ROWS),
               'affected_test_cases': len(affected), 'cases': sorted(affected)}
    (root / 'fixture-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
