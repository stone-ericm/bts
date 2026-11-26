"""Run the real selection, commit writer and journal on both double clamps."""
from contextlib import redirect_stderr
from dataclasses import asdict
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json

from bts import daily_decision, picks, scheduler, strategy
from bts.simulate import mdp
from commit_paths import state
from scripts.audit.season_walk import walk

END = date(2027, 9, 27)


def select(root, s, d, saver, p, allow=True, different=True):
    day = END - timedelta(days=d)
    frame = walk._predictions(day, p)
    if not different:
        frame['game_pk'] = frame['game_pk'].iloc[0]
    statuses = {pk: {'abstract': 'P', 'detailed': 'Scheduled'} for pk in frame.game_pk}
    with walk.offline(walk.REAL_DATETIME.combine(day, walk.REAL_DATETIME.min.time(), walk.ET)):
        return strategy.select_pick(frame, day.isoformat(), root, streak=s, saver_available=saver,
                                    allow_double=allow, game_statuses_detailed=statuses,
                                    require_detailed_statuses=True)


def line_for(selection):
    capture = StringIO()
    with redirect_stderr(capture):
        scheduler._log_policy_decision(selection)
    return capture.getvalue().strip()


def validate_line(selection, p, s, d, expected_saver):
    decision = selection.decision
    artifact = strategy._load_mdp()
    bounds, table = artifact['boundaries'], artifact['policy_table']
    q = sum(p >= b for b in bounds)
    donor = max(j for j in range(q) if table[s, min(180, d), int(selection.saver_available), j])
    assert mdp.lookup_action(table, bounds, s, d, selection.saver_available, p) == 'skip'
    assert decision.policy_action == mdp.ACTIONS[table[s, min(180, d), int(selection.saver_available), donor]]
    assert decision.guard['bin'] == q and decision.guard['from_bin'] == donor
    assert decision.streak == s and decision.days_effective == min(180, d)
    expected = (f'POLICY GUARD — At streak {s} with {min(180, d)} policy days left and saver input {expected_saver}, '
                f'the table skips bin {q}. Bin {donor} supplies {decision.policy_action}. Selected action is {decision.action}.')
    return line_for(selection), expected


with TemporaryDirectory() as tmp:
    root = Path(tmp)
    bounds = strategy._load_mdp()['boundaries']
    p = (bounds[0] + bounds[1]) / 2
    for allow, different, actual in [(True, True, 'double'), (False, True, 'single'), (True, False, 'single')]:
        selection = select(root, 1, 29, False, p, allow, different)
        decision = selection.decision
        daily = selection.pick_result.daily
        assert decision.policy_action == 'double' and decision.action == selection.action == actual
        assert (daily.double_down is not None) == (actual == 'double')
        assert daily.pick.batter_id == 100
        if actual == 'double':
            # Give the same probability a raw double cell, then run normal selection on the same slate.
            original = strategy._load_mdp()
            table = original['policy_table'].copy()
            table[1, 29, 0, 1] = 2
            with patch.object(strategy, '_load_mdp', return_value=original | {'policy_table': table}):
                donor_selection = select(root, 1, 29, False, p)
            ordinary = donor_selection.pick_result.daily
            assert donor_selection.decision.guard is None
            assert daily.pick == ordinary.pick and daily.double_down == ordinary.double_down
            assert daily.double_down.batter_id == 101 and daily.pick.game_pk != daily.double_down.game_pk
        got, expected = validate_line(selection, p, 1, 29, 'unavailable')
        assert got == expected
        namespace = root / str((allow, different))
        path = picks.save_pick(daily, namespace)
        current = state()
        current.date = daily.date
        scheduler._commit_decision_for_pick(namespace, daily.date, daily, selection=selection,
                                            delivery_status='private_locked', state=current)
        rec = daily_decision.load_decision(daily.date, namespace)
        assert rec['action'] == actual and rec['policy_guard'] == decision.guard
        assert json.loads(json.dumps(asdict(daily))) == json.loads(path.read_text())
        assert json.loads(json.dumps(rec))['policy_guard'] == decision.guard
        print(got, 'legs', [daily.pick.batter_id, daily.double_down.batter_id if daily.double_down else None])
    assert line_for(select(root, 50, 187, False, .83)) == ''
    for streak, days, saver, label in [(50, 187, False, 'unavailable'), (42, 38, True, 'available')]:
        selection = select(root, streak, days, saver, .96)
        got, expected = validate_line(selection, .96, streak, days, label)
        assert got == expected and selection.action == 'single'
        print(got)
    selection = select(root, 50, 187, '0', .96)
    got, expected = validate_line(selection, .96, 50, 187, 'unavailable')
    assert got != expected and 'saver input available' in got
    print('non-bool saver mismatch:', got, 'Expected:', expected)
