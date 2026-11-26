import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import daily_decision, picks, scheduler, strategy
from scripts.audit.boundary_shadow_census import CensusHalt, run_census
from scripts.audit.season_walk import walk

from records_and_interrupts import DAY, GUARD, NOW, Interrupt, fresh_pick, state


def private_restart(root):
    daily, _ = fresh_pick(root)
    config = {'orchestrator': {'picks_dir': str(root)}, 'scheduler': {'pick_delivery': 'private'}}
    with patch.object(scheduler, '_now_et', return_value=NOW), \
            patch.object(scheduler, '_void_committed_games', return_value={}), \
            patch.object(scheduler, '_trigger_live_forward_capture_on_lock', side_effect=Interrupt):
        try:
            scheduler._deliver_and_lock_pick(daily, config, root, state(), DAY, 'private crash')
        except Interrupt:
            pass
        else:
            raise AssertionError('expected a private crash after persisting the lock')
    loaded = picks.load_pick(DAY, root)
    restored = scheduler.load_state(DAY, root)
    assert restored.pick_locked and not restored.committed_pick_written
    assert not picks.pick_was_delivered(loaded) and not loaded.delivery_attempted
    assert daily_decision.load_decision(DAY, root) is None
    rebuilt = scheduler.carry_forward_skip_state(state(), restored)
    assert not rebuilt.pick_locked
    scheduler._write_classification_decision(root, DAY, action='single', delivered=False,
                                             primary=scheduler._row_from_daily(loaded.pick),
                                             double_down=None, daily=loaded, state=rebuilt)
    assert daily_decision.load_decision(DAY, root) is None
    print('private interruption leaves a saved pick with no proof for classification recovery')


def census(root):
    # Use a 2026 date with effective day 180 so the census's historical calendar is correct.
    day = '2026-03-25'
    picks_dir = root / 'picks'
    record = daily_decision.write_decision(
        day, picks_dir, action='single', source='mdp', objective='reach57',
        primary={'batter_id': 100, 'p_game_hit': 0.96}, streak=50, saver_available=False,
        delivery_status='delivered', scoreable=True, now=NOW)
    record['policy_guard'] = GUARD
    daily_decision.decision_path(day, picks_dir).write_text(json.dumps(record))
    ledger = root / 'contest_ledger.jsonl'
    ledger.write_text(json.dumps({'recorded_at': f'{day}T00:00:00Z', 'active_streak': 50}))
    try:
        run_census(picks_dir=picks_dir, ledger_path=ledger, policy_path='data/models/mdp_policy.npz',
                   saver_active=False, start=day, end=day, as_of=day, output=root / 'census.json')
    except CensusHalt as exc:
        assert str(exc) == 'unattributable action mismatch: raw=skip recorded=single'
        print('real census halted on a guarded production record')
    else:
        raise AssertionError('census accepted the guarded pick')


def calendar_controls():
    artifact = strategy._load_mdp()
    for day, streak in (('2027-09-17', 55), ('2027-09-21', 56)):
        for probability, expected in ((0.96, 'skip'), (0.83, 'single')):
            ctx = strategy.DecisionContext(probability, probability, True, day, True, artifact,
                                           streak, 'trusted')
            decision = strategy.resolve_policy_decision(ctx, streak, False)
            assert (decision.action, decision.objective) == (expected, 'reach57')
    print('both affected calendar replay assertions and their unchanged donor controls verified')


if __name__ == '__main__':
    with TemporaryDirectory(dir='.codex-review/wr-check-g') as name:
        folder = Path(name)
        with walk.offline(NOW):
            calendar_controls()
            private_restart(folder / 'private')
            census(folder / 'census')
