from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from bts import daily_decision, dm, picks, scheduler, strategy
from bts.simulate import mdp
from scripts.audit.season_walk import walk

from records_and_interrupts import DAY, NOW, fresh_pick, state


def loaded_narrow_table(root):
    artifact_path = root / 'narrow.npz'
    np.savez(artifact_path, policy_table=np.zeros((58, 181, 2, 2), dtype=np.int8),
             boundaries=[0.8, 0.8], season_length=180)
    with patch.object(mdp, 'DEFAULT_POLICY_PATH', artifact_path), patch.object(strategy, '_mdp_cache', None):
        artifact = strategy._load_mdp()
        assert artifact['base_error'] is None and artifact['tail'] is None
        lower = strategy.DecisionContext(0.79, 0.7, True, DAY, True, artifact)
        assert strategy.resolve_policy_decision(lower, 10, False).action == 'skip'
        higher = strategy.DecisionContext(0.8, 0.7, True, DAY, True, artifact)
        try:
            strategy.resolve_policy_decision(higher, 10, False)
        except IndexError:
            print('narrow NPZ loads, base reach-57 lookup raises at exact tied edge')
        else:
            raise AssertionError('the existing lookup must raise at column 2')


def clamp_donor():
    artifact = strategy._load_mdp()
    for probability, allow, different, want_policy, want_action in (
        (0.79, True, True, 'double', 'double'),
        (0.79, False, True, 'double', 'single'),
        (0.79, True, False, 'double', 'single'),
        (0.80, True, True, 'skip', 'skip'),
    ):
        context = strategy.DecisionContext(probability, 0.78 if different else None, different,
                                           '2027-08-29', allow, artifact)
        decision = strategy.resolve_policy_decision(context, 1, False)
        assert decision.days_effective == 29
        assert (decision.policy_action, decision.action) == (want_policy, want_action)
    print('both existing clamps work on the real donor cell (1, 29, 0, 0)')


def empty_ack(root):
    config = {'orchestrator': {'picks_dir': str(root)}, 'bluesky': {'dm_recipient': 'probe'},
              'scheduler': {'pick_delivery': 'dm'}}
    daily, selection = fresh_pick(root)
    with patch.object(scheduler, '_now_et', return_value=NOW), \
            patch.object(scheduler, '_void_committed_games', return_value={}), \
            patch.object(scheduler, '_trigger_live_forward_capture_on_lock'), \
            patch.object(dm, 'send_dm', return_value=''):
        assert scheduler._deliver_and_lock_pick(daily, config, root, state(), DAY, 'empty ack',
                                                selection=selection)
    record = daily_decision.load_decision(DAY, root)
    loaded = picks.load_pick(DAY, root)
    assert record['scoreable'] and record['delivery_status'] == 'delivered'
    assert loaded.delivery_attempted and not picks.pick_was_delivered(loaded)
    print('empty DM id follows the current delivered-record branch despite the false delivered-pick predicate')


if __name__ == '__main__':
    with TemporaryDirectory(dir='.codex-review/wr-check-g') as name:
        folder = Path(name)
        with walk.offline(NOW):
            loaded_narrow_table(folder)
            clamp_donor()
            empty_ack(folder / 'picks')
