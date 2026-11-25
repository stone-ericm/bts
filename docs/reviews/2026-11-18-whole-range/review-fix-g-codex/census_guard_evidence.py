"""Compare Gate B's response to an ordinary mismatch and an impossible guard."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from bts.daily_decision import write_decision
from scripts.audit.boundary_shadow_census import CensusHalt, run_census
from scripts.audit.season_walk import walk

DAY = '2026-03-25'
with TemporaryDirectory() as tmp:
    root = Path(tmp)
    ledger = root / 'ledger.jsonl'
    ledger.write_text(json.dumps({'recorded_at': DAY + 'T00:00:00Z', 'active_streak': 50}))
    clock = walk.REAL_DATETIME.fromisoformat(DAY + 'T12:00:00-04:00')
    with walk.offline(clock):
        for guard in (None, {'rule': 'monotone_skip_v1', 'bin': 999, 'from_bin': 998, 'table_action': 'skip'}):
            write_decision(DAY, root, action='double', scoreable=True, source='mdp',
                           streak=50, saver_available=False, delivery_status='delivered',
                           primary={'batter_id': 100, 'p_game_hit': .96},
                           double_down={'batter_id': 101, 'p_game_hit': .95},
                           policy_guard=guard)
            try:
                result = run_census(picks_dir=root, ledger_path=ledger, policy_path=Path('data/models/mdp_policy.npz'),
                                    saver_active=False, start=DAY, end=DAY, as_of=DAY, output=root / 'out.json')
            except CensusHalt as ex:
                assert guard is None
                print('no guard:', ex)
            else:
                assert guard is not None
                assert result['rows'][0]['parity'] == 'guarded_play_excluded'
                print('impossible guard:', result['rows'][0])
