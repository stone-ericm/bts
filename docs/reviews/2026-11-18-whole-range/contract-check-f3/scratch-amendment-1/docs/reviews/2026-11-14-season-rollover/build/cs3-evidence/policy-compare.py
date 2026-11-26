import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
scratch = root / '.codex-review/cs3'
if '--worker' in sys.argv:
    from datetime import datetime, timezone
    from tempfile import TemporaryDirectory
    from bts.contest_state import load_decision_streak_state
    from bts.simulate.mdp import load_policy
    from bts.strategy import DecisionContext, resolve_policy_decision
    with TemporaryDirectory(dir=scratch) as folder:
        picks = Path(folder)
        account = picks / 'account_state'
        account.mkdir()
        (account / 'saver_state.json').write_text(json.dumps({'season': 2027, 'state': 'not_earned'}))
        (account / 'contest_streak.json').write_text(json.dumps({
            'schema_version': 'bts_contest_streak_auto_v1', 'active_streak': 3,
            'best_streak': 3, 'source': 'mlb_bts_profile', 'source_date': '2027-08-28'}))
        state = load_decision_streak_state(picks, now=datetime(2027, 8, 29, 16, tzinfo=timezone.utc))
        table, boundaries, length = load_policy(root / 'data/models/mdp_policy.npz')
        ctx = DecisionContext(.80, .80, True, '2027-08-29', True, {
            'policy_table': table, 'boundaries': boundaries, 'season_length': length})
        decision = resolve_policy_decision(ctx, state.streak, state.saver_available)
        print(json.dumps([state.saver_available, decision.action, decision.source]))
else:
    results = {}
    for name, package in [('base', scratch / 'base/src'), ('branch', root / 'src')]:
        env = dict(os.environ, PYTHONPATH=str(package), PYTHONDONTWRITEBYTECODE='1')
        value = subprocess.check_output([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                        env=env, text=True)
        results[name] = json.loads(value)
    assert results == {'base': [False, 'skip', 'mdp'], 'branch': [True, 'double', 'mdp']}, results
    print(json.dumps(results))
