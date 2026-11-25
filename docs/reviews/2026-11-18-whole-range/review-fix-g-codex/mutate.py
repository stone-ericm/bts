"""Run a single source mutation against the original focused tests."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MDP = 'simulate/mdp.py'
STRATEGY = 'strategy.py'
RECORD = 'daily_decision.py'
SCHED = 'scheduler.py'
MUTATIONS = {
    'lowest': (MDP, 'range(q - 1, -1, -1)', 'range(q)'),
    'tail': (STRATEGY, 'source = "mdp"\n        tail =', '''source = "mdp"
        if ctx.mdp and ctx.mdp.get("policy_table") is not None:
            reach_action, reach_guard = _mdp_action_from(ctx.mdp, ctx.primary_p, streak, ctx.date, saver)
            if reach_guard is not None:
                policy_action, guard = reach_action, reach_guard
        tail ='''),
    'playing': (MDP, 'if action == "skip":\n        for j', 'if action in ACTIONS:\n        for j'),
    'clamp': (STRATEGY, 'pick_bar=pick_bar, guard=guard,',
              'pick_bar=pick_bar, guard=guard if action == policy_action else None,'),
    'production_raw': (STRATEGY, 'from bts.simulate.mdp import lookup_action_guarded\n',
                       'from bts.simulate.mdp import lookup_action as lookup_action_guarded\n'),
    'offline_guarded': (MDP, 'return "skip" if indices is None else ACTIONS[policy_table[indices]]',
                        'return lookup_action_guarded(policy_table, boundaries, streak, days_remaining, saver, top_pick_confidence, season_length)[0]'),
    'skip_omit': (RECORD, 'path = decision_path(date, picks_dir)\n',
                  'if action == "skip":\n            record.pop("policy_guard")\n        path = decision_path(date, picks_dir)\n'),
    'skip_guard': (RECORD, 'policy_guard if action != "skip" else None', 'policy_guard'),
    'recovery': (SCHED, 'if saved.get("guard") is not None:', 'if False:'),
    'numpy': (MDP, '"bin": int(q)', '"bin": np.int64(q)'),
    'journal_extra': (SCHED, 'if dec.objective == "reach57" and not dec.degraded_reason:',
                      'if guard is None:\n        print("  POLICY GUARD — unchanged", file=sys.stderr)\n    if dec.objective == "reach57" and not dec.degraded_reason:'),
    'journal_saver': (SCHED, 'saver = "available" if sel.saver_available else "unavailable"',
                      'saver = "unavailable"'),
    'public_guard': (SCHED, 'print(f"  LOCKED ({label}) — Posted to Bluesky: {uri}"',
                     'print(f"  LOCKED ({label}) — Posted to Bluesky: {uri}"'),
    'skip_eod_omit': (RECORD, 'path = decision_path(date, picks_dir)\n',
                      'if action == "skip" and policy_guard is None:\n            record.pop("policy_guard")\n        path = decision_path(date, picks_dir)\n'),
}
TESTS = ('tests/test_wr_g_guard.py tests/test_boundary_shadow_census_g.py '
         'tests/test_wr_a_calendar_replay_carbon.py tests/test_tail_policy_r3_fixes.py '
         'tests/test_scheduler_tail_objective.py tests/test_daily_decision.py tests/test_strategy.py '
         'tests/test_decide_action.py tests/test_orchestrator.py tests/test_cli_integration.py '
         'tests/scripts/season_walk tests/test_lightgbm_optional.py tests/test_web_saver.py '
         'tests/test_boundary_shadow_census.py').split()

if __name__ == '__main__':
    name = sys.argv[1]
    target = HERE / 'scratch' / name / 'src'
    shutil.copytree(ROOT / 'src', target)
    file, old, new = MUTATIONS[name]
    path = target / 'bts' / file
    body = path.read_text()
    if name == 'production_raw':
        old = 'return lookup_action_guarded(\n        mdp["policy_table"], mdp["boundaries"],\n        streak, days_remaining, saver, p_game_hit, mdp["season_length"],\n    )'
        body = body.replace('import lookup_action_guarded', 'import lookup_action as lookup_action_guarded', 1)
        new = old.replace('    )', '    ), None')
    if name == 'public_guard':
        old = '_record_commit("delivered")\n        print(f"  LOCKED ({label}) — Posted'
        new = 'daily.policy_decision = None\n        selection = None\n        _record_commit("delivered")\n        print(f"  LOCKED ({label}) — Posted'
    assert old in body and old != new, (name, old)
    path.write_text(body.replace(old, new, 1))
    env = os.environ | {'PYTHONPATH': str(target), 'PYTHONDONTWRITEBYTECODE': '1'}
    subprocess.run([sys.executable, '-c', 'import bts; print(bts.__file__)'],
                   env=env, cwd=ROOT, timeout=10, check=True)
    subset = sys.argv[2:]
    result = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '-q',
                             '--tb=short', '-x', '-k', 'not test_do_post_saver_transition_socket_wiring',
                             *(subset or TESTS)], env=env, cwd=ROOT, timeout=75)
    sys.exit(result.returncode)
