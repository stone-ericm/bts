import io
import re
import subprocess
import tarfile
from pathlib import Path


CHECK = Path(__file__).resolve().parent
ROOT = CHECK.parents[1]
SCRATCH = CHECK / 'scratch'
BASE = CHECK / 'base'


def extract(commit, destination, *paths):
    destination.mkdir(exist_ok=True)
    raw = subprocess.check_output(['git', 'archive', commit, *paths], cwd=ROOT, timeout=30)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(destination, filter='data')


def replace(path, before, after):
    path = SCRATCH / path
    source = path.read_text()
    assert before in source, (path, before)
    path.write_text(source.replace(before, after, 1))


extract('0a2bec10', BASE, 'src')
extract('0a2bec10', SCRATCH, 'src', 'scripts', 'tests', 'data', 'deploy', 'config', 'docs',
        'CLAUDE.md', 'README.md', 'INCIDENT.md', 'ARCHITECTURE.md', 'pyproject.toml', '.github')

replace('src/bts/simulate/tail_policy.py', 'OBJECTIVE_REACH57 =',
        'LONGEST_STREAK_FROM = 57\nLONGEST_STREAK_HORIZON = 180\nOBJECTIVE_LONGEST = "longest_streak"\n'
        'OBJECTIVE_REACH57 =')
replace('src/bts/simulate/tail_policy.py', '    d = effective_days(days_remaining, season_length)',
        '    if streak >= LONGEST_STREAK_FROM and days_remaining > 0:\n'
        '        return OBJECTIVE_LONGEST\n'
        '    d = effective_days(days_remaining, season_length)')
replace('src/bts/strategy.py', '    OBJECTIVE_REACH57, OBJECTIVE_TAIL,',
        '    LONGEST_STREAK_HORIZON, OBJECTIVE_LONGEST, OBJECTIVE_REACH57, OBJECTIVE_TAIL,')
replace('src/bts/strategy.py', '    return mdp_objective(streak, days, season_length), effective_days(days, season_length)',
        '    objective = mdp_objective(streak, days, season_length)\n'
        '    if objective == OBJECTIVE_LONGEST:\n'
        '        return objective, min(days, LONGEST_STREAK_HORIZON)\n'
        '    return objective, effective_days(days, season_length)')
replace('src/bts/strategy.py', '    if mdp is _UNSET:\n        mdp = _load_mdp()\n    try:',
        '    try:\n'
        '        if mdp is _UNSET:\n'
        '            if mdp_objective(streak, _days_remaining(date)) == OBJECTIVE_LONGEST:\n'
        '                return 0.0\n'
        '            mdp = _load_mdp()')
replace('src/bts/strategy.py', '        if objective == OBJECTIVE_TAIL:',
        '        if objective == OBJECTIVE_LONGEST:\n'
        '            return 0.0\n'
        '        if objective == OBJECTIVE_TAIL:')
replace('src/bts/strategy.py', '    if objective == OBJECTIVE_TAIL:\n        effective_m',
        '    if objective == OBJECTIVE_LONGEST:\n'
        '        tail = (ctx.mdp or {}).get("tail")\n'
        '        base_sha = (ctx.mdp or {}).get("base_sha256")\n'
        '        paired = bool(tail and base_sha and base_sha == tail.base_policy_sha256\n'
        '                      and len(tail.bin_p_hit) == 1)\n'
        '        source = "mdp"\n'
        '        g = 0.0\n'
        '        policy_action = "single"\n'
        '        if paired:\n'
        '            for _ in range(d_eff):\n'
        '                values = (float(tail.bin_p_hit[0]) * (1.0 + g),\n'
        '                          float(tail.bin_p_both[0]) * (2.0 + g), g)\n'
        '                g = max(values)\n'
        '                policy_action = ("single", "double", "skip")[values.index(g)]\n'
        '            tail_sha = tail.sha256\n'
        '        else:\n'
        '            degraded = "longest-streak rule unavailable (scratch); forced single"\n'
        '    elif objective == OBJECTIVE_TAIL:\n        effective_m')
replace('src/bts/daily_decision.py', 'OBJECTIVES = ("reach57", "emax_season_best")',
        'OBJECTIVES = ("reach57", "emax_season_best", "longest_streak")')

direct = re.compile(
    r'\bbts\.(?:strategy|simulate\.(?:mdp|tail_policy)|scheduler|daily_decision|orchestrator|'
    r'health\.(?:mdp_policy_alignment|tail_policy))\b|'
    r'from bts(?:\.simulate|\.health)? import[^\n]*\b(?:strategy|scheduler|daily_decision|'
    r'mdp|tail_policy|mdp_policy_alignment|orchestrator)\b|'
    r'\b(?:mdp_objective|lookup_action|resolve_policy_decision|effective_pick_bar|'
    r'policy_decision|run_single_check|run_and_pick|season_walk|decision_objective|'
    r'is_reach57_mdp_skip|emax_season_best|reach57|mdp_policy_alignment)\b|'
    r'from (?:bts\.cli|scripts\.audit\.(?:boundary_shadow_census|p05_tail_policy_audit|bin_collapse_read))\b'
)
paths = [p.relative_to(SCRATCH) for p in sorted((SCRATCH / 'tests').rglob('test*.py'))
         if direct.search(p.read_text()) or 'season_ledger' in p.parts]
(CHECK / 'test-files.txt').write_text(''.join(f'{p}\n' for p in paths))
print(f'scratch={SCRATCH} selected test files={len(paths)}')
