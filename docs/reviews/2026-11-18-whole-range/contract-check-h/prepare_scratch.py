import io
from pathlib import Path
import re
import subprocess
import tarfile


CHECK = Path(__file__).resolve().parent
ROOT = CHECK.parents[1]
SCRATCH = CHECK / 'scratch'
BASE = CHECK / 'base'


def extract(commit, destination, *paths):
    destination.mkdir(exist_ok=True)
    raw = subprocess.check_output(['git', 'archive', commit, *paths], cwd=ROOT, timeout=12)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(destination, filter='data')


extract('71590310', BASE, 'src')
extract('HEAD', SCRATCH, 'src', 'scripts', 'tests', 'data', 'deploy', 'config', 'docs',
        'CLAUDE.md', 'README.md', 'INCIDENT.md', 'ARCHITECTURE.md', 'pyproject.toml')

tail = SCRATCH / 'src/bts/simulate/tail_policy.py'
s = tail.read_text()
s = s.replace('OBJECTIVE_TAIL =', 'LONGEST_STREAK_FROM = 57\nOBJECTIVE_LONGEST = "longest_streak"\nOBJECTIVE_TAIL =', 1)
s = s.replace('    if d <= 0 or streak >= target:',
              '    if d > 0 and streak >= LONGEST_STREAK_FROM:\n        return OBJECTIVE_LONGEST\n'
              '    if d <= 0 or streak >= target:', 1)
tail.write_text(s)

strategy = SCRATCH / 'src/bts/strategy.py'
s = strategy.read_text().replace('    OBJECTIVE_REACH57, OBJECTIVE_TAIL,',
                                  '    OBJECTIVE_LONGEST, OBJECTIVE_REACH57, OBJECTIVE_TAIL,', 1)
s = s.replace('        if objective == OBJECTIVE_TAIL:',
              '        if objective == OBJECTIVE_LONGEST:\n            return 0.0\n'
              '        if objective == OBJECTIVE_TAIL:', 1)
s = s.replace('    if objective == OBJECTIVE_TAIL:\n        effective_m',
              '    if objective == OBJECTIVE_LONGEST:\n'
              '        tail = (ctx.mdp or {}).get("tail")\n'
              '        paired = tail is not None and ctx.mdp.get("base_sha256") == tail.base_policy_sha256\n'
              '        source = "mdp"\n'
              '        policy_action = ("double" if d_eff <= 3 else "single") if paired else "single"\n'
              '        tail_sha = tail.sha256 if paired else None\n'
              '        degraded = None if paired else "longest-streak rule unavailable (scratch); forced single"\n'
              '    elif objective == OBJECTIVE_TAIL:\n        effective_m', 1)
strategy.write_text(s)
decision = SCRATCH / 'src/bts/daily_decision.py'
s = decision.read_text().replace('OBJECTIVES = ("reach57", "emax_season_best")',
                                 'OBJECTIVES = ("reach57", "emax_season_best", "longest_streak")', 1)
decision.write_text(s)

direct = re.compile(r'\bbts\.(?:strategy\b|simulate\.(?:mdp|tail_policy)\b|scheduler\b|daily_decision\b|health\.(?:mdp_policy_alignment|tail_policy)\b)|'
                    r'from bts(?:\.simulate|\.health)? import[^\n]*\b(?:strategy|scheduler|daily_decision|mdp|tail_policy|mdp_policy_alignment)\b|'
                    r'\b(?:mdp_objective|lookup_action|resolve_policy_decision|effective_pick_bar|policy_decision|run_single_check|run_and_pick)\b|'
                    r'\bseason_walk\b|\bbts\.orchestrator\b')
paths = [p.relative_to(SCRATCH) for p in sorted((SCRATCH / 'tests').rglob('test*.py'))
         if direct.search(p.read_text())]
(CHECK / 'test-files.txt').write_text(''.join(f'{p}\n' for p in paths))
print(f'scratch={SCRATCH} selected test files={len(paths)}')
