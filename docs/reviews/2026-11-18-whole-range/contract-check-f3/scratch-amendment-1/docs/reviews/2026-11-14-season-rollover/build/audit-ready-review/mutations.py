import os
from pathlib import Path
import shutil
import subprocess
import sys

from review_runner import PYTHON, REVIEW, ROOT, archive, environment

CONTEXT = 'src/bts/shadow_eval.py'
POLICY = 'src/bts/skip_policy_shadow.py'
CLI = 'src/bts/cli.py'
CRON = 'scripts/cron-setup-hetzner.sh'
ADOPTED = 'tests/test_audit_f6_f8_review_probes.py'
POLICY_TEST = 'tests/test_skip_policy_shadow.py'
CONTEXT_TEST = 'tests/test_shadow_eval.py'
CURRENT_ET = 'datetime.now(ZoneInfo("America/New_York")).year'
GLOB_POLICY = 'Path(picks_dir).glob("*.policy_shadow.json")'


def section_glob(tree, function, replacement):
    path = tree / POLICY
    text = path.read_text()
    start = text.index(f'def {function}(')
    assert GLOB_POLICY in text[start:text.find('\ndef ', start + 5)]
    path.write_text(text[:start] + text[start:].replace(GLOB_POLICY, replacement, 1))


def replace(tree, path, a, b):
    file = tree / path
    before = file.read_text()
    assert before.count(a) == 1, (path, a)
    file.write_text(before.replace(a, b))


CASES = (
    ('no-enter-recipient', [(CRON, 'enter) FETCH_RECIPIENT=" --dm-recipient stonehengee.bsky.social"',
                            'enter) FETCH_RECIPIENT=""')],
     [ADOPTED, 'tests/test_entry_intent.py'], 'show_keeps_four or enter_install_adds'),
    ('research-recipient', [(CRON, 'FETCH_RECIPIENT=""\ncase',
                            'FETCH_RECIPIENT=" --dm-recipient stonehengee.bsky.social"\ncase')],
     [ADOPTED, 'tests/test_entry_intent.py'], 'show_keeps_four or enter_install_adds'),
    ('scope-aged', [('find_aged_contradictions', 'Path(picks_dir).glob(f\'{now.astimezone(ZoneInfo("America/New_York")).year}-*.policy_shadow.json\')')],
     [ADOPTED, POLICY_TEST], 'aged or policy_update'),
    ('scope-prune', [(POLICY, 'def prune_superseded(picks_dir, *, now: datetime | None = None)',
                      'def prune_superseded(picks_dir, *, now: datetime | None = None, season=None)'),
                     ('prune_superseded', 'Path(picks_dir).glob("*.policy_shadow.json" if season is None else f"{season}-*.policy_shadow.json")'),
                     (CLI, 'pruned = sps.prune_superseded(picks_path)',
                      'pruned = sps.prune_superseded(picks_path, season=season or season_of_date(date) or now.astimezone(sps.ZoneInfo("America/New_York")).year)')],
     [ADOPTED, POLICY_TEST], 'prun or historical or policy_update or monitors or three_commands'),
    ('scope-reconcile', [('reconcile_pending', 'Path(picks_dir).glob(f\'{now.astimezone(ZoneInfo("America/New_York")).year}-*.policy_shadow.json\') if now else Path(picks_dir).glob("*.policy_shadow.json")')],
     [ADOPTED, POLICY_TEST], 'reconcil or aged or historical or policy_update'),
    ('unscope-context', [(CONTEXT, 'picks_dir.glob(f"{season}-*.shadow.json")', 'picks_dir.glob("*.shadow.json")')],
     [ADOPTED, CONTEXT_TEST], 'three_commands or monitors or scopes_shadow or defaults_to_et'),
    ('unscope-policy', [(POLICY, 'Path(picks_dir).glob(f"{season}-*.policy_shadow.json")', GLOB_POLICY)],
     [ADOPTED, POLICY_TEST], 'three_commands or monitors or defaults_to_et or policy_update'),
    ('utc-context', [(CONTEXT, CURRENT_ET, 'datetime.now(timezone.utc).year')],
     [ADOPTED, CONTEXT_TEST], 'defaults_to_et or three_commands or monitors'),
    ('utc-policy-loader', [(POLICY, CURRENT_ET, 'datetime.now(timezone.utc).year')],
     [ADOPTED, POLICY_TEST], 'defaults_to_et or three_commands or monitors'),
    ('utc-policy-writer', [(POLICY, 'season = now.astimezone(ZoneInfo("America/New_York")).year', 'season = now.year')],
     [ADOPTED, POLICY_TEST], 'defaults_to_et or three_commands or monitors or aged or policy_update or reconcil'),
    ('drop-p03-pin', [('scripts/audit/p03_context_shadow_closeout.py', ', season=SNAPSHOT_SEASON', '')],
     ['tests/test_p03_context_shadow_closeout.py'], 'closeout'),
)


def main():
    if len(sys.argv) > 1:
        import bts.cli
        import pytest
        import sitecustomize

        assert Path(bts.cli.__file__).resolve().is_relative_to(Path.cwd() / 'src')
        assert sitecustomize.loopback('127.0.0.1')
        return pytest.main(['-p', 'no:cacheprovider', '-q', '--tb=short', *sys.argv[1:]])

    scratch = REVIEW / 'mutation-work'
    archive('HEAD', scratch, 'src', 'scripts', 'tests', 'pyproject.toml')
    guide = Path(__file__).resolve()
    results = {}
    for name, changes, tests, selection in CASES:
        tree = REVIEW / 'mutations' / name
        shutil.copytree(scratch, tree)
        for mutation in changes:
            if len(mutation) == 2:
                section_glob(tree, *mutation)
            else:
                replace(tree, *mutation)
        env = environment(tree)
        arguments = ['-k', selection, *tests]
        result = subprocess.run([str(PYTHON), str(guide), *arguments], cwd=tree, env=env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (tree / 'pytest.log').write_text(result.stdout)
        results[name] = result.returncode
        print(f'{name}: {result.returncode}', flush=True)
        for line in result.stdout.splitlines():
            if line.startswith(('FAILED ', 'ERROR ', '=')):
                print(line, flush=True)
    return int(any(code not in (0, 1) for code in results.values()))


if __name__ == '__main__':
    raise SystemExit(main())
