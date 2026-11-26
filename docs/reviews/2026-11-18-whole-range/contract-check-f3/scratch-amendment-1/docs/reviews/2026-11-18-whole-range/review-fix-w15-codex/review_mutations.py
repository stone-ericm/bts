import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
MUTANTS = ROOT / 'mutants'
PRODUCER = 'bts/features/probable_pitcher_cache.py'
READER = 'bts/probable_pitcher_status.py'
HEALTH = 'bts/health/probable_pitcher_cache.py'
INCIDENTS = 'tests/test_wr_w15_incidents.py'
HEALTH_TESTS = 'tests/test_wr_w15_health.py'
CORPUS = 'tests/test_wr_w15_corpus.py'
FULL = [CORPUS, INCIDENTS, HEALTH_TESTS]
BASE_SAVE = '''    if scanned and can_rewrite and not special:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
        try:
            temp.write_text(json.dumps({str(game): entry for game, entry in lookup.items()}))
            os.replace(temp, cache)
        finally:
            temp.unlink(missing_ok=True)
'''


def change(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f'Expected one occurrence of {before!r}')
    return text.replace(before, after)


def move_save(text, before_preserve):
    saved = BASE_SAVE.split('\n', 1)[1]
    save_helper = 'def save(cache, lookup):\n' + '\n'.join(line[4:] for line in saved.splitlines()) + '\n\n\n'
    text = change(text, BASE_SAVE, '')
    text = change(text, 'def build(cache, raw):', save_helper + 'def build(cache, raw):')
    if before_preserve:
        return change(text, '    can_rewrite = True\n',
                      '    if scanned and not special:\n        save(cache, lookup | scanned)\n    can_rewrite = True\n')
    text = change(text, 'def report(cache, original, special, reason, scanned, outcomes, counts):',
                  'def report(cache, original, special, reason, scanned, outcomes, counts, final_lookup):')
    text = text.replace('scanned, outcomes, counts)', 'scanned, outcomes, counts, lookup | scanned)')
    text = change(text, '    try:\n        publish(target, row)',
                  '    if scanned and not special and preserved_ok:\n        save(cache, final_lookup)\n'
                  '    try:\n        publish(target, row)')
    return change(text, '    lookup.update(scanned)\n',
                  '    if scanned and not special and not unreadable and not invalid:\n'
                  '        save(cache, lookup | scanned)\n    lookup.update(scanned)\n')


MUTATIONS = [
    ('accept-bool', PRODUCER, ('type(value) is int', 'isinstance(value, int)'), [f'{CORPUS}::test_cache_corpus', '-k', 'true']),
    ('skip-raw-validation', PRODUCER, ('if valid(entry):\n                scanned[game]', 'if True:\n                scanned[game]'),
     [f'{CORPUS}::test_raw_corpus']),
    ('replace-before-preserve', PRODUCER, lambda text: move_save(text, True), [f'{INCIDENTS}::test_interrupt_at_each_rename']),
    ('publish-after-replace', PRODUCER, lambda text: move_save(text, False), [f'{INCIDENTS}::test_interrupt_at_each_rename']),
    ('reuse-by-name', PRODUCER, ('        try:\n            with open_regular(candidate)',
                                '        if candidate.name.endswith(f".{original_sha[:12]}.json"):\n'
                                '            return candidate.name\n        try:\n            with open_regular(candidate)'),
     [f'{INCIDENTS}::test_occupied_evidence_and_status_destinations']),
    ('omit-date', PRODUCER, ('f"{day}.{reason}.{identity[:12]}.json"', 'f"{reason}.{identity[:12]}.json"'),
     [f'{INCIDENTS}::test_nonobject_features_and_status']),
    ('hash-sample-only', PRODUCER, ('dict(observation, outcomes=outcomes)', 'dict(observation, outcomes=outcomes[:10])'),
     [f'{INCIDENTS}::test_identity_includes_full_sha_and_all_per_game_outcomes']),
    ('drop-yesterday', HEALTH, ('(f"{today}.*.json", f"{today - timedelta(days=1)}.*.json")', '(f"{today}.*.json",)'),
     [f'{HEALTH_TESTS}::test_late_d_health_can_read_d_after_d_plus_one']),
    ('key-by-reason', HEALTH, (':{row[\'identity\']}', ':{row[\'reason\']}'),
     [f'{HEALTH_TESTS}::test_same_day_changed_incident_remains_visible']),
    ('follow-record-symlink', READER, ('with open_regular(path) as stream:',
                                       'with open_regular(path, follow_symlinks=True) as stream:'),
     [f'{HEALTH_TESTS}::test_malformed_status_files_warn', '-k', 'symlink']),
]


PYTEST = '''
import sys
from pathlib import Path
import bts
import pytest
assert Path(bts.__file__).resolve() == Path(sys.argv[1]).resolve() / 'bts/__init__.py'
print(f'Testing {bts.__file__}', flush=True)
raise SystemExit(pytest.main(sys.argv[2:]))
'''


def invoke(work, name, tests):
    source = work / 'src'
    stage = work / name
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / 'offline'), str(source)]),
               PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', BTS_W15_REAL_ARTIFACTS='0')
    args = [sys.executable, '-c', PYTEST, str(source), '-q', '-x', '-p', 'no:cacheprovider', '-o', 'addopts=',
            '--tb=short', '--basetemp', str(stage), *tests]
    with (work / f'{name}.log').open('w') as log:
        result = subprocess.run(args, cwd=REPO, env=env, timeout=85, stdout=log, stderr=subprocess.STDOUT)
    data = (work / f'{name}.log').read_text()
    if result.returncode == 1 and 'FAILED tests/' in data:
        return next(line for line in data.splitlines() if line.startswith('FAILED tests/'))
    if result.returncode:
        raise RuntimeError(f'{work.name} {name} test run exited {result.returncode}:\n{data[-2000:]}')
    return None


def main():
    MUTANTS.mkdir(exist_ok=True)
    survivors = []
    for name, filename, transform, targeted in MUTATIONS:
        work = MUTANTS / name
        shutil.copytree(REPO / 'src', work / 'src', ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
        path = work / 'src' / filename
        text = path.read_text()
        modified = transform(text) if callable(transform) else change(text, *transform)
        compile(modified, str(path), 'exec')
        path.write_text(modified)
        killer = invoke(work, 'targeted', targeted)
        if killer is None:
            killer = invoke(work, 'all-w15', FULL)
        if killer is None:
            survivors.append(name)
            print(f'{name}: SURVIVED all W15 tests', flush=True)
        else:
            print(f'{name}: killed by {killer.removeprefix("FAILED ")}', flush=True)
    print('Mutation survivors:', ', '.join(survivors) or 'none', flush=True)


if __name__ == '__main__':
    main()
