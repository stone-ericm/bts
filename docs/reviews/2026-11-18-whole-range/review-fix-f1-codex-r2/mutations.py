"""Change one decision at a time in a disposable source copy."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHANGES = {
    'integer': [(' and type(slot["game_pk"]) is int', '')],
    'shared': [('if len({slot["batter_id"] for slot in slots}) != len(slots):', 'if False:')],
    'prior': [('if not affected and prior_wrong_game and not qualification["all_confirmed"]:', 'if False:'),
              ('            if prior_wrong_game:\n                affected', '            if False:\n                affected')],
    'initial': [('            consumed.add("wrong_game")', '            consumed.update(("wrong_game", "initial"))')],
    'any-leg': [('not qualification["all_confirmed"]:',
                 'not any(leg["state"] == "confirmed" for leg in qualification["slots"]):')],
    'texts': [('        elif tier == "wrong_game":\n            msg =',
               '        elif tier != "wrong_game":\n            msg =')],
}


if __name__ == '__main__':
    name = sys.argv[1]
    with TemporaryDirectory(prefix=f'mutation-{name}-', dir=HERE) as tmp:
        src = Path(tmp) / 'src'
        shutil.copytree(REPO / 'src', src)
        cli = src / 'bts' / 'cli.py'
        code = cli.read_text()
        for old, new in CHANGES[name]:
            assert code.count(old) == 1, old
            code = code.replace(old, new, 1)
        cli.write_text(code)
        env = {**os.environ, 'PYTHONPATH': str(src)}
        result = subprocess.run([sys.executable, str(HERE / 'matched_tests.py'), 'f1',
                                 '-k', 'not fifo and not ordered_pairs'],
                                cwd=REPO, env=env, timeout=15)
        print(f'{name}: pytest exit {result.returncode}', flush=True)
        if result.returncode not in (0, 1):
            raise SystemExit(result.returncode)
