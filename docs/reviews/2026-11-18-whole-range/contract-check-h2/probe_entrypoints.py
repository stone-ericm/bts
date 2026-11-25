import os
from pathlib import Path
import sys


check = Path(__file__).resolve().parent
project = check.parents[1]
scratch = check / 'scratch'
mode, family, *args = sys.argv[1:]
archives = project / 'docs/reviews/2026-11-18-whole-range/contract-check-h'
os.chdir(scratch)
sys.path[:0] = [str((check / 'base' if mode == 'base' else scratch) / 'src'), str(scratch)]
import bts

print('bts', bts.__file__, family, flush=True)
source = archives / f'{family}.py'
prior = source.read_text()
if family == 'decision_probe':
    day_cases = ('for kind in ("dated", "automatic"):\n'
                 '    for text in ("2027-07-15", "2027-09-24", "2027-09-25", "2027-09-26"):\n'
                 '        with TemporaryDirectory() as temp:\n'
                 '            root = Path(temp)\n'
                 '            entrypoints(kind, date.fromisoformat(text))\n\n')
    first = prior.index('with TemporaryDirectory() as temp:\n    root = Path(temp)\n    entrypoints(')
    last = prior.index('with TemporaryDirectory() as temp:\n    root = Path(temp)\n    date_str', first)
    prior = prior[:first] + day_cases + prior[last:]
    prior = prior.replace('.observations[0]', '.rows[0]')
    prior = prior.replace('scheduler.SchedulerState(date=date_str)',
                          'scheduler.SchedulerState(date_str, None, [], [], [], False, None, None, None)')
elif family == 'delivery_probe':
    prior = prior.replace('next_wakeup=None)', 'next_wakeup=None)')
elif family == 'scheduler_probe':
    prior = prior.replace('day = date(2027, 7, 15)',
                          'day = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2027, 7, 15)')
    prior = prior.replace("walk.State('dated', 57, 57, 'used', .96)",
                          "walk.State(sys.argv[3] if len(sys.argv) > 3 else 'dated', 57, 57, 'used', .96)")
sys.argv = [__file__, *(args if family == 'loader_probe' else [mode, *args])]
exec(compile(prior, str(source), 'exec'), {'__file__': __file__, '__name__': '__main__'})
