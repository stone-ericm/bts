from pathlib import Path
import sys


runner = Path(__file__).with_name('probe_commits.py')
source = runner.read_text()
assert 'low = frame.assign(' in source and '    for operation in (' in source
source = source.replace('low = frame.assign(', 'low = frame.head(2).assign(', 1)
source = source.replace('    for operation in (', '    from copy import deepcopy\n'
                        '    original = deepcopy(daily)\n'
                        '    for operation in (', 1)
source = source.replace('        with ExitStack() as st:', '        daily = deepcopy(original)\n'
                        '        with ExitStack() as st:', 1)
sys.argv[1:] = ['scratch', 'fallback']
exec(compile(source, str(runner), 'exec'), {'__file__': str(runner), '__name__': '__main__'})
