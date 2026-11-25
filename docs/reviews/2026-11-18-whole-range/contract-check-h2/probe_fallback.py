from pathlib import Path
import sys


runner = Path(__file__).with_name('probe_commits.py')
source = runner.read_text()
old = 'low = frame.assign(p_game_hit=[.70, .69], batter_id=[900, 901], batter_name=[\'Fresh A\', \'Fresh B\'])'
assert old in source
source = source.replace(old, 'low = frame.head(2).assign(p_game_hit=[.70, .69], batter_id=[900, 901], '
                        'batter_name=[\'Fresh A\', \'Fresh B\'])')
sys.argv[1:] = ['scratch', 'fallback']
exec(compile(source, str(runner), 'exec'), {'__file__': str(runner), '__name__': '__main__'})
