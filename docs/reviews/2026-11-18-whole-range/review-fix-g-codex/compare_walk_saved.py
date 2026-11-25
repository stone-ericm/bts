"""The saved transcript has the walk output followed by a hand-written summary."""
from pathlib import Path

original = Path(__file__).with_name('compare_walk.py')
program = original.read_text().replace(
    "assert (RESULTS / 'walk-g.txt').read_bytes() == (HERE / 'walk.txt').read_bytes()",
    "assert (RESULTS / 'walk-g.txt').read_bytes().split(b'\\n\\nBaseline comparison', 1)[0] + b'\\n' == (HERE / 'walk.txt').read_bytes()",
).replace('Fresh JSON and text match the saved walk byte for byte (JSON decompressed).',
          'Fresh JSON matches the decompressed bytes; stdout matches the saved transcript before its appended summary.')
exec(compile(program, str(original), 'exec'))
