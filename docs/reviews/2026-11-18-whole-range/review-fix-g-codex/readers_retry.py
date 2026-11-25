"""Pass the nullable double-down and runner-up fields for the stale preview."""
from pathlib import Path

original = Path(__file__).with_name('readers.py')
program = original.read_text().replace(
    'pick=picks.pick_from_row(preview)), root)',
    'pick=picks.pick_from_row(preview), double_down=None, runner_up=None), root)',
)
exec(compile(program, str(original), 'exec'))
