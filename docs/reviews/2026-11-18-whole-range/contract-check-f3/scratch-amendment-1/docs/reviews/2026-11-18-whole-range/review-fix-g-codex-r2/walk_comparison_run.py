"""Compare the saved combined stream with the separate fresh stdout/stderr."""
from pathlib import Path

probe = Path(__file__).with_name("walk_comparison.py")
source = probe.read_text()
original = 'assert saved_body + b"\\n" == (HERE / "walk.txt").read_bytes()'
replacement = ('assert saved_body + b"\\n" == '
               '(HERE / "walk.stderr").read_bytes() + (HERE / "walk.txt").read_bytes()')
assert source.count(original) == 1
exec(compile(source.replace(original, replacement), str(probe), "exec"))
