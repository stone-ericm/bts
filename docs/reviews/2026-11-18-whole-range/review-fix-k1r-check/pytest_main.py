"""Run the new tests against the archived main package and print its location."""

import sys
from pathlib import Path

import bts
import pytest

print(f'bts.__file__={bts.__file__}', flush=True)
assert Path(bts.__file__).resolve() == Path.cwd() / 'src/bts/__init__.py'
raise SystemExit(pytest.main(sys.argv[1:]))
