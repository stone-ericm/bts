"""Run the prose and deferral checks with space for hyphenated option help."""

from pathlib import Path
from runpy import run_path
from unittest.mock import patch

from click.testing import CliRunner


invoke = CliRunner.invoke


def invoke_wide(self, *args, **kwargs):
    kwargs.setdefault('terminal_width', 105)
    return invoke(self, *args, **kwargs)


with patch.object(CliRunner, 'invoke', invoke_wide):
    run_path(str(Path(__file__).with_name('operator_text.py')), run_name='__main__')
