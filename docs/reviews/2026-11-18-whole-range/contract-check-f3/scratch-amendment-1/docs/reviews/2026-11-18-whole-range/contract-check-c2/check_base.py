"""Run the retained C2 probes with bounded I/O and a serializable poll result."""

from contextlib import redirect_stderr
from importlib import import_module
from io import StringIO
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import probe
import replay
import verify


def interrupt_case(directory):
    def patch_fixture(target, attribute, *args, **kwargs):
        if attribute == 'run_result_polling':
            kwargs['return_value'] = 'final'
        return patch.object(target, attribute, *args, **kwargs)

    with patch.object(verify, 'patch', SimpleNamespace(object=patch_fixture)):
        verify.interrupt_receipt(directory)


if __name__ == '__main__':
    signal.alarm(45)
    runner = import_module('run-review')
    journal = StringIO()
    try:
        with TemporaryDirectory(dir=probe.HERE) as tmp, redirect_stderr(journal), \
                patch.object(probe, 'run_day_cases', runner.restart_cases), \
                patch.object(replay, 'interrupt_receipt', interrupt_case):
            replay.main(Path(tmp))
    except Exception:
        print(journal.getvalue()[-3000:])
        raise
    finally:
        signal.alarm(0)
