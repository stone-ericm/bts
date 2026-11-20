"""Entry point for all C2 probes. Uses the real CLI live gate with valid config."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from unittest.mock import patch

import probe
import replay
from bts import scheduler


def invoke_with_required_state(*args, **kwargs):
    # The null-pick-ID fixture mocks the actual read on its first send. Its
    # subsequent CLI restarts exercise the production gate with live settings.
    with patch.object(scheduler, '_contest_state_required', lambda config: True):
        return original_invoke(*args, **kwargs)


original_invoke = probe.opening.invoke
original_run_day_cases = probe.run_day_cases


def restart_cases(directory):
    with patch.object(probe.opening, 'invoke', invoke_with_required_state):
        original_run_day_cases(directory)


if __name__ == '__main__':
    signal.alarm(45)
    journal = StringIO()
    try:
        with TemporaryDirectory(dir=probe.HERE) as tmp, redirect_stderr(journal), \
                patch.object(probe, 'run_day_cases', restart_cases):
            replay.main(Path(tmp))
    except Exception:
        print(journal.getvalue()[-3000:])
        raise
    finally:
        signal.alarm(0)
