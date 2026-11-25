import argparse
import signal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import probe_standing as standing


class ReadDeadline(RuntimeError):
    pass


def bounded(fn):
    def expired(*_):
        raise ReadDeadline('bounded file read')

    original = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, .08)
    try:
        return fn()
    except Exception as exc:
        return type(exc).__name__, str(exc).splitlines()[0]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollback', type=Path)
    args = parser.parse_args()
    print(standing.cs.__file__)
    # An OSError-like timeout can be swallowed by a reader's normal I/O recovery.
    # Interrupt with a different exception to surface the first blocking read.
    with patch.object(standing, 'describe', bounded):
        if args.rollback:
            standing.rollback(args.rollback)
        else:
            with TemporaryDirectory() as directory:
                root = Path(directory)
                standing.interruption(root / 'interrupt')
                standing.malformed(root / 'malformed')
                standing.rollover(root / 'year')
                standing.delivery(root / 'delivery')
