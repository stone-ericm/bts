"""Run decision-record comparisons with a deadline that escapes reader catches."""

from contextlib import contextmanager
import signal
from unittest.mock import patch

import probe
from bts import picks


class OpenExceededDeadline(BaseException):
    pass


@contextmanager
def bound_open():
    def expired(*_):
        raise OpenExceededDeadline

    prior = signal.signal(signal.SIGALRM, expired)
    signal.alarm(1)
    try:
        yield
    except OpenExceededDeadline:
        raise TimeoutError("decision reader blocked on opening a FIFO") from None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior)


def comparisons():
    with patch.object(probe, "deadline", bound_open):
        baseline = probe.run("fifo", "legacy")
        plain = probe.run("fifo", "plain")
        bounded = probe.run("fifo", "regular")
    assert len(baseline[1]) == 1 and plain[0] and len(bounded[1]) == 1
    for case in ("malformed", "wrong_date"):
        assert len(probe.run(case, "regular")[1]) == 1
    without_reuse = probe.run("attempt_poll", "legacy")
    with_reuse = probe.run("attempt_poll", "regular")
    assert not without_reuse[1] and any("OUTCOME UNKNOWN" in line for line in without_reuse[2])
    assert len(with_reuse[1]) == 1


def dangling_pick():
    with probe.offline.offline(probe.offline.instant(probe.DAY, 18, 25)) as (world, _):
        path = world.picks / f"{probe.DAY}.json"
        path.symlink_to(world.picks / "missing.json")
        assert picks.load_pick(probe.DAY, world.picks) is None
        assert picks.load_pick_bytes(probe.DAY, world.picks) == (None, None)
        assert path.is_symlink()
        print("PICK both readers treat a dangling link as absent; link untouched", flush=True)


if __name__ == "__main__":
    comparisons()
    dangling_pick()
