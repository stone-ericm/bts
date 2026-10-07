"""Red first: code review r3's two counterexamples (`docs/audit/2026-10-06-c2-2a-code-codex-r3.md`), replayed with the
reviewer's own injections against the revision-3 code (`dfbf286`). Each test asserts what the design requires, so
each is RED there. The revision-4 code has no flag helper and no `collect`, so these injections have no target in it;
their class is tested on revision 4 by `tests/c2_2a/test_r3_counterexamples.py` and the per-line sweep.

The precondition of both (the reviewer's): one PA record is appended through the real `collect` with an
append-then-raise proxy, so `collect` reports failure, and its error record is lost. Only the first pipeline PA read,
or only calibration's PA read, is affected.
- R3-1: a trace raises MemoryError at the `call` event of the actual `_mark_incomplete` code object, before the
  helper's body guard. Required: the forecast is the unfaulted one.
- R3-2: a trace raises MemoryError at the line `ok[0] = False` inside the helper, which contains it, so the flag is
  never cleared. Required: the affected part is withheld.

Run against an export of dfbf286 (its `src` and its `tests.c2_2a.test_local_witness` world), with this checkout's
interpreter; PYTHONPATH puts the export's `bts` ahead of the checkout's editable install:
    X=<scratch dir>; mkdir -p $X && git archive dfbf286 | tar -x -C $X && cp <this file> $X/ && cd $X
    TZ=America/New_York PYTHONPATH=$X/src:$X <checkout>/.venv/bin/python -B -m pytest r3_replay.py -p no:cacheprovider -q
(`python -c "import bts; print(bts.__file__)"` with the same PYTHONPATH shows the export's file.)
"""
import sys

import pytest

from bts.model import predict as P
from tests.c2_2a.test_local_witness import _run, world  # noqa: F401  (the r3 world fixture)

TARGETS = ("run_pipeline", "predict_local")


def _landed_then_raised(target: str):
    """P.collect for the first PA read made from `target`: the record lands, the append raises, the error is lost."""
    real, done = P.collect, []

    def collect(items, build, errors, what):
        if what != "PA input" or done or sys._getframe(2).f_code.co_name != target:
            return real(items, build, errors, what)
        done.append(target)

        class Proxy:
            def append(self, record):
                items.append(record)
                raise MemoryError("landed, then raised")
        return real(Proxy(), build, None, what)          # errors=None: the error record is lost
    return collect, done


def _trace_helper(event_kind: str, fired: list):
    code = P._mark_incomplete.__code__
    body_line = code.co_firstlineno + 3                  # `ok[0] = False`

    def local(frame, event, arg):                        # inside the helper only
        if event == "line" and frame.f_lineno == body_line and not fired:
            fired.append(frame.f_lineno)
            raise MemoryError("injected flag write")
        return local

    def trace(frame, event, arg):                        # the global trace: `call` events
        if frame.f_code is not code or fired:
            return None
        if event_kind == "call":
            fired.append(frame.f_lineno)
            raise MemoryError("injected flag helper invocation")
        return local
    return trace


def _faulted(world, monkeypatch, target, event_kind):
    collect, done = _landed_then_raised(target)
    monkeypatch.setattr(P, "collect", collect)
    fired = []
    sys.settrace(_trace_helper(event_kind, fired))
    try:
        out = _run(world)
    finally:
        sys.settrace(None)
    assert done == [target] and fired, "the injection did not fire"
    return out


@pytest.fixture
def calibrated(monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")


@pytest.mark.parametrize("target", TARGETS)
def test_r3_1_a_failed_flag_helper_invocation_keeps_the_forecast(world, monkeypatch, calibrated, target):
    plain = _run(world)
    out = _faulted(world, monkeypatch, target, "call")
    assert out is not None and list(out["p_game_hit"]) == list(plain["p_game_hit"])


@pytest.mark.parametrize("target", TARGETS)
def test_r3_2_a_failed_flag_write_still_withholds_the_part(world, monkeypatch, calibrated, target):
    out = _faulted(world, monkeypatch, target, "line")
    serving = out.attrs["serving"]
    part = serving["inputs"] if target == "run_pipeline" else serving["calibration"]["pa_input"]
    assert part is None, f"{target}: published {part!r} after its collection reported failure"
