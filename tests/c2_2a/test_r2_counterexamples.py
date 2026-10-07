"""C2 step 2a code review r2 (`docs/audit/2026-10-06-c2-2a-code-codex-r2.md`): each residual, as a test.

R2-1 the provenance-removal helper's dict allocation, and its invocation, were outside any guard: an injected
MemoryError there escaped predict_local and lost the forecast. R2-2 the PA collectors judged completeness by list
length, so an append that landed and then raised (its error record lost too) published clean-looking PA provenance."""
import inspect
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

try:
    import lightgbm  # noqa: F401  (bts.model.predict imports it at module level; it is an optional extra)
except (ImportError, OSError):
    pytest.skip("lightgbm unavailable (the optional model extra)", allow_module_level=True)

from bts import orchestrator as O
from bts import serving_witness as W
from bts.model import predict as P
from bts.slate import save_slate
from tests.c2_2a.test_local_witness import DATE, _expected_calibrated, _run, world  # noqa: F401
from tests.c2_2a.test_r1_counterexamples import _AppendThenRaise


def _line_of(func, pattern: str) -> int:
    """The one source line of `func` whose stripped text matches `pattern` (the test fails if it is gone)."""
    lines, start = inspect.getsourcelines(func)
    hits = [start + i for i, text in enumerate(lines) if re.fullmatch(pattern, text.strip())]
    assert len(hits) == 1, hits
    return hits[0]


def _raise_at(func, pattern: str, exc: BaseException, fired: list):
    """A trace function that raises `exc` once, at the start of that line of `func` (the reviewer's injection)."""
    code, line = func.__code__, _line_of(func, pattern)

    def local(frame, event, arg):
        if event == "line" and frame.f_lineno == line:
            fired.append(line)
            raise exc
        return local
    return lambda frame, event, arg: local if frame.f_code is code else None


def _copying_provenance_fails(monkeypatch):
    g = pd.DataFrame.__finalize__.__globals__
    real = g["deepcopy"]

    def failing(obj, *a, **k):
        if isinstance(obj, dict) and "serving_model" in obj:
            raise MemoryError("synthetic: copying attached provenance")
        return real(obj, *a, **k)
    monkeypatch.setitem(g, "deepcopy", failing)


# ---------------------------------------------------------------- R2-1

def test_a_failed_provenance_allocation_keeps_the_calibrated_forecast(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)
    fired = []
    sys.settrace(_raise_at(O._take_pipeline_provenance, r"parts(: dict)? = \{\}", MemoryError("synthetic: dict"), fired))
    try:
        out = _run(world)
    finally:
        sys.settrace(None)
    assert fired and out["p_game_hit"].tolist() == expected
    assert set(out.attrs) == {"serving"}                    # the pipeline's provenance never reached calibration
    w = out.attrs["serving"]
    assert w["model"] is None and w["inputs"] is None and w["calibration"]["status"] == "applied"
    assert any("run_pipeline provenance unavailable" in e for e in w["errors"])


def test_the_helper_itself_contains_its_allocation():
    """The helper alone, without predict_local's containment: the fault is contained, the parts are reported
    unavailable (None) and the frame is still cleared."""
    frame = pd.DataFrame({"p_game_hit": [0.8]})
    frame.attrs.update({"serving_errors": [], "serving_model": {"source": "cache"}, "serving_inputs": []})
    fired = []
    sys.settrace(_raise_at(O._take_pipeline_provenance, r"parts(: dict)? = \{\}", MemoryError("synthetic: dict"), fired))
    try:
        got = O._take_pipeline_provenance(frame, [])
    finally:
        sys.settrace(None)
    assert fired and got is None and frame.attrs == {}


def test_a_failed_provenance_take_still_clears_the_frame_before_calibration(world, monkeypatch):
    """The invocation itself fails, and copying any attached provenance would fail too: the forecast stays calibrated
    because the frame is cleared before calibration, never calibrated with the provenance still on it."""
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)
    monkeypatch.setattr(O, "_take_pipeline_provenance",
                        lambda *a, **k: (_ for _ in ()).throw(MemoryError("synthetic: invocation")))
    with monkeypatch.context() as m:
        _copying_provenance_fails(m)
        out = _run(world)
    assert out["p_game_hit"].tolist() == expected and set(out.attrs) == {"serving"}
    w = out.attrs["serving"]
    assert w["model"] is None and w["inputs"] is None and w["calibration"]["status"] == "applied"


# ---------------------------------------------------------------- R2-2

class _Lost(list):
    def append(self, x):
        raise MemoryError("synthetic: the error record is lost too")


def _pa_collects_land_then_raise(monkeypatch, which):
    real = W.collect
    seen = {"n": 0}

    def collect(items, build, errors, what):
        if what == "PA input" and items is not None:
            seen["n"] += 1
            if which(seen["n"]):
                return real(_AppendThenRaise(items), build, _Lost(), what)
        return real(items, build, errors, what)
    monkeypatch.setattr(P, "collect", collect)
    return seen


@pytest.mark.parametrize("nth, part", [(1, "inputs"), (2, "inputs"), (3, "pa_input")])
def test_a_pa_input_that_raised_after_appending_is_withheld(world, monkeypatch, tmp_path, nth, part):
    """Collects 1–2 are the pipeline's pa_2025/pa_2026, 3 is calibration's pa_2026 (sorted, pipeline first)."""
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)
    seen = _pa_collects_land_then_raise(monkeypatch, lambda n: n == nth)
    out = _run(world)
    assert seen["n"] == 3 and out["p_game_hit"].tolist() == expected
    w = json.loads(save_slate(out, DATE, tmp_path, "local").read_text())["serving"]
    c = w["calibration"]
    assert c["status"] == "applied" and c["applied"] is True and c["n_fit"] == 40
    if part == "inputs":
        assert w["inputs"] is None and c["pa_input"] is not None and c["pa_input"]["sha256"]
    else:
        assert c["pa_input"] is None and len(w["inputs"]) == 2 and all(i["sha256"] for i in w["inputs"])


def test_a_fallback_pa_input_that_raised_after_appending_is_withheld(world, monkeypatch, tmp_path):
    """The same on the path-parse fallback: every PA buffer fails, and every PA collect lands and then raises."""
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)
    real_rb = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(MemoryError("synthetic: buffer"))
                        if self.suffix == ".parquet" else real_rb(self))
    seen = _pa_collects_land_then_raise(monkeypatch, lambda n: True)
    out = _run(world)
    assert seen["n"] == 3 and out["p_game_hit"].tolist() == expected
    w = json.loads(save_slate(out, DATE, tmp_path, "local").read_text())["serving"]
    assert w["inputs"] is None and w["calibration"]["pa_input"] is None
    assert w["calibration"]["status"] == "applied"


def test_the_count_still_withholds_when_the_flag_cannot_be_cleared(world, monkeypatch, tmp_path):
    """Two layers: if a failed collection could not even clear the flag, the record count still withholds (r1 F3's
    rule, kept alongside the flag)."""
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    real = W.collect

    class NoAppend(list):
        def append(self, x):
            raise MemoryError("synthetic: append")

    def collect(items, build, errors, what):
        if what == "PA input" and items is not None:
            return real(NoAppend(), build, errors, what)
        return real(items, build, errors, what)
    monkeypatch.setattr(P, "collect", collect)
    monkeypatch.setattr(P, "_mark_incomplete", lambda ok: None)       # the flag write itself is lost
    w = json.loads(save_slate(_run(world), DATE, tmp_path, "local").read_text())["serving"]
    assert w["inputs"] is None and w["calibration"]["pa_input"] is None
    assert w["calibration"]["status"] == "applied"

