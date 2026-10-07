"""C2 step 2a code review r2 (`docs/audit/2026-10-06-c2-2a-code-codex-r2.md`): each residual, as a test.

R2-1 the provenance-removal helper's dict allocation, and its invocation, were outside any guard: an injected
MemoryError there escaped predict_local and lost the forecast. Since code review r4 the pipeline records its witness
beside the computation and attaches nothing to the frame, so that helper no longer exists; what remains testable is the
property R2-1 protected: calibration never runs with provenance attached, even when copying it would fail.
R2-2 the PA collectors judged completeness by list length, so an append that landed and then raised (its error record
lost too) published clean-looking PA provenance. Since r4 a record counts only once confirmed after the parse."""
import json
from pathlib import Path

import pandas as pd
import pytest

try:
    import lightgbm  # noqa: F401  (bts.model.predict imports it at module level; it is an optional extra)
except (ImportError, OSError):
    pytest.skip("lightgbm unavailable (the optional model extra)", allow_module_level=True)

from bts import serving_witness as W
from bts.model import predict as P
from bts.slate import save_slate
from tests.c2_2a.test_local_witness import DATE, _expected_calibrated, _run, world  # noqa: F401


# ---------------------------------------------------------------- R2-1

def test_calibration_never_runs_with_provenance_attached(world, monkeypatch):
    """Copying any attached provenance fails; the forecast stays calibrated because nothing is attached before the
    witness is sealed, after calibration."""
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)
    frames = []
    real = P.run_pipeline
    monkeypatch.setattr(P, "run_pipeline", lambda *a, **k: (lambda out: (frames.append(dict(out.attrs)), out)[1])(
        real(*a, **k)))
    g = pd.DataFrame.__finalize__.__globals__
    real_copy = g["deepcopy"]

    def failing(obj, *a, **k):
        if isinstance(obj, dict) and obj:
            raise MemoryError("synthetic: copying attached provenance")
        return real_copy(obj, *a, **k)
    with monkeypatch.context() as m:
        m.setitem(g, "deepcopy", failing)
        out = _run(world)
    assert frames == [{}] and out["p_game_hit"].tolist() == expected and set(out.attrs) == {"serving"}
    assert out.attrs["serving"]["calibration"]["status"] == "applied"


# ---------------------------------------------------------------- R2-2

def _pa_records(monkeypatch, which):
    """PA record appends (the pipeline's two, then calibration's one) that land and then raise when `which(n)`."""
    seen = {"n": 0}

    class LandsThenRaises(list):
        def append(self, rec):
            super().append(rec)
            seen["n"] += 1
            if which(seen["n"]):
                raise RuntimeError("synthetic: raised after appending")
    real = W.Ledger.__init__

    def init(self, errors, what):
        real(self, errors, what)
        if what in ("PA input", "calibration PA input"):
            self.records = LandsThenRaises()
    monkeypatch.setattr(W.Ledger, "__init__", init)
    return seen


@pytest.mark.parametrize("nth, part", [(1, "inputs"), (2, "inputs"), (3, "pa_input")])
def test_a_pa_input_that_raised_after_appending_is_withheld(world, monkeypatch, tmp_path, nth, part):
    """Appends 1–2 are the pipeline's pa_2025/pa_2026, 3 is calibration's pa_2026 (sorted, pipeline first)."""
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)
    seen = _pa_records(monkeypatch, lambda n: n == nth)
    out = _run(world)
    assert seen["n"] >= 3 and out["p_game_hit"].tolist() == expected
    w = json.loads(save_slate(out, DATE, tmp_path, "local").read_text())["serving"]
    c = w["calibration"]
    assert c["status"] == "applied" and c["applied"] is True and c["n_fit"] == 40
    if part == "inputs":
        assert w["inputs"] is None and c["pa_input"] is not None and c["pa_input"]["sha256"]
    else:
        assert c["pa_input"] is None and len(w["inputs"]) == 2 and all(i["sha256"] for i in w["inputs"])


def test_a_fallback_pa_input_that_raised_after_appending_is_withheld(world, monkeypatch, tmp_path):
    """The same on the path-parse fallback: every PA buffer fails, and every PA record lands and then raises."""
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)
    real_rb = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(MemoryError("synthetic: buffer"))
                        if self.suffix == ".parquet" else real_rb(self))
    _pa_records(monkeypatch, lambda n: True)
    out = _run(world)
    assert out["p_game_hit"].tolist() == expected
    w = json.loads(save_slate(out, DATE, tmp_path, "local").read_text())["serving"]
    assert w["inputs"] is None and w["calibration"]["pa_input"] is None
    assert w["calibration"]["status"] == "applied"


def test_a_record_that_never_landed_withholds_the_part(world, monkeypatch, tmp_path):
    """r2's two layers (the count, and the flag) are one rule since r4: nothing counts unless confirmed."""
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")

    class NoAppend(list):
        def append(self, x):
            raise MemoryError("synthetic: append")
    real = W.Ledger.__init__

    def init(self, errors, what):
        real(self, errors, what)
        if what in ("PA input", "calibration PA input"):
            self.records = NoAppend()
    monkeypatch.setattr(W.Ledger, "__init__", init)
    w = json.loads(save_slate(_run(world), DATE, tmp_path, "local").read_text())["serving"]
    assert w["inputs"] is None and w["calibration"]["pa_input"] is None
    assert w["calibration"]["status"] == "applied"
