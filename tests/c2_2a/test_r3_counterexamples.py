"""C2 step 2a code review r3 (`docs/audit/2026-10-06-c2-2a-code-codex-r3.md`): each counterexample, as a test, against
the class repair of code review r4 (Eric's row C2-2a-review-r4).

R3-1 a fault at the invocation of a new witness helper (outside any guard) lost the pipeline's forecast or switched
calibration off. Since r4 every hook is called from a one-statement guard beside the deployed statements; the per-line
sweep (`tests/c2_2a/sweep.py`) checks every new line, and these tests keep the reviewer's own injections.
R3-2 a record that landed and reported failure, together with a failed incomplete-mark, published as complete. Since r4
nothing is marked incomplete: a record counts only once confirmed, so every combination of those faults withholds."""
import json
import sys

import pytest

try:
    import lightgbm  # noqa: F401  (bts.model.predict imports it at module level; it is an optional extra)
except (ImportError, OSError):
    pytest.skip("lightgbm unavailable (the optional model extra)", allow_module_level=True)

from bts import serving_witness as W
from bts.slate import save_slate
from tests.c2_2a.test_local_witness import DATE, _expected_calibrated, _run, world  # noqa: F401


def _fault_at_call(code, nth: int, fired: list):
    """A trace function raising MemoryError at the nth `call` event of `code` (the reviewer's R3-1 injection)."""
    seen = {"n": 0}

    def trace(frame, event, arg):
        if event == "call" and frame.f_code is code:
            seen["n"] += 1
            if seen["n"] == nth:
                fired.append(nth)
                raise MemoryError(f"synthetic: invocation {nth}")
        return None
    return trace


def _cache_bytes(world):
    return (world[1] / f"blend_{DATE}.pkl").read_bytes()


# Ledger.hold / Ledger.confirm calls in a calibrated run: 1–2 the pipeline's pa_2025/pa_2026, 3 calibration's pa_2026
# (then the pick files).
@pytest.mark.parametrize("hook", ["hold", "confirm"])
@pytest.mark.parametrize("nth, side", [(1, "pipeline"), (2, "pipeline"), (3, "calibration")])
def test_a_failed_hook_invocation_keeps_the_calibrated_forecast(world, monkeypatch, tmp_path, hook, nth, side):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)
    fired = []
    sys.settrace(_fault_at_call(getattr(W.Ledger, hook).__code__, nth, fired))
    try:
        out = _run(world)
    finally:
        sys.settrace(None)
    assert fired == [nth] and out["p_game_hit"].tolist() == expected
    w = json.loads(save_slate(out, DATE, tmp_path, "local").read_text())["serving"]
    c = w["calibration"]
    assert c["status"] == "applied" and c["applied"] is True and c["n_fit"] == 40
    files = {1: "pa_2025.parquet", 2: "pa_2026.parquet", 3: "pa_2026.parquet"}
    if hook == "hold":                       # the path was parsed instead: that file is consumed with unknown bytes
        part = w["inputs"] if side == "pipeline" else [c["pa_input"]]
        assert {"file": files[nth], "bytes": None, "sha256": None} in part
    elif side == "pipeline":                 # the confirmation never ran: that file is unconfirmed, the part withheld
        assert w["inputs"] is None and c["pa_input"] is not None
    else:
        assert c["pa_input"] is None and w["inputs"] is not None


def test_the_cache_is_the_same_bytes_under_a_failed_hook_invocation(world):
    _run(world)
    plain = _cache_bytes(world)
    (world[1] / f"blend_{DATE}.pkl").unlink()
    fired = []
    sys.settrace(_fault_at_call(W.Ledger.hold.__code__, 1, fired))
    try:
        _run(world)
    finally:
        sys.settrace(None)
    assert fired == [1] and _cache_bytes(world) == plain


@pytest.mark.parametrize("side, what", [("pipeline", "PA input"), ("calibration", "calibration PA input")])
def test_a_landed_record_with_a_failed_confirmation_is_withheld(world, monkeypatch, tmp_path, side, what):
    """R3-2's combination: the record lands and reports failure, and the confirmation fails too."""
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)

    class LandsThenRaises(list):
        def append(self, rec):
            super().append(rec)
            raise RuntimeError("synthetic: raised after appending")
    real_init, real_confirm = W.Ledger.__init__, W.Ledger.confirm

    def init(self, errors, w):
        real_init(self, errors, w)
        if w == what:
            self.records = LandsThenRaises()
    monkeypatch.setattr(W.Ledger, "__init__", init)
    monkeypatch.setattr(W.Ledger, "confirm", lambda self, used: (_ for _ in ()).throw(MemoryError("synthetic"))
                        if self.what == what else real_confirm(self, used))
    monkeypatch.setattr(W, "note", lambda *a, **k: False)                # the error records are lost as well
    out = _run(world)
    assert out["p_game_hit"].tolist() == expected
    w = json.loads(save_slate(out, DATE, tmp_path, "local").read_text())["serving"]
    if side == "pipeline":
        assert w["inputs"] is None
    else:
        assert w["calibration"]["pa_input"] is None
    assert w["calibration"]["status"] == "applied"
