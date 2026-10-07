"""C2 step 2a code review r4 (Eric's row C2-2a-review-r4): the witness core's earned completeness, unit by unit.

A record counts only once confirmed, a confirmation stands only for the object the computation consumed, a part is
complete only against an independent listing, and no hook records anything outside an open witness."""
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from bts import serving_witness as W


@pytest.fixture
def files(tmp_path):
    for name, body in (("a.bin", b"alpha"), ("b.bin", b"bravo!")):
        (tmp_path / name).write_bytes(body)
    return tmp_path


def _buffer(raw, sha):
    return io.BytesIO(raw)


# ---------------------------------------------------------------- the ledger

def test_a_held_object_consumed_and_confirmed_makes_the_part_complete(files):
    led = W.Ledger([], "part")
    for name in ("a.bin", "b.bin"):
        used = led.hold(files / name, _buffer)
        assert isinstance(used, io.BytesIO) and used.getvalue() == (files / name).read_bytes()
        led.confirm(used)
    assert [r["file"] for r in led.records] == ["a.bin", "b.bin"] and led.records[0]["bytes"] == 5
    assert led.complete(["a.bin", "b.bin"]) and not led.complete(["a.bin"]) and not led.complete(["b.bin", "a.bin"])


def test_a_confirmation_stands_only_for_the_held_object(files):
    led = W.Ledger([], "part")
    led.hold(files / "a.bin", _buffer)
    led.confirm(files / "a.bin")                       # the computation consumed something else (the path)
    assert led.records[-1] == {"file": "a.bin", "bytes": None, "sha256": None}
    assert not led.complete(["a.bin"])                 # the held record stays unconfirmed beside the path's record


def test_a_confirmation_never_raises(files):
    class NoAppend(list):
        def append(self, x):
            raise RuntimeError("synthetic")
    errors = []
    led = W.Ledger(errors, "part")
    led.records = NoAppend()
    led.confirm(files / "a.bin")                      # its fallback record fails: contained, and nothing counts
    assert errors and led.confirmed == [] and not led.complete(["a.bin"])


def test_a_confirmation_stands_only_for_the_last_record(files):
    led = W.Ledger([], "part")
    used = led.hold(files / "a.bin", _buffer)
    led.records.append({"file": "b.bin", "bytes": 6, "sha256": None})     # a record the hold did not make
    led.confirm(used)
    assert not led.complete(["a.bin", "b.bin"])


def test_a_lost_record_still_returns_the_held_object(files):
    class NoAppend(list):
        def append(self, x):
            raise RuntimeError("synthetic")
    errors = []
    led = W.Ledger(errors, "part")
    led.records = NoAppend()
    used = led.hold(files / "a.bin", _buffer)
    assert isinstance(used, io.BytesIO) and errors                  # one read: the held bytes are still consumed
    led.records = []
    led.confirm(used)                                 # a held object whose record did not stand: nothing counts
    assert led.records == [] and led.confirmed == [] and not led.complete(["a.bin"])


def test_only_an_unreadable_file_takes_the_unreadable_stand_in(files, monkeypatch):
    seen = []
    led = W.Ledger([], "part")
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(PermissionError("synthetic"))
                        if self.name == "a.bin" else real(self))
    assert led.hold(files / "a.bin", _buffer, lambda p: (seen.append(p.name), "stand-in")[1]) == "stand-in"
    assert led.hold(files / "a.bin", _buffer) == files / "a.bin"            # no stand-in given: the path
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(MemoryError("synthetic")))
    assert led.hold(files / "b.bin", _buffer, lambda p: "stand-in") == files / "b.bin"   # not a read failure
    assert seen == ["a.bin"] and led.records == []


def test_the_unreadable_pick_stand_in_raises_oserror_from_c():
    stand_in = W._unreadable_pick(Path("2026-06-20.json"))
    with pytest.raises(OSError):
        stand_in.read_text()
    assert stand_in.name == "2026-06-20.json" and stand_in.sha256 is None


def test_the_listing_refuses_a_count_mismatch(files):
    assert W.names(files, "*.bin", 2) == ["a.bin", "b.bin"] and W.names(files, "*.bin") == ["a.bin", "b.bin"]
    with pytest.raises(LookupError):
        W.names(files, "*.bin", 3)


# ---------------------------------------------------------------- the current witness

def test_no_hook_records_without_an_open_witness(files):
    assert W.current() is None
    assert W.pa_open() is W._NULL_LEDGER and W.calibration_pa(files / "a.bin") == files / "a.bin"
    assert W.pick_file(files / "a.bin") == files / "a.bin"
    f = io.BytesIO()
    assert W.hashing_writer(f) is f


def test_the_cache_hooks_without_a_witness_are_the_deployed_load(files):
    """Total for any input: with no witness (predict_local could not make one), the deployed loader is returned and
    nothing is recorded."""
    deployed = object()
    assert W.cache_loader(files / "a.bin", deployed, None) is deployed and W.cache_used(deployed, None) is None


def test_the_cache_hash_stands_only_for_the_held_loader(files):
    """The held loader was made (its hash recorded), but the computation used the deployed one (a fault between the
    two): the path was read a second time, so the held bytes' hash is not the loaded bytes' and is withheld."""
    w = W.Serving()
    deployed = object()
    held = W.cache_loader(files / "a.bin", deployed, w)
    assert held is not deployed and w.cache is not None
    W.cache_used(deployed, w)
    assert w.cache_used is False and any("loaded from the path" in e for e in w.errors)
    w2 = W.Serving()
    W.cache_used(W.cache_loader(files / "a.bin", deployed, w2), w2)
    assert w2.cache_used is True and w2.errors == []


def test_a_withheld_save_digest_is_recorded_as_an_error():
    class ShortSink(io.BytesIO):
        def write(self, b):
            return super().write(bytes(b)[:2])
    w = W.Serving()
    token = W.begin(w)
    try:
        f = W.hashing_writer(ShortSink())
        f.write(b"abcdef")
        W.saved(f)
    finally:
        W.end(token)
    assert w.save_digest is None and any("digest withheld" in e for e in w.errors)


def test_a_sealed_witness_records_nothing(files):
    w = W.Serving()
    token = W.begin(w)
    try:
        w.sealed = True
        assert W.current() is None and W.pa_open() is W._NULL_LEDGER and w.pipeline_inputs is None
    finally:
        W.end(token)


def test_run_end_seals_attaches_and_stops_recording():
    w = W.Serving()
    token = W.begin(w)
    frame = pd.DataFrame({"p_game_hit": [0.8]})
    W.calibration_enabled(False)
    W.run_end(frame, w, token)
    assert w.sealed and W._CURRENT.get() is None and frame.attrs["serving"]["calibration"]["status"] == "off"
    json.dumps(frame.attrs["serving"], allow_nan=False)


def test_a_second_pipeline_ledger_cannot_name_the_first_ones_files(files, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "pa_2026.parquet").write_bytes(b"x")
    (files / "pa_2025.parquet").write_bytes(b"y")
    w = W.Serving()
    token = W.begin(w)
    try:
        first = W.pa_open()
        first.confirm(first.hold(files / "pa_2025.parquet", _buffer))
        W.pa_names(files, 1, first)
        second = W.pa_open()
        W.pa_names(other, 1, second)                   # a second run's listing never replaces the first run's
    finally:
        W.end(token)
    assert second is W._NULL_LEDGER and w.pipeline_names == ["pa_2025.parquet"]
    assert w.record()["inputs"] == first.records


def test_a_resolver_that_read_nothing_consumed_no_pick_files(tmp_path):
    from bts.model import calibrate as C
    picks = tmp_path / "picks"
    picks.mkdir()
    (picks / "2026-06-20.json").write_text("{}")
    w = W.Serving()
    token = W.begin(w)
    try:
        assert C._resolve_pick_outcomes(picks, pd.DataFrame(), pd.Timestamp("2026-06-30").date(), 30) == []
    finally:
        W.end(token)
    assert w.calibration.pick_names == [] and w.calibration.picks.complete([])


def test_an_unknown_enabled_flag_withholds_the_calibration_record():
    """If the hook recording whether calibration was enabled did not run, the record is withheld, never "off"."""
    c = W.Calibration()
    assert c.record() is None and c.errors


def test_an_unknown_status_is_never_published_as_not_applied():
    c = W.Calibration()
    c.enabled = True
    rec = c.record()
    assert rec["status"] is None and rec["applied"] is None
    c.applied = True
    assert c.record()["applied"] is True and c.record()["status"] == "applied"


def test_a_held_pick_whose_record_did_not_stand_is_not_counted(tmp_path):
    """Only a path the computation read is recorded as consumed-with-unknown-bytes; a held stand-in whose record was
    lost (it has a name too) is never re-recorded that way."""
    f = tmp_path / "2026-06-20.json"
    f.write_text("{}")

    class NoAppend(list):
        def append(self, x):
            raise RuntimeError("synthetic")
    led = W.Ledger([], "pick input")
    led.records = NoAppend()
    held = led.hold(f, W._pick_text(f))
    assert held.read_text() == "{}"
    led.records = []
    led.confirm(held)
    assert led.records == [] and not led.complete(["2026-06-20.json"])


def test_a_path_read_that_failed_is_never_counted_as_consumed(tmp_path, monkeypatch):
    """Capture preparation failed (the path is used), and then the path's own read failed: skipped as deployed,
    and that file is not recorded as consumed."""
    from bts.model import calibrate as C
    picks = tmp_path / "picks"
    picks.mkdir()
    (picks / "2026-06-20.json").write_text("{}")
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(MemoryError("synthetic buffer")))
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(PermissionError("synthetic")))
    w = W.Serving()
    token = W.begin(w)
    try:
        assert C._resolve_pick_outcomes(picks, pd.DataFrame({"batter_id": [1], "date": ["2026-06-20"],
                                                             "is_hit": [1], "is_resumed_portion": [False]}),
                                        pd.Timestamp("2026-06-30").date(), 30) == []
    finally:
        W.end(token)
    assert w.calibration.picks.records == [] and not w.calibration.picks.complete(w.calibration.pick_names)


def test_the_slate_drops_the_witness_from_the_frame_even_if_taking_it_failed(tmp_path, monkeypatch):
    """save_slate's backstop: if taking the witness fails, it is still dropped from the frame before the rows are
    extracted, so pandas never copies it (r1 F1), and the slate is written with a null witness."""
    from bts import slate as S
    frame = pd.DataFrame({"batter_id": [1], "game_pk": [10], "p_game_hit": [0.8]})
    frame.attrs["serving"] = {"schema": "x"}
    monkeypatch.setattr(S, "_take_serving", lambda predictions: (_ for _ in ()).throw(MemoryError("synthetic")))
    g = pd.DataFrame.__finalize__.__globals__
    real = g["deepcopy"]
    monkeypatch.setitem(g, "deepcopy", lambda obj, *a, **k: (_ for _ in ()).throw(MemoryError("copied the witness"))
                        if isinstance(obj, dict) and "serving" in obj else real(obj, *a, **k))
    path = S.save_slate(frame, "2026-06-30", tmp_path, "local")
    assert path is not None and json.loads(path.read_text())["serving"] is None and frame.attrs == {}
