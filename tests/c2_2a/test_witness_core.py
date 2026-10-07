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
    assert w.sealed and W.current() is None and frame.attrs["serving"]["calibration"]["status"] == "off"
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


def test_an_unknown_status_is_never_published_as_not_applied():
    c = W.Calibration()
    c.enabled = True
    rec = c.record()
    assert rec["status"] is None and rec["applied"] is None
    c.applied = True
    assert c.record()["applied"] is True and c.record()["status"] == "applied"
