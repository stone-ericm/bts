"""C2 step 2a code review r1 (`docs/audit/2026-10-06-c2-2a-code-codex-r1.md`): each counterexample, as a test.

F1 pandas copying the pipeline's provenance during calibration; F2 a decoder-preparation OSError; F3 an omitted input
whose error is also lost, and stale record fields; F4 error descriptions and records prepared outside the guards; F5 a
short successful write; F4 a swallowed package-version failure. Code review r4 (Eric's row C2-2a-review-r4): the same
behaviours, through the guarded hooks and the run's witness (bts.serving_witness)."""
import contextlib
import hashlib
import io
import json
import pickle

import pandas as pd
import pytest

try:
    import lightgbm  # noqa: F401  (bts.model.predict imports it at module level; it is an optional extra)
except (ImportError, OSError):
    pytest.skip("lightgbm unavailable (the optional model extra)", allow_module_level=True)

from bts import serving_witness as W
from bts.model import calibrate as C
from bts.model import predict as P
from bts.slate import save_slate
from tests.c2_2a.test_local_witness import DATE, RAW_P, _expected_calibrated, _run, world  # noqa: F401


class BadRepr(MemoryError):
    def __repr__(self):
        raise MemoryError("synthetic: formatting the witness error")

    def __str__(self):
        raise MemoryError("synthetic: formatting the witness error")


@contextlib.contextmanager
def witnessed():
    w = W.Serving()
    token = W.begin(w)
    try:
        yield w
    finally:
        W.end(token)


# ---------------------------------------------------------------- F1

def test_copying_the_pipeline_provenance_never_turns_calibration_off(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)
    g = pd.DataFrame.__finalize__.__globals__
    real = g["deepcopy"]

    def failing(obj, *a, **k):
        if isinstance(obj, dict) and ("serving_model" in obj or "serving" in obj):
            raise MemoryError("synthetic: copying attached provenance")
        return real(obj, *a, **k)
    with monkeypatch.context() as m:                     # the fault lives only while the pipeline runs
        m.setitem(g, "deepcopy", failing)
        out = _run(world)
    assert out["p_game_hit"].tolist() == expected
    assert out.attrs["serving"]["calibration"]["status"] == "applied"


def test_the_slate_never_copies_the_witness(world, monkeypatch, tmp_path):
    out = _run(world)
    g = pd.DataFrame.__finalize__.__globals__
    real = g["deepcopy"]

    def failing(obj, *a, **k):
        if isinstance(obj, dict) and "serving" in obj:
            raise MemoryError("synthetic: copying the witness")
        return real(obj, *a, **k)
    monkeypatch.setitem(g, "deepcopy", failing)
    path = save_slate(out, DATE, tmp_path, "local")
    assert path is not None and json.loads(path.read_text())["serving"]["schema"] == "bts_serving_witness_v2"


# ---------------------------------------------------------------- F2

def test_a_decoder_preparation_oserror_takes_the_fallback(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)

    class IoProxy:
        def __getattr__(self, name):
            return getattr(io, name)

        @staticmethod
        def TextIOWrapper(*a, **k):
            raise OSError("synthetic: decoder preparation, after a successful read")
    monkeypatch.setattr(W, "_io", IoProxy())
    out = _run(world)
    rec = out.attrs["serving"]["calibration"]
    assert out["p_game_hit"].tolist() == expected and rec["status"] == "applied" and rec["n_fit"] == 40
    assert rec["pick_inputs"] and all(i["sha256"] is None for i in rec["pick_inputs"])


def test_a_genuine_read_oserror_is_still_skipped_once(world, monkeypatch):
    picks = world[2]
    real_rb, real_rt, reads = P.Path.read_bytes, P.Path.read_text, []

    def rb(self):
        if self.name == "2026-06-29-00.json":
            reads.append("b")
            raise PermissionError("synthetic read error")
        return real_rb(self)
    monkeypatch.setattr(P.Path, "read_bytes", rb)
    monkeypatch.setattr(P.Path, "read_text", lambda self, *a, **k: (reads.append("t") if self.name ==
                                                                      "2026-06-29-00.json" else None,
                                                                      real_rt(self, *a, **k))[1])
    with witnessed() as w:
        samples = C._resolve_pick_outcomes(picks, pd.read_parquet(world[0] / "pa_2026.parquet"),
                                           pd.Timestamp(DATE).date(), 30)
    c = w.calibration
    assert reads == ["b"] and len(samples) == 39
    assert not c.picks.complete(c.pick_names)                       # the consumed inventory is incomplete


# ---------------------------------------------------------------- F3

def test_an_omitted_input_with_a_lost_error_is_never_published_as_complete(world, monkeypatch, tmp_path):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    picks = world[2]
    (picks / "2026-04-02.json").write_text(json.dumps({"date": "2026-04-02", "result": "hit",
                                                       "pick": {"batter_id": 1, "p_game_hit": 0.9}}))

    class DropsOne(list):
        def append(self, rec):
            if rec.get("file") == "2026-04-02.json":
                raise MemoryError("synthetic: append")
            super().append(rec)
    real_init = W.Ledger.__init__

    def init(self, errors, what):
        real_init(self, errors, what)
        if what == "pick input":
            self.records = DropsOne()
    monkeypatch.setattr(W.Ledger, "__init__", init)
    monkeypatch.setattr(W, "note", lambda *a, **k: False)              # every error record is lost
    out = _run(world)
    rec = json.loads(save_slate(out, DATE, tmp_path, "local").read_text())["serving"]["calibration"]
    assert rec["status"] == "applied" and rec["n_fit"] == 40
    assert rec["pick_inputs"] is None                                  # withheld: never a partial list as complete


def _bind_fault(monkeypatch, nth, landed):
    """The nth binding's append fails; when `landed`, it lands first and then raises."""
    real, seen = W.pick_bind, {"n": 0}

    class Lands(list):
        def append(self, x):
            super().append(x)
            raise MemoryError("synthetic: raised after appending")

    def bind(f, pick_date, slot_key, bid, sample):
        seen["n"] += 1
        if seen["n"] != nth:
            return real(f, pick_date, slot_key, bid, sample)
        c = W.current().calibration
        if not landed:
            raise MemoryError("synthetic: append")
        held, c.bindings = c.bindings, Lands(c.bindings)
        try:
            return real(f, pick_date, slot_key, bid, sample)
        finally:
            held[:] = list(c.bindings)
            c.bindings = held
    monkeypatch.setattr(W, "pick_bind", bind)


def test_a_lost_binding_withholds_the_samples_and_their_hash(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    _bind_fault(monkeypatch, 7, landed=False)
    rec = _run(world).attrs["serving"]["calibration"]
    assert rec["status"] == "applied" and rec["n_fit"] == 40
    assert rec["samples"] is None and rec["samples_sha256"] is None and rec["errors"]


def test_a_failed_record_build_attaches_null_calibration_not_a_stale_one(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)
    monkeypatch.setattr(W.Calibration, "record", lambda self: (_ for _ in ()).throw(MemoryError("synthetic")))
    out = _run(world)
    assert out["p_game_hit"].tolist() == expected and out.attrs["serving"]["calibration"] is None


# ---------------------------------------------------------------- F4

@pytest.mark.parametrize("target", ["pa_2025.parquet", "pa_2026.parquet"])
def test_an_undescribable_parquet_preparation_error_still_parses_the_path(world, monkeypatch, target):
    plain = _run(world)
    real = P.Path.read_bytes
    monkeypatch.setattr(P.Path, "read_bytes", lambda self: (_ for _ in ()).throw(BadRepr()) if self.name == target
                        else real(self))
    out = _run(world)
    assert out is not None and out["p_game_hit"].tolist() == plain["p_game_hit"].tolist()
    assert out.attrs["serving"]["inputs"] is not None


def test_an_undescribable_cache_preparation_error_still_loads_the_path(world, monkeypatch):
    from tests.c2_2a.test_local_witness import _cache
    cache = _cache(world)
    real = P.Path.read_bytes
    monkeypatch.setattr(P.Path, "read_bytes", lambda self: (_ for _ in ()).throw(BadRepr()) if self.name == cache.name
                        else real(self))
    out = _run(world)
    assert out is not None and out.attrs["serving"]["model"] == {"source": "cache", "sha256": None}


def test_an_undescribable_pick_preparation_error_still_takes_the_fallback(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)
    real = P.Path.read_bytes
    monkeypatch.setattr(P.Path, "read_bytes", lambda self: (_ for _ in ()).throw(BadRepr())
                        if self.parent.name == "picks" else real(self))
    out = _run(world)
    assert out["p_game_hit"].tolist() == expected


def test_an_undescribable_hash_error_keeps_the_forecast(world, monkeypatch):
    plain = _run(world)
    real = hashlib.sha256

    def sha(*a, **k):
        if a and isinstance(a[0], (bytes, bytearray)) and bytes(a[0][:4]) == b"PAR1":
            raise BadRepr()
        return real(*a, **k)
    monkeypatch.setattr(hashlib, "sha256", sha)
    out = _run(world)
    assert out["p_game_hit"].tolist() == plain["p_game_hit"].tolist()
    assert all(i["sha256"] is None for i in out.attrs["serving"]["inputs"])


def test_an_undescribable_writer_construction_error_still_saves(tmp_path, monkeypatch):
    class Broken:
        def __init__(self, *a, **k):
            raise BadRepr()
    monkeypatch.setattr(W, "HashingWriter", Broken)
    blend = {"m": [1, 2, 3]}
    with witnessed() as w:
        assert P.save_blend(blend, tmp_path / "b.pkl") is None
    assert (tmp_path / "b.pkl").read_bytes() == pickle.dumps(blend) and w.save_digest is None


def test_a_failed_package_query_is_recorded(monkeypatch):
    real = W.version
    monkeypatch.setattr(W, "version", lambda name: (_ for _ in ()).throw(LookupError(name)) if name == "pyarrow"
                        else real(name))
    w = W.build(model=None, inputs=None, calibration=None)
    assert w["packages"]["pyarrow"] is None and any("pyarrow" in e for e in w["errors"])


def test_note_formats_inside_its_guard():
    errors = []
    assert W.note(errors, "what", BadRepr()) is True and errors and "what" in errors[0]


# ---------------------------------------------------------------- F5

def test_a_short_successful_write_withholds_the_digest():
    class ShortSink(io.BytesIO):
        def write(self, b):
            return super().write(bytes(b)[:2])
    sink = ShortSink()
    w = W.HashingWriter(sink, hashlib.sha256())
    assert w.write(b"abcdef") == 2 and sink.getvalue() == b"ab"
    assert w.digest() is None                                # earned: 6 bytes hashed, 2 written


def test_a_full_write_keeps_the_digest():
    sink = io.BytesIO()
    w = W.HashingWriter(sink, hashlib.sha256())
    assert w.write(memoryview(b"abcdef")) == 6
    assert w.digest() == hashlib.sha256(b"abcdef").hexdigest()


def test_a_binding_that_raised_after_appending_is_still_withheld(world, monkeypatch):
    """Its append landed, with the right values, but reported failure: never confirmed, so withheld (earned)."""
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    _bind_fault(monkeypatch, 7, landed=True)
    rec = _run(world).attrs["serving"]["calibration"]
    assert rec["status"] == "applied" and rec["n_fit"] == 40
    assert rec["samples"] is None and rec["samples_sha256"] is None
