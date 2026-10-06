"""C2 step 2a code review r1 (`docs/audit/2026-10-06-c2-2a-code-codex-r1.md`): each counterexample, as a test.

F1 pandas copying the pipeline's provenance during calibration; F2 a decoder-preparation OSError; F3 an omitted input
whose error is also lost, and stale record fields; F4 error descriptions and records prepared outside the guards; F5 a
short successful write; F4 a swallowed package-version failure."""
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

from bts import orchestrator as O
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
    monkeypatch.setattr(C, "io", IoProxy())
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
    w, inputs, errors, status = {}, [], [], {}
    samples = C._resolve_pick_outcomes(picks, pd.read_parquet(world[0] / "pa_2026.parquet"),
                                       pd.Timestamp(DATE).date(), 30, bindings=[], inputs=inputs, errors=errors,
                                       status=status)
    assert reads == ["b"] and len(samples) == 39 and status["inputs_complete"] is False


# ---------------------------------------------------------------- F3

def test_an_omitted_input_with_a_lost_error_is_never_published_as_complete(world, monkeypatch, tmp_path):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    picks = world[2]
    (picks / "2026-04-02.json").write_text(json.dumps({"date": "2026-04-02", "result": "hit",
                                                       "pick": {"batter_id": 1, "p_game_hit": 0.9}}))
    real_collect = W.collect

    class NoAppend(list):
        def append(self, x):
            raise MemoryError("synthetic: append")

    def collect(items, build, errors, what):
        if items is not None and what == "pick input":
            try:
                rec = build()
            except Exception:
                rec = None
            if rec and rec.get("file") == "2026-04-02.json":
                return real_collect(NoAppend(), build, NoAppend(), what)
        return real_collect(items, build, errors, what)
    monkeypatch.setattr(C, "_collect", collect)
    monkeypatch.setattr(W, "note", lambda *a, **k: False)              # every error record is lost
    monkeypatch.setattr(C, "_note", lambda *a, **k: False)
    out = _run(world)
    rec = json.loads(save_slate(out, DATE, tmp_path, "local").read_text())["serving"]["calibration"]
    assert rec["status"] == "applied" and rec["n_fit"] == 40
    assert rec["pick_inputs"] is None                                  # withheld: never a partial list as complete


def test_a_lost_binding_withholds_the_samples_and_their_hash(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    real_collect = W.collect

    class NoAppend(list):
        def append(self, x):
            raise MemoryError("synthetic: append")
    seen = {"n": 0}

    def collect(items, build, errors, what):
        if what == "sample binding":
            seen["n"] += 1
            if seen["n"] == 7:
                return real_collect(NoAppend(), build, errors, what)
        return real_collect(items, build, errors, what)
    monkeypatch.setattr(C, "_collect", collect)
    rec = _run(world).attrs["serving"]["calibration"]
    assert rec["status"] == "applied" and rec["n_fit"] == 40
    assert rec["samples"] is None and rec["samples_sha256"] is None and rec["errors"]


def test_a_failed_record_build_attaches_null_calibration_not_a_stale_one(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)
    monkeypatch.setattr(O, "_calibration_record", lambda *a, **k: (_ for _ in ()).throw(MemoryError("synthetic")))
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
    monkeypatch.setattr(P, "HashingWriter", Broken)
    blend = {"m": [1, 2, 3]}
    assert P.save_blend(blend, tmp_path / "b.pkl", []) is None
    assert (tmp_path / "b.pkl").read_bytes() == pickle.dumps(blend)


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
    sink, errors = ShortSink(), []
    w = P.HashingWriter(sink, errors)
    assert w.write(b"abcdef") == 2 and sink.getvalue() == b"ab"
    assert w.hexdigest() is None and errors


def test_a_full_write_keeps_the_digest():
    sink = io.BytesIO()
    w = P.HashingWriter(sink, [])
    assert w.write(memoryview(b"abcdef")) == 6
    assert w.hexdigest() == hashlib.sha256(b"abcdef").hexdigest()


class _AppendThenRaise:
    """A collector whose append lands and then raises: the length looks complete; only the reported failure says not."""
    def __init__(self, target):
        self.target = target

    def append(self, x):
        self.target.append(x)
        raise MemoryError("synthetic: raised after appending")


def test_a_binding_that_raised_after_appending_is_still_withheld(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    real_collect = W.collect
    seen = {"n": 0}

    def collect(items, build, errors, what):
        if what == "sample binding" and items is not None:
            seen["n"] += 1
            if seen["n"] == 7:
                return real_collect(_AppendThenRaise(items), build, errors, what)
        return real_collect(items, build, errors, what)
    monkeypatch.setattr(C, "_collect", collect)
    rec = _run(world).attrs["serving"]["calibration"]
    assert rec["status"] == "applied" and rec["n_fit"] == 40
    assert rec["samples"] is None and rec["samples_sha256"] is None


def test_a_failed_pop_still_clears_the_pipeline_provenance(world, monkeypatch):
    """If taking an attr fails, the backstop clears them all before calibration (r1 F1)."""
    class PopFails(dict):
        def pop(self, *a):
            raise RuntimeError("synthetic: pop")

    class Frame(pd.DataFrame):
        held = PopFails()

        @property
        def _constructor(self):
            return pd.DataFrame

        @property
        def attrs(self):
            return Frame.held

        @attrs.setter
        def attrs(self, value):
            pass
    frame = Frame({"batter_id": [1], "game_pk": [10], "p_game_hit": [0.8]})
    Frame.held.update({"serving_errors": [], "serving_model": {"source": "cache", "sha256": None},
                       "serving_inputs": []})
    monkeypatch.setattr(P, "run_pipeline", lambda *a, **k: frame)
    out = _run(world)
    assert out is frame and set(Frame.held) == {"serving"}
