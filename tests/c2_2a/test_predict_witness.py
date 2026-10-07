"""C2 step 2a, design §3.0/§3.1/§4: the PA-parquet capture, the hashing blend save, run_pipeline's model provenance
and R10's explicit lineup state (`docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md`).

Code review r4 (Eric's row C2-2a-review-r4): the witness is recorded by guarded hooks beside the deployed statements,
into the run's witness (bts.serving_witness), and each part's completeness is earned. These tests run the real
statements of run_pipeline and save_blend, with and without an open witness."""
import builtins
import contextlib
import hashlib
import io
import json
import pickle

import numpy as np
import pandas as pd
import pytest

try:
    import lightgbm  # noqa: F401  (bts.model.predict imports it at module level; it is an optional extra)
except (ImportError, OSError):
    pytest.skip("lightgbm unavailable (the optional model extra)", allow_module_level=True)

from bts import serving_witness as W
from bts.model import predict as P


@contextlib.contextmanager
def witnessed():
    """An open witness for the code under test, as predict_local opens one."""
    w = W.Serving()
    token = W.begin(w)
    try:
        yield w
    finally:
        W.end(token)


class _Trained:
    pass


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    proc = tmp_path / "processed"
    proc.mkdir()
    for season in (2025, 2026):
        pd.DataFrame({"batter_id": [1, 2], "date": [f"{season}-06-01"] * 2, "is_hit": [1, 0]}).to_parquet(
            proc / f"pa_{season}.parquet")
    trained = []
    monkeypatch.setattr(P, "_refresh_season_data", lambda *a, **k: None)
    monkeypatch.setattr(P, "compute_all_features", lambda df: df)
    monkeypatch.setattr(P, "train_model", lambda df, feature_cols=None: (trained.append("model"), "MODEL")[1])
    monkeypatch.setattr(P, "train_blend", lambda df, **k: (trained.append("blend"), {"m1": ("B1", ["f"])})[1])
    monkeypatch.setattr(P, "_build_feature_lookups", lambda df: {})
    seen = {}

    def fake_predict(date, df, model, lookups, check_openers=True, blend=None, feature_cols=None):
        seen.update(model=model, blend=dict(blend) if blend is not None else None, rows=len(df))
        return pd.DataFrame({"batter_id": [1], "p_game_hit": [0.8]})
    monkeypatch.setattr(P, "predict", fake_predict)
    return proc, trained, seen


def _inputs(proc):
    return [{"file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(proc.glob("pa_*.parquet"))]


def _inputs(proc):
    return [{"file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(proc.glob("pa_*.parquet"))]


def _spy_read_parquet(monkeypatch):
    calls = []
    real = pd.read_parquet

    def spy(src, *a, **k):
        calls.append(src)
        return real(src, *a, **k)
    monkeypatch.setattr(P.pd, "read_parquet", spy)
    return calls


def _inputs_of(w):
    return w.record()["inputs"]


# ---------------------------------------------------------------- PA parquet capture (§3.0)

def test_parquet_is_parsed_once_from_the_hashed_buffer(pipeline, monkeypatch):
    proc, _, seen = pipeline
    calls = _spy_read_parquet(monkeypatch)
    with witnessed() as w:
        P.run_pipeline("2026-06-02", str(proc))
    assert len(calls) == 2 and all(isinstance(c, io.BytesIO) for c in calls) and seen["rows"] == 4
    assert _inputs_of(w) == _inputs(proc) and w.errors == []


def test_an_unwitnessed_run_reads_the_paths_exactly_as_deployed(pipeline, monkeypatch):
    proc, _, seen = pipeline
    calls = _spy_read_parquet(monkeypatch)
    reads = []
    real_rb = P.Path.read_bytes
    monkeypatch.setattr(P.Path, "read_bytes", lambda self: (reads.append(self.name), real_rb(self))[1])
    out = P.run_pipeline("2026-06-02", str(proc))
    assert calls == sorted(proc.glob("pa_*.parquet")) and reads == [] and out.attrs == {} and seen["rows"] == 4


def test_parquet_capture_preparation_failure_parses_the_original_path_once(pipeline, monkeypatch):
    proc, _, _ = pipeline
    calls = _spy_read_parquet(monkeypatch)
    monkeypatch.setattr(P.Path, "read_bytes", lambda self: (_ for _ in ()).throw(MemoryError("synthetic buffer")))
    with witnessed() as w:
        P.run_pipeline("2026-06-02", str(proc))
    assert calls == sorted(proc.glob("pa_*.parquet"))                        # the paths, once each
    assert _inputs_of(w) == [{"file": p.name, "bytes": None, "sha256": None} for p in sorted(proc.glob("pa_*"))]
    assert len(w.errors) == 2 and all("read from the path" in e for e in w.errors)


def test_parquet_hash_failure_still_parses_the_held_buffer(pipeline, monkeypatch):
    proc, _, _ = pipeline
    calls = _spy_read_parquet(monkeypatch)

    class Boom:
        def __init__(self, *a, **k):
            raise MemoryError("synthetic hash failure")
    monkeypatch.setattr(hashlib, "sha256", Boom)
    with witnessed() as w:
        P.run_pipeline("2026-06-02", str(proc))
    monkeypatch.undo()
    assert len(calls) == 2 and all(isinstance(c, io.BytesIO) for c in calls)
    got = _inputs_of(w)
    assert [i["sha256"] for i in got] == [None, None] and [i["bytes"] for i in got] == [
        p.stat().st_size for p in sorted(proc.glob("pa_*.parquet"))] and w.errors


def test_a_parquet_parser_error_propagates_without_a_second_parse(pipeline, monkeypatch):
    proc, _, _ = pipeline
    bad = proc / "pa_2024.parquet"
    bad.write_bytes(b"PAR1 this is not a parquet file")
    with pytest.raises(Exception) as deployed:
        pd.read_parquet(bad)
    calls = _spy_read_parquet(monkeypatch)
    with witnessed(), pytest.raises(type(deployed.value)):
        P.run_pipeline("2026-06-02", str(proc))
    assert len(calls) == 1                                                    # pa_2024 first, parsed once


def test_a_lost_record_still_parses_the_held_buffer_and_withholds_the_inputs(pipeline, monkeypatch):
    """A record append that fails is not a failed buffer: the held bytes are parsed (one read); the file is left
    unconfirmed, so the part is withheld (earned completeness)."""
    proc, _, _ = pipeline
    calls = _spy_read_parquet(monkeypatch)

    class NoAppend(list):
        def append(self, x):
            raise RuntimeError("synthetic collector failure")
    real_open = W.pa_open

    def open_broken():
        led = real_open()
        led.records = NoAppend()
        return led
    monkeypatch.setattr(W, "pa_open", open_broken)
    with witnessed() as w:
        P.run_pipeline("2026-06-02", str(proc))
    assert len(calls) == 2 and all(isinstance(c, io.BytesIO) for c in calls)
    assert _inputs_of(w) is None and "PA inputs incomplete; withheld" in w.errors


def test_a_second_run_pipeline_under_one_witness_records_nothing(pipeline, monkeypatch):
    proc, _, _ = pipeline
    with witnessed() as w:
        P.run_pipeline("2026-06-02", str(proc))
        first = list(w.pipeline_inputs.records)
        calls = _spy_read_parquet(monkeypatch)
        P.run_pipeline("2026-06-02", str(proc))
    assert calls == sorted(proc.glob("pa_*.parquet")) and w.pipeline_inputs.records == first


def test_a_changed_parquet_listing_withholds_the_inputs(pipeline, monkeypatch):
    """Completeness is checked against an independent listing after the loop: a file that appeared in between (or
    any mismatch) withholds the part."""
    proc, _, _ = pipeline
    real = W.names
    monkeypatch.setattr(W, "names", lambda d, pattern, count=None: real(d, pattern) + ["pa_2027.parquet"])
    with witnessed() as w:
        P.run_pipeline("2026-06-02", str(proc))
    assert _inputs_of(w) is None


# ---------------------------------------------------------------- the hashing save (§3.1)

def _blend():
    return {"a": (np.arange(50_000, dtype=np.float64), ["c1", "c2"]),     # > one pickle frame: direct writes
            "b": {"k": "v" * 1000}, "_model": [1, 2, 3]}


def _baseline_save(blend, path):
    """save_blend as deployed at f882411."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(blend, f)


def _baseline_save(blend, path):
    """save_blend as deployed at f882411."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(blend, f)


def test_save_writes_exactly_the_pickle_bytes_and_records_their_hash(tmp_path):
    path = tmp_path / "models" / "blend.pkl"
    with witnessed() as w:
        assert P.save_blend(_blend(), path) is None
    raw = path.read_bytes()
    assert raw == pickle.dumps(_blend())
    assert w.save_digest == hashlib.sha256(raw).hexdigest() and w.errors == []


def test_an_unwitnessed_save_writes_through_the_plain_file(tmp_path, monkeypatch):
    built = []
    real = W.HashingWriter
    monkeypatch.setattr(W, "HashingWriter", lambda *a, **k: (built.append(1), real(*a, **k))[1])
    P.save_blend(_blend(), tmp_path / "blend.pkl")
    assert (tmp_path / "blend.pkl").read_bytes() == pickle.dumps(_blend()) and built == []


@pytest.mark.parametrize("witness", [False, True], ids=["unwitnessed", "witnessed"])
def test_a_serialization_error_behaves_as_before(tmp_path, witness):
    bad = {"a": b"x" * 200_000, "b": lambda: 0}
    old, new = tmp_path / "old.pkl", tmp_path / "new.pkl"
    old.write_bytes(b"previous contents")
    new.write_bytes(b"previous contents")
    with pytest.raises(Exception) as e_old:
        _baseline_save(bad, old)
    with (witnessed() if witness else contextlib.nullcontext()), pytest.raises(type(e_old.value)) as e_new:
        P.save_blend(bad, new)
    assert str(e_new.value) == str(e_old.value)
    assert new.read_bytes() == old.read_bytes()            # truncated before serialization, same partial bytes


class _FailingFile:
    def __init__(self, f, after):
        self._f, self._left = f, after

    def write(self, b):
        if self._left <= 0:
            raise OSError(28, "No space left on device (synthetic)")
        self._left -= 1
        return self._f.write(b)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()
        return False


@pytest.mark.parametrize("witness", [False, True], ids=["unwitnessed", "witnessed"])
def test_a_partial_write_error_behaves_as_before(tmp_path, monkeypatch, witness):
    real_open = builtins.open

    def failing_open(path, mode="r", *a, **k):
        f = real_open(path, mode, *a, **k)
        return _FailingFile(f, after=1) if "w" in mode and str(path).endswith(".pkl") else f
    monkeypatch.setattr(builtins, "open", failing_open)
    old, new = tmp_path / "old.pkl", tmp_path / "new.pkl"
    with pytest.raises(OSError) as e_old:
        _baseline_save(_blend(), old)
    with (witnessed() if witness else contextlib.nullcontext()), pytest.raises(OSError) as e_new:
        P.save_blend(_blend(), new)
    assert e_new.value.args == e_old.value.args
    monkeypatch.setattr(builtins, "open", real_open)
    assert new.read_bytes() == old.read_bytes() and len(new.read_bytes()) > 0


def test_a_hash_update_failure_keeps_the_bytes_and_withholds_the_digest(tmp_path, monkeypatch):
    class BadHash:
        def update(self, b):
            raise MemoryError("synthetic hash update failure")

        def hexdigest(self):
            return "0" * 64
    monkeypatch.setattr(hashlib, "sha256", lambda *a: BadHash())
    with witnessed() as w:
        P.save_blend(_blend(), tmp_path / "blend.pkl")
    monkeypatch.undo()
    assert (tmp_path / "blend.pkl").read_bytes() == pickle.dumps(_blend())
    assert w.save_digest is None and w.errors                     # earned: 0 bytes hashed != bytes written


def test_a_hash_finalisation_failure_withholds_the_digest(tmp_path, monkeypatch):
    real = hashlib.sha256

    class BadFinal:
        def __init__(self):
            self._h = real()

        def update(self, b):
            self._h.update(b)

        def hexdigest(self):
            raise MemoryError("synthetic finalisation failure")
    monkeypatch.setattr(hashlib, "sha256", lambda *a: BadFinal())
    with witnessed() as w:
        P.save_blend(_blend(), tmp_path / "blend.pkl")
    assert w.save_digest is None and w.errors


def test_a_hashing_writer_construction_failure_saves_through_the_plain_file(tmp_path, monkeypatch):
    class Broken:
        def __init__(self, *a, **k):
            raise MemoryError("synthetic construction failure")
    monkeypatch.setattr(W, "HashingWriter", Broken)
    with witnessed() as w:
        P.save_blend(_blend(), tmp_path / "blend.pkl")
    assert (tmp_path / "blend.pkl").read_bytes() == pickle.dumps(_blend())
    assert w.save_digest is None and w.errors


def test_the_hashing_writer_forwards_and_returns_the_real_write_result():
    sink = io.BytesIO()
    w = W.HashingWriter(sink, hashlib.sha256())
    assert w.write(b"abc") == 3 and w.write(memoryview(b"de")) == 2
    assert sink.getvalue() == b"abcde" and w.digest() == hashlib.sha256(b"abcde").hexdigest()


def test_a_real_lightgbm_blend_round_trips_through_the_hashing_save(tmp_path):
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["f1", "f2", "f3"])
    y = (X["f1"] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    params = dict(n_estimators=5, num_leaves=4, random_state=7, deterministic=True, force_row_wise=True,
                  n_jobs=1, verbose=-1)
    blend = {"m1": (lgb.LGBMClassifier(**params).fit(X, y), ["f1", "f2", "f3"]),
             "_model": lgb.LGBMClassifier(**params).fit(X, y)}
    path = tmp_path / "blend.pkl"
    with witnessed() as w:
        P.save_blend(blend, path)
    raw = path.read_bytes()
    assert raw == pickle.dumps(blend) and w.save_digest == hashlib.sha256(raw).hexdigest()
    loaded = P.load_blend(path)
    np.testing.assert_array_equal(loaded["m1"][0].predict_proba(X), blend["m1"][0].predict_proba(X))
    np.testing.assert_array_equal(loaded["_model"].predict_proba(X), blend["_model"].predict_proba(X))


# ---------------------------------------------------------------- run_pipeline's model provenance (§3.1)

def test_a_cached_blend_is_witnessed_as_cache_with_the_held_bytes_hash(pipeline):
    proc, trained, seen = pipeline
    cached = {"m1": ("B1", ["f"]), "_model": "CACHED"}
    with witnessed() as w:
        loader = object()
        w.cache, w.cache_used = ("ab" * 32, loader), True          # predict_local's confirmed held load
        out = P.run_pipeline("2026-06-02", str(proc), cached_blend=cached)
    assert trained == [] and seen["model"] == "CACHED" and "_model" not in cached       # the existing pop
    assert w.model == {"source": "cache", "sha256": "ab" * 32} and out.attrs == {}
    assert _inputs_of(w) == _inputs(proc) and seen["rows"] == 4


def test_a_cache_hash_stands_only_when_the_held_load_was_used(pipeline):
    proc, _, _ = pipeline
    with witnessed() as w:
        w.cache, w.cache_used = ("ab" * 32, object()), False
        P.run_pipeline("2026-06-02", str(proc), cached_blend={"m1": ("B1", ["f"]), "_model": "CACHED"})
    assert w.model == {"source": "cache", "sha256": None}


def test_an_empty_cached_blend_trains_and_is_witnessed_as_trained(pipeline, tmp_path):
    proc, trained, seen = pipeline
    save = tmp_path / "models" / "blend_2026-06-02.pkl"
    with witnessed() as w:
        P.run_pipeline("2026-06-02", str(proc), cached_blend={}, save_blend_path=save)
    assert trained == ["model", "blend"]
    raw = save.read_bytes()
    assert raw == pickle.dumps({"m1": ("B1", ["f"]), "_model": "MODEL"})
    assert w.model == {"source": "trained", "sha256": hashlib.sha256(raw).hexdigest()}


def test_training_without_a_save_path_is_witnessed_as_unsaved(pipeline):
    proc, trained, _ = pipeline
    with witnessed() as w:
        P.run_pipeline("2026-06-02", str(proc))
    assert trained == ["model", "blend"]
    assert w.model == {"source": "trained_unsaved", "sha256": None}


def test_a_save_hash_failure_is_witnessed_as_trained_with_a_null_hash(pipeline, tmp_path, monkeypatch):
    proc, _, _ = pipeline

    class Broken:
        def __init__(self, *a, **k):
            raise MemoryError("synthetic construction failure")
    monkeypatch.setattr(W, "HashingWriter", Broken)
    save = tmp_path / "blend.pkl"
    with witnessed() as w:
        P.run_pipeline("2026-06-02", str(proc), save_blend_path=save)
    assert w.model == {"source": "trained", "sha256": None}
    assert w.errors and save.read_bytes() == pickle.dumps({"m1": ("B1", ["f"]), "_model": "MODEL"})


def test_run_pipeline_returns_the_predictions_untouched(pipeline, monkeypatch):
    """Since code review r4 run_pipeline attaches nothing to the frame (no attrs to copy or fail)."""
    proc, _, _ = pipeline

    class NoAttrs(pd.DataFrame):
        @property
        def attrs(self):
            raise RuntimeError("synthetic attrs failure")

        @attrs.setter
        def attrs(self, value):
            raise RuntimeError("synthetic attrs failure")
    frame = NoAttrs({"batter_id": [1], "p_game_hit": [0.8]})
    monkeypatch.setattr(P, "predict", lambda *a, **k: frame)
    with witnessed():
        assert P.run_pipeline("2026-06-02", str(proc)) is frame


def test_a_genuine_parquet_failure_still_raises_from_run_pipeline(pipeline):
    proc, _, _ = pipeline
    (proc / "pa_2024.parquet").write_bytes(b"PAR1 corrupt")
    with witnessed(), pytest.raises(Exception):
        P.run_pipeline("2026-06-02", str(proc))


# ---------------------------------------------------------------- R10: explicit lineup state (§4)

def _schedule(pk):
    return {"dates": [{"games": [{
        "gamePk": pk, "gameType": "R", "officialDate": "2026-06-02", "gameDate": "2026-06-02T23:05:00Z",
        "status": {"detailedState": "Pre-Game"},
        "teams": {"away": {"probablePitcher": {"id": 900, "fullName": "Away Pitcher"}},
                  "home": {"probablePitcher": {"id": 901, "fullName": "Home Pitcher"}}}}]}]}


def _feed():
    posted = {f"ID{i}": {"battingOrder": str(i * 100), "person": {"id": 100 + i, "fullName": f"Away {i}"}}
              for i in range(1, 10)}
    return {"gameData": {"venue": {"id": 3, "fieldInfo": {"roofType": "Open"}}, "weather": {}, "officials": [],
                         "teams": {"away": {"abbreviation": "AWY", "id": 10},
                                   "home": {"abbreviation": "HOM", "id": 11}}},
            "liveData": {"boxscore": {"teams": {"away": {"players": posted}, "home": {"players": {}}}},
                         "plays": {"allPlays": []}}}


def test_every_slot_carries_an_explicit_projected_boolean(monkeypatch):
    responses = {"schedule": _schedule(777), "feed": _feed()}

    class Resp:
        def __init__(self, body):
            self._b = json.dumps(body).encode()

        def read(self):
            return self._b

    def fake_urlopen(url, timeout=None):
        return Resp(responses["feed"] if "/feed/live" in url else responses["schedule"])
    monkeypatch.setattr(P, "urlopen", fake_urlopen)
    monkeypatch.setattr(P, "_fetch_prior_lineup", lambda team_id, season: [
        {"batter_id": 200 + i, "batter_name": f"Home {i}", "lineup": i} for i in range(1, 10)])
    slots = P._fetch_game_slots("2026-06-02")
    away = [s for s in slots if s["team"] == "AWY"]
    home = [s for s in slots if s["team"] == "HOM"]
    assert len(away) == 9 and len(home) == 9
    assert all(s["projected"] is False for s in away)
    assert all(s["projected"] is True for s in home)
