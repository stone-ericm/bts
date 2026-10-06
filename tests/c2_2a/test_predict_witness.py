"""C2 step 2a, design §3.0/§3.1/§4: the PA-parquet capture, the hashing blend save, run_pipeline's model provenance
and R10's explicit lineup state (`docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md`)."""
import builtins
import hashlib
import io
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from bts.model import predict as P


# ---------------------------------------------------------------- PA parquet capture (§3.0)

@pytest.fixture
def parquet(tmp_path):
    path = tmp_path / "pa_2026.parquet"
    pd.DataFrame({"batter_id": [1, 2, 3], "date": ["2026-06-01"] * 3, "is_hit": [1, 0, 1]}).to_parquet(path)
    return path


def _spy_read_parquet(monkeypatch):
    calls = []
    real = pd.read_parquet

    def spy(src, *a, **k):
        calls.append(src)
        return real(src, *a, **k)
    monkeypatch.setattr(P.pd, "read_parquet", spy)
    return calls


def test_parquet_is_parsed_once_from_the_hashed_buffer(parquet, monkeypatch):
    expected = pd.read_parquet(parquet)
    calls = _spy_read_parquet(monkeypatch)
    inputs, errors = [], []
    got = P._read_pa_parquet(parquet, inputs, errors)
    pd.testing.assert_frame_equal(got, expected)
    assert len(calls) == 1 and isinstance(calls[0], io.BytesIO)
    raw = parquet.read_bytes()
    assert inputs == [{"file": "pa_2026.parquet", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}]
    assert errors == []


def test_parquet_capture_preparation_failure_parses_the_original_path_once(parquet, monkeypatch):
    expected = pd.read_parquet(parquet)
    calls = _spy_read_parquet(monkeypatch)

    def no_buffer(self):
        raise MemoryError("synthetic buffer failure")
    monkeypatch.setattr(P.Path, "read_bytes", no_buffer)
    inputs, errors = [], []
    pd.testing.assert_frame_equal(P._read_pa_parquet(parquet, inputs, errors), expected)
    assert calls == [parquet]
    assert inputs == [{"file": "pa_2026.parquet", "bytes": None, "sha256": None}]
    assert len(errors) == 1 and "parsed from path" in errors[0]


def test_parquet_hash_failure_still_parses_the_held_buffer(parquet, monkeypatch):
    calls = _spy_read_parquet(monkeypatch)

    class Boom:
        def __init__(self, *a, **k):
            raise MemoryError("synthetic hash failure")
    monkeypatch.setattr(hashlib, "sha256", Boom)
    inputs, errors = [], []
    P._read_pa_parquet(parquet, inputs, errors)
    assert len(calls) == 1 and isinstance(calls[0], io.BytesIO)
    assert inputs[0]["sha256"] is None and inputs[0]["bytes"] == parquet.stat().st_size and errors


def test_a_parquet_parser_error_propagates_without_a_second_parse(tmp_path, monkeypatch):
    bad = tmp_path / "pa_2026.parquet"
    bad.write_bytes(b"PAR1 this is not a parquet file")
    with pytest.raises(Exception) as original:
        pd.read_parquet(bad)
    calls = _spy_read_parquet(monkeypatch)
    with pytest.raises(type(original.value)):
        P._read_pa_parquet(bad, [], [])
    assert len(calls) == 1


def test_a_parquet_collector_failure_returns_the_frame(parquet):
    class NoAppend(list):
        def append(self, x):
            raise RuntimeError("synthetic collector failure")
    errors = []
    pd.testing.assert_frame_equal(P._read_pa_parquet(parquet, NoAppend(), errors), pd.read_parquet(parquet))
    assert errors


# ---------------------------------------------------------------- the hashing save (§3.1)

def _blend():
    return {"a": (np.arange(50_000, dtype=np.float64), ["c1", "c2"]),     # > one pickle frame: direct writes
            "b": {"k": "v" * 1000}, "_model": [1, 2, 3]}


def _baseline_save(blend, path):
    """save_blend as deployed at f882411."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(blend, f)


def test_save_writes_exactly_the_pickle_bytes_and_returns_their_hash(tmp_path):
    path = tmp_path / "models" / "blend.pkl"
    errors = []
    digest = P.save_blend(_blend(), path, errors)
    raw = path.read_bytes()
    assert raw == pickle.dumps(_blend())
    assert digest == hashlib.sha256(raw).hexdigest() and errors == []


def test_save_keeps_the_original_signature_for_existing_callers(tmp_path):
    P.save_blend(_blend(), tmp_path / "blend.pkl")
    assert (tmp_path / "blend.pkl").read_bytes() == pickle.dumps(_blend())


def test_a_serialization_error_behaves_as_before(tmp_path):
    bad = {"a": b"x" * 200_000, "b": lambda: 0}
    old, new = tmp_path / "old.pkl", tmp_path / "new.pkl"
    old.write_bytes(b"previous contents")
    new.write_bytes(b"previous contents")
    with pytest.raises(Exception) as e_old:
        _baseline_save(bad, old)
    with pytest.raises(type(e_old.value)) as e_new:
        P.save_blend(bad, new, [])
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


def test_a_partial_write_error_behaves_as_before(tmp_path, monkeypatch):
    real_open = builtins.open

    def failing_open(path, mode="r", *a, **k):
        f = real_open(path, mode, *a, **k)
        return _FailingFile(f, after=1) if "w" in mode and str(path).endswith(".pkl") else f
    monkeypatch.setattr(builtins, "open", failing_open)
    old, new = tmp_path / "old.pkl", tmp_path / "new.pkl"
    with pytest.raises(OSError) as e_old:
        _baseline_save(_blend(), old)
    with pytest.raises(OSError) as e_new:
        P.save_blend(_blend(), new, [])
    assert e_new.value.args == e_old.value.args
    monkeypatch.setattr(builtins, "open", real_open)
    assert new.read_bytes() == old.read_bytes() and len(new.read_bytes()) > 0


def test_a_hash_update_failure_keeps_the_bytes_and_nulls_the_digest(tmp_path, monkeypatch):
    class BadHash:
        def update(self, b):
            raise MemoryError("synthetic hash update failure")

        def hexdigest(self):
            return "0" * 64
    monkeypatch.setattr(hashlib, "sha256", lambda *a: BadHash())
    errors = []
    assert P.save_blend(_blend(), tmp_path / "blend.pkl", errors) is None
    monkeypatch.undo()
    assert (tmp_path / "blend.pkl").read_bytes() == pickle.dumps(_blend()) and errors


def test_a_hash_finalisation_failure_nulls_the_digest(tmp_path, monkeypatch):
    real = hashlib.sha256

    class BadFinal:
        def __init__(self):
            self._h = real()

        def update(self, b):
            self._h.update(b)

        def hexdigest(self):
            raise MemoryError("synthetic finalisation failure")
    monkeypatch.setattr(hashlib, "sha256", lambda *a: BadFinal())
    errors = []
    assert P.save_blend(_blend(), tmp_path / "blend.pkl", errors) is None and errors


def test_a_hashing_writer_construction_failure_saves_through_the_plain_file(tmp_path, monkeypatch):
    class Broken:
        def __init__(self, *a, **k):
            raise MemoryError("synthetic construction failure")
    monkeypatch.setattr(P, "HashingWriter", Broken)
    errors = []
    assert P.save_blend(_blend(), tmp_path / "blend.pkl", errors) is None
    assert (tmp_path / "blend.pkl").read_bytes() == pickle.dumps(_blend()) and errors


def test_the_hashing_writer_forwards_and_returns_the_real_write_result():
    sink = io.BytesIO()
    w = P.HashingWriter(sink, [])
    assert w.write(b"abc") == 3 and w.write(memoryview(b"de")) == 2
    assert sink.getvalue() == b"abcde" and w.hexdigest() == hashlib.sha256(b"abcde").hexdigest()


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
    digest = P.save_blend(blend, path, [])
    raw = path.read_bytes()
    assert raw == pickle.dumps(blend) and digest == hashlib.sha256(raw).hexdigest()
    loaded = P.load_blend(path)
    np.testing.assert_array_equal(loaded["m1"][0].predict_proba(X), blend["m1"][0].predict_proba(X))
    np.testing.assert_array_equal(loaded["_model"].predict_proba(X), blend["_model"].predict_proba(X))


# ---------------------------------------------------------------- run_pipeline's provenance (§3.1)

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


def test_a_cached_blend_is_witnessed_as_cache_with_the_callers_hash(pipeline):
    proc, trained, seen = pipeline
    cached = {"m1": ("B1", ["f"]), "_model": "CACHED"}
    out = P.run_pipeline("2026-06-02", str(proc), cached_blend=cached, cached_blend_sha256="ab" * 32)
    assert trained == [] and seen["model"] == "CACHED" and "_model" not in cached       # the existing pop
    assert out.attrs["serving_model"] == {"source": "cache", "sha256": "ab" * 32}
    assert out.attrs["serving_inputs"] == _inputs(proc) and out.attrs["serving_errors"] == []
    assert seen["rows"] == 4


def test_an_empty_cached_blend_trains_and_is_witnessed_as_trained(pipeline, tmp_path):
    proc, trained, seen = pipeline
    save = tmp_path / "models" / "blend_2026-06-02.pkl"
    out = P.run_pipeline("2026-06-02", str(proc), cached_blend={}, save_blend_path=save,
                         cached_blend_sha256="cd" * 32)
    assert trained == ["model", "blend"]
    raw = save.read_bytes()
    assert raw == pickle.dumps({"m1": ("B1", ["f"]), "_model": "MODEL"})
    assert out.attrs["serving_model"] == {"source": "trained", "sha256": hashlib.sha256(raw).hexdigest()}


def test_training_without_a_save_path_is_witnessed_as_unsaved(pipeline):
    proc, trained, _ = pipeline
    out = P.run_pipeline("2026-06-02", str(proc))
    assert trained == ["model", "blend"]
    assert out.attrs["serving_model"] == {"source": "trained_unsaved", "sha256": None}


def test_a_save_hash_failure_is_witnessed_as_trained_with_a_null_hash(pipeline, tmp_path, monkeypatch):
    proc, _, _ = pipeline

    class Broken:
        def __init__(self, *a, **k):
            raise MemoryError("synthetic construction failure")
    monkeypatch.setattr(P, "HashingWriter", Broken)
    save = tmp_path / "blend.pkl"
    out = P.run_pipeline("2026-06-02", str(proc), save_blend_path=save)
    assert out.attrs["serving_model"] == {"source": "trained", "sha256": None}
    assert out.attrs["serving_errors"] and save.read_bytes() == pickle.dumps({"m1": ("B1", ["f"]), "_model": "MODEL"})


def test_an_attrs_failure_returns_the_predictions(pipeline, monkeypatch):
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
    assert P.run_pipeline("2026-06-02", str(proc)) is frame


def test_a_genuine_parquet_failure_still_raises_from_run_pipeline(pipeline):
    proc, _, _ = pipeline
    (proc / "pa_2024.parquet").write_bytes(b"PAR1 corrupt")
    with pytest.raises(Exception):
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
