"""C2 step 2a, design §3.0–§3.2: predict_local's cache capture, calibration record and witness attachment.

run_pipeline is real (its parquet reads, cache/train branch and save run), with only the heavy leaves stubbed."""
import hashlib
import io
import json
import pickle
import sys
from datetime import date, timedelta

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

DATE = "2026-06-30"
RAW_P = [0.86, 0.81, 0.74]


@pytest.fixture
def world(tmp_path, monkeypatch):
    data, models, picks = tmp_path / "processed", tmp_path / "models", tmp_path / "picks"
    for d in (data, models, picks):
        d.mkdir()
    pa_rows = []
    for i in range(40):                                  # 40 resolved picks in the 30-day window
        d = (date(2026, 6, 29) - timedelta(days=i % 29)).isoformat()
        bid = 1000 + i
        (picks / f"{d}-{i:02d}.json").write_text(json.dumps(
            {"date": d, "result": "hit", "pick": {"batter_id": bid, "p_game_hit": 0.6 + 0.3 * (i % 2)}}))
        pa_rows.append({"batter_id": bid, "date": d, "is_hit": int(i % 3 != 0), "is_resumed_portion": False})
    pd.DataFrame(pa_rows).to_parquet(data / "pa_2026.parquet")
    pd.DataFrame(pa_rows[:5]).to_parquet(data / "pa_2025.parquet")
    trained = []
    monkeypatch.setattr(P, "_refresh_season_data", lambda *a, **k: None)
    monkeypatch.setattr(P, "compute_all_features", lambda df: df)
    monkeypatch.setattr(P, "train_model", lambda df, feature_cols=None: (trained.append("model"), "MODEL")[1])
    monkeypatch.setattr(P, "train_blend", lambda df, **k: (trained.append("blend"), {"m1": ("B1", ["f"])})[1])
    monkeypatch.setattr(P, "_build_feature_lookups", lambda df: {})
    monkeypatch.setattr(P, "predict", lambda *a, **k: pd.DataFrame(
        {"batter_id": [1, 2, 3], "game_pk": [10, 11, 12], "p_game_hit": RAW_P}))
    monkeypatch.delenv("BTS_USE_CALIBRATION", raising=False)
    return data, models, picks, trained


def _run(world):
    data, models, picks, _ = world
    return O.predict_local(DATE, data_dir=str(data), models_dir=str(models), picks_dir=str(picks))


def _cache(world, blend=None):
    _, models, _, _ = world
    path = models / f"blend_{DATE}.pkl"
    path.write_bytes(pickle.dumps(blend if blend is not None else {"m1": ("B1", ["f"]), "_model": "CACHED"}))
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- the model (§3.0 cached blend, §3.1)

def test_a_cached_blend_is_loaded_from_the_hashed_bytes_once(world, monkeypatch):
    cache = _cache(world)
    reads, loads = [], []
    real_rb = P.Path.read_bytes
    monkeypatch.setattr(P.Path, "read_bytes", lambda self: (reads.append(self.name), real_rb(self))[1])
    monkeypatch.setattr(P, "load_blend", lambda path: loads.append(path))
    out = _run(world)
    assert reads.count(cache.name) == 1 and loads == [] and world[3] == []
    w = out.attrs["serving"]
    assert w["model"] == {"source": "cache", "sha256": _sha(cache)} and w["errors"] == []
    assert list(out["p_game_hit"]) == RAW_P


def test_a_cache_capture_preparation_failure_loads_the_original_path_once(world, monkeypatch):
    cache = _cache(world)
    real_rb, loads = P.Path.read_bytes, []
    real_load = P.load_blend

    def rb(self):
        if self.name == cache.name:
            raise MemoryError("synthetic buffer failure")
        return real_rb(self)
    monkeypatch.setattr(P.Path, "read_bytes", rb)
    monkeypatch.setattr(P, "load_blend", lambda path: (loads.append(path), real_load(path))[1])
    out = _run(world)
    assert loads == [cache] and world[3] == []
    w = out.attrs["serving"]
    assert w["model"] == {"source": "cache", "sha256": None} and any("loaded from path" in e for e in w["errors"])


def test_a_cache_hash_failure_still_loads_the_held_bytes(world, monkeypatch):
    _cache(world)
    real = hashlib.sha256

    def picky(data=b"", *a, **k):
        if isinstance(data, bytes) and data.startswith(b"\x80"):        # the pickle's bytes only
            raise MemoryError("synthetic hash failure")
        return real(data, *a, **k)
    monkeypatch.setattr(hashlib, "sha256", picky)
    out = _run(world)
    w = out.attrs["serving"]
    assert w["model"] == {"source": "cache", "sha256": None} and w["errors"] and world[3] == []


def test_a_genuine_unpickling_failure_propagates_once_as_before(world, monkeypatch):
    cache = _cache(world)
    cache.write_bytes(b"\x80\x04 corrupt")
    with pytest.raises(Exception) as today:
        P.load_blend(cache)                                     # what f882411's predict_local raised
    with pytest.raises(type(today.value)) as now:
        _run(world)
    assert str(now.value) == str(today.value)
    calls = []

    def stateful(raw):
        calls.append(len(raw))
        raise pickle.UnpicklingError("synthetic stateful failure")
    monkeypatch.setattr(pickle, "loads", stateful)
    monkeypatch.setattr(P, "load_blend", lambda path: calls.append("load_blend"))
    with pytest.raises(pickle.UnpicklingError, match="stateful"):
        _run(world)
    assert calls == [len(cache.read_bytes())]


def test_a_cold_train_is_witnessed_as_trained_with_the_saved_bytes_hash(world):
    out = _run(world)
    _, models, _, trained = world
    cache = models / f"blend_{DATE}.pkl"
    assert trained == ["model", "blend"]
    assert out.attrs["serving"]["model"] == {"source": "trained", "sha256": _sha(cache)}


def test_an_empty_cached_dict_trains_and_is_witnessed_as_trained(world):
    cache = _cache(world, blend={})
    empty = cache.read_bytes()
    out = _run(world)
    assert world[3] == ["model", "blend"] and cache.read_bytes() != empty
    assert out.attrs["serving"]["model"] == {"source": "trained", "sha256": _sha(cache)}


def test_inputs_and_only_the_serving_key_remain_in_attrs(world):
    data = world[0]
    out = _run(world)
    assert set(out.attrs) == {"serving"}
    assert out.attrs["serving"]["inputs"] == [
        {"file": p.name, "bytes": p.stat().st_size, "sha256": _sha(p)} for p in sorted(data.glob("pa_*.parquet"))]


def test_a_prediction_failure_returns_none_as_before(world, monkeypatch):
    monkeypatch.setattr(P, "predict", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("synthetic")))
    assert _run(world) is None


# ---------------------------------------------------------------- calibration (§3.2)

EMPTY = {"pa_input": None, "pick_inputs": None, "n_fit": None, "samples": None, "samples_sha256": None,
         "map": None, "map_sha256": None}


def test_calibration_off(world):
    cal = _run(world).attrs["serving"]["calibration"]
    assert cal == {"enabled": False, "applied": False, "status": "off", **EMPTY, "errors": []}


def test_calibration_without_a_pa_file(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    (world[0] / "pa_2026.parquet").unlink()
    out = _run(world)
    assert out.attrs["serving"]["calibration"] == {"enabled": True, "applied": False, "status": "no_pa_file",
                                                   **EMPTY, "errors": []}
    assert list(out["p_game_hit"]) == RAW_P


def _expected_calibrated(world):
    data, _, picks, _ = world
    cal = C.fit_calibrator_from_picks(picks, pd.read_parquet(data / "pa_2026.parquet"), today=date(2026, 6, 30))
    return C.apply_calibrator_series(pd.Series(RAW_P), cal).tolist(), cal


def test_calibration_applied_records_every_part(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, cal = _expected_calibrated(world)
    out = _run(world)
    assert out["p_game_hit"].tolist() == expected and out["p_game_hit_raw"].tolist() == RAW_P
    rec = out.attrs["serving"]["calibration"]
    pa = world[0] / "pa_2026.parquet"
    assert rec["enabled"] is True and rec["applied"] is True and rec["status"] == "applied" and rec["errors"] == []
    assert rec["pa_input"] == {"file": "pa_2026.parquet", "bytes": pa.stat().st_size, "sha256": _sha(pa)}
    assert rec["n_fit"] == 40 == len(rec["samples"]) and len(rec["pick_inputs"]) == 40
    assert rec["samples_sha256"] == W.canon_sha256(rec["samples"])
    assert rec["map"]["X_thresholds"] == cal.X_thresholds_.tolist() and rec["map_sha256"] == W.canon_sha256(rec["map"])
    json.dumps(out.attrs["serving"], allow_nan=False)


def test_calibration_with_insufficient_support(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    for f in sorted(world[2].glob("*.json"))[5:]:
        f.unlink()
    out = _run(world)
    rec = out.attrs["serving"]["calibration"]
    assert (rec["enabled"], rec["applied"], rec["status"], rec["n_fit"]) == (True, False, "insufficient_support", 5)
    assert rec["pa_input"] and len(rec["samples"]) == 5 and rec["samples_sha256"] and rec["map"] is None
    assert list(out["p_game_hit"]) == RAW_P and "p_game_hit_raw" not in out


def test_calibration_without_sklearn(world, monkeypatch):
    import builtins
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda name, *a, **k: (_ for _ in ()).throw(ImportError(name))
                        if name.startswith("sklearn") else real(name, *a, **k))
    rec = _run(world).attrs["serving"]["calibration"]
    assert (rec["enabled"], rec["applied"], rec["status"]) == (True, False, "no_sklearn")


def test_a_genuine_calibration_failure_before_assignment_is_failed(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    (world[2] / "2026-06-29-zz.json").write_bytes(b"\xff\xfe not utf-8")
    out = _run(world)
    rec = out.attrs["serving"]["calibration"]
    assert (rec["applied"], rec["status"]) == (False, "failed") and any("UnicodeDecodeError" in e for e in rec["errors"])
    assert list(out["p_game_hit"]) == RAW_P and "p_game_hit_raw" not in out


def test_a_failure_after_assignment_keeps_calibrated_probabilities_and_applied(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)

    class Stderr(io.StringIO):
        def write(self, s):
            if "Applied calibration" in s:
                raise OSError("synthetic stderr failure")
            return super().write(s)
    monkeypatch.setattr(sys, "stderr", Stderr())
    out = _run(world)
    rec = out.attrs["serving"]["calibration"]
    assert out["p_game_hit"].tolist() == expected
    assert (rec["applied"], rec["status"]) == (True, "applied") and any("synthetic stderr" in e for e in rec["errors"])


def test_a_calibration_pa_capture_failure_parses_the_path_and_still_applies(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)
    real_rb = P.Path.read_bytes
    seen = []

    def rb(self):
        if self.name == "pa_2026.parquet":
            seen.append(1)
            if len(seen) == 2:                                   # run_pipeline reads first; calibration second
                raise MemoryError("synthetic buffer failure")
        return real_rb(self)
    monkeypatch.setattr(P.Path, "read_bytes", rb)
    out = _run(world)
    rec = out.attrs["serving"]["calibration"]
    assert out["p_game_hit"].tolist() == expected and rec["status"] == "applied"
    assert rec["pa_input"] == {"file": "pa_2026.parquet", "bytes": None, "sha256": None} and rec["errors"]


def test_a_witness_collection_failure_never_changes_the_calibration(world, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    expected, _ = _expected_calibrated(world)
    monkeypatch.setattr(C, "_canon_sha256", lambda o: (_ for _ in ()).throw(ValueError("synthetic")))
    out = _run(world)
    assert out["p_game_hit"].tolist() == expected
    rec = out.attrs["serving"]["calibration"]
    assert rec["status"] == "applied" and rec["samples_sha256"] is None and rec["errors"]


# ---------------------------------------------------------------- attachment (§3.0 containment)

def test_a_build_failure_attaches_null_and_returns_the_forecast(world, monkeypatch):
    monkeypatch.setattr(W, "build", lambda **k: (_ for _ in ()).throw(MemoryError("synthetic build failure")))
    out = _run(world)
    assert out.attrs.get("serving") is None and list(out["p_game_hit"]) == RAW_P


def test_missing_run_pipeline_provenance_is_recorded(world, monkeypatch):
    monkeypatch.setattr(P, "run_pipeline", lambda *a, **k: pd.DataFrame({"p_game_hit": RAW_P}))
    w = _run(world).attrs["serving"]
    assert w["model"] is None and w["inputs"] is None and len([e for e in w["errors"] if "missing" in e]) == 3


def test_an_attrs_failure_returns_the_forecast(world, monkeypatch):
    class NoAttrs(pd.DataFrame):
        @property
        def attrs(self):
            raise RuntimeError("synthetic attrs failure")

        @attrs.setter
        def attrs(self, value):
            raise RuntimeError("synthetic attrs failure")
    frame = NoAttrs({"batter_id": [1], "p_game_hit": [0.8]})
    monkeypatch.setattr(P, "run_pipeline", lambda *a, **k: frame)
    assert _run(world) is frame
