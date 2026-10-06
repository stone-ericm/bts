"""The serving witness (C1 rank 4a, X-E1 prerequisite): what recipe, model artifact and inputs produced a slate."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

try:                                    # bts.model.predict imports lightgbm (an optional extra; Pi5 has none)
    import lightgbm  # noqa: F401
except (ImportError, OSError):
    pytest.skip("lightgbm unavailable", allow_module_level=True)

from bts import serving_witness as W
from bts.model.predict import _read_pa_parquets
from bts.orchestrator import predict_local


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fake_pkg(root: Path) -> Path:
    for rel, text in {"model/a.py": "A = 1\n", "model/sub/b.py": "B = 2\n", "features/f.py": "F = 3\n",
                      "data/d.py": "D = 4\n", "picks.py": "P = 5\n", "model/notes.txt": "not code\n"}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def test_recipe_hashes_every_py_file_of_the_recipe_packages_and_nothing_else(tmp_path):
    r = W.recipe(root=fake_pkg(tmp_path))
    assert r["files"] == {"data/d.py": sha(b"D = 4\n"), "features/f.py": sha(b"F = 3\n"), "model/a.py": sha(b"A = 1\n"),
                          "model/sub/b.py": sha(b"B = 2\n")}
    assert set(r["functions"]) == set(W.RECIPE_FUNCTIONS)
    assert all(len(v) == 64 for v in r["functions"].values())
    body = json.dumps({"files": r["files"], "functions": r["functions"]}, sort_keys=True, separators=(",", ":"))
    assert r["sha256"] == sha(body.encode())


def test_any_recipe_file_edit_changes_the_fingerprint(tmp_path):
    root = fake_pkg(tmp_path)
    before = W.recipe(root=root)["sha256"]
    (root / "model" / "sub" / "b.py").write_text("B = 3\n")
    assert W.recipe(root=root)["sha256"] != before
    (root / "model" / "sub" / "b.py").write_text("B = 2\n")
    assert W.recipe(root=root)["sha256"] == before
    (root / "picks.py").write_text("P = 6\n")                     # outside the recipe packages: not a recipe file
    assert W.recipe(root=root)["sha256"] == before


def test_a_serving_function_edit_changes_the_fingerprint(tmp_path, monkeypatch):
    root = fake_pkg(tmp_path)
    before = W.recipe(root=root)
    monkeypatch.setattr(W, "_function_source", lambda spec: "changed" if spec == W.RECIPE_FUNCTIONS[0] else spec)
    after = W.recipe(root=root)
    assert after["functions"][W.RECIPE_FUNCTIONS[0]] != before["functions"][W.RECIPE_FUNCTIONS[0]]
    assert after["sha256"] != before["sha256"]


def test_the_real_recipe_covers_the_serving_code():
    r = W.recipe()
    assert "model/predict.py" in r["files"] and "features/compute.py" in r["files"] and "data/schema.py" in r["files"]
    assert set(r["functions"]) == {"bts.orchestrator:predict_local", "bts.picks:is_resume_date_game",
                                   "bts.util:is_regular_season_game"}
    assert all(v and len(v) == 64 for v in r["functions"].values())


def test_env_records_the_allowlisted_flags_and_only_the_names_of_others(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("BTS_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("BTS_LGBM_RANDOM_STATE", "7")
    monkeypatch.setenv("BTS_BLUESKY_PASSWORD", "hunter2")
    w = W.build(model=None, inputs=None, calibration=None)
    assert w["env"]["BTS_LGBM_RANDOM_STATE"] == "7"
    assert w["env"]["BTS_USE_CALIBRATION"] is None                  # unset is recorded as unset
    assert set(w["env"]) == set(W.RECIPE_ENV)
    assert w["env_names"] == ["BTS_BLUESKY_PASSWORD", "BTS_LGBM_RANDOM_STATE"]
    assert "hunter2" not in json.dumps(w)


def test_build_never_raises_and_names_what_failed(monkeypatch):
    def boom(**_):
        raise OSError("disk gone")
    monkeypatch.setattr(W, "recipe", boom)
    w = W.build(model={"source": "trained"}, inputs=[], calibration={"enabled": False, "applied": False})
    assert w["recipe"] is None and w["schema"] == W.SCHEMA
    assert any("recipe" in e and "disk gone" in e for e in w["errors"])
    assert w["packages"]["python"] and w["packages"]["lightgbm"]
    json.dumps(w)


def test_read_pa_parquets_hashes_the_bytes_it_parses(tmp_path):
    pd.DataFrame({"x": [1, 2]}).to_parquet(tmp_path / "pa_2026.parquet")
    pd.DataFrame({"x": [3]}).to_parquet(tmp_path / "pa_2027.parquet")
    (tmp_path / "other.parquet").write_bytes(b"ignored")
    dfs, inputs = _read_pa_parquets(tmp_path)
    assert [len(d) for d in dfs] == [2, 1]
    assert inputs == [{"file": f"pa_{y}.parquet", "bytes": (tmp_path / f"pa_{y}.parquet").stat().st_size,
                       "sha256": sha((tmp_path / f"pa_{y}.parquet").read_bytes())} for y in (2026, 2027)]


def _frame(inputs):
    df = pd.DataFrame([{"batter_id": 1, "p_game_hit": 0.8}])
    df.attrs["serving_inputs"] = inputs
    return df


def test_predict_local_witnesses_a_cached_blend_by_the_bytes_it_loaded(tmp_path, monkeypatch):
    monkeypatch.delenv("BTS_USE_CALIBRATION", raising=False)
    raw = __import__("pickle").dumps({"_model": "m", "k": 1})
    (tmp_path / "blend_2027-04-01.pkl").write_bytes(raw)
    seen = {}

    def fake_run(date, data_dir, cached_blend=None, save_blend_path=None):
        seen.update(cached=cached_blend, save=save_blend_path)
        return _frame([{"file": "pa_2027.parquet", "bytes": 1, "sha256": "a" * 64}])
    with patch("bts.model.predict.run_pipeline", side_effect=fake_run):
        out = predict_local(date="2027-04-01", models_dir=str(tmp_path))
    assert seen == {"cached": {"_model": "m", "k": 1}, "save": None}
    w = out.attrs["serving"]
    assert w["model"] == {"source": "cache", "file": "blend_2027-04-01.pkl", "sha256": sha(raw)}
    assert w["inputs"] == [{"file": "pa_2027.parquet", "bytes": 1, "sha256": "a" * 64}]
    assert "serving_inputs" not in out.attrs
    assert w["calibration"] == {"enabled": False, "applied": False}
    assert w["tier_type"] == "local"


def test_predict_local_witnesses_a_trained_blend_by_its_saved_bytes(tmp_path):
    def fake_run(date, data_dir, cached_blend=None, save_blend_path=None):
        Path(save_blend_path).write_bytes(b"trained-bytes")
        return _frame([])
    with patch("bts.model.predict.run_pipeline", side_effect=fake_run):
        out = predict_local(date="2027-04-01", models_dir=str(tmp_path))
    assert out.attrs["serving"]["model"] == {"source": "trained", "file": "blend_2027-04-01.pkl",
                                             "sha256": sha(b"trained-bytes")}


def test_predict_local_with_an_unsaved_trained_blend_records_no_hash(tmp_path):
    with patch("bts.model.predict.run_pipeline", return_value=_frame(None)):
        out = predict_local(date="2027-04-01", models_dir=str(tmp_path))
    assert out.attrs["serving"]["model"]["sha256"] is None
    assert out.attrs["serving"]["inputs"] is None


def test_predict_local_records_an_applied_calibration(tmp_path, monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    data = tmp_path / "processed"
    data.mkdir()
    pd.DataFrame({"x": [1]}).to_parquet(data / "pa_2027.parquet")
    with patch("bts.model.predict.run_pipeline", return_value=_frame([])), \
         patch("bts.model.calibrate.fit_calibrator_from_picks", return_value=object()), \
         patch("bts.model.calibrate.apply_calibrator_series", side_effect=lambda s, c: s * 0 + 0.5):
        out = predict_local(date="2027-04-01", data_dir=str(data), models_dir=str(tmp_path))
    assert out["p_game_hit"].tolist() == [0.5]
    assert out.attrs["serving"]["calibration"] == {"enabled": True, "applied": True}


def test_predict_local_still_returns_predictions_when_the_witness_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "build", lambda **_: (_ for _ in ()).throw(RuntimeError("witness bug")))
    with patch("bts.model.predict.run_pipeline", return_value=_frame([])):
        out = predict_local(date="2027-04-01", models_dir=str(tmp_path))
    assert out is not None and out["p_game_hit"].tolist() == [0.8]
    assert "serving" not in out.attrs
