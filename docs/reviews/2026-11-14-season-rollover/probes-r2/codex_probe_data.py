"""S3-S5 real build CLI, base parity matrix and concurrent/interrupted writes."""
import copy
import ast
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pytest
from click.testing import CliRunner
import bts.data.build as build
import bts.data.pull as pull
import bts.features.compute as compute
import bts.model.predict as predict
from bts.cli import cli
from tests.conftest import sample_game_feed

HERE = Path(__file__).resolve().parent
base = types.ModuleType("bts.data._review_base_build")
sys.modules[base.__name__] = base
exec(compile(subprocess.check_output(["git", "show", "12dbaaa:src/bts/data/build.py"], text=True),
             "12dbaaa:src/bts/data/build.py", "exec"), base.__dict__)
out = {"build_matrix": []}

# Evaluate the exact base function against the same numeric cache and sidecar.
tree = ast.parse(subprocess.check_output(["git", "show", "12dbaaa:src/bts/features/compute.py"], text=True))
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_probable_pitcher_lookup")
namespace = dict(compute.__dict__)
exec(compile(ast.Module(body=[node], type_ignores=[]), "12dbaaa:probable_lookup", "exec"), namespace)
with tempfile.TemporaryDirectory(dir=HERE) as directory, pytest.MonkeyPatch.context() as mp:
    root = Path(directory)
    (root / "raw" / "2027").mkdir(parents=True)
    feed = sample_game_feed.__wrapped__()
    (root / "raw" / "2027" / "9001.json").write_text(json.dumps(feed))
    namespace["_probable_pitcher_cache_path"] = root / "base_cache.json"
    mp.setattr(compute, "_probable_pitcher_cache_path", root / "head_cache.json")
    before = namespace["_build_probable_pitcher_lookup"](str(root / "raw"))
    after = compute._build_probable_pitcher_lookup(str(root / "raw"))
    assert before == after
    assert (root / "base_cache.json").read_bytes() == (root / "head_cache.json").read_bytes()
    (root / "raw" / "2027" / "9001_weather.json").write_text("not JSON")
    try:
        namespace["_build_probable_pitcher_lookup"](str(root / "raw"))
        raise AssertionError("base sidecar did not fail")
    except ValueError:
        pass
    assert compute._build_probable_pitcher_lookup(str(root / "raw")) == after
    out["probable_pitcher_cache"] = "base bytes identical for numeric stems; sidecar crashes base, skipped at HEAD"

def outcome(fn):
    try:
        result = fn()
        return "noop" if result is None else f"rows:{len(result)}"
    except Exception as exc:
        return type(exc).__name__

for kind in ("missing", "empty", "spring", "weather", "untyped", "corrupt", "regular", "file"):
    for existing in (False, True):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            raw, target = root / "raw", root / "processed" / "pa_2027.parquet"
            season = raw / "2027"
            if kind == "file":
                raw.mkdir()
                season.write_text("not a directory")
            elif kind != "missing":
                season.mkdir(parents=True)
            if kind in ("spring", "regular"):
                feed = copy.deepcopy(sample_game_feed.__wrapped__())
                feed["gameData"]["game"]["type"] = "S" if kind == "spring" else "R"
                (season / "9001.json").write_text(json.dumps(feed))
            elif kind == "weather":
                (season / "9001_weather.json").write_text("not a feed")
            elif kind == "untyped":
                (season / "9001.json").write_text("{}")
            elif kind == "corrupt":
                (season / "9001.json").write_text("{")
            if existing:
                target.parent.mkdir()
                target.write_bytes(b"existing parquet sentinel")
            before = outcome(lambda: base.build_season(raw, target, 2027))
            # Restore target shape so both implementations get identical input.
            if existing:
                target.write_bytes(b"existing parquet sentinel")
            elif target.exists():
                target.unlink()
            after = outcome(lambda: build.build_season(raw, target, 2027))
            result = CliRunner().invoke(cli, ["data", "build", "--seasons", "2027",
                                            "--raw-dir", str(raw), "--out-dir", str(target.parent)])
            if kind in ("missing", "empty", "spring", "weather"):
                assert before == "noop" and after == ("ValueError" if existing else "noop")
                assert result.exit_code == (1 if existing else 0)
            elif kind == "file":
                assert after == "NotADirectoryError" and result.exit_code == 1
            else:
                assert before == after
            out["build_matrix"].append({"kind": kind, "existing": existing, "base": before,
                                         "head": after, "cli_exit": result.exit_code})

with tempfile.TemporaryDirectory(dir=HERE) as directory, pytest.MonkeyPatch.context() as mp:
    root = Path(directory)
    barrier = threading.Barrier(2)
    bodies = [b'{"revision": 1}', b'{"revision": 2}']
    local = threading.local()
    def network(*args, **kwargs):
        barrier.wait(timeout=10)
        return io.BytesIO(bodies[local.index])
    mp.setattr(pull, "urlopen", network)
    def download(index):
        local.index = index
        return pull.download_game_feed(9001, root)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(download, (0, 1)))
    assert results == [root / "9001.json"] * 2
    assert (root / "9001.json").read_bytes() in bodies
    assert list(root.iterdir()) == [root / "9001.json"]
    out["concurrent_download"] = "two complete bodies, one complete target, no temporary residue"

# Inject interruption while the temporary file exists, observe both real scans.
with tempfile.TemporaryDirectory(dir=HERE) as directory, pytest.MonkeyPatch.context() as mp:
    root = Path(directory)
    season = root / "raw" / "2027"
    season.mkdir(parents=True)
    mp.setattr(pull, "urlopen", lambda *a, **k: io.BytesIO(b'{}'))
    mp.setattr(compute, "_probable_pitcher_cache_path", root / "cache.json")
    original = tempfile.NamedTemporaryFile
    observed = []
    def temporary(*args, **kwargs):
        result = original(*args, **kwargs)
        original_write = result.write
        def write(data):
            original_write(data[:1])
            result.flush()
            observed.append(Path(result.name).name)
            assert list(season.glob("*.json")) == []
            assert build.build_season(root / "raw", root / "pa_2027.parquet", 2027) is None
            assert compute._build_probable_pitcher_lookup(str(root / "raw")) == {}
            raise KeyboardInterrupt("synthetic interrupted writer")
        result.write = write
        return result
    mp.setattr(tempfile, "NamedTemporaryFile", temporary)
    try:
        pull.download_game_feed(9001, season)
        raise AssertionError("interruption missing")
    except KeyboardInterrupt:
        pass
    assert list(season.iterdir()) == [] and len(observed) == 1
    out["interrupted_download"] = {"temporary": observed, "cleanup": True, "scans_ignore_temporary": True}

# Existing broken feeds are intentionally kept. Refresh propagates new build errors.
with tempfile.TemporaryDirectory(dir=HERE) as directory, pytest.MonkeyPatch.context() as mp:
    root = Path(directory)
    raw, proc = root / "raw", root / "processed"
    (raw / "2027").mkdir(parents=True)
    proc.mkdir()
    target = proc / "pa_2027.parquet"
    target.write_bytes(b"existing parquet sentinel")
    mp.setattr(pull, "discover_games", lambda *a: [])
    out["prediction_refresh_empty_restore"] = outcome(lambda: predict._refresh_season_data(
        "2027-06-01", raw_dir=str(raw), processed_dir=str(proc)))
    assert out["prediction_refresh_empty_restore"] == "ValueError"
    assert target.read_bytes() == b"existing parquet sentinel"
    (proc / ".refreshed_2027_through_2027-05-31").write_text("marker")
    out["same_day_marker_restore_limit"] = outcome(lambda: predict._refresh_season_data(
        "2027-06-01", raw_dir=str(raw), processed_dir=str(proc)))
    assert out["same_day_marker_restore_limit"] == "noop"

(HERE / "data-results.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
