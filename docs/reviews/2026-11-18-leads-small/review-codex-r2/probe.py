"""Exercise archived cache builders and the prediction CLI with synthetic feeds."""

from contextlib import redirect_stderr, redirect_stdout
import errno
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest.mock import patch


REVIEW = Path(__file__).resolve().parent
SOURCE = Path("src/bts/features/compute.py")
CACHE_NAME = "probable_pitcher_lookup.json"


def load(name):
    path = REVIEW / "sources" / name / SOURCE
    spec = importlib.util.spec_from_file_location(f"{name}_compute", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entry(away, home):
    return {"away": away, "home": home, "away_tid": 51, "home_tid": 52}


def feed(raw, pk, away, home, season="2025"):
    directory = raw / season
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{pk}.json").write_text(json.dumps({"gameData": {
        "teams": {"away": {"id": 51}, "home": {"id": 52}},
        "probablePitchers": {"away": {"id": away}, "home": {"id": home}},
    }}))


def setup(root):
    cache = root / "models" / CACHE_NAME
    cache.parent.mkdir(parents=True)
    return cache, root / "raw"


def invoke(module, cache, raw):
    out, err = StringIO(), StringIO()
    with patch.object(module, "_probable_pitcher_cache_path", cache):
        with redirect_stdout(out), redirect_stderr(err):
            result = module._build_probable_pitcher_lookup(str(raw))
    return result, out.getvalue(), err.getvalue()


def bytes_for(lookup):
    return json.dumps(lookup).encode()


def cache_state(cache):
    stat = cache.stat()
    return cache.read_bytes(), stat.st_ino, stat.st_mtime_ns


def parity_sequence(module, root):
    cache, raw = setup(root)
    feed(raw, 100, 21, 22, "2024")
    feed(raw, 101, 23, None)
    (raw / "2025" / "999.json").write_text("\n{")
    (raw / "2025" / "notes.json").write_text("{}")
    expected = {100: entry(21, 22), 101: entry(23, None)}
    stages = {}

    def record(stage, raw_path, expected_lookup):
        result, stdout, stderr = invoke(module, cache, raw_path)
        assert result == expected_lookup, stage
        assert stdout == stderr == "", stage
        published = cache.read_bytes() if cache.exists() else None
        assert published == (bytes_for(expected_lookup) if expected_lookup else None), stage
        stages[stage] = result, published

    record("first", raw, expected)
    (raw / "2024" / "100.json").unlink()
    feed(raw, 101, 128, 129)
    feed(raw, 102, 24, 25, "2026")
    expected = expected | {102: entry(24, 25)}
    record("incremental", raw, expected)
    state = cache_state(cache)
    record("nothing new", raw, expected)
    assert cache_state(cache) == state
    record("no raw directory, cache present", root / "absent", expected)
    assert cache_state(cache) == state
    cache.unlink()
    record("no raw directory, cache absent", root / "absent", {})
    return stages


def exception_for(error_type):
    if error_type is OSError:
        return OSError(errno.ENOSPC, "injected write failure")
    return error_type("injected interruption")


def fault_probe(module, root, stage, error_type, *, direct_write=False, fresh=False):
    cache, raw = setup(root)
    old_lookup = {100: entry(21, 22)}
    old_bytes = None if fresh else bytes_for(old_lookup)
    new_bytes = bytes_for(({} if fresh else old_lookup) | {101: entry(23, 24)})
    if not fresh:
        feed(raw, 100, 21, 22)
        assert invoke(module, cache, raw) == (old_lookup, "", "")
        assert cache.read_bytes() == old_bytes
    feed(raw, 101, 23, 24)
    temp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
    original_write = Path.write_text
    original_replace = os.replace
    calls = []
    error = exception_for(error_type)

    def published():
        return cache.read_bytes() if cache.exists() else None

    def torn_write(path, content, *args, **kwargs):
        assert path == (cache if direct_write else temp)
        assert content.encode() == new_bytes
        calls.append("write")
        original_write(path, content[:len(content) // 2], *args, **kwargs)
        if not direct_write:
            assert published() == old_bytes
        raise error

    def replace_then_raise(src, dst):
        assert (src, dst) == (temp, cache)
        assert src.read_bytes() == new_bytes
        assert published() == old_bytes
        calls.append("replace")
        if stage == "after-replace":
            original_replace(src, dst)
            assert published() == new_bytes
        raise error

    if stage == "write":
        fault = patch.object(Path, "write_text", torn_write)
    else:
        fault = patch.object(os, "replace", replace_then_raise)
    with fault:
        try:
            invoke(module, cache, raw)
        except error_type as caught:
            assert caught is error
        else:
            raise AssertionError("fault did not propagate")

    assert calls == ["write" if stage == "write" else "replace"]
    if direct_write:
        result = published()
        assert result == new_bytes[:len(new_bytes) // 2]
        try:
            json.loads(result)
        except json.JSONDecodeError:
            pass
        else:
            raise AssertionError("base unexpectedly left valid JSON")
    else:
        assert published() == (new_bytes if stage == "after-replace" else old_bytes)
    assert set(cache.parent.iterdir()) == ({cache} if cache.exists() else set())


def bad_cache_probe(module, root, *, expect_warning):
    for label, broken, exception_name in (
        ("truncated", b'{"100": {"away": 2', "JSONDecodeError"),
        ("invalid-utf8", b'{"100": "\xff"}', "UnicodeDecodeError"),
        ("array", b'[{"100": 1}]', "AttributeError"),
    ):
        cache, raw = setup(root / label)
        feed(raw, 101, 23, 24)
        cache.write_bytes(broken)
        before = cache_state(cache)
        result, stdout, stderr = invoke(module, cache, raw)
        expected = {101: entry(23, 24)}
        assert result == expected and stdout == ""
        if expect_warning:
            assert stderr.splitlines() == [
                f"WARNING: unreadable probable-pitcher cache {cache} "
                f"({exception_name}); rebuilding it from raw feeds"
            ]
        else:
            assert stderr == ""
        after = cache_state(cache)
        assert after[0] == bytes_for(expected) and before[0] != after[0]
        assert invoke(module, cache, raw) == (expected, "", "")
        assert cache_state(cache) == after
        print(f"bad cache ({'head' if expect_warning else 'base'}): {label} rebuilt")

    cache, raw = setup(root / "no-usable-raw")
    cache.write_text("{")
    before = cache_state(cache)
    result, stdout, stderr = invoke(module, cache, raw)
    assert result == {} and stdout == ""
    assert len(stderr.splitlines()) == int(expect_warning)
    assert cache_state(cache) == before
    raw.mkdir()
    (raw / "2025").mkdir()
    (raw / "2025" / "100.json").write_text("{")
    result, stdout, stderr = invoke(module, cache, raw)
    assert result == {} and stdout == ""
    assert len(stderr.splitlines()) == int(expect_warning)
    assert cache_state(cache) == before


def prediction_probe(root):
    import pandas as pd
    from click.testing import CliRunner

    from bts.cli import cli
    from bts.features import compute
    from bts.model import predict
    from bts import orchestrator

    cache, raw = setup(root)
    feed(raw, 101, 23, 24)
    cache.write_text("{")
    pick = {"batter_id": 503, "p_game_hit": 0.73}

    def run_pipeline(*args, **kwargs):
        assert compute._build_probable_pitcher_lookup(str(raw)) == {101: entry(23, 24)}
        return pd.DataFrame([pick])

    assert Path(compute.__file__).resolve() == REVIEW.parent.parent / SOURCE
    with patch.object(compute, "_probable_pitcher_cache_path", cache):
        with patch.object(predict, "run_pipeline", run_pipeline):
            result = CliRunner().invoke(cli, ["predict-json", "--date", "2025-06-15",
                                              "--models-dir", str(cache.parent)])
    assert result.exit_code == 0, result.exception
    assert json.loads(result.stdout) == [pick]
    assert sum(line.startswith("WARNING: unreadable probable-pitcher cache ")
               for line in result.stderr.splitlines()) == 1
    completed = CompletedProcess(["ssh"], result.exit_code, result.stdout, result.stderr)
    with patch.object(orchestrator.subprocess, "run", return_value=completed):
        output = orchestrator.ssh_predict("offline-fixture", "/unused", "2025-06-15")
    assert output.to_dict(orient="records") == [pick]
    print("predict-json emitted valid stdout JSON and SSH orchestration returned the row")


def main():
    base, head = load("base"), load("head")
    with TemporaryDirectory(prefix="probe-", dir=REVIEW) as name:
        root = Path(name)
        base_stages = parity_sequence(base, root / "base-parity")
        head_stages = parity_sequence(head, root / "head-parity")
        assert head_stages == base_stages
        for stage, (lookup, cache) in head_stages.items():
            cache_summary = f"{len(cache)} bytes" if cache is not None else "no file"
            print(f"parity: {stage}, {len(lookup)} entries, {cache_summary}")

        for error in (OSError, KeyboardInterrupt, SystemExit):
            for stage in ("write", "before-replace", "after-replace"):
                for fresh in (False, True):
                    fault_probe(head, root / f"head-{error.__name__}-{stage}-{fresh}",
                                stage, error, fresh=fresh)
                print(f"head fault: {error.__name__} at {stage}, existing and fresh cache")
            for fresh in (False, True):
                fault_probe(base, root / f"base-{error.__name__}-{fresh}", "write",
                            error, direct_write=True, fresh=fresh)
            print(f"base fault: {error.__name__} during direct write left truncated JSON")

        bad_cache_probe(head, root / "head-bad", expect_warning=True)
        bad_cache_probe(base, root / "base-bad", expect_warning=False)
        print("corrupt cache with missing or unusable raw feeds remained on disk")
        prediction_probe(root / "predict-json")


if __name__ == "__main__":
    main()
