"""Compare the archived and current production cache code using local files only."""

import contextlib
import errno
import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL_RAW = Path("/Users/eric/projects/bts/data/raw/2025")
NAME = "probable_pitcher_lookup.json"
ENTRY = {"away": 21, "home": 22, "away_tid": 51, "home_tid": 52}
REAL_IDS = (776135, 776136, 776137, 776138, 776139, 776140)


def module(version):
    src = ROOT / "src" if version == "head" else HERE / "base" / "src"
    spec = importlib.util.spec_from_file_location(f"compute_{version}", src / "bts/features/compute.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def feed(root, number, away=21):
    season = root / "2025"
    season.mkdir(parents=True, exist_ok=True)
    (season / f"{number}.json").write_text(json.dumps({"gameData": {
        "teams": {"away": {"id": 51}, "home": {"id": 52}},
        "probablePitchers": {"away": {"id": away}, "home": {"id": 22}},
    }}))


def setup(root, seed=True):
    cache = root / "models" / NAME
    raw = root / "raw"
    cache.parent.mkdir(parents=True)
    feed(raw, 101)
    if seed:
        cache.write_text(json.dumps({"100": ENTRY}))
    return cache, raw


def build(compute, cache, raw):
    with patch.object(compute, "_probable_pitcher_cache_path", cache):
        return compute._build_probable_pitcher_lookup(str(raw))


def compare_bytes(versions, scratch, copied_feeds, kind, seed):
    stages = {}
    for version, compute in versions.items():
        root = scratch / f"{kind}-{seed}-{version}"
        cache = root / "models" / NAME
        raw = root / "raw"
        cache.parent.mkdir(parents=True)
        if seed:
            seed_data = json.loads((HERE / "fixtures" / NAME).read_bytes())
            for number in REAL_IDS:
                assert seed_data.pop(str(number), None) is not None
            cache.write_text(json.dumps(seed_data))

        def add(numbers):
            if kind == "synthetic":
                for number in numbers:
                    feed(raw, number)
            else:
                season = raw / "2025"
                season.mkdir(parents=True, exist_ok=True)
                for number in numbers:
                    shutil.copyfile(copied_feeds / f"{number}.json", season / f"{number}.json")

        add(REAL_IDS[:3])
        initial = (build(compute, cache, raw), cache.read_bytes())
        add(REAL_IDS[3:])
        if kind == "synthetic":
            feed(raw, REAL_IDS[0], away=9000)
            (raw / "2025" / "101_weather.json").write_text("not a feed")
            (raw / "2025" / "102.json").write_text("{torn")
        update = (build(compute, cache, raw), cache.read_bytes())
        before = cache.stat().st_mtime_ns
        noop = (build(compute, cache, raw), cache.read_bytes())
        absent = (build(compute, cache, root / "missing"), cache.read_bytes())
        assert update == noop == absent and cache.stat().st_mtime_ns == before
        if kind == "synthetic":
            assert update[0][REAL_IDS[0]]["away"] == 21
        if seed:
            assert set(update[0]) - set(REAL_IDS) == {int(key) for key in seed_data}
            assert build(compute, HERE / "fixtures" / NAME, root / "missing") == {
                int(key): value for key, value in json.loads((HERE / "fixtures" / NAME).read_bytes()).items()
            }
        unset = root / "uncreated" / NAME
        assert build(compute, unset, root / "missing") == {} and not unset.exists()
        stages[version] = (initial, update, noop, absent)
    assert stages["base"] == stages["head"]
    print("parity", kind, f"seed={seed}", [(len(data), hashlib.sha256(content).hexdigest()[:12])
                                           for data, content in stages["head"]])


def fault_matrix(versions, scratch):
    real_write, real_replace = Path.write_text, os.replace
    for version, compute in versions.items():
        for stage in ("before_write", "partial_write", "before_replace", "after_replace"):
            for error in (OSError(errno.ENOSPC, "injected full disk"), KeyboardInterrupt(), SystemExit(3)):
                cache, raw = setup(scratch / f"{version}-{stage}-{type(error).__name__}")
                old = cache.read_bytes()
                new = json.dumps({"100": ENTRY, "101": ENTRY}).encode()
                during = []

                def write(path, data, *args, **kwargs):
                    if stage == "before_write":
                        raise error
                    if stage == "partial_write":
                        real_write(path, data[: len(data) // 2], *args, **kwargs)
                        during.append(cache.read_bytes())
                        raise error
                    return real_write(path, data, *args, **kwargs)

                def replace(src, dst):
                    assert Path(src).read_bytes() == new and cache.read_bytes() == old
                    if stage == "before_replace":
                        raise error
                    real_replace(src, dst)
                    if stage == "after_replace":
                        raise error

                raised = False
                with patch.object(Path, "write_text", write), patch.object(os, "replace", replace):
                    try:
                        build(compute, cache, raw)
                    except (OSError, KeyboardInterrupt, SystemExit) as caught:
                        assert caught is error
                        raised = True
                expected = {
                    "before_write": old,
                    "partial_write": old if version == "head" else new[: len(new) // 2],
                    "before_replace": old if version == "head" else new,
                    "after_replace": new,
                }[stage]
                assert cache.read_bytes() == expected
                assert raised == (version == "head" or stage in ("before_write", "partial_write"))
                assert list(cache.parent.iterdir()) == [cache]
                if during:
                    assert during == [expected]
                state = "old" if expected == old else "new" if expected == new else "partial"
                print("fault", version, stage, type(error).__name__, f"raised={raised}", state, "temps=0")


def process_writer(version, cache, raw, gate):
    compute = module(version)
    real_write = Path.write_text

    def split_write(path, data, *args, **kwargs):
        real_write(path, data[: len(data) // 2], *args, **kwargs)
        print(path, flush=True)
        assert sys.stdin.readline() == f"{gate}\n"
        return real_write(path, data, *args, **kwargs)

    with patch.object(Path, "write_text", split_write):
        assert build(compute, cache, raw)


@contextlib.contextmanager
def writer(version, cache, raw):
    command = [sys.executable, str(Path(__file__).resolve()), "worker", version, str(cache), str(raw)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as proc:
        try:
            yield proc
        finally:
            if proc.poll() is None:
                proc.kill()


def process_cases(versions, scratch):
    for version, compute in versions.items():
        cache, raw = setup(scratch / f"{version}-kill")
        with writer(version, cache, raw) as child:
            stage_path = Path(child.stdout.readline().strip())
            assert stage_path.is_file(), child.stderr.read() if child.poll() is not None else stage_path
            child.kill()
            child.wait(timeout=10)
        files = sorted(path.name for path in cache.parent.iterdir())
        if version == "head":
            assert json.loads(cache.read_bytes()) == {"100": ENTRY}
            assert files == [f".{NAME}.{child.pid}.tmp", NAME]
        else:
            assert files == [NAME]
            try:
                json.loads(cache.read_bytes())
            except json.JSONDecodeError:
                pass
            else:
                raise AssertionError("expected truncated cache")
        print("SIGKILL", version, "files", files)

        cache, raw = setup(scratch / f"{version}-race")
        other_raw = raw.with_name("other")
        feed(other_raw, 102)
        with writer(version, cache, raw) as first:
            first_path = Path(first.stdout.readline().strip())
            assert first_path.is_file()
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                read = build(compute, cache, raw.with_name("missing"))
            with writer(version, cache, other_raw) as second:
                second_path = Path(second.stdout.readline().strip())
                assert second_path.is_file()
                for child in (first, second):
                    child.communicate("go\n", timeout=10)
                    assert child.returncode == 0
                    intermediate = json.loads(cache.read_bytes())
                    print("process race", version, "published", sorted(intermediate), "reader", sorted(read))
        keys = sorted(json.loads(cache.read_bytes()))
        if version == "head":
            assert first_path != second_path and keys == ["100", "102"]
            assert sorted(read) == [100] and not stderr.getvalue()
        else:
            assert first_path == second_path == cache and keys == ["102"]
            assert read == {} and not stderr.getvalue()


def same_pid_cases(compute, scratch):
    cache, raw = setup(scratch / "threads")
    other_raw = raw.with_name("other")
    feed(other_raw, 102)
    real_write = Path.write_text
    first_wrote, second_wrote, first_published = Event(), Event(), Event()

    def write(path, data, *args, **kwargs):
        if '"101"' in data:
            real_write(path, data, *args, **kwargs)
            first_wrote.set()
            assert second_wrote.wait(10)
        else:
            real_write(path, data, *args, **kwargs)
            second_wrote.set()
            assert first_published.wait(10)

    with (patch.object(Path, "write_text", write),
          patch.object(compute, "_probable_pitcher_cache_path", cache),
          ThreadPoolExecutor(max_workers=2) as pool):
        first = pool.submit(compute._build_probable_pitcher_lookup, str(raw))
        assert first_wrote.wait(10)
        second = pool.submit(compute._build_probable_pitcher_lookup, str(other_raw))
        assert sorted(first.result(timeout=10)) == [100, 101]
        first_published.set()
        try:
            second.result(timeout=10)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected PID-named temp collision")
    assert sorted(json.loads(cache.read_bytes())) == ["100", "102"]
    print("same PID threads: first returned id 101, published 102; second raised FileNotFoundError")

    cache, raw = setup(scratch / "stale")
    stale = cache.with_name(f".{NAME}.{os.getpid()}.tmp")
    stale.write_text("partial abandoned by an old process with this PID")
    assert sorted(build(compute, cache, raw)) == [100, 101] and not stale.exists()
    print("reused PID with stale regular temp: overwritten and removed")


def filesystem_cases(versions, scratch):
    from bts.data import sync

    class FakeClient:
        def __init__(self):
            self.uploads = {}

        def upload_file(self, path, key):
            self.uploads[key] = path.read_bytes()

    for version, compute in versions.items():
        cache, raw = setup(scratch / f"{version}-readonly")
        cache.parent.chmod(0o555)
        try:
            try:
                result = sorted(build(compute, cache, raw))
            except PermissionError:
                result = "PermissionError"
            after = sorted(json.loads(cache.read_bytes()))
            assert (result, after) == (([100, 101], ["100", "101"]) if version == "base"
                                       else ("PermissionError", ["100"]))
            assert list(cache.parent.iterdir()) == [cache]
        finally:
            cache.parent.chmod(0o755)
        print("read-only models directory, writable existing file", version, result, after)

        cache, raw = setup(scratch / f"{version}-mode-link")
        target = cache.parent.parent / "shared.json"
        cache.rename(target)
        cache.symlink_to(target)
        other_link = target.with_name("other.json")
        other_link.symlink_to(target)
        target.chmod(0o600)
        previous_mask = os.umask(0o022)
        try:
            assert sorted(build(compute, cache, raw)) == [100, 101]
        finally:
            os.umask(previous_mask)
        mode = stat.S_IMODE(cache.stat().st_mode)
        target_ids = sorted(json.loads(other_link.read_bytes()))
        if version == "head":
            assert not cache.is_symlink() and mode == 0o644 and target_ids == ["100"]
        else:
            assert cache.is_symlink() and mode == 0o600 and target_ids == ["100", "101"]
        client = FakeClient()
        with (patch.object(sync, "read_manifest", return_value=None),
              patch.object(sync, "write_manifest_atomic")):
            manifest = sync.sync_to_r2(client, cache.parent / "missing", cache.parent)
        manifest_entry = manifest["files"][f"models/{NAME}"]
        assert client.uploads[manifest_entry["key"]] == cache.read_bytes()
        print("symlink", version, f"is_link={cache.is_symlink()}", oct(mode), "target", target_ids,
              "synced", sorted(json.loads(client.uploads[manifest_entry["key"]])))

        cache, raw = setup(scratch / f"{version}-mode")
        cache.chmod(0o644)
        previous_mask = os.umask(0o077)
        try:
            assert sorted(build(compute, cache, raw)) == [100, 101]
        finally:
            os.umask(previous_mask)
        mode = stat.S_IMODE(cache.stat().st_mode)
        assert mode == (0o600 if version == "head" else 0o644)
        print("regular cache 0644, umask 077", version, oct(mode))


def warning_cases(versions, scratch):
    for version, compute in versions.items():
        for label, content in {
            "torn": b'{"100":', "invalid_utf8": b'\xff', "array": b'[]', "null": b'null',
            "non_numeric": b'{"100": {}, "bad": {}}', "non_object_value": b'{"100": 3}',
            "read_error": b'{"100": {}}',
        }.items():
            cache, raw = setup(scratch / f"{version}-malformed-{label}")
            cache.write_bytes(content)
            stdout, stderr = io.StringIO(), io.StringIO()
            real_read = Path.read_text

            def read(path, *args, **kwargs):
                if path == cache:
                    raise PermissionError(errno.EACCES, "injected unreadable cache")
                return real_read(path, *args, **kwargs)

            simulated_read_error = patch.object(Path, "read_text", read) if label == "read_error" else contextlib.nullcontext()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), simulated_read_error:
                missing_results = [build(compute, cache, raw.with_name("missing")) for _ in range(2)]
                result = build(compute, cache, raw)
            expected = {100: 3} if label == "non_object_value" else {}
            assert missing_results == [expected, expected] and result == expected | {101: ENTRY}
            assert stdout.getvalue() == ""
            warning = stderr.getvalue().splitlines()
            assert len(warning) == (3 if version == "head" and not expected else 0)
            assert all(line.startswith(f"WARNING: unreadable probable-pitcher cache {cache}") for line in warning)
            after_rebuild = io.StringIO()
            with contextlib.redirect_stderr(after_rebuild):
                assert build(compute, cache, raw) == result
            assert not after_rebuild.getvalue()
            print("reads", version, label, f"warnings={len(warning)}", "after", sorted(result))


def main():
    versions = {version: module(version) for version in ("base", "head")}
    with TemporaryDirectory(prefix="cache-probe-", dir=HERE) as directory:
        scratch = Path(directory)
        copied_feeds = scratch / "original-feeds"
        copied_feeds.mkdir()
        for number in REAL_IDS:
            shutil.copyfile(ORIGINAL_RAW / f"{number}.json", copied_feeds / f"{number}.json")
        for kind, seed in (("synthetic", False), ("real", False), ("real", True)):
            compare_bytes(versions, scratch, copied_feeds, kind, seed)
        fault_matrix(versions, scratch)
        process_cases(versions, scratch)
        same_pid_cases(versions["head"], scratch)
        filesystem_cases(versions, scratch)
        warning_cases(versions, scratch)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _, _, version, cache, raw = sys.argv
        process_writer(version, Path(cache), Path(raw), "go")
    else:
        main()
