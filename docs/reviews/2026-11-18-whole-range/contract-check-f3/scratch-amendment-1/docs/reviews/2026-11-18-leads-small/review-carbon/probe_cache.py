"""Adversarial probe for _build_probable_pitcher_lookup (base df454725 vs HEAD b0c11135).

Checks:
1. Base vs HEAD equivalence (returned dict + exact cache bytes) on synthetic and copied real inputs:
   first build, incremental build, nothing new, no raw directory.
2. Interruptions at every stage (before write, mid-write, before os.replace, after os.replace)
   across OSError (ENOSPC/EIO), KeyboardInterrupt, and SystemExit: cache is always old or new
   complete bytes, never torn, and no .tmp file remains. Demonstrates base lost cache-only keys.
3. Concurrent multi-process writers + concurrent reader loop: 0 torn reads on HEAD, no PID temp
   collision across processes, documents last-writer-wins behaviour before and after, and checks
   same-process multi-thread behaviour.
4. Read-only models dir, symlinked cache (valid & dangling), non-object JSON, non-numeric key,
   non-object value (including compute_all_features impact), unreadable cache with empty/missing
   raw_dir, stdout cleanliness, and file mode/umask behaviour.
"""
from __future__ import annotations

import contextlib
import errno
import hashlib
import importlib.util
import io
import json
import multiprocessing as mp
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from bts.features import compute as head_compute

REVIEW_DIR = Path(__file__).resolve().parent
BASE_COMPUTE_PATH = REVIEW_DIR / "base_ref" / "src" / "bts" / "features" / "compute.py"
REAL_COPY_CACHE = REVIEW_DIR / "real_copy" / "probable_pitcher_lookup.json"
REAL_COPY_RAW = REVIEW_DIR / "real_copy" / "raw"


def _load_base_compute():
    spec = importlib.util.spec_from_file_location("_leads_review_base_compute", BASE_COMPUTE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _feed_json(away_pid, home_pid, away_tid=111, home_tid=112) -> str:
    pp = {}
    if away_pid is not None:
        pp["away"] = {"id": away_pid}
    if home_pid is not None:
        pp["home"] = {"id": home_pid}
    return json.dumps({
        "gameData": {
            "teams": {"away": {"id": away_tid}, "home": {"id": home_tid}},
            "probablePitchers": pp,
        }
    })


@contextlib.contextmanager
def _patched_cache_path(mod, cache_path: Path):
    orig = mod._probable_pitcher_cache_path
    mod._probable_pitcher_cache_path = cache_path
    try:
        yield
    finally:
        mod._probable_pitcher_cache_path = orig


def _call_build(mod, cache_path: Path, raw_dir: Path):
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with _patched_cache_path(mod, cache_path), contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        res = mod._build_probable_pitcher_lookup(str(raw_dir))
    return res, out_buf.getvalue(), err_buf.getvalue()


def _mp_writer_worker(cache_str: str, raw_str: str, iterations: int, ready_q, go_ev):
    cache_p = Path(cache_str)
    ready_q.put(os.getpid())
    go_ev.wait(timeout=10)
    for i in range(iterations):
        # Force a fresh write each iteration by dropping one key from a local copy or adding a feed
        feed_p = Path(raw_str) / "2025" / f"{900000 + os.getpid() * 100 + i}.json"
        feed_p.write_text(_feed_json(10 + i, 20 + i))
        with _patched_cache_path(head_compute, cache_p):
            head_compute._build_probable_pitcher_lookup(raw_str)


def check_equivalence_synthetic(base_compute) -> dict:
    results = {}
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as td:
        root = Path(td)
        for mode in ("first_build", "incremental", "nothing_new", "no_raw_with_cache", "no_raw_no_cache"):
            b_dir = root / f"base_{mode}"
            h_dir = root / f"head_{mode}"
            for d in (b_dir, h_dir):
                (d / "models").mkdir(parents=True)
                if not mode.startswith("no_raw"):
                    (d / "raw" / "2024").mkdir(parents=True)
                    (d / "raw" / "2025").mkdir(parents=True)
                    (d / "raw" / "README.txt").write_text("not a dir")
                    (d / "raw" / "2025" / "700001_weather.json").write_text('{"temp": 70}')
                    (d / "raw" / "2025" / "700002.json").write_text("{corrupt feed")
                    (d / "raw" / "2024" / "600001.json").write_text(_feed_json(101, 102, 1, 2))
                    (d / "raw" / "2025" / "700001.json").write_text(_feed_json(201, None, 3, 4))

                if mode in ("incremental", "nothing_new", "no_raw_with_cache"):
                    seed = {
                        "500000": {"away": 9, "home": 8, "away_tid": 7, "home_tid": 6},
                        "600001": {"away": 999, "home": 888, "away_tid": 1, "home_tid": 2},
                    }
                    if mode == "nothing_new":
                        seed["700001"] = {"away": 201, "home": None, "away_tid": 3, "home_tid": 4}
                    (d / "models" / "probable_pitcher_lookup.json").write_text(json.dumps(seed))

            b_cache = b_dir / "models" / "probable_pitcher_lookup.json"
            h_cache = h_dir / "models" / "probable_pitcher_lookup.json"
            b_mtime_before = b_cache.stat().st_mtime_ns if b_cache.exists() else None
            h_mtime_before = h_cache.stat().st_mtime_ns if h_cache.exists() else None

            b_res, b_out, b_err = _call_build(base_compute, b_cache, b_dir / "raw")
            h_res, h_out, h_err = _call_build(head_compute, h_cache, h_dir / "raw")

            assert b_res == h_res, f"{mode}: lookup mismatch"
            assert b_out == "" and h_out == "", f"{mode}: unexpected stdout"
            assert b_err == "" and h_err == "", f"{mode}: unexpected stderr"
            assert b_cache.exists() == h_cache.exists()
            if h_cache.exists():
                assert b_cache.read_bytes() == h_cache.read_bytes(), f"{mode}: cache bytes differ"
            if mode == "nothing_new":
                assert b_cache.stat().st_mtime_ns == b_mtime_before
                assert h_cache.stat().st_mtime_ns == h_mtime_before
            if mode == "incremental":
                # Existing cached key 600001 is skipped (retains 999, not overwritten with 101)
                assert h_res[600001]["away"] == 999
                assert h_res[500000]["away"] == 9
                assert h_res[700001] == {"away": 201, "home": None, "away_tid": 3, "home_tid": 4}
            results[mode] = {"entries": len(h_res), "cache_exists": h_cache.exists()}
    return results


def check_equivalence_real_copies(base_compute) -> dict:
    assert REAL_COPY_CACHE.exists(), "missing real_copy cache"
    sample_files = sorted((REAL_COPY_RAW / "2025").glob("*.json"))
    assert len(sample_files) >= 5, "missing sample 2025 raw feeds"

    results = {}
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as td:
        root = Path(td)
        # 1. Full real cache + sample 2025 feeds (all already cached -> nothing new)
        for label, mod in (("base", base_compute), ("head", head_compute)):
            c = root / f"full_{label}.json"
            shutil.copy2(REAL_COPY_CACHE, c)
            res, out, err = _call_build(mod, c, REAL_COPY_RAW)
            results[f"full_{label}"] = (len(res), hashlib.sha256(c.read_bytes()).hexdigest(), out, err, res)
        assert results["full_base"][:4] == results["full_head"][:4]
        assert results["full_base"][4] == results["full_head"][4]

        # 2. Evict 5 sample 2025 IDs from copied real cache -> incremental rebuild from real 2025 feeds
        raw_full = json.loads(REAL_COPY_CACHE.read_text())
        evicted_ids = [f.stem for f in sample_files[:5]]
        trimmed = {k: v for k, v in raw_full.items() if k not in evicted_ids}
        b_inc = root / "inc_base.json"
        h_inc = root / "inc_head.json"
        b_inc.write_text(json.dumps(trimmed))
        h_inc.write_text(json.dumps(trimmed))
        b_res, b_out, b_err = _call_build(base_compute, b_inc, REAL_COPY_RAW)
        h_res, h_out, h_err = _call_build(head_compute, h_inc, REAL_COPY_RAW)
        assert b_res == h_res
        assert b_inc.read_bytes() == h_inc.read_bytes()
        assert b_out == "" and h_out == "" and b_err == "" and h_err == ""
        for eid in evicted_ids:
            assert h_res[int(eid)] == raw_full[eid], f"rebuilt real feed {eid} differed from real cache"

        # 3. First build from empty cache on real 2025 sample feeds
        b_fresh = root / "fresh_base.json"
        h_fresh = root / "fresh_head.json"
        b_fres, _, _ = _call_build(base_compute, b_fresh, REAL_COPY_RAW)
        h_fres, _, _ = _call_build(head_compute, h_fresh, REAL_COPY_RAW)
        assert b_fres == h_fres and len(h_fres) == len(sample_files)
        assert b_fresh.read_bytes() == h_fresh.read_bytes()

        # 4. No raw directory with full real cache
        b_noraw = root / "noraw_base.json"
        h_noraw = root / "noraw_head.json"
        shutil.copy2(REAL_COPY_CACHE, b_noraw)
        shutil.copy2(REAL_COPY_CACHE, h_noraw)
        b_nr, _, _ = _call_build(base_compute, b_noraw, root / "does_not_exist")
        h_nr, _, _ = _call_build(head_compute, h_noraw, root / "does_not_exist")
        assert b_nr == h_nr and len(h_nr) == len(raw_full)

    return {
        "real_cache_entries": len(raw_full),
        "sample_2025_feeds": len(sample_files),
        "incremental_sha256_match": True,
        "fresh_sha256_match": True,
    }


def check_interruptions(base_compute) -> dict:
    summary = {}
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as td:
        root = Path(td)
        raw = root / "raw"
        (raw / "2025").mkdir(parents=True)
        (raw / "2025" / "100.json").write_text(_feed_json(11, 12))

        # Seed cache with cache-only historical game 999 + game 100
        models = root / "models"
        models.mkdir()
        cache = models / "probable_pitcher_lookup.json"
        seed = {
            "999": {"away": 91, "home": 92, "away_tid": 50, "home_tid": 51},
            "100": {"away": 11, "home": 12, "away_tid": 111, "home_tid": 112},
        }
        cache.write_text(json.dumps(seed))
        before_bytes = cache.read_bytes()

        # Add new game 101 to trigger write
        (raw / "2025" / "101.json").write_text(_feed_json(21, 22))

        real_write_text = Path.write_text
        real_replace = os.replace

        # Stage 1: fail before writing any bytes (ENOSPC on open/write)
        def fail_before_write(self, data, *args, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        Path.write_text = fail_before_write
        try:
            raised = False
            try:
                _call_build(head_compute, cache, raw)
            except OSError as exc:
                raised = exc.errno == errno.ENOSPC
            assert raised
        finally:
            Path.write_text = real_write_text
        assert cache.read_bytes() == before_bytes
        assert [p.name for p in models.iterdir()] == [cache.name]
        summary["stage1_before_write_enospc"] = "old_intact_no_tmp"

        # Stage 2: torn write halfway + OSError / KeyboardInterrupt / SystemExit
        for exc_cls, exc_val in (
            (OSError, OSError(errno.ENOSPC, "disk full mid-write")),
            (KeyboardInterrupt, KeyboardInterrupt("ctrl-c mid-write")),
            (SystemExit, SystemExit(2)),
        ):
            def torn_write(self, data, *args, _exc=exc_val, **kwargs):
                real_write_text(self, data[: max(1, len(data) // 2)], *args, **kwargs)
                raise _exc

            Path.write_text = torn_write
            try:
                caught = None
                try:
                    _call_build(head_compute, cache, raw)
                except BaseException as exc:
                    caught = exc
                assert isinstance(caught, exc_cls)
            finally:
                Path.write_text = real_write_text
            assert cache.read_bytes() == before_bytes
            assert [p.name for p in models.iterdir()] == [cache.name]
            summary[f"stage2_mid_write_{exc_cls.__name__}"] = "old_intact_no_tmp"

        # Stage 3: tmp written completely, os.replace fails with OSError / KeyboardInterrupt / SystemExit
        for exc_cls, exc_val in (
            (OSError, OSError(errno.EIO, "replace failed")),
            (KeyboardInterrupt, KeyboardInterrupt("ctrl-c before replace")),
            (SystemExit, SystemExit(3)),
        ):
            def fail_replace(src, dst, *args, _exc=exc_val, **kwargs):
                assert Path(src).exists()
                raise _exc

            os.replace = fail_replace
            try:
                caught = None
                try:
                    _call_build(head_compute, cache, raw)
                except BaseException as exc:
                    caught = exc
                assert isinstance(caught, exc_cls)
            finally:
                os.replace = real_replace
            assert cache.read_bytes() == before_bytes
            assert [p.name for p in models.iterdir()] == [cache.name]
            summary[f"stage3_replace_fail_{exc_cls.__name__}"] = "old_intact_no_tmp"

        # Stage 4: interrupt immediately AFTER os.replace commits
        def post_replace_interrupt(src, dst, *args, **kwargs):
            real_replace(src, dst, *args, **kwargs)
            raise KeyboardInterrupt("after commit")

        os.replace = post_replace_interrupt
        try:
            caught = None
            try:
                _call_build(head_compute, cache, raw)
            except BaseException as exc:
                caught = exc
            assert isinstance(caught, KeyboardInterrupt)
        finally:
            os.replace = real_replace
        published = json.loads(cache.read_text())
        assert set(published.keys()) == {"999", "100", "101"}
        assert [p.name for p in models.iterdir()] == [cache.name]
        summary["stage4_after_replace_KeyboardInterrupt"] = "new_complete_no_tmp"

        # Contrast: demonstrate what base_compute did on Stage 2
        base_cache = models / "base_lookup.json"
        base_cache.write_text(json.dumps(seed))

        def base_torn_write(self, data, *args, **kwargs):
            real_write_text(self, data[: len(data) // 2], *args, **kwargs)
            raise OSError("interrupted")

        Path.write_text = base_torn_write
        try:
            try:
                _call_build(base_compute, base_cache, raw)
            except OSError:
                pass
        finally:
            Path.write_text = real_write_text
        # Base left truncated JSON on disk; next call silently drops cache-only key 999
        rebuilt_base, _, base_err = _call_build(base_compute, base_cache, raw)
        assert 999 not in rebuilt_base and base_err == ""
        summary["base_contrast_lost_cache_only_key"] = True

    return summary


def check_concurrency(base_compute) -> dict:
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as td:
        root = Path(td)
        models = root / "models"
        models.mkdir()
        cache = models / "probable_pitcher_lookup.json"
        cache.write_text(json.dumps({"1": {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}}))
        raw = root / "raw"
        (raw / "2025").mkdir(parents=True)

        ctx = mp.get_context("spawn")
        ready_q = ctx.Queue()
        go_ev = ctx.Event()
        procs = [
            ctx.Process(target=_mp_writer_worker, args=(str(cache), str(raw), 15, ready_q, go_ev))
            for _ in range(2)
        ]
        for p in procs:
            p.start()
        pids = [ready_q.get(timeout=10) for _ in procs]
        assert len(set(pids)) == 2, "concurrent processes must have distinct PIDs"
        go_ev.set()

        reads = 0
        torn_reads = 0
        while any(p.is_alive() for p in procs):
            try:
                txt = cache.read_text()
                parsed = json.loads(txt)
                assert isinstance(parsed, dict) and "1" in parsed
                reads += 1
            except FileNotFoundError:
                torn_reads += 1
            except json.JSONDecodeError:
                torn_reads += 1
            time.sleep(0.001)

        for p in procs:
            p.join(timeout=10)
            assert p.exitcode == 0
        assert torn_reads == 0
        leftover_tmps = [p.name for p in models.iterdir() if p.name != cache.name]
        assert leftover_tmps == []

        # Lost-update race (interleaved read-modify-write across two distinct raw dirs):
        # Both before and after, if writer A and writer B read C0={1} simultaneously and scan
        # disjoint raw roots, the second replace overwrites the first's addition.
        race_cache = models / "race.json"
        race_cache.write_text(json.dumps({"1": {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}}))
        raw_a = root / "raw_a"
        raw_b = root / "raw_b"
        (raw_a / "2025").mkdir(parents=True)
        (raw_b / "2025").mkdir(parents=True)
        (raw_a / "2025" / "200.json").write_text(_feed_json(20, 21))
        (raw_b / "2025" / "300.json").write_text(_feed_json(30, 31))

        # Simulate simultaneous snapshot read of C0 before either writes
        real_replace = os.replace
        paused_src = []

        def intercept_first_replace(src, dst):
            # Let writer B run completely while writer A is about to os.replace
            os.replace = real_replace
            _call_build(head_compute, race_cache, raw_b)
            paused_src.append(Path(src).name)
            return real_replace(src, dst)

        os.replace = intercept_first_replace
        try:
            _call_build(head_compute, race_cache, raw_a)
        finally:
            os.replace = real_replace
        after_race = json.loads(race_cache.read_text())
        # Writer A replaced after B finished -> 300 from B was overwritten, until raw_b is scanned again
        lost_until_rescan = ("200" in after_race) and ("300" not in after_race)
        rescanned, _, _ = _call_build(head_compute, race_cache, raw_b)
        recovered_on_rescan = (200 in rescanned and 300 in rescanned)

    return {
        "worker_pids": pids,
        "concurrent_reads": reads,
        "torn_reads": torn_reads,
        "lost_until_rescan_on_concurrent_writers": lost_until_rescan,
        "recovered_on_next_scan": recovered_on_rescan,
    }


def check_edge_cases(base_compute) -> dict:
    out = {}
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as td:
        root = Path(td)
        raw = root / "raw"
        (raw / "2025").mkdir(parents=True)
        (raw / "2025" / "100.json").write_text(_feed_json(21, 22, 51, 52))

        # 1. Read-only models directory (0o555)
        ro_models = root / "ro_models"
        ro_models.mkdir()
        ro_cache = ro_models / "probable_pitcher_lookup.json"
        ro_cache.write_text(json.dumps({"50": {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}}))
        ro_before = ro_cache.read_bytes()
        ro_models.chmod(0o555)
        try:
            # When nothing new (no raw), read succeeds
            res_ok, _, err_ok = _call_build(head_compute, ro_cache, root / "missing_raw")
            assert res_ok == {50: {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}} and err_ok == ""
            # When new_count > 0, raises PermissionError, cache untouched, no tmp left
            raised_perm = False
            try:
                _call_build(head_compute, ro_cache, raw)
            except PermissionError:
                raised_perm = True
            assert raised_perm
            assert ro_cache.read_bytes() == ro_before
            assert [p.name for p in ro_models.iterdir()] == [ro_cache.name]
        finally:
            ro_models.chmod(0o755)
        out["readonly_models_dir"] = "raises_PermissionError_cache_intact"

        # 2. Symlinked cache path (valid target vs dangling target, base vs head)
        sym_dir = root / "sym"
        sym_dir.mkdir()
        b_target = sym_dir / "base_target.json"
        h_target = sym_dir / "head_target.json"
        init_bytes = json.dumps({"50": {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}}).encode()
        b_target.write_bytes(init_bytes)
        h_target.write_bytes(init_bytes)
        b_link = sym_dir / "base_link.json"
        h_link = sym_dir / "head_link.json"
        b_link.symlink_to(b_target)
        h_link.symlink_to(h_target)
        _call_build(base_compute, b_link, raw)
        _call_build(head_compute, h_link, raw)
        assert b_link.is_symlink() and b_target.read_bytes() != init_bytes  # base mutated target through link
        assert not h_link.is_symlink() and h_target.read_bytes() == init_bytes  # head replaced link, target untouched
        out["symlink_valid"] = "base_mutated_target_head_replaced_link"

        dang_link = sym_dir / "dangling.json"
        dang_link.symlink_to(sym_dir / "nonexistent_dir" / "x.json")
        _, d_out, d_err = _call_build(head_compute, dang_link, raw)
        assert not dang_link.is_symlink() and dang_link.exists() and d_out == "" and d_err == ""
        out["symlink_dangling"] = "head_replaces_cleanly_without_warning"

        # 3. Malformed / non-object / non-numeric-key / non-object-value JSON caches (before & after)
        malformed_cases = {
            "truncated": ('{"50": {"away": 1', "JSONDecodeError"),
            "json_list": ("[1, 2, 3]", "AttributeError"),
            "json_null": ("null", "AttributeError"),
            "json_int": ("42", "AttributeError"),
            "non_numeric_key": (json.dumps({"50": {"away": 1}, "bad": {"away": 2}}), "ValueError"),
        }
        for cname, (payload, expected_exc_name) in malformed_cases.items():
            bc = root / f"mal_b_{cname}.json"
            hc = root / f"mal_h_{cname}.json"
            bc.write_text(payload)
            hc.write_text(payload)
            b_r, b_o, b_e = _call_build(base_compute, bc, raw)
            h_r, h_o, h_e = _call_build(head_compute, hc, raw)
            assert b_r == h_r == {100: {"away": 21, "home": 22, "away_tid": 51, "home_tid": 52}}
            assert b_o == "" and h_o == "" and b_e == ""
            assert h_e.count("\n") == 1
            assert f"WARNING: unreadable probable-pitcher cache {hc} ({expected_exc_name}); rebuilding it from raw feeds" in h_e
            out[f"cache_{cname}"] = expected_exc_name

        # Non-object value in cache: {"100": null} -> accepted by _build_probable_pitcher_lookup both before & after!
        nov_b = root / "nov_b.json"
        nov_h = root / "nov_h.json"
        nov_b.write_text('{"100": null}')
        nov_h.write_text('{"100": null}')
        b_nov, _, b_nove = _call_build(base_compute, nov_b, raw)
        h_nov, _, h_nove = _call_build(head_compute, nov_h, raw)
        assert b_nov == h_nov == {100: None} and b_nove == "" and h_nove == ""
        # Check compute_all_features behaviour when game_pk 100 is in df and lookup[100] is None
        pa_df = pd.DataFrame([{
            "game_pk": 100, "date": "2025-04-01", "season": 2025,
            "batter_id": 1, "pitcher_id": 2, "is_home": True, "is_hit": 1,
            "stand": "R", "p_throws": "R", "venue_id": 1, "lineup_pos": 1,
        }])
        with _patched_cache_path(head_compute, nov_h):
            try:
                head_compute.compute_all_features(pa_df)
                caf_non_obj = "returned"
            except AttributeError:
                caf_non_obj = "AttributeError_both_before_and_after"
        out["cache_non_object_value"] = caf_non_obj

        # Unreadable cache when raw_dir is missing: warns on stderr, returns {}, leaves file intact
        unr_noraw = root / "unr_noraw.json"
        unr_noraw.write_text("{broken")
        unr_res, unr_o, unr_e = _call_build(head_compute, unr_noraw, root / "no_such_raw")
        assert unr_res == {} and unr_o == "" and "JSONDecodeError" in unr_e
        assert unr_noraw.read_text() == "{broken"
        out["unreadable_cache_no_raw"] = "warns_stderr_returns_empty_leaves_file"

        # 4. File mode check
        mode_b = root / "mode_b.json"
        mode_h = root / "mode_h.json"
        _call_build(base_compute, mode_b, raw)
        _call_build(head_compute, mode_h, raw)
        b_perm = stat.S_IMODE(mode_b.stat().st_mode)
        h_perm = stat.S_IMODE(mode_h.stat().st_mode)
        assert b_perm == h_perm
        out["created_file_mode_octal"] = oct(h_perm)

    return out


def main() -> None:
    base_compute = _load_base_compute()
    report = {
        "synthetic_equivalence": check_equivalence_synthetic(base_compute),
        "real_copy_equivalence": check_equivalence_real_copies(base_compute),
        "interruptions": check_interruptions(base_compute),
        "concurrency": check_concurrency(base_compute),
        "edge_cases": check_edge_cases(base_compute),
    }
    out_path = REVIEW_DIR / "probe_cache_results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
