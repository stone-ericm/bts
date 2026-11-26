"""Follow-up cache concurrency (distinct-PID vs same-PID) and filesystem/JSON edge-case probe."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import multiprocessing as mp
import os
import stat
import tempfile
from pathlib import Path

import pandas as pd

from bts.data.build import parse_game_feed
from bts.features import compute as head_compute

REVIEW_DIR = Path(__file__).resolve().parent
BASE_COMPUTE_PATH = REVIEW_DIR / "base_ref" / "src" / "bts" / "features" / "compute.py"
REAL_COPY_RAW = REVIEW_DIR / "real_copy" / "raw"


def _load_base_compute():
    spec = importlib.util.spec_from_file_location("_leads_review_base_compute_edges", BASE_COMPUTE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _feed_json(away_pid, home_pid, away_tid=111, home_tid=112) -> str:
    return json.dumps({
        "gameData": {
            "teams": {"away": {"id": away_tid}, "home": {"id": home_tid}},
            "probablePitchers": {"away": {"id": away_pid}, "home": {"id": home_pid}},
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


def _child_build_once(cache_str: str, raw_str: str, q):
    cache_p = Path(cache_str)
    with _patched_cache_path(head_compute, cache_p):
        res = head_compute._build_probable_pitcher_lookup(raw_str)
    q.put((os.getpid(), sorted(res.keys())))


def check_two_process_vs_same_pid_races(base_compute) -> dict:
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as td:
        root = Path(td)
        models = root / "models"
        models.mkdir()
        raw_a = root / "raw_a"
        raw_b = root / "raw_b"
        (raw_a / "2025").mkdir(parents=True)
        (raw_b / "2025").mkdir(parents=True)
        (raw_a / "2025" / "200.json").write_text(_feed_json(20, 21))
        (raw_b / "2025" / "300.json").write_text(_feed_json(30, 31))

        # 1. Two DISTINCT OS processes (parent PID_A and spawned child PID_B):
        # Parent prepares tmp `.race_mp.json.<PID_A>.tmp`, pauses before os.replace,
        # while Child (PID_B != PID_A) reads C0={1}, scans raw_b={300}, writes `.race_mp.json.<PID_B>.tmp`,
        # and atomically replaces race_mp.json with {1, 300}. Then Parent finishes os.replace with {1, 200}.
        race_mp = models / "race_mp.json"
        race_mp.write_text(json.dumps({"1": {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}}))
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        real_replace = os.replace
        seen_tmps = []

        def pause_for_child(src, dst):
            os.replace = real_replace
            p = ctx.Process(target=_child_build_once, args=(str(race_mp), str(raw_b), q))
            p.start()
            child_pid, child_keys = q.get(timeout=15)
            p.join(timeout=15)
            assert p.exitcode == 0
            assert child_pid != os.getpid()
            assert child_keys == [1, 300]
            # Parent's own temp file is still intact because child used child_pid!
            assert Path(src).exists()
            seen_tmps.append((Path(src).name, f".{race_mp.name}.{child_pid}.tmp"))
            return real_replace(src, dst)

        os.replace = pause_for_child
        try:
            _call_build(head_compute, race_mp, raw_a)
        finally:
            os.replace = real_replace

        after_mp = json.loads(race_mp.read_text())
        assert set(after_mp.keys()) == {"1", "200"}
        assert [p.name for p in models.iterdir()] == ["race_mp.json"]
        # Next scan on raw_b recovers 300
        rescanned, _, _ = _call_build(head_compute, race_mp, raw_b)
        assert sorted(rescanned.keys()) == [1, 200, 300]

        # Same lost-update check on base_compute (last writer also overwrote)
        race_base = models / "race_base.json"
        race_base.write_text(json.dumps({"1": {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}}))
        real_wt = Path.write_text

        def pause_base_wt(self, data, *args, **kwargs):
            Path.write_text = real_wt
            _call_build(base_compute, race_base, raw_b)
            return real_wt(self, data, *args, **kwargs)

        Path.write_text = pause_base_wt
        try:
            _call_build(base_compute, race_base, raw_a)
        finally:
            Path.write_text = real_wt
        assert set(json.loads(race_base.read_text()).keys()) == {"1", "200"}

        # 2. Same-PID concurrent calls (e.g., two threads in one process sharing os.getpid()):
        # Inner call uses the identical `.race_same_pid.json.<pid>.tmp` name and unlinks it before
        # outer call runs os.replace -> FileNotFoundError!
        race_same = models / "race_same_pid.json"
        race_same.write_text(json.dumps({"1": {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}}))

        def same_pid_collision(src, dst):
            os.replace = real_replace
            _call_build(head_compute, race_same, raw_b)
            return real_replace(src, dst)

        os.replace = same_pid_collision
        same_pid_err = None
        try:
            try:
                _call_build(head_compute, race_same, raw_a)
            except FileNotFoundError as exc:
                same_pid_err = type(exc).__name__
        finally:
            os.replace = real_replace
        assert same_pid_err == "FileNotFoundError"

    return {
        "distinct_pid_tmps": seen_tmps[0],
        "distinct_pid_lost_update_last_writer_wins_cleanly": True,
        "base_also_last_writer_wins": True,
        "recovered_on_next_scan": True,
        "same_pid_thread_collision": same_pid_err,
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
            res_ok, _, err_ok = _call_build(head_compute, ro_cache, root / "missing_raw")
            assert res_ok == {50: {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}} and err_ok == ""
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
        out["readonly_models_dir_0555"] = "read_ok_write_raises_PermissionError_cache_intact"

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
        assert b_link.is_symlink() and b_target.read_bytes() != init_bytes
        assert not h_link.is_symlink() and h_target.read_bytes() == init_bytes
        out["symlink_valid"] = "base_mutated_target_head_replaced_link_with_regular_file"

        dang_link = sym_dir / "dangling.json"
        dang_link.symlink_to(sym_dir / "nonexistent_dir" / "x.json")
        _, d_out, d_err = _call_build(head_compute, dang_link, raw)
        assert not dang_link.is_symlink() and dang_link.exists() and d_out == "" and d_err == ""
        out["symlink_dangling"] = "head_replaces_dangling_link_without_warning"

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

        # Non-object value in cache: {"776135": null} -> accepted by _build_probable_pitcher_lookup both before & after
        sample_feed_path = next(
            p for p in sorted((REAL_COPY_RAW / "2025").glob("*.json"))
            if parse_game_feed(json.loads(p.read_text()))
        )
        sample_pk = int(sample_feed_path.stem)
        pa_rows = parse_game_feed(json.loads(sample_feed_path.read_text()))
        pa_df = pd.DataFrame(pa_rows)

        nov_b = root / "nov_b.json"
        nov_h = root / "nov_h.json"
        nov_b.write_text(json.dumps({str(sample_pk): None}))
        nov_h.write_text(json.dumps({str(sample_pk): None}))
        b_nov, _, b_nove = _call_build(base_compute, nov_b, root / "no_raw")
        h_nov, _, h_nove = _call_build(head_compute, nov_h, root / "no_raw")
        assert b_nov == h_nov == {sample_pk: None} and b_nove == "" and h_nove == ""

        # Both base and head raise AttributeError in compute_all_features when lookup[pk] is None
        for label, mod, cpath in (("base", base_compute, nov_b), ("head", head_compute, nov_h)):
            with _patched_cache_path(mod, cpath):
                raised_attr = False
                try:
                    mod.compute_all_features(pa_df)
                except AttributeError:
                    raised_attr = True
                assert raised_attr
        out["cache_non_object_value"] = "lookup_returns_None_both_raise_AttributeError_in_compute_all_features"

        # Unreadable cache when raw_dir is missing or empty: warns on stderr, returns {}, leaves file intact
        unr_noraw = root / "unr_noraw.json"
        unr_noraw.write_text("{broken")
        unr_res, unr_o, unr_e = _call_build(head_compute, unr_noraw, root / "no_such_raw")
        assert unr_res == {} and unr_o == "" and "JSONDecodeError" in unr_e
        assert unr_noraw.read_text() == "{broken"
        out["unreadable_cache_no_raw"] = "warns_stderr_returns_empty_leaves_file"

        # 4. File mode check (fresh creation + overwrite of 0o600 existing file)
        mode_b = root / "mode_b.json"
        mode_h = root / "mode_h.json"
        _call_build(base_compute, mode_b, raw)
        _call_build(head_compute, mode_h, raw)
        b_perm = stat.S_IMODE(mode_b.stat().st_mode)
        h_perm = stat.S_IMODE(mode_h.stat().st_mode)
        assert b_perm == h_perm

        # Overwrite a 0o600 cache when a new feed arrives: base kept inode mode 0o600, head creates 0o666 & ~umask
        (raw / "2025" / "101.json").write_text(_feed_json(31, 32, 51, 52))
        mode_b.chmod(0o600)
        mode_h.chmod(0o600)
        _call_build(base_compute, mode_b, raw)
        _call_build(head_compute, mode_h, raw)
        out["fresh_file_mode_octal"] = oct(h_perm)
        out["overwrite_0600_base_vs_head_mode"] = (
            oct(stat.S_IMODE(mode_b.stat().st_mode)),
            oct(stat.S_IMODE(mode_h.stat().st_mode)),
        )

    return out


def main() -> None:
    base_compute = _load_base_compute()
    report = {
        "races": check_two_process_vs_same_pid_races(base_compute),
        "edge_cases": check_edge_cases(base_compute),
    }
    out_path = REVIEW_DIR / "probe_cache_edges_results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
