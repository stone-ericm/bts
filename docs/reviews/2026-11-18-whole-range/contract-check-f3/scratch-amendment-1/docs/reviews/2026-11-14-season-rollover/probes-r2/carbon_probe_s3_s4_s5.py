"""Adversarial probes for S3, S4, S5 comparing 12dbaaa vs HEAD."""
from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest.mock as mock
from pathlib import Path

from click.testing import CliRunner


def load_base_module(rel_path: str, mod_name: str):
    src = subprocess.check_output(["git", "show", f"12dbaaa:{rel_path}"], text=True)
    tmp = Path(".codex-review/sr-review/scratch") / f"{mod_name}.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location(mod_name, tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probe_s3(tmp_root: Path) -> None:
    print("=== 1. S3 Probable-Pitcher Scan ===")
    import bts.features.compute as head_compute
    base_compute = load_base_module("src/bts/features/compute.py", "_base_compute")

    # Test 1: numeric + weather sidecar + non-digit .json -> cache parity vs numeric-only at base
    feed = {
        "gameData": {
            "teams": {"away": {"id": 134}, "home": {"id": 121}},
            "probablePitchers": {"away": {"id": 3001}, "home": {"id": 3002}},
        }
    }
    raw_clean = tmp_root / "s3_clean" / "2026"
    raw_clean.mkdir(parents=True)
    (raw_clean / "745001.json").write_text(json.dumps(feed))

    raw_sidecar = tmp_root / "s3_sidecar" / "2026"
    raw_sidecar.mkdir(parents=True)
    (raw_sidecar / "745001.json").write_text(json.dumps(feed))
    (raw_sidecar / "745001_weather.json").write_text(json.dumps({"surface_pressure": 1013.2}))
    (raw_sidecar / "notes.json").write_text("not even json")

    with mock.patch.object(base_compute, "_probable_pitcher_cache_path", tmp_root / "c_base.json"):
        base_lookup = base_compute._build_probable_pitcher_lookup(str(raw_clean.parent))
        try:
            base_compute._build_probable_pitcher_lookup(str(raw_sidecar.parent))
            base_err = None
        except Exception as e:
            base_err = type(e).__name__

    with mock.patch.object(head_compute, "_probable_pitcher_cache_path", tmp_root / "c_head.json"):
        head_lookup = head_compute._build_probable_pitcher_lookup(str(raw_sidecar.parent))

    print(f"  Base on sidecar: {base_err}; Head lookup == Base clean lookup: {head_lookup == base_lookup}")
    print(f"  Cache bytes equal: {(tmp_root / 'c_base.json').read_text() == (tmp_root / 'c_head.json').read_text()}")

    # Test 2: Unicode superscript digit stem '².json' -> f.stem.isdigit() is True, int('²') raises ValueError!
    raw_uni = tmp_root / "s3_uni" / "2026"
    raw_uni.mkdir(parents=True)
    (raw_uni / "745001.json").write_text(json.dumps(feed))
    (raw_uni / "².json").write_text("{}")
    with mock.patch.object(head_compute, "_probable_pitcher_cache_path", tmp_root / "c_uni.json"):
        try:
            head_compute._build_probable_pitcher_lookup(str(raw_uni.parent))
            uni_res = "ok"
        except Exception as e:
            uni_res = f"{type(e).__name__}: {e}"
    print(f"  Unicode superscript '².json' ('²'.isdigit()={'²'.isdigit()}): {uni_res}")


def probe_s4(tmp_root: Path) -> None:
    print("\n=== 2. S4 Atomic & Validated Feed Download ===")
    from bts.data.pull import download_game_feed, pull_feeds

    out_dir = tmp_root / "s4" / "2027"
    valid_bytes = b'{"gameData": {"game": {"pk": 9001, "type": "R"}}}'

    # 1. Truncated / non-object responses (3 attempts -> raises, no file or .tmp left)
    for name, bad_payload in [
        ("truncated", b'{"gameData": '),
        ("array", b'[1, 2, 3]'),
        ("scalar_int", b'42'),
        ("null", b'null'),
        ("html_502", b'<html>502 Bad Gateway</html>'),
    ]:
        d = tmp_root / f"s4_{name}"
        with (
            mock.patch("bts.data.pull.urlopen", side_effect=lambda *a, **k: io.BytesIO(bad_payload)),
            mock.patch("bts.data.pull.time.sleep"),
        ):
            try:
                download_game_feed(9001, d)
                outcome = "returned"
            except Exception as e:
                outcome = type(e).__name__
        leftover = list(d.iterdir()) if d.exists() else []
        print(f"  Bad payload {name}: outcome={outcome}, leftover_files={leftover}")

    # 2. Disk-full during write() and os.replace() failure -> no .tmp left behind
    d_err = tmp_root / "s4_replace_err"
    with (
        mock.patch("bts.data.pull.urlopen", side_effect=lambda *a, **k: io.BytesIO(valid_bytes)),
        mock.patch("bts.data.pull.os.replace", side_effect=OSError("ENOSPC simulated")),
        mock.patch("bts.data.pull.time.sleep"),
    ):
        try:
            download_game_feed(9001, d_err)
        except OSError:
            pass
    print(f"  os.replace OSError leftover files: {list(d_err.iterdir())}")

    # 3. Valid write: exact bytes + file mode check + existing corrupt file untouched
    with mock.patch("bts.data.pull.urlopen", side_effect=lambda *a, **k: io.BytesIO(valid_bytes)):
        p = download_game_feed(9001, out_dir)
    mode_oct = oct(stat.S_IMODE(p.stat().st_mode))
    print(f"  Valid download: bytes_equal={p.read_bytes() == valid_bytes}, mode={mode_oct}, leftovers={[x.name for x in out_dir.iterdir()]}")

    corrupt_existing = out_dir / "9002.json"
    corrupt_existing.write_text("{corrupt")
    with mock.patch("bts.data.pull.urlopen", side_effect=AssertionError("should not fetch")):
        p2 = download_game_feed(9002, out_dir)
    print(f"  Existing corrupt file: returned={p2.name}, contents_unchanged={p2.read_text() == '{corrupt'}")


def probe_s5(tmp_root: Path) -> None:
    print("\n=== 3. S5 build_season Matrix (12dbaaa vs HEAD) & Mid-Season Restore ===")
    import bts.data.build as head_build
    base_build = load_base_module("src/bts/data/build.py", "_base_build")
    from tests.conftest import sample_game_feed

    r_feed = sample_game_feed.__wrapped__()
    s_feed = json.loads(json.dumps(r_feed))
    s_feed["gameData"]["game"]["type"] = "S"
    untyped_feed = json.loads(json.dumps(r_feed))
    del untyped_feed["gameData"]["game"]["type"]

    def setup_case(base_dir: Path, case: str) -> Path:
        raw = base_dir / "raw"
        sdir = raw / "2027"
        if case == "missing_dir":
            raw.mkdir(parents=True)
        elif case == "empty_dir":
            sdir.mkdir(parents=True)
        elif case == "season_is_file":
            raw.mkdir(parents=True)
            sdir.write_text("not a dir")
        elif case == "raw_is_file":
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text("not a dir")
        elif case == "dangling_symlink":
            raw.mkdir(parents=True)
            sdir.symlink_to(base_dir / "nonexistent_target")
        elif case == "unreadable_dir":
            sdir.mkdir(parents=True)
            (sdir / "1.json").write_text(json.dumps(r_feed))
            sdir.chmod(0)
        elif case == "weather_only":
            sdir.mkdir(parents=True)
            (sdir / "1_weather.json").write_text("{}")
        elif case == "spring_only":
            sdir.mkdir(parents=True)
            (sdir / "1.json").write_text(json.dumps(s_feed))
        elif case == "untyped_only":
            sdir.mkdir(parents=True)
            (sdir / "1.json").write_text(json.dumps(untyped_feed))
        elif case == "dh_7inning_only":
            sdir.mkdir(parents=True)
            dh_feed = json.loads(json.dumps(r_feed))
            dh_feed["gameData"]["game"]["doubleHeader"] = "Y"
            for play in dh_feed["liveData"]["plays"]["allPlays"]:
                play["about"]["inning"] = min(play["about"]["inning"], 7)
            (sdir / "1.json").write_text(json.dumps(dh_feed))
        elif case == "regular_valid":
            sdir.mkdir(parents=True)
            (sdir / "1.json").write_text(json.dumps(r_feed))
        return raw

    cases = [
        "missing_dir", "empty_dir", "season_is_file", "raw_is_file",
        "dangling_symlink", "unreadable_dir", "weather_only", "spring_only",
        "untyped_only", "dh_7inning_only", "regular_valid",
    ]
    for case in cases:
        for out_exists in (False, True):
            row = []
            for label, mod in (("BASE", base_build), ("HEAD", head_build)):
                d = tmp_root / f"s5_{case}_{out_exists}_{label}"
                raw = setup_case(d, case)
                out = d / "proc" / "pa_2027.parquet"
                if out_exists:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(b"EXISTING_PARQUET")
                try:
                    res = mod.build_season(raw, out, 2027)
                    kept = out.read_bytes() == b"EXISTING_PARQUET" if out_exists and res is None else out.exists()
                    row.append(f"{label}:ret={'None' if res is None else f'DF({len(res)})'}(out={kept})")
                except Exception as e:
                    kept = (out.read_bytes() == b"EXISTING_PARQUET") if out_exists else (not out.exists())
                    row.append(f"{label}:RAISE({type(e).__name__},untouched={kept})")
                finally:
                    sdir = raw / "2027"
                    if case == "unreadable_dir" and sdir.exists():
                        sdir.chmod(0o755)
            print(f"  {case:18s} out_exists={str(out_exists):5s} | {' | '.join(row)}")

    # Mid-season restore check via _refresh_season_data when MLB pull returns 0 feeds (outage or All-Star window)
    from bts.model.predict import _refresh_season_data
    d_restore = tmp_root / "midseason_restore"
    proc = d_restore / "processed"
    proc.mkdir(parents=True)
    (proc / "pa_2027.parquet").write_bytes(b"RESTORED_FROM_R2")
    with mock.patch("bts.data.pull.pull_feeds", return_value=[]):
        try:
            _refresh_season_data("2027-07-16", str(d_restore / "raw"), str(proc))
            r_out = "ok"
        except Exception as e:
            r_out = f"{type(e).__name__}: {e}"
    print(f"\n  Mid-season restore + _refresh_season_data when pull_feeds returns []: {r_out}")
    print(f"  Restored parquet untouched: {(proc / 'pa_2027.parquet').read_bytes() == b'RESTORED_FROM_R2'}")


if __name__ == "__main__":
    scratch = Path(".codex-review/sr-review/scratch/tmp_s3_s4_s5")
    shutil.rmtree(scratch, ignore_errors=True)
    probe_s3(scratch)
    probe_s4(scratch)
    probe_s5(scratch)
