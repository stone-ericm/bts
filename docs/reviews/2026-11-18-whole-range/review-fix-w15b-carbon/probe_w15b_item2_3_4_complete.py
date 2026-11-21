"""Complete Probe for Items 2, 3, and 4: Whole-branch parity/sequences vs base/r1/r2, W15b diff/display/stderr, and operator texts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_SRC = HERE.parents[1] / "src"
BASE_SRC = HERE / "archives/base/src"
R1_SRC = HERE / "archives/r1/src"
R2_SRC = HERE / "archives/r2/src"
INPUTS = HERE / "inputs"

SEED = {"away": 1, "home": 22, "away_tid": 51, "home_tid": 52}
RECOVERY = {"away": 2, "home": 23, "away_tid": 61, "home_tid": 62}
TICK = chr(96)


FEATURE_CHILD = r"""
import hashlib, json, os, resource, shutil, sys
from pathlib import Path
import pandas as pd

expected_src = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
work = Path(sys.argv[3]).resolve()
os.chdir(work)

import bts
assert Path(bts.__file__).resolve() == expected_src / "bts/__init__.py"
from bts.features import compute, park_drag
park_drag.get_table = lambda: None

SEED = {"away": 1, "home": 22, "away_tid": 51, "home_tid": 52}
RECOVERY = {"away": 2, "home": 23, "away_tid": 61, "home_tid": 62}

def pa(game, day, pitcher, home, hit=0):
    return dict(
        pa_id=0, date=day, game_pk=game, batter_id=11, pitcher_id=pitcher,
        pitch_hand="R", venue_id=31, is_home=home, is_hit=hit, pitch_calls=["X"],
        pitch_types=["FF"], pitch_speeds=[93.0], pitch_spin_rates=[2300.0],
        pitch_extensions=[6.0], pitch_break_vertical=[15.0],
        pitch_break_horizontal=[5.0], pitch_px=[0.0], pitch_pz=[2.5],
        sz_top=3.5, sz_bottom=1.5, final_count_balls=0, final_count_strikes=0,
        launch_speed=91.0, launch_angle=10.0,
    )

def feed_json(entry):
    pp = {}
    if entry.get("away", "MISSING") != "OMIT":
        pp["away"] = {"id": entry["away"]}
    if entry.get("home", "MISSING") != "OMIT":
        pp["home"] = {"id": entry["home"]}
    return json.dumps({
        "gameData": {
            "probablePitchers": pp,
            "teams": {"away": {"id": entry["away_tid"]}, "home": {"id": entry["home_tid"]}},
        }
    })

def sig(lookup):
    return [
        [k, type(k).__name__, [[mk, type(mv).__name__, mv] for mk, mv in v.items()]]
        for k, v in lookup.items()
    ]

if mode == "nonobject_defect":
    shapes = {"null": None, "int": 7, "str": "bad", "list": [1], "true": True, "false": False}
    out = {}
    rows = [pa(100, "2025-04-01", 1, True), pa(100, "2025-04-01", 22, False)]
    for name, val in shapes.items():
        for with_feed in (False, True):
            sub = work / f"{name}_{with_feed}"
            (sub / "data/models").mkdir(parents=True)
            (sub / "data/raw/2025").mkdir(parents=True)
            cache = sub / "data/models/probable_pitcher_lookup.json"
            cache.write_text(json.dumps({"100": val, "999": SEED}))
            if with_feed:
                (sub / "data/raw/2025/100.json").write_text(feed_json(RECOVERY))
            os.chdir(sub)
            compute._probable_pitcher_cache_path = cache
            df = pd.DataFrame(rows)
            df["pa_id"] = range(len(df))
            try:
                res = compute.compute_all_features(df)
                out[f"{name}|{with_feed}"] = {
                    "status": "ok",
                    "opp_team": [None if pd.isna(x) else int(x) for x in res["opp_pitching_team_id"].tolist()],
                }
            except Exception as exc:
                out[f"{name}|{with_feed}"] = {"status": type(exc).__name__}
    print(json.dumps(out))

elif mode == "parity":
    inputs_dir = Path(sys.argv[4]).resolve()
    out = {}
    for fname in ("probable_pitcher_lookup.json", "probable_pitcher_lookup.2017-2025.json"):
        sub = work / fname
        (sub / "data/models").mkdir(parents=True)
        (sub / "data/raw").mkdir(parents=True)
        cache = sub / "data/models/probable_pitcher_lookup.json"
        shutil.copy2(inputs_dir / fname, cache)
        before_bytes = cache.read_bytes()
        before_mtime = cache.stat().st_mtime_ns
        compute._probable_pitcher_cache_path = cache
        r_fd, w_fd = os.pipe()
        saved = os.dup(2)
        try:
            os.dup2(w_fd, 2)
            lookup = compute._build_probable_pitcher_lookup(str(sub / "data/raw"))
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            os.close(w_fd)
        err = os.read(r_fd, 65536).decode("utf-8")
        os.close(r_fd)
        out[fname] = {
            "len": len(lookup),
            "sig_sha256": hashlib.sha256(json.dumps(sig(lookup)).encode()).hexdigest(),
            "bytes_unchanged": cache.read_bytes() == before_bytes,
            "mtime_unchanged": cache.stat().st_mtime_ns == before_mtime,
            "stderr": err,
            "extra_files": [p.name for p in (sub / "data/models").iterdir() if p.name != cache.name],
            "health_exists": (sub / "data/health_state").exists(),
        }

    for scan_new in (False, True):
        sub = work / f"synth_{scan_new}"
        (sub / "data/models").mkdir(parents=True)
        (sub / "data/raw/2024").mkdir(parents=True)
        (sub / "data/raw/2025").mkdir(parents=True)
        cache = sub / "data/models/probable_pitcher_lookup.json"
        rev_seed = {"home_tid": 52, "away_tid": 51, "home": None, "away": 1}
        big_id = 10**80 + 7
        raw_text = (
            '{"00100": {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}, '
            '"99": ' + json.dumps(rev_seed) + ', '
            '" 100\\n": ' + json.dumps(RECOVERY) + ', '
            '"-50": {"away": null, "home": null, "away_tid": null, "home_tid": null}, '
            '"0": {"away": -5, "home": 0, "away_tid": ' + str(big_id) + ', "home_tid": 9}}'
        )
        cache.write_text(raw_text)
        os.utime(cache, ns=(1_700_000_000_123_456_789, 1_700_000_000_123_456_789))
        before_bytes = cache.read_bytes()
        before_mtime = cache.stat().st_mtime_ns
        (sub / "data/raw/2025/99.json").write_text("{broken json for already-cached game")
        if scan_new:
            (sub / "data/raw/2024/201.json").write_text(feed_json({"away": "OMIT", "home": 88, "away_tid": 10, "home_tid": 20}))
            (sub / "data/raw/2025/202.json").write_text(feed_json(SEED))
        compute._probable_pitcher_cache_path = cache
        lookup = compute._build_probable_pitcher_lookup(str(sub / "data/raw"))
        out[f"synth_{scan_new}"] = {
            "sig": sig(lookup),
            "disk_bytes": cache.read_bytes().decode("utf-8"),
            "bytes_unchanged": cache.read_bytes() == before_bytes,
            "mtime_unchanged": cache.stat().st_mtime_ns == before_mtime,
            "mtime_advanced": cache.stat().st_mtime_ns != before_mtime,
            "extra_files": [p.name for p in (sub / "data/models").iterdir() if p.name != cache.name],
            "health_exists": (sub / "data/health_state").exists(),
        }

    sub = work / "full_df"
    (sub / "data/models").mkdir(parents=True)
    (sub / "data/raw/2025").mkdir(parents=True)
    cache = sub / "data/models/probable_pitcher_lookup.json"
    cache.write_text(json.dumps({str(100 + i): SEED for i in range(10)}))
    (sub / "data/raw/2025/110.json").write_text(feed_json(SEED))
    (sub / "data/raw/2025/111.json").write_text(feed_json(RECOVERY))
    os.chdir(sub)
    compute._probable_pitcher_cache_path = cache
    dates = pd.date_range("2025-04-01", periods=12)
    rows = [pa(100 + i, d, 3 if i < 10 else (1 if i == 10 else 99), True, i % 2) for i, d in enumerate(dates)]
    df = pd.DataFrame(rows)
    df["pa_id"] = range(len(df))
    feat_df = compute.compute_all_features(df)
    out["full_df_json"] = feat_df.to_json(orient="split", date_format="iso", double_precision=15)
    print(json.dumps(out))

elif mode == "losses_and_r1":
    out = {}
    dates = pd.date_range("2025-04-01", periods=11)
    rows = [pa(100 + i, d, 3 if i < 10 else 1, True, i % 2) for i, d in enumerate(dates)]

    # Accepted losses: extra key, integral float, missing home_tid
    for loss_name, bad_100 in [
        ("extra_key", dict(SEED, extra=99)),
        ("integral_float", dict(SEED, away_tid=51.0)),
        ("missing_home_tid", {"away": 1, "home": 22, "away_tid": 51}),
    ]:
        sub = work / loss_name
        (sub / "data/models").mkdir(parents=True)
        (sub / "data/raw").mkdir(parents=True)
        c = sub / "data/models/probable_pitcher_lookup.json"
        vals = {str(100 + i): dict(SEED) for i in range(11)}
        vals["100"] = bad_100
        c.write_text(json.dumps(vals))
        os.chdir(sub)
        compute._probable_pitcher_cache_path = c
        df = pd.DataFrame(rows)
        df["pa_id"] = range(len(df))
        val = compute.compute_all_features(df).iloc[-1]["opp_bullpen_hr_30g"]
        out[loss_name] = None if pd.isna(val) else float(val)

    # Round-1 directory fd pressure (64 numeric .json dirs ahead of 109.json under RLIMIT_NOFILE=48)
    sub_fd = work / "fd_pressure"
    (sub_fd / "data/models").mkdir(parents=True)
    (sub_fd / "data/raw/2025").mkdir(parents=True)
    c_fd = sub_fd / "data/models/probable_pitcher_lookup.json"
    c_fd.write_text(json.dumps({str(g): SEED for g in range(100, 111) if g != 109}))
    for g in range(1, 65):
        (sub_fd / f"data/raw/2025/{g:04d}.json").mkdir()
    (sub_fd / "data/raw/2025/109.json").write_text(feed_json(SEED))
    os.chdir(sub_fd)
    compute._probable_pitcher_cache_path = c_fd
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (48, hard))
    fds_before = len(os.listdir("/dev/fd"))
    df = pd.DataFrame(rows)
    df["pa_id"] = range(len(df))
    val_fd = compute.compute_all_features(df).iloc[-1]["opp_bullpen_hr_30g"]
    fds_after = len(os.listdir("/dev/fd"))
    out["fd_pressure"] = {
        "bp": None if pd.isna(val_fd) else float(val_fd),
        "fd_delta": fds_after - fds_before,
        "cached_109": "109" in json.loads(c_fd.read_text()),
    }
    print(json.dumps(out))

elif mode == "stderr_encoding":
    sub = work / "enc_test"
    (sub / "data/models").mkdir(parents=True)
    (sub / "data/raw").mkdir(parents=True)
    c_utf = sub / "data/models/caf\u00e9_\u26a1.json"
    c_utf.write_text("{")
    compute._probable_pitcher_cache_path = c_utf
    compute._build_probable_pitcher_lookup(str(sub / "data/raw"))
    surr = "surrogate_test:" + chr(0xDCFF) + "_end"
    try:
        from bts.features import probable_pitcher_cache as ppc
        c_drop = sub / f"data/models/drop_caf\u00e9_{chr(96)}.json"
        c_drop.write_text('{"100": null}')
        compute._probable_pitcher_cache_path = c_drop
        compute._build_probable_pitcher_lookup(str(sub / "data/raw"))
        ppc.warn(surr)
    except ImportError:
        print(f"WARNING: {surr}", file=sys.stderr, flush=True)
"""


def run_feat_child(src_dir: Path, mode: str, *extra_args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    with TemporaryDirectory(dir=HERE, prefix=f"tmp-f-{mode}-") as tmp:
        script = Path(tmp) / "fchild.py"
        script.write_text(FEATURE_CHILD, encoding="utf-8")
        env = {
            **os.environ,
            "PYTHONPATH": str(src_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
            **(env_extra or {}),
        }
        return subprocess.run(
            [sys.executable, str(script), str(src_dir), mode, tmp, *extra_args],
            env=env,
            capture_output=True,
            timeout=60,
            check=True,
        )


def check_item2_parity_and_sequences():
    print("=== 2A. Base AttributeError vs HEAD Recovery/Drop ===")
    base_def = json.loads(run_feat_child(BASE_SRC, "nonobject_defect").stdout)
    head_def = json.loads(run_feat_child(REPO_SRC, "nonobject_defect").stdout)
    for k in sorted(base_def):
        assert base_def[k]["status"] == "AttributeError", (k, base_def[k])
        assert head_def[k]["status"] == "ok", (k, head_def[k])
        with_feed = k.endswith("True")
        assert head_def[k]["opp_team"] == ([61, 62] if with_feed else [None, None]), (k, head_def[k])
    print(f"  Verified all {len(base_def)} non-object x feed cases: base -> AttributeError, HEAD -> exact [61,62] / [None,None]")

    print("\n=== 2B. Parity on Real Caches, Synthetic Caches, and Full Feature Frame ===")
    base_par = json.loads(run_feat_child(BASE_SRC, "parity", str(INPUTS)).stdout)
    head_par = json.loads(run_feat_child(REPO_SRC, "parity", str(INPUTS)).stdout)
    for fname in ("probable_pitcher_lookup.json", "probable_pitcher_lookup.2017-2025.json"):
        assert base_par[fname] == head_par[fname], (fname, base_par[fname], head_par[fname])
        assert head_par[fname]["bytes_unchanged"] and head_par[fname]["mtime_unchanged"]
        assert head_par[fname]["stderr"] == "" and head_par[fname]["extra_files"] == [] and not head_par[fname]["health_exists"]
        print(f"  {fname}: {head_par[fname]['len']} entries, exact parity (dict/key-orders/types/bytes/mtime/stderr/no-artifacts)")

    assert base_par["synth_False"] == head_par["synth_False"]
    assert head_par["synth_False"]["bytes_unchanged"] and head_par["synth_False"]["mtime_unchanged"]
    assert base_par["synth_True"]["sig"] == head_par["synth_True"]["sig"]
    assert base_par["synth_True"]["disk_bytes"] == head_par["synth_True"]["disk_bytes"]
    assert head_par["synth_True"]["mtime_advanced"] and not head_par["synth_True"]["health_exists"]
    print("  Synthetic valid caches (aliases, reversed inner keys, nulls, negative/80-digit ints, with & without scan): exact parity")

    df_base = pd.read_json(io.StringIO(base_par["full_df_json"]), orient="split")
    df_head = pd.read_json(io.StringIO(head_par["full_df_json"]), orient="split")
    pd.testing.assert_frame_equal(df_base, df_head, check_exact=True)
    print(f"  Full compute_all_features frame ({df_head.shape[0]}x{df_head.shape[1]}): exact equality with base")

    print("\n=== 2C. Accepted Losses, R1 Regressions (FD Pressure & Closed Stderr), and Sequences ===")
    base_loss = json.loads(run_feat_child(BASE_SRC, "losses_and_r1").stdout)
    r1_loss = json.loads(run_feat_child(R1_SRC, "losses_and_r1").stdout)
    head_loss = json.loads(run_feat_child(REPO_SRC, "losses_and_r1").stdout)
    for loss_name in ("extra_key", "integral_float", "missing_home_tid"):
        assert base_loss[loss_name] == 0.5 and head_loss[loss_name] is None
    assert base_loss["fd_pressure"] == {"bp": 0.5, "fd_delta": 0, "cached_109": True}
    assert r1_loss["fd_pressure"]["bp"] is None and not r1_loss["fd_pressure"]["cached_109"]
    assert head_loss["fd_pressure"] == {"bp": 0.5, "fd_delta": 0, "cached_109": True}
    print("  Accepted losses (extra_key, integral_float, missing_home_tid): base=0.5 -> HEAD=NaN (documented W15-S2)")
    print("  FD pressure (64 .json dirs, RLIMIT_NOFILE=48): base=0.5, r1=NaN (leaked), HEAD=0.5 (fd_delta=0)")

    # Closed stderr reader EPIPE: r1 exits 120, HEAD exits 0
    epipe_code = (
        "import os, sys; from pathlib import Path; os.chdir(sys.argv[1]); "
        "from bts.features import compute; c = Path('data/models/probable_pitcher_lookup.json'); "
        "c.parent.mkdir(parents=True, exist_ok=True); c.write_text('{'); "
        "compute._probable_pitcher_cache_path = c; compute._build_probable_pitcher_lookup('data/raw'); "
        "print('OK', flush=True)"
    )
    for label, src_p, expected_rc in [("r1 (a924d748)", R1_SRC, 120), ("HEAD (284dc97a)", REPO_SRC, 0)]:
        with TemporaryDirectory(dir=HERE, prefix="tmp-epipe-") as tmp:
            r_fd, w_fd = os.pipe()
            os.close(r_fd)
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", epipe_code, tmp],
                    env={**os.environ, "PYTHONPATH": str(src_p), "PYTHONDONTWRITEBYTECODE": "1"},
                    stdout=subprocess.PIPE, stderr=w_fd, text=True, timeout=15,
                )
            finally:
                os.close(w_fd)
            assert proc.stdout == "OK\n" and proc.returncode == expected_rc, (label, proc.returncode)
            print(f"  Closed-reader stderr EPIPE on {label}: stdout='OK', rc={proc.returncode}")

    # Hard os._exit(99) before/after each of the 3 renames in child subprocesses, then fresh recovery run
    hard_exit_code = """
import json, os, sys
from pathlib import Path
os.chdir(sys.argv[1])
stop_target = sys.argv[2]
stop_after = sys.argv[3] == "1"
from bts.features import compute
cache = Path("data/models/probable_pitcher_lookup.json")
compute._probable_pitcher_cache_path = cache
real_replace = os.replace
def hooked(src, dst, *a, **kw):
    dst_p = Path(dst)
    kind = "cache" if dst_p == cache else ("evidence" if dst_p.parent.name == "rejected" else "status")
    if kind == stop_target:
        if stop_after:
            real_replace(src, dst, *a, **kw)
        os._exit(99)
    return real_replace(src, dst, *a, **kw)
os.replace = hooked
compute._build_probable_pitcher_lookup("data/raw")
"""
    for stop_target in ("evidence", "status", "cache"):
        for stop_after in (False, True):
            with TemporaryDirectory(dir=HERE, prefix="tmp-kill-") as tmp:
                p = Path(tmp)
                (p / "data/models").mkdir(parents=True)
                (p / "data/raw/2025").mkdir(parents=True)
                orig = b'{"100": null, "200": {"away": 1, "home": 22, "away_tid": 51, "home_tid": 52}}'
                (p / "data/models/probable_pitcher_lookup.json").write_bytes(orig)
                (p / "data/raw/2025/101.json").write_text(json.dumps({
                    "gameData": {
                        "probablePitchers": {"away": {"id": 2}, "home": {"id": 23}},
                        "teams": {"away": {"id": 61}, "home": {"id": 62}},
                    }
                }))
                script = p / "kill_step.py"
                script.write_text(hard_exit_code)
                res = subprocess.run(
                    [sys.executable, str(script), str(p), stop_target, "1" if stop_after else "0"],
                    env={**os.environ, "PYTHONPATH": str(REPO_SRC), "PYTHONDONTWRITEBYTECODE": "1"},
                    timeout=15,
                )
                assert res.returncode == 99
                active_bytes = (p / "data/models/probable_pitcher_lookup.json").read_bytes()
                if stop_target == "cache" and stop_after:
                    assert json.loads(active_bytes) == {"200": SEED, "101": RECOVERY}
                else:
                    assert active_bytes == orig
                retry_code = (
                    "import json, os, sys; from pathlib import Path; os.chdir(sys.argv[1]); "
                    "from bts.features import compute; compute._probable_pitcher_cache_path = Path('data/models/probable_pitcher_lookup.json'); "
                    "print(json.dumps(compute._build_probable_pitcher_lookup('data/raw')))"
                )
                retry = subprocess.run(
                    [sys.executable, "-c", retry_code, str(p)],
                    env={**os.environ, "PYTHONPATH": str(REPO_SRC), "PYTHONDONTWRITEBYTECODE": "1"},
                    capture_output=True, text=True, timeout=15, check=True,
                )
                assert json.loads(retry.stdout) == {"200": SEED, "101": RECOVERY}
                ev_files = [f for f in (p / "data/models/rejected").iterdir() if not f.name.startswith(".") and f.suffix == ".json"]
                st_files = [f for f in (p / "data/health_state/probable_pitcher_cache").iterdir() if f.suffix == ".json"]
                assert len(ev_files) == 1 and ev_files[0].read_bytes() == orig
                assert len(st_files) == 1
    print("  All 6 hard-kill (os._exit) rename stop points + fresh retries preserved entries, evidence, and dated status")


def check_item3_and_4():
    print("\n=== 3A. display() Injectivity / Reversibility & Scope ===")
    sys.path.insert(0, str(REPO_SRC))
    from bts.probable_pitcher_status import display

    def undisplay(s: str) -> str:
        out = []
        i = 0
        while i < len(s):
            if s[i] == "\\":
                if s[i : i + 2] == "\\\\":
                    out.append("\\")
                    i += 2
                elif s[i : i + 4] == "\\x60":
                    out.append("`")
                    i += 4
                else:
                    raise ValueError(f"Invalid escape at {i}: {s!r}")
            else:
                out.append(s[i])
                i += 1
        return "".join(out)

    rng = random.Random(20261121)
    alphabet = ["a", "/", ".", "\\", "`", "x", "6", "0", "\\x60", "\\\\", "\n", "\u00e9", "\u26a1", chr(0xDCFF)]
    for _ in range(2000):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 25)))
        enc = display(raw)
        assert TICK not in enc
        if "\\" not in raw and "`" not in raw:
            assert enc == raw
        assert undisplay(enc) == raw
    print("  display() verified injective & reversible over 2,000 adversarial strings (including \\x60, \\\\, Unicode, lone surrogates)")

    print("\n=== 3B. warn() Stream Encoding Comparison (UTF-8 vs ASCII, Base vs R2 vs HEAD) ===")
    for io_enc in ("utf-8", "utf-8:strict", "ascii"):
        b_err = run_feat_child(BASE_SRC, "stderr_encoding", env_extra={"PYTHONIOENCODING": io_enc}).stderr
        r2_err = run_feat_child(R2_SRC, "stderr_encoding", env_extra={"PYTHONIOENCODING": io_enc}).stderr
        hd_err = run_feat_child(REPO_SRC, "stderr_encoding", env_extra={"PYTHONIOENCODING": io_enc}).stderr
        print(f"  PYTHONIOENCODING={io_enc}:")
        print(f"    base: {b_err.splitlines()}")
        print(f"    r2  : {r2_err.splitlines()}")
        print(f"    HEAD: {hd_err.splitlines()}")
        if io_enc == "utf-8":
            assert b_err.splitlines()[0] == r2_err.splitlines()[0] == hd_err.splitlines()[0]
            assert b"\\udcff" in b_err and b"\\udcff" in hd_err and b"surrogate_test:?_end" in r2_err
        elif io_enc == "ascii":
            assert b_err.splitlines()[0] == hd_err.splitlines()[0]
            assert b"caf\\xe9" in hd_err and b"caf?" in r2_err

    print("\n=== 3C. Standalone Process Logging Route (logging.lastResort -> stderr) ===")
    log_route_code = (
        "import json, sys; from datetime import date; from pathlib import Path; "
        "import bts.cli, bts.scheduler; from bts.health import probable_pitcher_cache as h; "
        "st = Path(sys.argv[1]) / 'data/health_state/probable_pitcher_cache'; st.mkdir(parents=True); "
        "(st / '2026-12-31.bad`file.json').write_text('{'); "
        "a, = h.check(st, date(2026, 12, 31)); print(a.message)"
    )
    with TemporaryDirectory(dir=HERE, prefix="tmp-logroute-") as tmp:
        lr = subprocess.run(
            [sys.executable, "-c", log_route_code, tmp],
            env={**os.environ, "PYTHONPATH": str(REPO_SRC), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, timeout=15, check=True,
        )
        assert "2026-12-31.bad\\x60file.json" in lr.stderr and TICK not in lr.stderr
        assert "details in this health run's log and in health_state/probable_pitcher_cache beside the picks directory." in lr.stdout
        print("  Standalone process with bts.cli + bts.scheduler imported: log.warning routes to stderr via lastResort, backticks escaped")

    print("\n=== 4. Exact Rendered Operator Texts Added/Changed in W15b ===")
    from bts.health import probable_pitcher_cache as health
    summary = health.summarize(date(2026, 12, 31), ["a" * 64] * 1000, {"invalid entries", "unreadable cache", "status unreadable"})
    print("  Longest summary (1000 obs, 3 kinds):", summary.message)
    print("  Longest summary len:", len(summary.message))


if __name__ == "__main__":
    check_item2_parity_and_sequences()
    check_item3_and_4()
    print("\nALL ITEM 2-4 PROBES COMPLETED.")
