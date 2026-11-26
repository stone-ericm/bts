"""Remaining sections of Items 1D, 2C, 3, and 4."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import random
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_SRC = HERE.parents[1] / "src"
BASE_SRC = HERE / "archives/base/src"
R1_SRC = HERE / "archives/r1/src"
R2_SRC = HERE / "archives/r2/src"

DAY = date(2026, 12, 31)
SOURCE = "probable_pitcher_cache"
SEED = {"away": 1, "home": 22, "away_tid": 51, "home_tid": 52}
RECOVERY = {"away": 2, "home": 23, "away_tid": 61, "home_tid": 62}
TICK = chr(96)


def run_1d():
    print("=== 1D. Exhaustive Summary Sweep & Hostile Disk Inputs ===")
    sys.path.insert(0, str(REPO_SRC))
    from bts.features import probable_pitcher_cache as producer
    from bts.health import alert, attention, probable_pitcher_cache as health, runner

    labels_all = ("invalid entries", "status unreadable", "unreadable cache")
    max_len_1000 = 0
    for r in range(1, 4):
        for subset in itertools.combinations(labels_all, r):
            for count in list(range(1, 1001)) + [9999, 10**6, 10**25, 10**26]:
                noun = "observation" if count == 1 else "observations"
                if count <= 1000:
                    a = health.summarize(DAY, ["x"] * count, set(subset))
                    text = a.message
                    if len(text) > max_len_1000:
                        max_len_1000 = len(text)
                else:
                    a = health.summarize(DAY, ["x"] * 2, set(subset))
                    text = a.message.replace("2 observations", f"{count} {noun}", 1)
                assert not attention.OOM_EVIDENCE_RE.search(f"{SOURCE} {text}".lower()), text
                assert TICK not in text
                if count <= 10**25:
                    assert len(text) <= 600, (count, len(text))
                elif r == 3:
                    assert len(text) == 601
    worst_alert = health.summarize(DAY, ["x"] * 1000, set(labels_all))
    streaked = attention._with_streak(worst_alert, 365)
    dm_body = alert.format_health_dm_body([], [streaked])
    resurface = alert._resurface_note({"status": "failed", "critical_count": 3})
    full_dm = f"{resurface}\n\n{dm_body}"
    assert len(full_dm) < 1000, len(full_dm)
    print(f"  Max summary len (count<=1000): {max_len_1000} (<=600 up to count=10**25); worst full DM with H6 banner & 365d streak: {len(full_dm)} chars (<1000)")

    with TemporaryDirectory(dir=HERE, prefix="tmp-hostile-") as tmp:
        identity = "a" * 64
        for sep_label, sep_char in [("backtick", TICK), ("newline", "\n")]:
            st_sub = Path(tmp) / sep_label / "oom/data/health_state" / SOURCE
            st_sub.mkdir(parents=True)
            crafted = dict(
                schema_version="bts_probable_pitcher_cache_status_v1",
                reason="invalid_entries",
                recorded_at=f"2026-12-31{sep_char}14:00:00+00:00",
                recorded_date_et="2026-12-31",
                cache_path="data/models/probable_pitcher_lookup.json",
                dropped=1, recovered_from_cache=0, rescanned=0, unusable_feed=0, without_feed=1,
                game_ids=[100], identity=identity, preserved=None, original_sha256="c" * 64,
            )
            rec_path = st_sub / f"2026-12-31.invalid_entries.{identity[:12]}.json"
            rec_path.write_text(json.dumps(crafted))
            logs = StringIO()
            h = logging.StreamHandler(logs)
            logger = logging.getLogger(health.__name__)
            logger.addHandler(h)
            try:
                alerts = health.check(st_sub, DAY)
            finally:
                logger.removeHandler(h)
            logged_text = logs.getvalue()
            print(
                f"  Crafted recorded_at ({sep_label}): read_record accepted=True, "
                f"sep_in_log={sep_char in logged_text}, sep_in_summary={sep_char in alerts[0].message}, "
                f"oom_in_summary={bool(attention.OOM_EVIDENCE_RE.search(alerts[0].message))}"
            )
            same_except_ts = dict(crafted, recorded_at="2026-12-31T15:00:00+00:00")
            producer.publish(rec_path, same_except_ts)
            persisted_ts = json.loads(rec_path.read_text())["recorded_at"]
            print(f"    publish() coalesced over crafted recorded_at: kept_crafted={persisted_ts == crafted['recorded_at']}")

        st_crash = Path(tmp) / "crash_sweep/oom/data/health_state" / SOURCE
        st_crash.mkdir(parents=True)
        hostile_payloads = {
            "non_utf8": b"\xff\xfe",
            "oversize": b"{" + b" " * 65536 + b"}",
            "huge_int_5000_digits": ('{"x": ' + ("9" * 5000) + "}").encode(),
            "deep_nesting": ("[" * 2500 + "]" * 2500).encode(),
            "unhashable_reason": json.dumps({
                "schema_version": "bts_probable_pitcher_cache_status_v1",
                "reason": ["unhashable"],
                "recorded_at": "2026-12-31T14:00:00+00:00",
                "recorded_date_et": "2026-12-31",
                "cache_path": "x",
                "dropped": 1, "recovered_from_cache": 0, "rescanned": 0, "unusable_feed": 0, "without_feed": 1,
                "game_ids": [100], "identity": "a" * 64, "preserved": None, "original_sha256": "c" * 64,
            }).encode(),
            "unhashable_game_id": json.dumps({
                "schema_version": "bts_probable_pitcher_cache_status_v1",
                "reason": "invalid_entries",
                "recorded_at": "2026-12-31T14:00:00+00:00",
                "recorded_date_et": "2026-12-31",
                "cache_path": "x",
                "dropped": 1, "recovered_from_cache": 0, "rescanned": 0, "unusable_feed": 0, "without_feed": 1,
                "game_ids": [{"bad": 1}], "identity": "a" * 64, "preserved": None, "original_sha256": "c" * 64,
            }).encode(),
            "overflow_timestamp": json.dumps({
                "schema_version": "bts_probable_pitcher_cache_status_v1",
                "reason": "invalid_entries",
                "recorded_at": "0001-01-01T01:00:00+00:00",
                "recorded_date_et": "0001-01-01",
                "cache_path": "x",
                "dropped": 1, "recovered_from_cache": 0, "rescanned": 0, "unusable_feed": 0, "without_feed": 1,
                "game_ids": [100], "identity": "a" * 64, "preserved": None, "original_sha256": "c" * 64,
            }).encode(),
        }
        for name, data_bytes in hostile_payloads.items():
            (st_crash / f"{DAY}.{name}.json").write_bytes(data_bytes)
        os.mkfifo(st_crash / f"{DAY}.fifo.json")
        (st_crash / f"{DAY}.dir.json").mkdir()
        (st_crash / f"{DAY}.sym.json").symlink_to(st_crash / f"{DAY}.non_utf8.json")

        safe_alerts = runner._safe_run("probable_pitcher_cache", lambda: health.check(st_crash, DAY))
        assert len(safe_alerts) == 1 and safe_alerts[0].level == "WARN" and safe_alerts[0].source == SOURCE
        assert not attention.OOM_EVIDENCE_RE.search(f"{safe_alerts[0].source} {safe_alerts[0].message}".lower())
        print(f"  Hostile crash-resistance sweep ({len(hostile_payloads) + 3} files under oom/data): 0 crashes, 1 clean WARN")


CHILD_CODE = r"""
import json, os, resource, sys
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
    return json.dumps({
        "gameData": {
            "probablePitchers": {"away": {"id": entry["away"]}, "home": {"id": entry["home"]}},
            "teams": {"away": {"id": entry["away_tid"]}, "home": {"id": entry["home_tid"]}},
        }
    })

if mode == "losses_and_r1":
    out = {}
    dates = pd.date_range("2025-04-01", periods=11)
    rows = [pa(100 + i, d, 3 if i < 10 else 1, True, i % 2) for i, d in enumerate(dates)]

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
    try:
        val_fd = compute.compute_all_features(df).iloc[-1]["opp_bullpen_hr_30g"]
        fds_after = len(os.listdir("/dev/fd"))
        out["fd_pressure"] = {
            "bp": None if pd.isna(val_fd) else float(val_fd),
            "fd_delta": fds_after - fds_before,
            "cached_109": "109" in json.loads(c_fd.read_text()),
        }
    except OSError as exc:
        out["fd_pressure"] = {"error": f"{type(exc).__name__}:{exc.errno}"}
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


def run_sub(src_dir: Path, mode: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    with TemporaryDirectory(dir=HERE, prefix=f"tmp-rem-{mode}-") as tmp:
        script = Path(tmp) / "sub.py"
        script.write_text(CHILD_CODE, encoding="utf-8")
        env = {
            **os.environ,
            "PYTHONPATH": str(src_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
            **(env_extra or {}),
        }
        return subprocess.run(
            [sys.executable, str(script), str(src_dir), mode, tmp],
            env=env,
            capture_output=True,
            timeout=60,
            check=True,
        )


def run_2c_3_4():
    print("\n=== 2C. Accepted Losses, R1 Regressions (FD Pressure & Closed Stderr), and Sequences ===")
    base_loss = json.loads(run_sub(BASE_SRC, "losses_and_r1").stdout)
    r1_loss = json.loads(run_sub(R1_SRC, "losses_and_r1").stdout)
    head_loss = json.loads(run_sub(REPO_SRC, "losses_and_r1").stdout)
    for loss_name in ("extra_key", "integral_float", "missing_home_tid"):
        assert base_loss[loss_name] == 0.5 and head_loss[loss_name] is None
    assert base_loss["fd_pressure"] == {"bp": 0.5, "fd_delta": 0, "cached_109": True}
    assert "error" in r1_loss["fd_pressure"] or r1_loss["fd_pressure"].get("bp") is None
    assert head_loss["fd_pressure"] == {"bp": 0.5, "fd_delta": 0, "cached_109": True}
    print(f"  Accepted losses (extra_key, integral_float, missing_home_tid): base=0.5 -> HEAD=NaN")
    print(f"  FD pressure (64 .json dirs, RLIMIT_NOFILE=48): base={base_loss['fd_pressure']}, r1={r1_loss['fd_pressure']}, HEAD={head_loss['fd_pressure']}")

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

    # Two calls on one ET day with different outcomes + D record read after midnight with D+1 present
    from bts import dm
    from bts.features import compute, probable_pitcher_cache as producer
    from bts.health import alert, probable_pitcher_cache as health, runner

    with TemporaryDirectory(dir=HERE, prefix="tmp-seq-") as tmp:
        p = Path(tmp)
        (p / "data/models").mkdir(parents=True)
        (p / "data/raw/2025").mkdir(parents=True)
        cache = p / "data/models/probable_pitcher_lookup.json"
        cache.write_text('{"100": null}')
        compute._probable_pitcher_cache_path = cache

        class ClockD(datetime):
            @classmethod
            def now(cls, tz=None):
                dt = cls(2027, 1, 1, 4, 59, tzinfo=timezone.utc)  # 2026-12-31 23:59 ET
                return dt.astimezone(tz) if tz is not None else dt.replace(tzinfo=None)

        class ClockD1(datetime):
            @classmethod
            def now(cls, tz=None):
                dt = cls(2027, 1, 1, 5, 1, tzinfo=timezone.utc)  # 2027-01-01 00:01 ET
                return dt.astimezone(tz) if tz is not None else dt.replace(tzinfo=None)

        # Call 1 on D: no feed -> preserved=None
        with patch.object(producer, "datetime", ClockD):
            assert compute._build_probable_pitcher_lookup(str(p / "data/raw")) == {}
        # Call 2 on D: feed arrives -> preserved=<copy>, rewritten cache
        (p / "data/raw/2025/100.json").write_text(json.dumps({
            "gameData": {
                "probablePitchers": {"away": {"id": 2}, "home": {"id": 23}},
                "teams": {"away": {"id": 61}, "home": {"id": 62}},
            }
        }))
        with patch.object(producer, "datetime", ClockD):
            assert compute._build_probable_pitcher_lookup(str(p / "data/raw")) == {100: RECOVERY}
        # Call 3 on D+1: new bad cache -> D+1 record also on disk
        cache.write_text('{"200": null}')
        with patch.object(producer, "datetime", ClockD1):
            assert compute._build_probable_pitcher_lookup(str(p / "data/raw")) == {100: RECOVERY}

        st_dir = p / "data/health_state" / SOURCE
        all_recs = [json.loads(f.read_text()) for f in sorted(st_dir.glob("*.json"))]
        assert len(all_recs) == 3
        d_recs = [r for r in all_recs if r["recorded_date_et"] == "2026-12-31"]
        d1_recs = [r for r in all_recs if r["recorded_date_et"] == "2027-01-01"]
        assert len(d_recs) == 2 and len(d1_recs) == 1
        assert {r["preserved"] is not None for r in d_recs} == {False, True}

        sent_dms = []
        safe_run = runner._safe_run
        with (
            patch.object(runner, "_safe_run", lambda n, fn: safe_run(n, fn) if n == SOURCE else []),
            patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None),
            patch.object(runner, "datetime", ClockD1),  # Wall clock is after midnight (00:01 ET Jan 1)!
            patch.object(alert, "datetime", ClockD1),
            patch.object(alert, "send_dm", lambda rec, body: (sent_dms.append(body), "id-1")[1]),
        ):
            # Late D run after midnight sees ONLY the 2 D records (ignores D+1!)
            d_alerts = runner.run_all_checks(p / "data/picks", p / "data/models", "did:plc:fake", today=DAY)
            assert len(d_alerts) == 1 and "2 observations (invalid entries)" in d_alerts[0].message
            # Subsequent D+1 run sees all 3 records (D + D+1) and is NOT suppressed by D's receipt!
            d1_alerts = runner.run_all_checks(p / "data/picks", p / "data/models", "did:plc:fake", today=DAY + timedelta(days=1))
            assert len(d1_alerts) == 1 and "3 observations (invalid entries)" in d1_alerts[0].message
        assert len(sent_dms) == 2
        print("  Same-day changed outcomes (2 records on D) + post-midnight late-D health with D+1 present: verified")

    print("\n=== 3A. display() Injectivity / Reversibility & Scope ===")
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
        b_err = run_sub(BASE_SRC, "stderr_encoding", env_extra={"PYTHONIOENCODING": io_enc}).stderr
        r2_err = run_sub(R2_SRC, "stderr_encoding", env_extra={"PYTHONIOENCODING": io_enc}).stderr
        hd_err = run_sub(REPO_SRC, "stderr_encoding", env_extra={"PYTHONIOENCODING": io_enc}).stderr
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
    summary = health.summarize(date(2026, 12, 31), ["a" * 64] * 1000, {"invalid entries", "unreadable cache", "status unreadable"})
    print("  Longest summary (1000 obs, 3 kinds):", summary.message)
    print("  Longest summary len:", len(summary.message))


if __name__ == "__main__":
    run_1d()
    run_2c_3_4()
    print("\nALL REMAINING PROBES COMPLETED.")
