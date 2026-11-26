"""Probe family 1: base AttributeError, parity (real + synthetic caches + features), corpus, and sequences."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
BASE_SRC = REVIEW_DIR / "base_src" / "src"
HEAD_SRC = REPO_ROOT / "src"
FIXTURES = REVIEW_DIR / "fixtures"

VALID_A = {"away": 10, "home": 20, "away_tid": 110, "home_tid": 120}
VALID_B = {"away": 11, "home": 21, "away_tid": 111, "home_tid": 121}


def make_feed(entry: dict) -> str:
    return json.dumps({
        "gameData": {
            "probablePitchers": {
                side: ({"id": entry[side]} if entry[side] is not None else {})
                for side in ("away", "home")
                if entry[side] is not None
            },
            "teams": {
                side: {"id": entry[f"{side}_tid"]}
                for side in ("away", "home")
            },
        }
    })


def pa_row(game: int, day: str, pitcher: int, is_home: bool, hit: int = 0, batter: int = 7) -> dict:
    return {
        "pa_id": 0,
        "date": day,
        "game_pk": game,
        "batter_id": batter,
        "pitcher_id": pitcher,
        "pitch_hand": "R",
        "venue_id": 31,
        "is_home": is_home,
        "is_hit": hit,
        "pitch_calls": ["C", "X"],
        "pitch_types": ["FF", "SL"],
        "pitch_speeds": [94.0, 86.0],
        "pitch_spin_rates": [2350.0, 2500.0],
        "pitch_extensions": [6.2, 6.1],
        "pitch_break_vertical": [16.0, 3.0],
        "pitch_break_horizontal": [6.0, -4.0],
        "pitch_px": [0.8, 0.1],
        "pitch_pz": [2.5, 2.4],
        "sz_top": 3.5,
        "sz_bottom": 1.5,
        "final_count_balls": 1,
        "final_count_strikes": 1,
        "launch_speed": 95.0,
        "launch_angle": 12.0,
    }


CHILD_RUNNER = """
import json, os, sys
from datetime import datetime
from pathlib import Path
import pandas as pd

mode = sys.argv[1]
action = sys.argv[2]
work = Path(sys.argv[3])
os.chdir(work)

import bts
from bts.features import compute, park_drag
park_drag.get_table = lambda: None

cache = work / "data/models/probable_pitcher_lookup.json"
compute._probable_pitcher_cache_path = cache

if len(sys.argv) > 4 and sys.argv[4] != "none":
    stop = sys.argv[4]
    real_replace = os.replace
    def hooked_replace(src, dst, *a, **kw):
        dst_p = Path(dst)
        label = "cache" if dst_p == cache else ("evidence" if dst_p.parent.name == "rejected" else "status")
        if stop == f"before-{label}":
            os._exit(91)
        res = real_replace(src, dst, *a, **kw)
        if stop == f"after-{label}":
            os._exit(92)
        return res
    os.replace = hooked_replace

if len(sys.argv) > 5 and sys.argv[5] != "none":
    ts = datetime.fromisoformat(sys.argv[5])
    try:
        from bts.features import probable_pitcher_cache as ppc
        class Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return ts.astimezone(tz) if tz is not None else ts.replace(tzinfo=None)
        ppc.datetime = Frozen
    except ImportError:
        pass

if action == "build":
    d = compute._build_probable_pitcher_lookup("data/raw")
    sig = [
        [k, type(k).__name__, [[mk, type(mv).__name__, mv] for mk, mv in v.items()]]
        for k, v in d.items()
    ]
    print(json.dumps({"bts_file": bts.__file__, "sig": sig}))
elif action == "features":
    rows = json.loads((work / "rows.json").read_text())
    df = pd.DataFrame(rows)
    df["pa_id"] = range(len(df))
    try:
        out = compute.compute_all_features(df)
        out.to_pickle(work / "features_out.pkl")
        print(json.dumps({"bts_file": bts.__file__, "status": "ok", "cols": list(out.columns)}))
    except Exception as e:
        print(json.dumps({"bts_file": bts.__file__, "status": "err", "exc": type(e).__name__, "msg": str(e)}))
"""


def run_sub(which: str, action: str, work: Path, stop: str = "none", clock: str = "none") -> subprocess.CompletedProcess:
    src = str(BASE_SRC if which == "base" else HEAD_SRC)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = src
    return subprocess.run(
        [sys.executable, "-c", CHILD_RUNNER, which, action, str(work), stop, clock],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def check_defect_repro(root: Path) -> None:
    print("=== 1. Defect reproduction on base vs HEAD ===")
    rows = [pa_row(100, "2025-04-01", 10, True), pa_row(100, "2025-04-01", 20, False)]
    for shape_name, bad_val in [
        ("null", None), ("int", 7), ("str", "bad"), ("list", [1, 2]), ("bool_true", True), ("bool_false", False)
    ]:
        for with_feed in (False, True):
            for which in ("base", "head"):
                work = root / f"repro_{shape_name}_{with_feed}_{which}"
                (work / "data/models").mkdir(parents=True)
                (work / "data/raw/2025").mkdir(parents=True)
                (work / "data/models/probable_pitcher_lookup.json").write_text(
                    json.dumps({"100": bad_val, "999": VALID_B})
                )
                if with_feed:
                    (work / "data/raw/2025/100.json").write_text(make_feed(VALID_A))
                (work / "rows.json").write_text(json.dumps(rows))
                res = run_sub(which, "features", work, clock="2027-01-01T04:59:00+00:00")
                assert res.returncode == 0, res.stderr
                payload = json.loads(res.stdout)
                if which == "base":
                    assert Path(payload["bts_file"]).resolve() == (BASE_SRC / "bts/__init__.py").resolve()
                    assert payload["status"] == "err" and payload["exc"] == "AttributeError", payload
                else:
                    assert Path(payload["bts_file"]).resolve() == (HEAD_SRC / "bts/__init__.py").resolve()
                    assert payload["status"] == "ok", payload
                    # Check status record & stderr
                    assert "dropped 1 invalid entry" in res.stderr
                    recs = list((work / "data/health_state/probable_pitcher_cache").glob("*.json"))
                    assert len(recs) == 1
                    rec = json.loads(recs[0].read_text())
                    assert rec["rescanned"] == (1 if with_feed else 0)
                    assert rec["without_feed"] == (0 if with_feed else 1)
    print("PASS: All 6 non-object shapes raise AttributeError on base (even with feed) and succeed on HEAD.")


def check_parity(root: Path) -> None:
    print("=== 1(a). Parity: real caches + synthetic caches + raw feeds + compute_all_features ===")
    real_main = FIXTURES / "probable_pitcher_lookup.json"
    real_frozen = FIXTURES / "probable_pitcher_lookup.2017-2025.json"
    assert real_main.is_file() and real_frozen.is_file()

    synthetic_payloads = {
        "nulls_and_reversed_keys": json.dumps({
            "500": {"home_tid": 52, "away_tid": 51, "home": None, "away": None},
            "501": {"away": None, "home": 22, "away_tid": None, "home_tid": None},
            "502": {"home": 33, "away": 44, "away_tid": 1, "home_tid": 2},
        }),
        "aliases_and_signs": (
            '{"00100": {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}, '
            '"99": {"home": 9, "away": 8, "home_tid": 7, "away_tid": 6}, '
            '" 100\\n": {"away": 10, "home": 20, "away_tid": 30, "home_tid": 40}, '
            '"+100": {"away": 11, "home": 21, "away_tid": 31, "home_tid": 41}, '
            '"-50": {"away": 0, "home": -1, "away_tid": 5, "home_tid": 6}, '
            '"0": {"away": null, "home": 0, "away_tid": 1, "home_tid": 2}}'
        ),
        "large_ids": json.dumps({
            str(2**80 + 7): {"away": 2**70, "home": -(2**65), "away_tid": 10**30, "home_tid": 999999999999},
            "100": VALID_A,
        }),
    }

    cases = [
        ("real_main", real_main.read_bytes()),
        ("real_frozen", real_frozen.read_bytes()),
        *[(k, v.encode("utf-8")) for k, v in synthetic_payloads.items()],
    ]

    for name, raw_bytes in cases:
        for with_uncached_feeds in (False, True):
            results = {}
            for which in ("base", "head"):
                work = root / f"parity_{name}_{with_uncached_feeds}_{which}"
                (work / "data/models").mkdir(parents=True)
                cache_file = work / "data/models/probable_pitcher_lookup.json"
                cache_file.write_bytes(raw_bytes)
                os.utime(cache_file, ns=(1_700_000_000_123_456_789, 1_700_000_000_123_456_789))
                before_mtime = cache_file.stat().st_mtime_ns
                if with_uncached_feeds:
                    for season, gpk, ent in [
                        ("2025", "99999901", VALID_A),
                        ("2025", "99999902", {"away": None, "home": 55, "away_tid": 110, "home_tid": 120}),
                        ("2026", "99999903", VALID_B),
                    ]:
                        p = work / "data/raw" / season / f"{gpk}.json"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_text(make_feed(ent))
                else:
                    (work / "data/raw").mkdir(parents=True)

                res = run_sub(which, "build", work)
                assert res.returncode == 0, f"{which} {name}: {res.stderr}"
                out = json.loads(res.stdout)
                after_bytes = cache_file.read_bytes()
                after_mtime = cache_file.stat().st_mtime_ns
                assert res.stderr == "", f"unexpected stderr on {which} {name}: {res.stderr}"
                assert not (work / "data/models/rejected").exists()
                assert not (work / "data/health_state").exists()
                assert list((work / "data/models").iterdir()) == [cache_file]
                if not with_uncached_feeds:
                    assert after_bytes == raw_bytes
                    assert after_mtime == before_mtime
                results[which] = (out["sig"], after_bytes)

            base_sig, base_bytes = results["base"]
            head_sig, head_bytes = results["head"]
            assert base_sig == head_sig, f"Lookup signature mismatch for {name} feeds={with_uncached_feeds}"
            assert base_bytes == head_bytes, f"Rewritten cache bytes mismatch for {name} feeds={with_uncached_feeds}"
            print(f"  parity OK: {name} (uncached_feeds={with_uncached_feeds}, entries={len(base_sig)})")

    # Feature-level parity on a 15-game fixture exercising bullpen rolling, starters, relievers, null probable
    dates = pd.date_range("2025-04-01", periods=15)
    rows = []
    cache_obj = {}
    for idx, dt in enumerate(dates):
        gpk = 5000 + idx
        day_s = dt.strftime("%Y-%m-%d")
        if idx == 12:
            # Uncached game supplied via raw feed
            pass
        elif idx == 5:
            cache_obj[str(gpk)] = {"away": None, "home": 20, "away_tid": 110, "home_tid": 120}
        else:
            cache_obj[str(gpk)] = {"away": 10, "home": 20, "away_tid": 110, "home_tid": 120}
        # Reliever PA (pitcher 99 != probable 10) and starter PA (pitcher 20 == probable 20)
        rows.append(pa_row(gpk, day_s, 99, True, hit=idx % 2, batter=7))
        rows.append(pa_row(gpk, day_s, 20, False, hit=(idx + 1) % 2, batter=8))

    frames = {}
    for which in ("base", "head"):
        work = root / f"feat_parity_{which}"
        (work / "data/models").mkdir(parents=True)
        (work / "data/raw/2025").mkdir(parents=True)
        (work / "data/models/probable_pitcher_lookup.json").write_text(json.dumps(cache_obj))
        (work / "data/raw/2025/5012.json").write_text(make_feed({"away": 10, "home": 20, "away_tid": 110, "home_tid": 120}))
        (work / "rows.json").write_text(json.dumps(rows))
        res = run_sub(which, "features", work)
        assert res.returncode == 0 and res.stderr == "", res.stderr
        frames[which] = pd.read_pickle(work / "features_out.pkl")

    df_base, df_head = frames["base"], frames["head"]
    assert list(df_base.columns) == list(df_head.columns)
    assert df_base.dtypes.to_dict() == df_head.dtypes.to_dict()
    pd.testing.assert_frame_equal(df_base, df_head, check_exact=True)
    assert not pd.isna(df_head.iloc[-2]["opp_bullpen_hr_30g"])
    print(f"  compute_all_features parity OK: {df_head.shape[0]} rows x {df_head.shape[1]} cols exact match")


def check_accepted_loss_and_uninvolved_games(root: Path) -> None:
    print("=== 1(b). Accepted loss vs uninvolved valid games ===")
    # Verify extra key and integral float are the accepted-loss cases when no feed exists,
    # while uninvolved valid games in the SAME cache NEVER lose their bullpen history!
    dates = pd.date_range("2025-04-01", periods=12)
    rows = [pa_row(100 + i, d.strftime("%Y-%m-%d"), 99, True, hit=i % 2) for i, d in enumerate(dates)]

    for shape_name, bad_entry in [
        ("extra_key", {**VALID_A, "extra": 1}),
        ("integral_float_pid", {**VALID_A, "away": 10.0}),
        ("integral_float_tid", {**VALID_A, "away_tid": 110.0}),
    ]:
        # Case 1: game 100 itself has extra_key/integral_float and no feed -> accepted loss (9 prior reliever games < 10 min_periods)
        cache_1 = {str(100 + i): VALID_A for i in range(12)}
        cache_1["100"] = bad_entry
        out = {}
        for which in ("base", "head"):
            work = root / f"loss_{shape_name}_{which}"
            (work / "data/models").mkdir(parents=True)
            (work / "data/raw").mkdir(parents=True)
            (work / "data/models/probable_pitcher_lookup.json").write_text(json.dumps(cache_1))
            (work / "rows.json").write_text(json.dumps(rows[:11]))  # 11 days: 10 prior days if 100 kept, 9 if 100 dropped
            res = run_sub(which, "features", work, clock="2027-01-01T04:59:00+00:00")
            assert res.returncode == 0
            df = pd.read_pickle(work / "features_out.pkl")
            out[which] = df.iloc[-1]["opp_bullpen_hr_30g"]
        assert out["base"] == 0.5 and math.isnan(out["head"]), (shape_name, out)

        # Case 2: game 9999 has bad_entry, games 100..110 are valid -> HEAD must keep exact 0.5 for games 100..110!
        cache_2 = {str(100 + i): VALID_A for i in range(11)}
        cache_2["9999"] = bad_entry
        work = root / f"uninvolved_{shape_name}"
        (work / "data/models").mkdir(parents=True)
        (work / "data/raw").mkdir(parents=True)
        (work / "data/models/probable_pitcher_lookup.json").write_text(json.dumps(cache_2))
        (work / "rows.json").write_text(json.dumps(rows[:11]))
        res = run_sub("head", "features", work, clock="2027-01-01T04:59:00+00:00")
        assert res.returncode == 0
        df = pd.read_pickle(work / "features_out.pkl")
        assert df.iloc[-1]["opp_bullpen_hr_30g"] == 0.5
        assert "dropped 1 invalid entry" in res.stderr
    print("PASS: Accepted loss matches contract (extra key, integral float), and valid games in same cache are unaffected.")


def check_ordinary_sequences(root: Path) -> None:
    print("=== 1(c). Ordinary sequences: scan+feed removal, 6 rename stops, crash between preserve & publish, same-day changes ===")
    # Sequence A: bad member 999 + scan 109 + remove feed 109 + fresh process
    dates = pd.date_range("2025-04-01", periods=11)
    rows = [pa_row(100 + i, d.strftime("%Y-%m-%d"), 99 if i < 10 else 10, True, hit=i % 2) for i, d in enumerate(dates)]
    init_cache = {str(100 + i): VALID_A for i in range(11) if 100 + i != 109}
    init_cache["999"] = "corrupt"

    work = root / "seq_feed_removal"
    (work / "data/models").mkdir(parents=True)
    (work / "data/raw/2025").mkdir(parents=True)
    cache_p = work / "data/models/probable_pitcher_lookup.json"
    cache_p.write_text(json.dumps(init_cache))
    orig_bytes = cache_p.read_bytes()
    feed_109 = work / "data/raw/2025/109.json"
    feed_109.write_text(make_feed(VALID_A))
    (work / "rows.json").write_text(json.dumps(rows))

    r1 = run_sub("head", "features", work, clock="2027-01-01T04:59:00+00:00")
    assert r1.returncode == 0
    assert pd.read_pickle(work / "features_out.pkl").iloc[-1]["opp_bullpen_hr_30g"] == 0.5
    feed_109.unlink()
    r2 = run_sub("head", "features", work, clock="2027-01-01T05:10:00+00:00")
    assert r2.returncode == 0 and r2.stderr == ""
    assert pd.read_pickle(work / "features_out.pkl").iloc[-1]["opp_bullpen_hr_30g"] == 0.5
    rewritten = json.loads(cache_p.read_bytes())
    assert "109" in rewritten and "999" not in rewritten
    rej = list((work / "data/models/rejected").glob("*.json"))
    assert len(rej) == 1 and rej[0].read_bytes() == orig_bytes

    # Sequence B: hard process exit (_exit) at all 6 rename points + retry in fresh process
    expected_states = {
        "before-evidence": (False, False, False, 91),
        "after-evidence":  (True,  False, False, 92),
        "before-status":   (True,  False, False, 91),
        "after-status":    (True,  True,  False, 92),
        "before-cache":    (True,  True,  False, 91),
        "after-cache":     (True,  True,  True,  92),
    }
    for stop, (exp_ev, exp_st, exp_ca, exp_rc) in expected_states.items():
        w = root / f"stop_{stop}"
        (w / "data/models").mkdir(parents=True)
        (w / "data/raw/2025").mkdir(parents=True)
        cp = w / "data/models/probable_pitcher_lookup.json"
        cp.write_text(json.dumps(init_cache))
        ob = cp.read_bytes()
        (w / "data/raw/2025/109.json").write_text(make_feed(VALID_A))
        res = run_sub("head", "build", w, stop=stop, clock="2027-01-01T04:59:00+00:00")
        assert res.returncode == exp_rc, (stop, res.returncode, res.stderr)
        evs = [p for p in (w / "data/models/rejected").glob("*.json")] if (w / "data/models/rejected").exists() else []
        sts = [p for p in (w / "data/health_state/probable_pitcher_cache").glob("*.json")] if (w / "data/health_state/probable_pitcher_cache").exists() else []
        cur = json.loads(cp.read_bytes())
        assert (len(evs) == 1, len(sts) == 1, "109" in cur) == (exp_ev, exp_st, exp_ca), stop
        if not exp_ca:
            assert cp.read_bytes() == ob
        if exp_ev:
            assert evs[0].read_bytes() == ob
        # Fresh process retry after hard kill (even with leftover temp files in directory!)
        retry = run_sub("head", "build", w, clock="2027-01-01T04:59:30+00:00")
        assert retry.returncode == 0, retry.stderr
        cur_after = json.loads(cp.read_bytes())
        assert "109" in cur_after and "999" not in cur_after and "100" in cur_after
        evs_after = list((w / "data/models/rejected").glob("*.json"))
        sts_after = list((w / "data/health_state/probable_pitcher_cache").glob("*.json"))
        assert len(evs_after) == 1 and evs_after[0].read_bytes() == ob
        assert len(sts_after) == 1
        rec = json.loads(sts_after[0].read_text())
        assert rec["preserved"] == evs_after[0].name

    # Sequence C: two calls on the same ET day with different outcomes (no-scan -> scan of unrelated 109)
    w_same = root / "same_day_diff"
    (w_same / "data/models").mkdir(parents=True)
    (w_same / "data/raw/2025").mkdir(parents=True)
    cp_same = w_same / "data/models/probable_pitcher_lookup.json"
    cp_same.write_text(json.dumps({"999": None, "100": VALID_A}))
    r_a = run_sub("head", "build", w_same, clock="2027-01-01T04:10:00+00:00")
    assert r_a.returncode == 0
    (w_same / "data/raw/2025/109.json").write_text(make_feed(VALID_B))
    r_b = run_sub("head", "build", w_same, clock="2027-01-01T04:20:00+00:00")
    assert r_b.returncode == 0
    sts_same = sorted((w_same / "data/health_state/probable_pitcher_cache").glob("*.json"))
    assert len(sts_same) == 2
    rows_same = [json.loads(p.read_text()) for p in sts_same]
    assert {r["preserved"] is None for r in rows_same} == {True, False}
    assert len({r["identity"] for r in rows_same}) == 2
    print("PASS: All ordinary sequences (feed removal, 6 crash points + retry with leftover temps, same-day changes) verified.")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="probe1_") as tmp:
        root = Path(tmp)
        check_defect_repro(root)
        check_parity(root)
        check_accepted_loss_and_uninvolved_games(root)
        check_ordinary_sequences(root)
    print("ALL PROBE 1 CHECKS PASSED")


if __name__ == "__main__":
    main()
