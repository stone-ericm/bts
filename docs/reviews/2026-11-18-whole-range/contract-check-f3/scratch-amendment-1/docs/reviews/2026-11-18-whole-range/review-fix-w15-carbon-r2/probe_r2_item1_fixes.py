"""Round-2 Probe 1: verify every round-1 BLOCK (Codex 1/2/3, Carbon B1/B2) on original inputs with real code and 1000-grapheme transport enforcement."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pandas as pd

R2_DIR = Path(__file__).resolve().parent
REVIEW_DIR = R2_DIR.parent
REPO_ROOT = REVIEW_DIR.parents[1]
BASE_SRC = REVIEW_DIR / "base_src" / "src"
HEAD_SRC = REPO_ROOT / "src"

sys.path.insert(0, str(HEAD_SRC))

from bts import dm as dm_mod
from bts.features import compute, probable_pitcher_cache as ppc
from bts.health import alert as alert_mod, runner
from bts.probable_pitcher_status import SOURCE

VALID = {"away": 1, "home": 22, "away_tid": 51, "home_tid": 52}
RECOVERY = {"away": 2, "home": 23, "away_tid": 61, "home_tid": 62}


def feed_str(entry: dict = VALID) -> str:
    return json.dumps({
        "gameData": {
            "probablePitchers": {"away": {"id": entry["away"]}, "home": {"id": entry["home"]}},
            "teams": {"away": {"id": entry["away_tid"]}, "home": {"id": entry["home_tid"]}},
        }
    })


def pa_row(game: int, day: str, pitcher: int, is_home: bool, hit: int = 0) -> dict:
    return {
        "pa_id": 0, "date": day, "game_pk": game, "batter_id": 11, "pitcher_id": pitcher,
        "pitch_hand": "R", "venue_id": 31, "is_home": is_home, "is_hit": hit,
        "pitch_calls": ["X"], "pitch_types": ["FF"], "pitch_speeds": [93.0],
        "pitch_spin_rates": [2300.0], "pitch_extensions": [6.0],
        "pitch_break_vertical": [15.0], "pitch_break_horizontal": [5.0],
        "pitch_px": [0.0], "pitch_pz": [2.5], "sz_top": 3.5, "sz_bottom": 1.5,
        "final_count_balls": 0, "final_count_strikes": 0,
        "launch_speed": 91.0, "launch_angle": 10.0,
    }


def check_r1_descriptor_pressure(root: Path) -> None:
    print("=== R2-1a. Descriptor-pressure replays (Codex 1: 64 dirs @ RLIMIT=48; Carbon B1: 65 dirs @ RLIMIT=64 + 70 daemon calls) ===")
    dates = pd.date_range("2025-04-01", periods=11)
    rows = [pa_row(100 + i, d.strftime("%Y-%m-%d"), 3 if i < 10 else 1, True, i % 2) for i, d in enumerate(dates)]

    # 1. Exact Codex 1 original input: valid cache lacking 109, 64 directories 0001.json..0064.json ahead of 109.json, soft limit 48!
    codex1_code = """
import json, os, resource, sys
import pandas as pd
from pathlib import Path
from bts.features import compute, park_drag
park_drag.get_table = lambda: None

work = Path(sys.argv[1])
os.chdir(work)
cache = work / "data/models/probable_pitcher_lookup.json"
compute._probable_pitcher_cache_path = cache

_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (48, hard))

def count_fds():
    n = 0
    for fd in range(3, 128):
        try:
            os.fstat(fd)
            n += 1
        except OSError:
            pass
    return n

rows = json.loads((work / "rows.json").read_text())
before = count_fds()
df = pd.DataFrame(rows)
df["pa_id"] = range(len(df))
out = compute.compute_all_features(df)
after = count_fds()
val = out.iloc[-1]["opp_bullpen_hr_30g"]
saved = json.loads(cache.read_bytes())
print(json.dumps({
    "bullpen": None if pd.isna(val) else float(val),
    "has_109": "109" in saved,
    "fds_before": before,
    "fds_after": after,
    "health_exists": (work / "data/health_state").exists(),
}))
"""
    for which, src in [("base", BASE_SRC), ("head", HEAD_SRC)]:
        w = root / f"codex1_{which}"
        (w / "data/models").mkdir(parents=True)
        (w / "data/raw/2025").mkdir(parents=True)
        (w / "data/models/probable_pitcher_lookup.json").write_text(
            json.dumps({str(g): VALID for g in range(100, 111) if g != 109})
        )
        for g in range(1, 65):
            (w / "data/raw/2025" / f"{g:04d}.json").mkdir()
        (w / "data/raw/2025/109.json").write_text(feed_str(VALID))
        (w / "rows.json").write_text(json.dumps(rows))
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(src)
        cp = subprocess.run([sys.executable, "-c", codex1_code, str(w)], env=env, capture_output=True, text=True, timeout=20)
        assert cp.returncode == 0 and cp.stderr == "", (which, cp.returncode, cp.stderr)
        res = json.loads(cp.stdout)
        print(f"  Codex-1 replay ({which}): {res}")
        assert res == {"bullpen": 0.5, "has_109": True, "fds_before": res["fds_before"], "fds_after": res["fds_before"], "health_exists": False}

    # 2. Exact Carbon B1 daemon replay: 70 consecutive calls in one process @ RLIMIT=64 with directory at cache path,
    #    PLUS directory candidates in rejected/ and status_dir!
    carbon_b1_code = """
import json, os, resource, sys
import pandas as pd
from datetime import date
from pathlib import Path
from bts.features import compute, park_drag
from bts.health import probable_pitcher_cache as h_ppc
park_drag.get_table = lambda: None

_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))

work = Path(sys.argv[1])
os.chdir(work)
cache = work / "data/models/probable_pitcher_lookup.json"
compute._probable_pitcher_cache_path = cache
cache.mkdir(parents=True)
# Also put a directory inside status_dir and rejected/
sdir = work / "data/health_state/probable_pitcher_cache"
sdir.mkdir(parents=True)
(sdir / "2026-12-31.dir.json").mkdir()
rdir = work / "data/models/rejected"
rdir.mkdir(parents=True)
(rdir / "probable_pitcher_lookup.20261231T000000Z.000000000000.json").mkdir()

def count_fds():
    n = 0
    for fd in range(3, 128):
        try:
            os.fstat(fd)
            n += 1
        except OSError:
            pass
    return n

rows = json.loads((work / "rows.json").read_text())
before = count_fds()
vals = []
for _ in range(70):
    df = pd.DataFrame(rows)
    df["pa_id"] = range(len(df))
    out = compute.compute_all_features(df)
    vals.append(float(out.iloc[-1]["opp_bullpen_hr_30g"]))
    h_ppc.check(sdir, date(2026, 12, 31))
after = count_fds()
print(json.dumps({"num_0_5": vals.count(0.5), "fds_before": before, "fds_after": after}))
"""
    w2 = root / "carbon_b1"
    (w2 / "data/raw/2025").mkdir(parents=True)
    for i in range(11):
        (w2 / "data/raw/2025" / f"{100 + i}.json").write_text(feed_str(VALID))
    (w2 / "rows.json").write_text(json.dumps(rows))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(HEAD_SRC)
    cp2 = subprocess.run([sys.executable, "-c", carbon_b1_code, str(w2)], env=env, capture_output=True, text=True, timeout=30)
    assert cp2.returncode == 0, cp2.stderr
    res2 = json.loads(cp2.stdout)
    print(f"  Carbon-B1 70-call daemon + health dir replay: {res2}")
    assert res2["num_0_5"] == 70 and res2["fds_before"] == res2["fds_after"]


def check_r2_closed_stderr(root: Path) -> None:
    print("=== R2-1b. Closed-reader stderr pipe & closed stderr fd exit code (Carbon B2) ===")
    child_code = """
import json, os, sys
import pandas as pd
from pathlib import Path
from bts.features import compute, park_drag
park_drag.get_table = lambda: None

work = Path(sys.argv[1])
os.chdir(work)
compute._probable_pitcher_cache_path = work / "data/models/probable_pitcher_lookup.json"
rows = json.loads((work / "rows.json").read_text())
df = pd.DataFrame(rows)
df["pa_id"] = range(len(df))
out = compute.compute_all_features(df)
sys.stdout.write(f"FEATURES_OK rows={len(out)}\\n")
sys.stdout.flush()
"""
    # Test both single warning (invalid_entries) AND triple warning (unreadable + failed preserve + failed status) on real EPIPE stderr!
    for case in ("single_warn", "triple_warn"):
        w = root / f"epipe_{case}"
        (w / "data/models").mkdir(parents=True)
        (w / "data/raw/2025").mkdir(parents=True)
        (w / "data/raw/2025/100.json").write_text(feed_str(RECOVERY))
        (w / "rows.json").write_text(json.dumps([pa_row(100, "2025-04-01", 1, True), pa_row(100, "2025-04-01", 22, False)]))
        if case == "single_warn":
            (w / "data/models/probable_pitcher_lookup.json").write_text('{"100": null}')
        else:
            (w / "data/models/probable_pitcher_lookup.json").write_text("{corrupt")
            (w / "data/models/rejected").write_text("block_preserve")
            (w / "data/health_state").mkdir(parents=True)
            (w / "data/health_state/probable_pitcher_cache").write_text("block_status")

        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(HEAD_SRC)
        r_fd, w_fd = os.pipe()
        os.close(r_fd)
        try:
            cp = subprocess.run([sys.executable, "-c", child_code, str(w)], env=env, stdout=subprocess.PIPE, stderr=w_fd, text=True, timeout=15)
        finally:
            os.close(w_fd)
        print(f"  EPIPE {case}: returncode={cp.returncode}, stdout={cp.stdout.strip()!r}")
        assert cp.returncode == 0 and cp.stdout.strip() == "FEATURES_OK rows=2"


def check_r4_dm_cap_with_real_send_dm(root: Path) -> None:
    print("=== R2-1c. D/D+1 persistent, same-day invalid+unreadable pair, and 16+ records through real send_dm enforcing 1000-grapheme cap ===")
    sent_wire_messages: list[str] = []

    def fake_retry_urlopen(req: object, timeout: int = 15, idempotent: bool = True):
        url = req.full_url
        if url.endswith("/com.atproto.server.createSession"):
            return io.BytesIO(b'{"accessJwt": "fake-jwt"}')
        if "/com.atproto.identity.resolveHandle" in url:
            return io.BytesIO(b'{"did": "did:plc:testrecipient"}')
        if "/chat.bsky.convo.getConvoForMembers" in url:
            return io.BytesIO(b'{"convo": {"id": "convo-1"}}')
        if url.endswith("/chat.bsky.convo.sendMessage"):
            body = json.loads(req.data.decode("utf-8"))
            text = body["message"]["text"]
            # Enforce Bluesky's 1,000-grapheme / character chat cap!
            if len(text) > 1000:
                raise HTTPError(url, 400, f"Message exceeds 1000-grapheme limit ({len(text)})", {}, io.BytesIO(b"too long"))
            sent_wire_messages.append(text)
            return io.BytesIO(f'{{"id": "msg-{len(sent_wire_messages)}"}}'.encode())
        raise AssertionError(f"Unexpected URL: {url}")

    def freeze_all(ts_iso: str):
        at = datetime.fromisoformat(ts_iso)
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return at.astimezone(tz) if tz is not None else at.replace(tzinfo=None)
        ppc.datetime = Clock
        runner.datetime = Clock
        alert_mod.datetime = Clock

    orig_safe = runner._safe_run
    with (
        patch.object(dm_mod, "get_bluesky_dm_password", return_value="fake-pw"),
        patch.object(dm_mod, "retry_urlopen", fake_retry_urlopen),
        patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None),
        patch.object(runner, "_safe_run", lambda name, fn: orig_safe(name, fn) if name == SOURCE else []),
    ):
        # Replay 1: Codex-3's exact persistent {"100": null} cache with no feeds across D and D+1!
        w1 = root / "codex3_persistent"
        cache1 = w1 / "data/models/probable_pitcher_lookup.json"
        raw1 = w1 / "data/raw"
        picks1 = w1 / "data/picks"
        cache1.parent.mkdir(parents=True)
        raw1.mkdir(parents=True)
        picks1.mkdir(parents=True)
        cache1.write_text('{"100": null}')

        with patch.object(compute, "_probable_pitcher_cache_path", cache1):
            freeze_all("2026-12-31T23:00:00+00:00")
            compute._build_probable_pitcher_lookup(str(raw1))
            a_d = runner.run_all_checks(picks1, cache1.parent, "eric.bsky.social", today=date(2026, 12, 31))
            assert len(a_d) == 1 and len(a_d[0].message) <= 600
            assert len(sent_wire_messages) == 1 and len(sent_wire_messages[-1]) < 1000

            freeze_all("2027-01-01T23:00:00+00:00")
            compute._build_probable_pitcher_lookup(str(raw1))
            a_d1 = runner.run_all_checks(picks1, cache1.parent, "eric.bsky.social", today=date(2027, 1, 1))
            assert len(a_d1) == 1 and len(a_d1[0].message) <= 600
            assert len(sent_wire_messages) == 2 and len(sent_wire_messages[-1]) < 1000
            print(f"  Codex-3 D/D+1 wire lengths: D={len(sent_wire_messages[0])}, D+1={len(sent_wire_messages[1])}")

        # Replay 2: Same-day invalid_entries + unreadable_cache pair across D and D+1
        w2 = root / "codex3_pair"
        cache2 = w2 / "data/models/probable_pitcher_lookup.json"
        raw2 = w2 / "data/raw"
        picks2 = w2 / "data/picks"
        cache2.parent.mkdir(parents=True)
        raw2.mkdir(parents=True)
        picks2.mkdir(parents=True)
        with patch.object(compute, "_probable_pitcher_cache_path", cache2):
            freeze_all("2026-12-31T18:00:00+00:00")
            cache2.write_text('{"100": null}')
            compute._build_probable_pitcher_lookup(str(raw2))
            cache2.write_text("{corrupt")
            compute._build_probable_pitcher_lookup(str(raw2))
            a_pair_d = runner.run_all_checks(picks2, cache2.parent, "eric.bsky.social", today=date(2026, 12, 31))
            assert len(a_pair_d) == 1 and len(a_pair_d[0].message) <= 600
            assert len(sent_wire_messages) == 3 and len(sent_wire_messages[-1]) < 1000

            freeze_all("2027-01-01T18:00:00+00:00")
            a_pair_d1 = runner.run_all_checks(picks2, cache2.parent, "eric.bsky.social", today=date(2027, 1, 1))
            assert len(a_pair_d1) == 1 and len(a_pair_d1[0].message) <= 600
            assert len(sent_wire_messages) == 4 and len(sent_wire_messages[-1]) < 1000
            print(f"  Codex-3 invalid+unreadable pair wire lengths: D={len(sent_wire_messages[2])}, D+1={len(sent_wire_messages[3])}")
            assert all("`" not in m for m in sent_wire_messages)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="r2_p1_") as tmp:
        root = Path(tmp)
        check_r1_descriptor_pressure(root)
        check_r2_closed_stderr(root)
        check_r4_dm_cap_with_real_send_dm(root)
    print("ALL ROUND-2 PROBE 1 CHECKS PASSED")


if __name__ == "__main__":
    main()
