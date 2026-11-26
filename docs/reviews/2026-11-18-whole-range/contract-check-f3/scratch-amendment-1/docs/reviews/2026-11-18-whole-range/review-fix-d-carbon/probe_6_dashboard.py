#!/usr/bin/env python3
"""Item 6 probe: dashboard GET byte-identity vs b128854d, saver POST 409 while held (including pre-loaded page and arm race), zero writes."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path
from urllib.parse import urlencode

REPO = Path(__file__).resolve().parents[2]
BASE_SRC = Path(__file__).resolve().parent / "base" / "src"
PYTHON = sys.executable
PINNED_NOW = datetime(2026, 6, 18, 16, 0, 0, tzinfo=timezone.utc)

GET_ROUTES = [
    "/",
    "/health",
    "/api/live?date=2026-06-18",
    "/api/live?date=2026-06-19",
    "/api/live-html?date=2026-06-18",
    "/api/audit-progress?provider=hetzner&dir=run1",
    "/api/leaderboard",
    "/unknown-route",
]


def snapshot_tree(root: Path) -> dict[str, str]:
    snap = {}
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        if p.is_dir():
            snap[rel] = "dir"
        else:
            snap[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return snap


def populate_fixture_data(data_root: Path) -> Path:
    picks = data_root / "picks"
    acct = picks / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    (acct / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 12,
        "best_streak": 12,
        "source": "mlb_bts_profile",
        "source_date": "2026-06-18",
    }))
    from bts.saver_state import transition_saver_state
    transition_saver_state(picks, expected_prior="uninitialized", new_state="active", season=2026, source="fixture")

    # Add a historical resolved pick so render_page exercises table rendering
    (picks / "2026-06-17.json").write_text(json.dumps({
        "date": "2026-06-17",
        "result": "hit",
        "streak_after": 12,
        "pick": {
            "batter_id": 660271, "batter_name": "Shohei Ohtani", "team": "LAD",
            "opponent": "COL", "pitcher_name": "Austin Gomber", "p_game_hit": 0.812,
            "game_pk": 746123, "game_time": "2026-06-17T20:10:00Z",
        },
    }))
    hb = data_root / ".heartbeat"
    hb.write_text(json.dumps({
        "state": "sleeping",
        "timestamp": "2026-06-18T15:55:00+00:00",
        "sleeping_until": "2026-06-18T18:00:00+00:00",
    }))
    (data_root / "leaderboard").mkdir(exist_ok=True)
    (data_root / "health_state").mkdir(exist_ok=True)
    (data_root / "validation").mkdir(exist_ok=True)
    return picks


def dump_gets_worker(data_root_str: str, guard_dir_str: str) -> int:
    import bts
    import bts.web as web

    data_root = Path(data_root_str)
    if guard_dir_str:
        os.environ["BTS_TEST_DEPLOY_GUARD_DIR"] = guard_dir_str
    else:
        os.environ.pop("BTS_TEST_DEPLOY_GUARD_DIR", None)

    class PinnedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return PINNED_NOW.astimezone(tz) if tz else PINNED_NOW.replace(tzinfo=None)

    web.datetime = PinnedClock
    web.PICKS_DIR = data_root / "picks"
    web.LEADERBOARD_DIR = data_root / "leaderboard"
    web.HEALTH_STATE_DIR = data_root / "health_state"
    web.VALIDATION_DIR = data_root / "validation"
    web.HEARTBEAT_PATH = data_root / ".heartbeat"
    web.PROJECT_ROOT = data_root
    web.Handler.log_message = lambda *a, **kw: None

    srv = HTTPServer(("127.0.0.1", 0), web.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        port = srv.server_address[1]
        captured = {"bts_file": str(Path(bts.__file__).resolve()), "routes": {}}
        for route in GET_ROUTES:
            conn = HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", route)
            resp = conn.getresponse()
            body = resp.read()
            conn.close()
            captured["routes"][route] = {
                "status": resp.status,
                "content_type": resp.getheader("Content-Type"),
                "sha256": hashlib.sha256(body).hexdigest(),
                "len": len(body),
                "html_sample": body.decode("utf-8", errors="replace") if route == "/" else "",
            }
        print(json.dumps(captured))
        return 0
    finally:
        srv.shutdown()
        srv.server_close()


def run_dump_subprocess(data_root: Path, guard_dir: Path | None, use_base: bool) -> dict:
    env = dict(os.environ)
    if use_base:
        env["PYTHONPATH"] = str(BASE_SRC)
    else:
        env["PYTHONPATH"] = str(REPO / "src")
    res = subprocess.run(
        [PYTHON, __file__, "--dump-gets", str(data_root), str(guard_dir) if guard_dir else ""],
        env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def post_form(port: int, form_data: dict[str, str], host_hdr: str | None = None) -> tuple[int, bytes]:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if host_hdr:
        headers["Origin"] = f"http://{host_hdr}"
    conn.request("POST", "/saver/transition", urlencode(form_data), headers)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--dump-gets":
        return dump_gets_worker(sys.argv[2], sys.argv[3])

    work_dir = Path(tempfile.mkdtemp(prefix="p6_", dir=Path(__file__).resolve().parent))
    try:
        data_root = work_dir / "data_fixture"
        picks_dir = populate_fixture_data(data_root)
        guard_dir = work_dir / "guard"
        guard_dir.mkdir(mode=0o755)
        for lk in ("writers.lock", "transaction.lock", "state.lock"):
            (guard_dir / lk).touch(mode=0o644)

        # 1. Compare GETs across BASE (b128854d), HEAD (clear), and HEAD (held)
        base_dump = run_dump_subprocess(data_root, None, use_base=True)
        assert Path(base_dump["bts_file"]).is_relative_to(BASE_SRC.resolve()), base_dump["bts_file"]
        head_clear_dump = run_dump_subprocess(data_root, guard_dir, use_base=False)
        (guard_dir / "hold").write_text('{"state":"pending"}')
        head_held_dump = run_dump_subprocess(data_root, guard_dir, use_base=False)
        (guard_dir / "hold").unlink()

        for route in GET_ROUTES:
            b_r = {k: v for k, v in base_dump["routes"][route].items() if k != "html_sample"}
            hc_r = {k: v for k, v in head_clear_dump["routes"][route].items() if k != "html_sample"}
            hh_r = {k: v for k, v in head_held_dump["routes"][route].items() if k != "html_sample"}
            assert b_r == hc_r == hh_r, f"GET mismatch on {route}: {b_r} vs {hc_r} vs {hh_r}"

        # Extract saver form fields from page loaded BEFORE hold!
        html_before_hold = head_clear_dump["routes"]["/"]["html_sample"]
        assert 'action="/saver/transition"' in html_before_hold
        prior_m = re.search(r'name="expected_prior"\s+value="([^"]+)"', html_before_hold)
        new_m = re.search(r'name="new_state"\s+value="([^"]+)"', html_before_hold)
        assert prior_m and new_m
        preloaded_form = {"expected_prior": prior_m.group(1), "new_state": new_m.group(1)}

        # 2. Live HTTP server for saver POST tests (held states, unavailable guard, and concurrent arm race)
        import bts.web as web

        class PinnedClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return PINNED_NOW.astimezone(tz) if tz else PINNED_NOW.replace(tzinfo=None)

        orig_dt = web.datetime
        orig_picks = web.PICKS_DIR
        orig_log = web.Handler.log_message
        web.datetime = PinnedClock
        web.PICKS_DIR = picks_dir
        web.Handler.log_message = lambda *a, **kw: None
        os.environ["BTS_TEST_DEPLOY_GUARD_DIR"] = str(guard_dir)

        srv = HTTPServer(("127.0.0.1", 0), web.Handler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            port = srv.server_address[1]
            expected_409 = (409, b"deploy hold present: saver changes are refused")

            # Test every held / unavailable state over real HTTP socket with pre-loaded form!
            held_cases = {}
            for cname, setup_fn, cleanup_fn in [
                ("pending_hold", lambda: (guard_dir / "hold").write_text('{"state":"pending"}'), lambda: (guard_dir / "hold").unlink()),
                ("activating_hold", lambda: (guard_dir / "hold").write_text('{"state":"activating"}'), lambda: (guard_dir / "hold").unlink()),
                ("failed_hold", lambda: (guard_dir / "hold").write_text('{"state":"failed"}'), lambda: (guard_dir / "hold").unlink()),
                ("fifo_hold", lambda: os.mkfifo(guard_dir / "hold"), lambda: (guard_dir / "hold").unlink()),
                ("dir_hold", lambda: (guard_dir / "hold").mkdir(), lambda: (guard_dir / "hold").rmdir()),
                ("symlink_hold", lambda: (guard_dir / "hold").symlink_to(work_dir / "dangling"), lambda: (guard_dir / "hold").unlink()),
                ("missing_writers_lock", lambda: (guard_dir / "writers.lock").unlink(), lambda: (guard_dir / "writers.lock").touch(mode=0o644)),
                ("world_writable_lock", lambda: os.chmod(guard_dir / "writers.lock", 0o666), lambda: os.chmod(guard_dir / "writers.lock", 0o644)),
                ("missing_guard_dir", lambda: os.environ.__setitem__("BTS_TEST_DEPLOY_GUARD_DIR", str(work_dir / "absent")),
                 lambda: os.environ.__setitem__("BTS_TEST_DEPLOY_GUARD_DIR", str(guard_dir))),
            ]:
                setup_fn()
                try:
                    before_snap = snapshot_tree(data_root)
                    status, body = post_form(port, preloaded_form, f"127.0.0.1:{port}")
                    after_snap = snapshot_tree(data_root)
                    assert (status, body) == expected_409, (cname, status, body)
                    assert before_snap == after_snap, f"{cname} wrote to disk!"
                    held_cases[cname] = {"status": status, "body": body.decode()}
                finally:
                    cleanup_fn()

            # 3. Real threaded race: POST enters saver_transition_response holding SH lock -> concurrent arm EX lock blocked!
            entered_evt = threading.Event()
            release_evt = threading.Event()
            orig_saver_fn = web.saver_transition_response

            def slow_saver(*a, **kw):
                entered_evt.set()
                assert release_evt.wait(timeout=5)
                return orig_saver_fn(*a, **kw)

            web.saver_transition_response = slow_saver
            post_outcome = []
            t_post = threading.Thread(target=lambda: post_outcome.append(post_form(port, preloaded_form, f"127.0.0.1:{port}")))
            t_post.start()
            assert entered_evt.wait(timeout=5)
            # While POST is mid-transition, try non-blocking LOCK_EX on writers.lock (exact arm admission step)
            with (guard_dir / "writers.lock").open("rb") as lf:
                try:
                    fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    arm_blocked_during_post = False
                except BlockingIOError:
                    arm_blocked_during_post = True
            release_evt.set()
            t_post.join(timeout=5)
            web.saver_transition_response = orig_saver_fn
            assert arm_blocked_during_post is True
            assert post_outcome[0][0] == 303

            # Reverse race: arm holds LOCK_EX on writers.lock before writing hold -> POST arrives and gets 409 immediately!
            before_race2 = snapshot_tree(data_root)
            with (guard_dir / "writers.lock").open("rb") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                r2_status, r2_body = post_form(port, {"expected_prior": "used", "new_state": "active"}, f"127.0.0.1:{port}")
            assert (r2_status, r2_body) == expected_409
            assert snapshot_tree(data_root) == before_race2
        finally:
            srv.shutdown()
            srv.server_close()
            web.datetime = orig_dt
            web.PICKS_DIR = orig_picks
            web.Handler.log_message = orig_log
            os.environ.pop("BTS_TEST_DEPLOY_GUARD_DIR", None)

        report = {
            "base_bts_file": base_dump["bts_file"],
            "get_routes_byte_identical": len(GET_ROUTES),
            "held_and_unavailable_409_cases": len(held_cases),
            "arm_blocked_while_post_in_flight": arm_blocked_during_post,
            "post_refused_409_while_arm_holds_ex_lock": True,
        }
        out_path = Path(__file__).resolve().parent / "probe_6_dashboard.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
