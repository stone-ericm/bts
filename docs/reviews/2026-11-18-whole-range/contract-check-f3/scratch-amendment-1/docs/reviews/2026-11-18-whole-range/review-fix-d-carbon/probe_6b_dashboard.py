#!/usr/bin/env python3
"""Item 6 probe (6b, socketless for sandbox): dashboard GET byte-identity vs b128854d, saver POST 409 while held (pre-loaded page + arm race), zero writes."""
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
from datetime import datetime, timezone
from io import BytesIO
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
        snap[rel] = "dir" if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
    return snap


def populate_fixture_data(data_root: Path) -> Path:
    picks = data_root / "picks"
    acct = picks / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    (acct / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 12, "best_streak": 12,
        "source": "mlb_bts_profile", "source_date": "2026-06-18",
    }))
    from bts.saver_state import transition_saver_state
    transition_saver_state(picks, expected_prior="uninitialized", new_state="active", season=2026, source="fixture")
    (picks / "2026-06-17.json").write_text(json.dumps({
        "date": "2026-06-17", "result": "hit", "streak_after": 12,
        "pick": {
            "batter_id": 660271, "batter_name": "Shohei Ohtani", "team": "LAD",
            "opponent": "COL", "pitcher_name": "Austin Gomber", "p_game_hit": 0.812,
            "game_pk": 746123, "game_time": "2026-06-17T20:10:00Z",
        },
    }))
    (data_root / ".heartbeat").write_text(json.dumps({
        "state": "sleeping",
        "timestamp": "2026-06-18T15:55:00+00:00",
        "sleeping_until": "2026-06-18T18:00:00+00:00",
    }))
    (data_root / "leaderboard").mkdir(exist_ok=True)
    (data_root / "health_state").mkdir(exist_ok=True)
    (data_root / "validation").mkdir(exist_ok=True)
    return picks


def invoke_handler_get(web, route: str) -> tuple[int, dict[str, str], bytes]:
    req = object.__new__(web.Handler)
    req.rfile = BytesIO(b"")
    req.wfile = BytesIO()
    req.path = route
    req.headers = {"Host": "localhost:3003"}
    req.client_address = ("127.0.0.1", 12345)
    status = []
    headers = {}
    req.send_response = status.append
    req.send_header = lambda k, v: headers.__setitem__(k, v)
    req.end_headers = lambda: None
    req.do_GET()
    return status[0], headers, req.wfile.getvalue()


def invoke_handler_post(web, form_data: dict[str, str], path: str = "/saver/transition") -> tuple[int, dict[str, str], bytes]:
    encoded = urlencode(form_data).encode()
    req = object.__new__(web.Handler)
    req.rfile = BytesIO(encoded)
    req.wfile = BytesIO()
    req.path = path
    req.headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Content-Length": str(len(encoded)),
        "Host": "localhost:3003",
        "Origin": "http://localhost:3003",
    }
    req.client_address = ("127.0.0.1", 12345)
    status = []
    headers = {}
    req.send_response = status.append
    req.send_header = lambda k, v: headers.__setitem__(k, v)
    req.end_headers = lambda: None
    req.do_POST()
    return status[0], headers, req.wfile.getvalue()


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

    captured = {"bts_file": str(Path(bts.__file__).resolve()), "routes": {}}
    for route in GET_ROUTES:
        code, hdrs, body = invoke_handler_get(web, route)
        captured["routes"][route] = {
            "status": code,
            "headers": hdrs,
            "sha256": hashlib.sha256(body).hexdigest(),
            "len": len(body),
            "html": body.decode("utf-8", errors="replace") if route == "/" else "",
        }
    print(json.dumps(captured))
    return 0


def run_dump_subprocess(data_root: Path, guard_dir: Path | None, use_base: bool) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BASE_SRC) if use_base else str(REPO / "src")
    res = subprocess.run(
        [PYTHON, __file__, "--dump-gets", str(data_root), str(guard_dir) if guard_dir else ""],
        env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--dump-gets":
        return dump_gets_worker(sys.argv[2], sys.argv[3])

    work_dir = Path(tempfile.mkdtemp(prefix="p6b_", dir=Path(__file__).resolve().parent))
    try:
        data_root = work_dir / "data_fixture"
        picks_dir = populate_fixture_data(data_root)
        guard_dir = work_dir / "guard"
        guard_dir.mkdir(mode=0o755)
        for lk in ("writers.lock", "transaction.lock", "state.lock"):
            (guard_dir / lk).touch(mode=0o644)

        base_dump = run_dump_subprocess(data_root, None, use_base=True)
        assert Path(base_dump["bts_file"]).is_relative_to(BASE_SRC.resolve()), base_dump["bts_file"]
        head_clear_dump = run_dump_subprocess(data_root, guard_dir, use_base=False)
        (guard_dir / "hold").write_text('{"state":"pending"}')
        head_held_dump = run_dump_subprocess(data_root, guard_dir, use_base=False)
        (guard_dir / "hold").unlink()

        for route in GET_ROUTES:
            b_r = {k: v for k, v in base_dump["routes"][route].items() if k != "html"}
            hc_r = {k: v for k, v in head_clear_dump["routes"][route].items() if k != "html"}
            hh_r = {k: v for k, v in head_held_dump["routes"][route].items() if k != "html"}
            assert b_r == hc_r == hh_r, f"GET mismatch on {route}: {b_r} vs {hc_r} vs {hh_r}"

        html_before_hold = head_clear_dump["routes"]["/"]["html"]
        assert 'action="/saver/transition"' in html_before_hold
        prior_m = re.search(r'name="expected_prior"\s+value="([^"]+)"', html_before_hold)
        new_m = re.search(r'name="new_state"\s+value="([^"]+)"', html_before_hold)
        assert prior_m and new_m
        preloaded_form = {"expected_prior": prior_m.group(1), "new_state": new_m.group(1)}

        import bts.web as web

        class PinnedClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return PINNED_NOW.astimezone(tz) if tz else PINNED_NOW.replace(tzinfo=None)

        orig_dt, orig_picks = web.datetime, web.PICKS_DIR
        web.datetime = PinnedClock
        web.PICKS_DIR = picks_dir
        os.environ["BTS_TEST_DEPLOY_GUARD_DIR"] = str(guard_dir)
        try:
            expected_409 = (409, b"deploy hold present: saver changes are refused")
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
                    code, _, body = invoke_handler_post(web, preloaded_form)
                    after_snap = snapshot_tree(data_root)
                    assert (code, body) == expected_409, (cname, code, body)
                    assert before_snap == after_snap, f"{cname} wrote to disk!"
                    held_cases[cname] = {"status": code, "body": body.decode()}
                finally:
                    cleanup_fn()

            # Real threaded flock race between saver POST and arm!
            entered_evt = threading.Event()
            release_evt = threading.Event()
            orig_saver_fn = web.saver_transition_response

            def slow_saver(*a, **kw):
                entered_evt.set()
                assert release_evt.wait(timeout=5)
                return orig_saver_fn(*a, **kw)

            web.saver_transition_response = slow_saver
            post_outcome = []
            t_post = threading.Thread(target=lambda: post_outcome.append(invoke_handler_post(web, preloaded_form)))
            t_post.start()
            assert entered_evt.wait(timeout=5)
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

            before_race2 = snapshot_tree(data_root)
            with (guard_dir / "writers.lock").open("rb") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                r2_code, _, r2_body = invoke_handler_post(web, {"expected_prior": "used", "new_state": "active"})
            assert (r2_code, r2_body) == expected_409
            assert snapshot_tree(data_root) == before_race2
        finally:
            web.datetime = orig_dt
            web.PICKS_DIR = orig_picks
            os.environ.pop("BTS_TEST_DEPLOY_GUARD_DIR", None)

        report = {
            "base_bts_file": base_dump["bts_file"],
            "get_routes_byte_identical": len(GET_ROUTES),
            "held_and_unavailable_409_cases": len(held_cases),
            "arm_blocked_while_post_in_flight": arm_blocked_during_post,
            "post_refused_409_while_arm_holds_ex_lock": True,
        }
        out_path = Path(__file__).resolve().parent / "probe_6b_dashboard.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
