"""Adversarial probe for CS6b-1 items 1, 2, and 6:
1. Full-day base (1bf518a2) vs branch (0e0c49a3) comparison through real run_day (with EOD health).
2. Exhaustive fault injection (7 exception classes x 15 sites + status hazards + outer interrupt non-swallow)
   and real bts.dm.send_dm worst-case HTTP stall timing at the 10-minute margin.
6. Row 5 diff, function-body byte identity, and AST live-variable analysis of run_day.
"""
from __future__ import annotations

import ast
import contextlib
import datetime as dt
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import Mock
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE_SRC = HERE / "base_tree" / "src"
BRANCH_SRC = REPO / "src"
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
REAL_DT = dt.datetime
REAL_DATE = dt.date
DAY = "2027-07-15"


def install_network_guard() -> None:
    def audit(event: str, args: tuple) -> None:
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo",
                     "socket.gethostbyname", "socket.gethostbyaddr", "socket.sendto"}:
            raise AssertionError(f"forbidden network call: {event} {args!r}")

    sys.addaudithook(audit)


def check_row5_and_scope() -> dict:
    diff_names = subprocess.check_output(
        ["git", "-C", str(REPO), "diff", "--name-only", "1bf518a2", "HEAD"],
        text=True,
    ).splitlines()
    assert diff_names == ["src/bts/scheduler.py", "tests/test_morning_state_cs6b1.py"], diff_names

    base_text = (BASE_SRC / "bts" / "scheduler.py").read_text(encoding="utf-8")
    branch_text = (BRANCH_SRC / "bts" / "scheduler.py").read_text(encoding="utf-8")
    base_tree = ast.parse(base_text)
    branch_tree = ast.parse(branch_text)

    def func_map(tree: ast.Module, text: str) -> dict[str, str]:
        lines = text.splitlines(keepends=True)
        out: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out[node.name] = "".join(lines[node.lineno - 1:node.end_lineno])
        return out

    base_funcs = func_map(base_tree, base_text)
    branch_funcs = func_map(branch_tree, branch_text)

    named_unchanged = [
        "run_single_check",
        "_deliver_and_lock_pick",
        "_alert_contest_state_failure",
        "_warn_stale_contest_reading",
        "resolve_fallback_deadline_min",
        "_fallback_min_with_floor",
        "plan_fallback_action",
        "_should_defer_at_fallback",
        "_refresh_is_genuine_skip",
        "_refresh_is_genuine_pick",
        "_refresh_pick_at_fallback_decision",
        "_refresh_pick_at_fallback",
        "_defer_pick_at_fallback",
        "_capture_fallback_skip",
    ]
    for name in named_unchanged:
        assert base_funcs[name] == branch_funcs[name], f"function {name} changed!"

    changed_top = sorted(
        name for name in set(base_funcs) | set(branch_funcs)
        if base_funcs.get(name) != branch_funcs.get(name)
    )
    assert changed_top == [
        "_morning_state_check",
        "_print_morning_check",
        "_report_morning_check_failure",
        "run_day",
    ], changed_top

    # Check untouched modules byte-for-byte
    untouched_rel = [
        "bts/strategy.py",
        "bts/contest_state.py",
        "bts/season_contract.py",
        "bts/picks.py",
        "bts/daily_decision.py",
        "bts/contest_fetch.py",
        "bts/cli.py",
        "bts/dm.py",
        "bts/posting.py",
        "bts/health/alert.py",
        "bts/health/runner.py",
        "bts/health/contest_state.py",
        "bts/health/attention.py",
    ]
    for rel in untouched_rel:
        assert (BASE_SRC / rel).read_bytes() == (BRANCH_SRC / rel).read_bytes(), f"{rel} changed!"

    # AST analysis of `now` and `first_check_et` in branch `run_day`
    run_day_node = next(
        n for n in branch_tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_day"
    )
    morning_try_idx = None
    for idx, stmt in enumerate(run_day_node.body):
        if isinstance(stmt, ast.Try) and any(
            isinstance(sub, ast.Name) and sub.id == "first_check_et"
            for sub in ast.walk(stmt)
        ):
            morning_try_idx = idx
            break
    assert morning_try_idx is not None
    after_stmts = run_day_node.body[morning_try_idx + 1:]

    first_check_uses_after = [
        (sub.lineno, type(sub.ctx).__name__)
        for stmt in after_stmts
        for sub in ast.walk(stmt)
        if isinstance(sub, ast.Name) and sub.id == "first_check_et"
    ]
    assert first_check_uses_after == [], first_check_uses_after

    nested_defs = [
        sub.name
        for stmt in after_stmts
        for sub in ast.walk(stmt)
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
    ]

    # Verify definite-assignment of `now` on all paths after morning_try_idx:
    # 1) Inside `for run_idx, run_info in enumerate(runs):`, stmt[0] assigns `target`, stmt[1] assigns `now`
    loop_stmt = after_stmts[1]
    assert isinstance(loop_stmt, ast.For)
    assert isinstance(loop_stmt.body[1], ast.Assign)
    assert [t.id for t in loop_stmt.body[1].targets] == ["now"]
    assert not any(
        isinstance(sub, ast.Name) and sub.id == "now"
        for sub in ast.walk(loop_stmt.body[0])
    )
    assert not any(
        isinstance(sub, ast.Name) and sub.id == "now"
        for sub in ast.walk(loop_stmt.iter)
    )

    # 2) After loop_stmt (even when `runs` is empty -> 0 loop iterations), find every top-level statement using `now`
    post_loop_now_stmts = []
    for stmt in after_stmts[2:]:
        names = [
            (sub.lineno, type(sub.ctx).__name__)
            for sub in ast.walk(stmt)
            if isinstance(sub, ast.Name) and sub.id == "now"
        ]
        if names:
            post_loop_now_stmts.append({"stmt_line": stmt.lineno, "type": type(stmt).__name__, "occurrences": names})
            # In each If block that touches `now`, verify the first occurrence is a Store before any Load!
            assert names[0][1] == "Store", f"Load before Store in post-loop stmt at {stmt.lineno}: {names}"

    return {
        "diff_files": diff_names,
        "changed_scheduler_defs": changed_top,
        "named_unchanged_verified": len(named_unchanged),
        "untouched_modules_verified": len(untouched_rel),
        "first_check_et_uses_after_block": first_check_uses_after,
        "nested_defs_after_block": nested_defs,
        "post_loop_now_definite_rebind": post_loop_now_stmts,
    }


def child_day_scenario(mode: str, scenario: str, workdir: Path) -> None:
    src = BASE_SRC if mode == "base" else BRANCH_SRC
    sys.path.insert(0, str(src))
    install_network_guard()

    import pandas as pd
    from bts import (
        contest_state, daily_decision, dm, heartbeat, orchestrator,
        picks, posting, scheduler, slate, strategy,
    )
    from bts.health import alert, memory_growth, post_failure, runner, unit_drift

    start_dt = REAL_DT(2027, 7, 15, 10, 0, 0, tzinfo=ET)
    if scenario == "thin_slate_1305":
        start_dt = REAL_DT(2027, 7, 15, 10, 0, 30, tzinfo=ET)
    elif scenario == "restart_after_delivery":
        start_dt = REAL_DT(2027, 7, 15, 12, 0, 0, tzinfo=ET)
    elif scenario == "restart_midday_before_delivery":
        start_dt = REAL_DT(2027, 7, 15, 14, 0, 0, tzinfo=ET)

    now_box = [start_dt]

    class PinnedDatetime(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            cur = now_box[0]
            return cur.astimezone(tz) if tz is not None else cur.replace(tzinfo=None)

        @classmethod
        def utcnow(cls):
            return now_box[0].astimezone(dt.timezone.utc).replace(tzinfo=None)

    for mod in (contest_state, daily_decision, picks, strategy, scheduler, alert, heartbeat, slate, post_failure):
        if hasattr(mod, "datetime"):
            mod.datetime = PinnedDatetime
    scheduler._now_et = lambda: now_box[0]
    posting._now_et = lambda: now_box[0]
    scheduler.time = Mock(monotonic=lambda: 0.0, sleep=lambda s: None)
    picks._git_head_sha = lambda *_: "b" * 40

    # Deterministic EOD system checks inside sandbox
    runner.get_self_pid = lambda: None
    runner.read_systemd_nrestarts = lambda *_: 0
    unit_drift.check = lambda **_: []
    memory_growth.check = lambda **_: []

    picks_dir = workdir / "data" / "picks"
    models_dir = workdir / "data" / "models"
    hb_path = workdir / "data" / "heartbeat.json"
    picks_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Seed contest state per scenario
    def write_account(name: str, payload: dict) -> None:
        p = picks_dir / "account_state" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2) + "\n")

    if scenario in {"normal_dated", "mdp_skip", "thin_slate_1305", "restart_after_delivery"}:
        write_account("contest_streak.manual.json", {
            "schema_version": "bts_contest_streak_manual_v3",
            "season": 2027,
            "effective_date": DAY,
            "active_streak": 30,
            "best_streak": 30,
            "saver_state": "used",
            "recorded_at": REAL_DT(2027, 7, 15, 7, 0, tzinfo=ET).isoformat(),
            "expires_at": f"{DAY}T23:59:59.999999-04:00",
            "source": "manual_cli",
        })
    elif scenario in {"normal_fresh_auto", "normal_stale_auto"}:
        age_h = 2.0 if scenario == "normal_fresh_auto" else 20.0
        rec_at = (start_dt - dt.timedelta(hours=age_h)).astimezone(UTC).isoformat().replace("+00:00", "Z")
        write_account("season_contract.json", {
            "schema_version": "bts_season_contract_v1",
            "season": 2027,
            "status": "automatic",
            "activation_id": "a" * 32,
            "switched_at": REAL_DT(2027, 3, 24, 9, 0, tzinfo=ET).isoformat(),
            "revoked_at": None,
            "note": None,
        })
        write_account("contest_streak.json", {
            "schema_version": "bts_contest_streak_auto_v2",
            "season": 2027,
            "activation_id": "a" * 32,
            "active_streak": 30,
            "best_streak": 30,
            "source": "mlb_bts_profile",
            "source_date": "2027-07-14",
            "recorded_at": rec_at,
        })
        write_account("saver_state.json", {
            "season": 2027,
            "state": "used",
            "source": "cli",
            "updated_at": REAL_DT(2027, 7, 1, 12, 0, tzinfo=UTC).isoformat(),
        })
    # refused_no_preview, refused_with_preview, restart_midday_before_delivery, no_games: no usable state!

    if scenario == "no_games":
        games = []
    elif scenario == "thin_slate_1305":
        games = [{
            "gamePk": 101,
            "gameDate": f"{DAY}T17:05:00Z",  # 13:05 ET -> thin -180m = 10:05 ET (4m30s from 10:00:30)
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
            "teams": {"away": {"team": {"name": "CLE"}}, "home": {"team": {"name": "NYY"}}},
        }]
    else:
        games = [
            {
                "gamePk": 101,
                "gameDate": f"{DAY}T23:10:00Z",  # 19:10 ET
                "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                "teams": {"away": {"team": {"name": "CLE"}}, "home": {"team": {"name": "NYY"}}},
            },
            {
                "gamePk": 102,
                "gameDate": f"{DAY}T23:40:00Z",  # 19:40 ET
                "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                "teams": {"away": {"team": {"name": "NYM"}}, "home": {"team": {"name": "LAD"}}},
            },
        ]

    if scenario == "refused_with_preview":
        # Seed an undelivered preview pick (chosen_on required=False)
        preview_pick = picks.DailyPick(
            date=DAY,
            run_time=REAL_DT(2027, 7, 15, 3, 30, tzinfo=ET).isoformat(),
            pick=picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                            0.82, ["PROJECTED"], True, 101, f"{DAY}T23:10:00Z"),
            double_down=None,
            runner_up=None,
            chosen_on={
                "schema": "bts_pick_chosen_on_v1",
                "required": False,
                "kind": "model",
                "streak": 0,
                "saver_available": True,
                "dated": None,
            },
        )
        picks.save_pick(preview_pick, picks_dir)

    if scenario == "restart_after_delivery":
        delivered = picks.DailyPick(
            date=DAY,
            run_time=REAL_DT(2027, 7, 15, 11, 0, tzinfo=ET).isoformat(),
            pick=picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                            0.99, [], False, 101, f"{DAY}T23:10:00Z"),
            double_down=None,
            runner_up=None,
            notification_sent=True,
            notification_channel="bluesky_dm",
            notification_id="msg-prev",
            delivery_attempted=True,
            delivered_at=REAL_DT(2027, 7, 15, 11, 0, tzinfo=ET).isoformat(),
            chosen_on={
                "schema": "bts_pick_chosen_on_v1",
                "required": True,
                "kind": "dated",
                "streak": 30,
                "saver_available": False,
                "dated": {
                    "effective_date": DAY,
                    "active_streak": 30,
                    "best_streak": 30,
                    "saver_state": "used",
                },
            },
        )
        picks.save_pick(delivered, picks_dir)
        prev_state = scheduler.SchedulerState(
            date=DAY,
            schedule_fetched_at=REAL_DT(2027, 7, 15, 10, 0, tzinfo=ET).isoformat(),
            games=[],
            confirmed_game_pks=[101],
            runs_completed=[],
            pick_locked=True,
            pick_locked_at=REAL_DT(2027, 7, 15, 11, 0, tzinfo=ET).isoformat(),
            result_status=None,
            next_wakeup=None,
            committed_pick_written=True,
        )
        scheduler.save_state(prev_state, picks_dir)
        daily_decision.write_decision(
            DAY, picks_dir, action="single", source="mdp",
            primary={"batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "game_pk": 101, "p_game_hit": 0.99},
            delivery_status="delivered", scoreable=True, streak=30, saver_available=False,
            now=REAL_DT(2027, 7, 15, 11, 0, tzinfo=ET),
        )
    elif scenario == "restart_midday_before_delivery":
        prev_state = scheduler.SchedulerState(
            date=DAY,
            schedule_fetched_at=REAL_DT(2027, 7, 15, 10, 0, tzinfo=ET).isoformat(),
            games=[],
            confirmed_game_pks=[],
            runs_completed=[],
            pick_locked=False,
            pick_locked_at=None,
            result_status=None,
            next_wakeup=None,
        )
        scheduler.save_state(prev_state, picks_dir)

    p_hit = [0.10, 0.09] if scenario == "mdp_skip" else [0.99, 0.81]
    g1_time = games[0]["gameDate"] if games else f"{DAY}T23:10:00Z"
    g2_time = games[1]["gameDate"] if len(games) > 1 else g1_time
    rows = [
        {"batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
         "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
         "p_hit_pa": 0.28, "p_game_hit": p_hit[0], "flags": "", "projected_lineup": False,
         "game_pk": 101, "game_time": g1_time},
    ]
    if len(games) > 1:
        rows.append(
            {"batter_id": 665742, "batter_name": "Juan Soto", "team": "NYM", "lineup": 2,
             "pitcher_id": 605483, "pitcher_name": "Blake Snell", "opp": "LAD",
             "p_hit_pa": 0.23, "p_game_hit": p_hit[1], "flags": "", "projected_lineup": False,
             "game_pk": 102, "game_time": g2_time}
        )
    pred_df = pd.DataFrame(rows)

    heartbeats = []
    real_write_hb = scheduler.write_heartbeat

    def spy_write_hb(path, state, now_utc=None, sleeping_until=None, extra=None):
        eff_utc = now_utc or now_box[0].astimezone(UTC)
        heartbeats.append({
            "at": now_box[0].isoformat(),
            "state": state,
            "sleeping_until": sleeping_until.isoformat() if sleeping_until else None,
            "extra": extra,
        })
        return real_write_hb(path, state=state, now_utc=eff_utc, sleeping_until=sleeping_until, extra=extra)

    scheduler.write_heartbeat = spy_write_hb
    scheduler.heartbeat_watchdog = lambda *a, **k: contextlib.nullcontext()
    scheduler.notify_ready = lambda *a, **k: None
    scheduler.notify_watchdog = lambda *a, **k: None
    scheduler.fetch_schedule = lambda d: games if d == DAY else []
    scheduler.check_confirmed_lineups = lambda pks: {pk: {"away", "home"} for pk in pks}
    scheduler._idle_until_next_wakeup = lambda *a, **k: None
    scheduler._watchdog_ping_sleep = lambda secs, *a, **k: now_box.__setitem__(
        0, now_box[0] + dt.timedelta(seconds=max(0.0, float(secs)))
    )
    scheduler.run_result_polling = lambda *a, **k: "final"

    check_events = []
    real_single_check = scheduler.run_single_check

    def spy_single_check(*args, **kwargs):
        check_events.append({"at": now_box[0].isoformat(), "date": kwargs.get("date")})
        return real_single_check(*args, **kwargs)

    scheduler.run_single_check = spy_single_check
    orchestrator.run_cascade = lambda *a, **k: (pred_df.copy(), "offline")
    statuses = {g["gamePk"]: {"abstract": "P", "detailed": "Scheduled", "code": "S"} for g in games}
    picks.get_game_statuses_detailed = lambda *a: statuses
    strategy.get_game_statuses_detailed = lambda *a: statuses
    picks.get_game_statuses = lambda *a: {pk: "P" for pk in statuses}

    pick_dms = []
    health_dms = []
    posts = []
    dm.send_dm = lambda rec, body: (pick_dms.append({"at": now_box[0].isoformat(), "to": rec, "body": body}) or "pick-msg-1")
    alert.send_dm = lambda rec, body: (health_dms.append({"at": now_box[0].isoformat(), "to": rec, "body": body}) or "health-msg-1")
    posting.post_to_bluesky = lambda body: (posts.append({"at": now_box[0].isoformat(), "body": body}) or "at://post/1")

    config = {
        "orchestrator": {
            "picks_dir": str(picks_dir),
            "heartbeat_path": str(hb_path),
            "models_dir": str(models_dir),
            "data_dir": str(workdir / "data" / "processed"),
        },
        "bluesky": {"dm_recipient": "operator.test"},
        "scheduler": {
            "contest_state_required": True,
            "pick_delivery": "dm",
            "live_forward_capture_on_lock": False,
        },
        "health_checks": {"enabled": True},
        "tiers": [],
    }

    err_buf = io.StringIO()
    with contextlib.redirect_stderr(err_buf):
        scheduler.run_day(DAY, config)

    files = {}
    data_root = workdir / "data"
    for p in sorted(data_root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(data_root))
            raw = p.read_bytes().replace(str(workdir).encode("utf-8"), b"<WORKDIR>")
            files[rel] = raw.decode("utf-8", errors="replace")

    out = {
        "files": files,
        "heartbeats": heartbeats,
        "check_events": check_events,
        "pick_dms": pick_dms,
        "health_dms": [
            {**d, "body": d["body"].replace(str(workdir), "<WORKDIR>")}
            for d in health_dms
        ],
        "posts": posts,
        "stderr_lines": [
            line.replace(str(workdir), "<WORKDIR>")
            for line in err_buf.getvalue().splitlines()
        ],
    }
    (workdir / "result.json").write_text(json.dumps(out, indent=2))


def run_day_comparisons() -> dict:
    scenarios = [
        "normal_dated",
        "normal_fresh_auto",
        "normal_stale_auto",
        "refused_no_preview",
        "refused_with_preview",
        "mdp_skip",
        "no_games",
        "thin_slate_1305",
        "restart_after_delivery",
        "restart_midday_before_delivery",
    ]
    scratch = HERE / "scratch_day_cmp"
    if scratch.exists():
        shutil.rmtree(scratch)
    results = {}

    for sc in scenarios:
        pair = {}
        for mode in ("base", "branch"):
            wdir = scratch / sc / mode
            wdir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, __file__, "child_day", mode, sc, str(wdir)],
                cwd=REPO,
                check=True,
                env=os.environ | {"TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1"},
            )
            pair[mode] = json.loads((wdir / "result.json").read_text())

        base, branch = pair["base"], pair["branch"]
        assert base["heartbeats"] == branch["heartbeats"], (sc, "heartbeats differ", base["heartbeats"], branch["heartbeats"])
        assert base["check_events"] == branch["check_events"], (sc, "check_events differ")
        assert base["pick_dms"] == branch["pick_dms"], (sc, "pick_dms differ")
        assert base["posts"] == branch["posts"], (sc, "posts differ")

        file_diffs = {}
        for k in sorted(set(base["files"]) | set(branch["files"])):
            if base["files"].get(k) != branch["files"].get(k):
                file_diffs[k] = {
                    "in_base": k in base["files"],
                    "in_branch": k in branch["files"],
                }

        # Verify shared health_dm_delivery_status.json is IDENTICAL in EVERY scenario!
        shared_key = "health_state/health_dm_delivery_status.json"
        assert shared_key not in file_diffs, f"{sc}: shared health status differed! {file_diffs}"
        for core_prefix in ("picks/", "heartbeat.json"):
            bad = [k for k in file_diffs if k.startswith(core_prefix) or k == core_prefix]
            assert not bad, f"{sc}: core day artifact differed: {bad}"

        stderr_added = [l for l in branch["stderr_lines"] if l not in base["stderr_lines"]]
        stderr_removed = [l for l in base["stderr_lines"] if l not in branch["stderr_lines"]]
        assert stderr_removed == [], (sc, "base stderr lines missing!", stderr_removed)

        if sc in {"normal_dated", "normal_fresh_auto", "mdp_skip"}:
            assert file_diffs == {}, (sc, file_diffs)
            assert base["health_dms"] == branch["health_dms"], sc
            assert len(stderr_added) == 1 and stderr_added[0].startswith("  Morning state check: "), (sc, stderr_added)
        elif sc in {"no_games", "restart_after_delivery"}:
            assert file_diffs == {}, (sc, file_diffs)
            assert base["health_dms"] == branch["health_dms"], sc
            assert stderr_added == [], (sc, stderr_added)
        elif sc == "thin_slate_1305":
            assert file_diffs == {}, (sc, file_diffs)
            assert base["health_dms"] == branch["health_dms"], sc
            assert stderr_added == ["  Morning state check skipped: first lineup check is due within 10 minutes."], stderr_added
        elif sc in {"refused_no_preview", "refused_with_preview", "restart_midday_before_delivery"}:
            assert set(file_diffs) == {"health_state/contest_morning_check_dm_delivery_status.json"}, (sc, file_diffs)
            assert branch["health_dms"][1:] == base["health_dms"], (sc, branch["health_dms"], base["health_dms"])
            assert "Morning check for 2027-07-15: no usable contest state was found at startup." in branch["health_dms"][0]["body"]
            assert len(stderr_added) == 1 and stderr_added[0].startswith("  MORNING STATE CHECK — no pick can be made yet: "), stderr_added
        elif sc == "normal_stale_auto":
            assert set(file_diffs) == {"health_state/contest_stale_reading_dm_delivery_status.json"}, (sc, file_diffs)
            # Base sent 4c WARN at 16:10 + EOD health DM; branch sent morning WARN at 10:00 + same EOD health DM (no second 4c WARN at 16:10!)
            assert len(branch["health_dms"]) == len(base["health_dms"]), sc
            assert branch["health_dms"][1:] == base["health_dms"][1:], sc
            assert branch["health_dms"][0]["at"].startswith("2027-07-15T10:00:00"), branch["health_dms"][0]
            assert "the automatic contest reading was stale at startup" in branch["health_dms"][0]["body"]
            assert len(stderr_added) == 1 and stderr_added[0].startswith("  MORNING STATE CHECK — stale contest reading: "), stderr_added

        results[sc] = {
            "matching_files": sum(1 for k in branch["files"] if base["files"].get(k) == branch["files"].get(k)),
            "file_diffs": list(file_diffs.keys()),
            "base_health_dm_count": len(base["health_dms"]),
            "branch_health_dm_count": len(branch["health_dms"]),
            "pick_dm_count": len(branch["pick_dms"]),
            "check_count": len(branch["check_events"]),
            "heartbeat_count": len(branch["heartbeats"]),
            "stderr_added": stderr_added,
        }
    return results


def run_fault_injections_and_stall() -> dict:
    sys.path.insert(0, str(BRANCH_SRC))
    install_network_guard()

    from bts import contest_state, daily_decision, dm, heartbeat, orchestrator, picks, posting, scheduler, slate, strategy
    from bts.health import alert

    class CustomBaseException(BaseException):
        pass

    class UnprintableError(BaseException):
        def __str__(self) -> str:
            raise RuntimeError("__str__ exploded")

        def __repr__(self) -> str:
            raise RuntimeError("__repr__ exploded")

    exc_factories = {
        "RuntimeError": lambda: RuntimeError("injected runtime\n" + "é" * 240),
        "KeyboardInterrupt": lambda: KeyboardInterrupt("injected kbi"),
        "SystemExit": lambda: SystemExit(7),
        "GeneratorExit": lambda: GeneratorExit("injected genexit"),
        "CustomBaseException": lambda: CustomBaseException("injected base"),
        "BaseExceptionGroup": lambda: BaseExceptionGroup("beg", [KeyboardInterrupt("kbi-in-group")]),
        "UnprintableError": lambda: UnprintableError(),
    }

    sites = [
        "_now_et",
        "runs_missing_time_et",
        "runs_naive_time_et",
        "_contest_state_required",
        "is_scoreable_commit",
        "load_pick",
        "pick_was_delivered",
        "state_read",
        "automatic_reading_record",
        "dispatcher",
        "transport_in_dispatcher",
        "stderr_skip_empty",
        "stderr_skip_due",
        "stderr_refusal",
        "stderr_stale",
        "stderr_accepted",
        "stderr_permanent",
    ]

    scratch = HERE / "scratch_faults"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    games_default = [{
        "gamePk": 101,
        "gameDate": f"{DAY}T23:10:00Z",  # 19:10 ET -> thin check 16:10 ET
        "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
        "teams": {"away": {"team": {"name": "CLE"}}, "home": {"team": {"name": "NYY"}}},
    }]

    def execute_one(case_dir: Path, site: str | None, exc_name: str | None, status_hazard: str | None = None) -> dict:
        start = REAL_DT(2027, 7, 15, 10, 0, 0, tzinfo=ET)
        now_box = [start]

        class Clock(REAL_DT):
            @classmethod
            def now(cls, tz=None):
                return now_box[0].astimezone(tz) if tz is not None else now_box[0].replace(tzinfo=None)

        picks_dir = case_dir / "data" / "picks"
        hb_path = case_dir / "data" / "hb.json"
        picks_dir.mkdir(parents=True, exist_ok=True)

        # Seed state where needed so site is actually reached!
        if site in {"automatic_reading_record", "stderr_accepted", "stderr_stale"}:
            age_h = 20.0 if site == "stderr_stale" else 2.0
            rec = (start - dt.timedelta(hours=age_h)).astimezone(UTC).isoformat().replace("+00:00", "Z")
            acct = picks_dir / "account_state"
            acct.mkdir(parents=True, exist_ok=True)
            (acct / "season_contract.json").write_text(json.dumps({
                "schema_version": "bts_season_contract_v1", "season": 2027, "status": "automatic",
                "activation_id": "a" * 32, "switched_at": start.isoformat(), "revoked_at": None, "note": None,
            }))
            (acct / "contest_streak.json").write_text(json.dumps({
                "schema_version": "bts_contest_streak_auto_v2", "season": 2027, "activation_id": "a" * 32,
                "active_streak": 12, "best_streak": 12, "source": "mlb_bts_profile",
                "source_date": "2027-07-14", "recorded_at": rec,
            }))
        if site == "pick_was_delivered":
            picks.save_pick(picks.DailyPick(
                date=DAY, run_time=start.isoformat(),
                pick=picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                                0.82, [], False, 101, f"{DAY}T23:10:00Z"),
                double_down=None, runner_up=None,
            ), picks_dir)

        health_dir = picks_dir.parent / "health_state"
        morning_status = health_dir / "contest_morning_check_dm_delivery_status.json"
        if status_hazard == "dir":
            morning_status.mkdir(parents=True, exist_ok=True)
        elif status_hazard == "corrupt_syntax":
            health_dir.mkdir(parents=True, exist_ok=True)
            morning_status.write_text("{corrupt")
        elif status_hazard == "corrupt_list":
            health_dir.mkdir(parents=True, exist_ok=True)
            morning_status.write_text("[]")
        elif status_hazard == "corrupt_deep_json":
            health_dir.mkdir(parents=True, exist_ok=True)
            morning_status.write_text("[" * 1500 + "]" * 1500)
        elif status_hazard == "unwritable_file":
            health_dir.mkdir(parents=True, exist_ok=True)
            morning_status.write_text("{}")
            morning_status.chmod(0o444)
        elif status_hazard == "unwritable_dir":
            health_dir.mkdir(parents=True, exist_ok=True)
            health_dir.chmod(0o555)

        first_check_state = {}
        injected = [False]

        with contextlib.ExitStack() as stack:
            def patch(obj, attr, val):
                orig = getattr(obj, attr)
                setattr(obj, attr, val)
                stack.callback(setattr, obj, attr, orig)

            for mod in (contest_state, daily_decision, picks, strategy, scheduler, alert, heartbeat, slate):
                if hasattr(mod, "datetime"):
                    patch(mod, "datetime", Clock)
            patch(scheduler, "_now_et", lambda: now_box[0])
            patch(scheduler, "time", Mock(monotonic=lambda: 0.0))
            patch(scheduler, "heartbeat_watchdog", lambda *a, **k: contextlib.nullcontext())
            patch(scheduler, "notify_ready", lambda *a, **k: None)
            patch(scheduler, "notify_watchdog", lambda *a, **k: None)
            patch(scheduler, "fetch_schedule", lambda d: games_default if d == DAY else [])
            patch(scheduler, "check_confirmed_lineups", lambda *_: {})
            patch(scheduler, "_idle_until_next_wakeup", lambda *_: None)
            patch(scheduler, "_watchdog_ping_sleep", lambda secs, *a, **k: now_box.__setitem__(
                0, now_box[0] + dt.timedelta(seconds=max(0.0, float(secs)))
            ))
            patch(scheduler, "run_result_polling", lambda *a, **k: "final")
            patch(dm, "send_dm", Mock(return_value="p1"))
            patch(alert, "send_dm", Mock(return_value="h1"))
            patch(posting, "post_to_bluesky", Mock(return_value="u1"))

            def on_cascade(*a, **k):
                if not first_check_state:
                    st_bytes = (picks_dir / DAY / "scheduler_state.json").read_bytes()
                    first_check_state["at"] = now_box[0].isoformat()
                    first_check_state["scheduler_state"] = json.loads(st_bytes)
                return None, None

            patch(orchestrator, "run_cascade", on_cascade)

            if site == "runs_missing_time_et":
                # Only runs[0] lacks time_et on startup? No: if runs[0] lacks time_et, loop line 2950 `run_info["time_et"]` would also fail on base!
                # Instead return a custom iterable or plan_run_times where run[1] lacks time_et? No, loop iterates all runs.
                # To test `min((run["time_et"] for run in runs))` raising inside the morning block while loop stays valid,
                # return a run dict whose __getitem__ raises on the FIRST read (inside min) and returns real target thereafter!
                target_dt = REAL_DT(2027, 7, 15, 16, 10, tzinfo=ET)

                class RunDict(dict):
                    def __init__(self):
                        super().__init__(time_et=target_dt, game_pks=[101])
                        self.calls = 0

                    def __getitem__(self, key):
                        if key == "time_et":
                            self.calls += 1
                            # Call 1 is run_day's print loop at L2893; Call 2 is morning block `min(...)`!
                            if self.calls == 2:
                                injected[0] = True
                                raise KeyError("time_et")
                        return super().__getitem__(key)

                patch(scheduler, "plan_run_times", lambda *a, **k: [RunDict()])
            elif site == "runs_naive_time_et":
                target_dt = REAL_DT(2027, 7, 15, 16, 10, tzinfo=ET)

                class NaiveOnceRunDict(dict):
                    def __init__(self):
                        super().__init__(time_et=target_dt, game_pks=[101])
                        self.calls = 0

                    def __getitem__(self, key):
                        if key == "time_et":
                            self.calls += 1
                            if self.calls == 2:
                                injected[0] = True
                                return REAL_DT(2027, 7, 15, 16, 10)  # naive!
                        return super().__getitem__(key)

                patch(scheduler, "plan_run_times", lambda *a, **k: [NaiveOnceRunDict()])
            elif site == "_now_et":
                calls = [0]

                def faulty_now():
                    calls[0] += 1
                    # L2870 (1), L2889 (2), L2909 (3), L2934 morning (4)!
                    if calls[0] == 4:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return now_box[0]

                patch(scheduler, "_now_et", faulty_now)
            elif site == "_contest_state_required":
                orig_req = scheduler._contest_state_required

                def faulty_req(cfg):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return orig_req(cfg)

                patch(scheduler, "_contest_state_required", faulty_req)
            elif site == "is_scoreable_commit":
                orig_isc = daily_decision.is_scoreable_commit

                def faulty_isc(*a, **k):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return orig_isc(*a, **k)

                patch(daily_decision, "is_scoreable_commit", faulty_isc)
            elif site == "load_pick":
                orig_lp = picks.load_pick

                def faulty_lp(*a, **k):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return orig_lp(*a, **k)

                patch(picks, "load_pick", faulty_lp)
            elif site == "pick_was_delivered":
                orig_pwd = picks.pick_was_delivered

                def faulty_pwd(*a, **k):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return orig_pwd(*a, **k)

                patch(picks, "pick_was_delivered", faulty_pwd)
            elif site == "state_read":
                orig_ldss = contest_state.load_decision_streak_state

                def faulty_ldss(*a, **k):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return orig_ldss(*a, **k)

                patch(contest_state, "load_decision_streak_state", faulty_ldss)
            elif site == "automatic_reading_record":
                orig_arr = contest_state.automatic_reading_record

                def faulty_arr(*a, **k):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return orig_arr(*a, **k)

                patch(contest_state, "automatic_reading_record", faulty_arr)
            elif site == "dispatcher":
                orig_disp = alert.dispatch_dm_for_health_alerts

                def faulty_disp(*a, **k):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return orig_disp(*a, **k)

                patch(alert, "dispatch_dm_for_health_alerts", faulty_disp)
            elif site == "transport_in_dispatcher":
                def faulty_send(*a, **k):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return "h1"

                patch(alert, "send_dm", faulty_send)
            elif site in {"stderr_skip_empty", "stderr_skip_due", "stderr_refusal", "stderr_stale", "stderr_accepted", "stderr_permanent"}:
                if site == "stderr_skip_empty":
                    patch(scheduler, "plan_run_times", lambda *a, **k: [])
                elif site == "stderr_skip_due":
                    patch(scheduler, "plan_run_times", lambda *a, **k: [
                        {"time_et": start + dt.timedelta(minutes=5), "game_pks": [101]},
                    ])

                def faulty_print(*args, **kwargs):
                    msg = str(args[0]) if args else ""
                    if ("Morning state check" in msg or "MORNING STATE CHECK" in msg):
                        if site == "stderr_permanent" or not msg.startswith("  Morning state check failed:"):
                            injected[0] = True
                            raise exc_factories[exc_name]()
                    return print(*args, **kwargs)

                scheduler.print = faulty_print
                stack.callback(delattr, scheduler, "print")

            config = {
                "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(hb_path)},
                "bluesky": {"dm_recipient": "operator.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm", "live_forward_capture_on_lock": False},
                "health_checks": {"enabled": False},
                "tiers": [],
            }
            err_buf = io.StringIO()
            try:
                with contextlib.redirect_stderr(err_buf):
                    scheduler.run_day(DAY, config)
            finally:
                if health_dir.exists():
                    health_dir.chmod(0o755)
                if morning_status.exists() and morning_status.is_file():
                    morning_status.chmod(0o644)

        return {
            "injected": injected[0],
            "first_check_state": first_check_state,
            "stderr": err_buf.getvalue(),
        }

    # Baseline clean reference state at first check
    ref_normal = execute_one(scratch / "ref_normal", None, None)
    assert ref_normal["first_check_state"]["at"] == "2027-07-15T16:10:00-04:00"

    total_fault_runs = 0
    for site in sites:
        excs_for_site = ["RuntimeError"] if site in {"runs_missing_time_et", "runs_naive_time_et"} else list(exc_factories)
        for exc_name in excs_for_site:
            res = execute_one(scratch / f"{site}_{exc_name}", site, exc_name)
            assert res["injected"] is True, (site, exc_name)
            if site == "stderr_skip_empty":
                # Empty plan has 0 checks, completes cleanly
                pass
            elif site == "stderr_skip_due":
                assert res["first_check_state"]["at"] == "2027-07-15T10:05:00-04:00"
            else:
                assert res["first_check_state"] == ref_normal["first_check_state"], (
                    site, exc_name, res["first_check_state"], ref_normal["first_check_state"]
                )
            total_fault_runs += 1

    # Status file hazards
    hazards = ["dir", "corrupt_syntax", "corrupt_list", "corrupt_deep_json", "unwritable_file", "unwritable_dir"]
    for hz in hazards:
        res = execute_one(scratch / f"hazard_{hz}", None, None, status_hazard=hz)
        assert res["first_check_state"] == ref_normal["first_check_state"], (hz, res["first_check_state"])

    # Verify non-morning interrupts ARE NOT swallowed (before block in fetch_schedule; after block in run_single_check)
    outer_propagated = {}
    for where in ("before_block", "in_loop"):
        for exc_cls in (KeyboardInterrupt, SystemExit):
            case_dir = scratch / f"outer_{where}_{exc_cls.__name__}"
            picks_dir = case_dir / "data" / "picks"
            picks_dir.mkdir(parents=True, exist_ok=True)
            cfg = {
                "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(case_dir / "hb.json")},
                "bluesky": {"dm_recipient": "operator.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm", "live_forward_capture_on_lock": False},
                "health_checks": {"enabled": False},
                "tiers": [],
            }
            with contextlib.ExitStack() as stack:
                def patch(obj, attr, val):
                    orig = getattr(obj, attr)
                    setattr(obj, attr, val)
                    stack.callback(setattr, obj, attr, orig)

                patch(scheduler, "_now_et", lambda: REAL_DT(2027, 7, 15, 10, 0, tzinfo=ET))
                patch(scheduler, "notify_ready", lambda *a, **k: None)
                patch(scheduler, "notify_watchdog", lambda *a, **k: None)
                patch(scheduler, "_watchdog_ping_sleep", lambda *a, **k: None)
                patch(alert, "send_dm", Mock(return_value="h1"))
                if where == "before_block":
                    patch(scheduler, "fetch_schedule", Mock(side_effect=exc_cls("outer-before")))
                else:
                    patch(scheduler, "fetch_schedule", lambda d: games_default if d == DAY else [])
                    patch(scheduler, "run_single_check", Mock(side_effect=exc_cls("outer-loop")))
                caught = None
                try:
                    with contextlib.redirect_stderr(io.StringIO()):
                        scheduler.run_day(DAY, cfg)
                except BaseException as err:
                    caught = type(err).__name__
                assert caught == exc_cls.__name__, (where, exc_cls, caught)
                outer_propagated[f"{where}:{exc_cls.__name__}"] = caught

    # Real bts.dm.send_dm worst-case timeout & backoff timing against exact 10-minute margin
    from bts import util

    def measure_real_dm_stall(recipient: str, mode: str) -> dict:
        t0 = REAL_DT(2027, 7, 15, 10, 0, 0, tzinfo=ET)
        first_target = t0 + dt.timedelta(minutes=10)  # 10:10:00 ET (exact 600s minimum lead)
        clock = [t0]
        http_calls = []
        counts: dict[str, int] = {}

        def fake_urlopen(req: urllib.request.Request, timeout: float = 15):
            url = req.full_url
            counts[url] = counts.get(url, 0) + 1
            attempt = counts[url]
            clock[0] += dt.timedelta(seconds=timeout)
            http_calls.append({"url": url, "attempt": attempt, "timeout": timeout, "at": clock[0].isoformat()})
            if mode == "fail_auth_all_retries":
                raise urllib.error.URLError("simulated 15s connect/read timeout")
            # Worst-case 4-step stall: fail attempts 1 & 2 on every idempotent endpoint, succeed at 15s on attempt 3,
            # and stall 15s on non-idempotent sendMessage!
            if "sendMessage" in url:
                raise urllib.error.URLError("simulated 15s sendMessage timeout")
            if attempt < 3:
                raise urllib.error.URLError("simulated 15s transient timeout")
            if "createSession" in url:
                body = b'{"accessJwt": "jwt-tok"}'
            elif "resolveHandle" in url:
                body = b'{"did": "did:plc:op123"}'
            elif "getConvoForMembers" in url:
                body = b'{"convo": {"id": "convo-1"}}'
            else:
                raise AssertionError(url)
            return io.BytesIO(body)

        case_dir = scratch / f"stall_{mode}_{recipient[:3]}"
        picks_dir = case_dir / "data" / "picks"
        picks_dir.mkdir(parents=True, exist_ok=True)
        loop_entered_at = []
        check_ran_at = []

        with contextlib.ExitStack() as stack:
            def patch(obj, attr, val):
                orig = getattr(obj, attr)
                setattr(obj, attr, val)
                stack.callback(setattr, obj, attr, orig)

            patch(os, "environ", os.environ | {"BTS_BLUESKY_APP_PASSWORD": "offline-secret"})
            patch(util, "urlopen", fake_urlopen)
            patch(util.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + dt.timedelta(seconds=s)))
            # Keep REAL alert.send_dm == bts.dm.send_dm!
            patch(alert, "send_dm", dm.send_dm)
            patch(scheduler, "_now_et", lambda: clock[0])
            patch(scheduler, "notify_ready", lambda *a, **k: None)
            patch(scheduler, "notify_watchdog", lambda *a, **k: None)
            patch(scheduler, "fetch_schedule", lambda d: games_default if d == DAY else [])
            patch(scheduler, "plan_run_times", lambda *a, **k: [{"time_et": first_target, "game_pks": [101]}])
            patch(scheduler, "_idle_until_next_wakeup", lambda *a, **k: None)

            def spy_sleep(secs, *a, **k):
                loop_entered_at.append({"before_sleep": clock[0].isoformat(), "sleep_secs": secs})
                clock[0] += dt.timedelta(seconds=max(0.0, float(secs)))

            patch(scheduler, "_watchdog_ping_sleep", spy_sleep)

            def spy_check(**kwargs):
                check_ran_at.append(clock[0].isoformat())
                return {"new_lineups": 0, "skipped": True, "pick_result": None, "should_post": False}

            patch(scheduler, "run_single_check", spy_check)

            cfg = {
                "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(case_dir / "hb.json")},
                "bluesky": {"dm_recipient": recipient},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm", "live_forward_capture_on_lock": False},
                "health_checks": {"enabled": False},
                "tiers": [],
            }
            with contextlib.redirect_stderr(io.StringIO()):
                scheduler.run_day(DAY, cfg)

        entered = dt.datetime.fromisoformat(loop_entered_at[0]["before_sleep"])
        elapsed_s = (entered - t0).total_seconds()
        remaining_s = (first_target - entered).total_seconds()
        return {
            "elapsed_sec": elapsed_s,
            "remaining_lead_sec": remaining_s,
            "http_call_count": len(http_calls),
            "loop_entered_at": loop_entered_at[0],
            "check_ran_at": check_ran_at,
        }

    stall_auth_only = measure_real_dm_stall("operator.bsky.social", "fail_auth_all_retries")
    stall_worst_handle = measure_real_dm_stall("operator.bsky.social", "worst_4_step")
    stall_worst_did = measure_real_dm_stall("did:plc:operator", "worst_4_step")
    assert stall_auth_only["elapsed_sec"] == 60.0
    assert stall_worst_handle["elapsed_sec"] == 195.0
    assert stall_worst_handle["remaining_lead_sec"] == 405.0
    assert stall_worst_handle["check_ran_at"] == ["2027-07-15T10:10:00-04:00"]
    assert stall_worst_did["elapsed_sec"] == 135.0

    return {
        "fault_matrix_runs": total_fault_runs,
        "status_hazards_verified": hazards,
        "outer_interrupts_propagated": outer_propagated,
        "stall_auth_only": stall_auth_only,
        "stall_worst_handle": stall_worst_handle,
        "stall_worst_did": stall_worst_did,
    }


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "child_day":
        _, _, mode, sc, wdir = sys.argv
        child_day_scenario(mode, sc, Path(wdir))
        return

    row5 = check_row5_and_scope()
    print("[OK] Row 5 & run_day AST scope verified:", json.dumps(row5))
    days = run_day_comparisons()
    print("[OK] 10 full-day base-vs-branch comparisons verified:", list(days.keys()))
    faults = run_fault_injections_and_stall()
    print("[OK] Fault matrix & real send_dm stall verified:", json.dumps({
        "fault_runs": faults["fault_matrix_runs"],
        "worst_handle_sec": faults["stall_worst_handle"]["elapsed_sec"],
        "remaining_lead_sec": faults["stall_worst_handle"]["remaining_lead_sec"],
    }))

    summary = {"row5_and_scope": row5, "day_comparisons": days, "faults_and_stall": faults}
    (HERE / "probe_day_and_isolation.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
