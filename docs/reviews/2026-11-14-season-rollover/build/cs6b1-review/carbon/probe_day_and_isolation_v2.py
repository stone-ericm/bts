"""Adversarial probe v2 for CS6b-1 items 1, 2, and 6:
Uses an identical working path for base and branch subprocesses (so path-bearing SHA-256 hashes in
health_dm_delivery_status.json are compared literally byte-for-byte) and pins datetime.datetime before
importing bts.* so every EOD health module (contest_state, restart_spike, post_failure, etc.) sees the
exact pinned clock.
"""
from __future__ import annotations

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


def child_day_scenario(mode: str, scenario: str, workdir: Path, out_json: Path) -> None:
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

    class PinnedDate(REAL_DATE):
        @classmethod
        def today(cls):
            return now_box[0].date()

    # Pin stdlib datetime BEFORE importing any bts module so every `from datetime import datetime` binds PinnedDatetime!
    dt.datetime = PinnedDatetime
    dt.date = PinnedDate

    src = BASE_SRC if mode == "base" else BRANCH_SRC
    sys.path.insert(0, str(src))
    install_network_guard()

    import pandas as pd
    from bts import (
        daily_decision, dm, orchestrator, picks, posting, scheduler, strategy,
    )
    from bts.health import alert, memory_growth, runner, unit_drift

    scheduler._now_et = lambda: now_box[0]
    posting._now_et = lambda: now_box[0]
    scheduler.time = Mock(monotonic=lambda: 0.0, sleep=lambda s: None)
    picks._git_head_sha = lambda *_: "b" * 40

    runner.get_self_pid = lambda: None
    runner.read_systemd_nrestarts = lambda *_: 0
    unit_drift.check = lambda **_: []
    memory_growth.check = lambda **_: []

    picks_dir = workdir / "data" / "picks"
    models_dir = workdir / "data" / "models"
    hb_path = workdir / "data" / "heartbeat.json"
    picks_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Seed clean operating files so healthy days produce 0 EOD health alerts (also test sparse alerts on refused days!)
    if scenario in {"normal_dated", "normal_fresh_auto", "normal_stale_auto", "mdp_skip", "thin_slate_1305", "restart_after_delivery"}:
        (models_dir / "blend_2027-07-16.pkl").write_bytes(b"blend")
        shutil.copyfile(REPO / "data" / "models" / "mdp_policy.npz", models_dir / "mdp_policy.npz")
        shutil.copyfile(REPO / "data" / "models" / "mdp_tail_policy.npz", models_dir / "mdp_tail_policy.npz")
        (picks_dir / "streak.json").write_text(json.dumps({
            "streak": 30, "saver_available": False, "updated": start_dt.astimezone(UTC).isoformat(),
            "applied_dates": [], "season": 2027,
        }))

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

    if scenario == "no_games":
        games = []
    elif scenario == "thin_slate_1305":
        games = [{
            "gamePk": 101,
            "gameDate": f"{DAY}T17:05:00Z",
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
            "teams": {"away": {"team": {"name": "CLE"}}, "home": {"team": {"name": "NYY"}}},
        }]
    else:
        games = [
            {
                "gamePk": 101,
                "gameDate": f"{DAY}T23:10:00Z",
                "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                "teams": {"away": {"team": {"name": "CLE"}}, "home": {"team": {"name": "NYY"}}},
            },
            {
                "gamePk": 102,
                "gameDate": f"{DAY}T23:40:00Z",
                "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                "teams": {"away": {"team": {"name": "NYM"}}, "home": {"team": {"name": "LAD"}}},
            },
        ]

    if scenario == "refused_with_preview":
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
            files[rel] = p.read_bytes().hex()

    out = {
        "files_hex": files,
        "heartbeats": heartbeats,
        "check_events": check_events,
        "pick_dms": pick_dms,
        "health_dms": health_dms,
        "posts": posts,
        "stderr_lines": err_buf.getvalue().splitlines(),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "child_day":
        _, _, mode, sc, wdir, out_json = sys.argv
        child_day_scenario(mode, sc, Path(wdir), Path(out_json))
        return

    import probe_day_and_isolation as p1

    row5 = p1.check_row5_and_scope()
    print("[OK] Row 5 & run_day AST scope verified:", json.dumps(row5))

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
    scratch = HERE / "scratch_day_cmp_v2"
    if scratch.exists():
        shutil.rmtree(scratch)
    active = scratch / "active_workdir"
    results = {}

    for sc in scenarios:
        pair = {}
        for mode in ("base", "branch"):
            if active.exists():
                shutil.rmtree(active)
            active.mkdir(parents=True, exist_ok=True)
            out_json = scratch / sc / f"{mode}.json"
            subprocess.run(
                [sys.executable, __file__, "child_day", mode, sc, str(active), str(out_json)],
                cwd=REPO,
                check=True,
                env=os.environ | {"TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1"},
            )
            pair[mode] = json.loads(out_json.read_text())

        base, branch = pair["base"], pair["branch"]
        assert base["heartbeats"] == branch["heartbeats"], (sc, "heartbeats differ")
        assert base["check_events"] == branch["check_events"], (sc, "check_events differ")
        assert base["pick_dms"] == branch["pick_dms"], (sc, "pick_dms differ")
        assert base["posts"] == branch["posts"], (sc, "posts differ")

        file_diffs = [
            k for k in sorted(set(base["files_hex"]) | set(branch["files_hex"]))
            if base["files_hex"].get(k) != branch["files_hex"].get(k)
        ]
        # Exact byte-for-byte match on shared health_dm_delivery_status.json and all picks/models/heartbeat!
        assert "health_state/health_dm_delivery_status.json" not in file_diffs, (sc, file_diffs)
        for k in file_diffs:
            assert not k.startswith("picks/") and not k.startswith("models/") and k != "heartbeat.json", (sc, k)

        stderr_added = [l for l in branch["stderr_lines"] if l not in base["stderr_lines"]]
        stderr_removed = [l for l in base["stderr_lines"] if l not in branch["stderr_lines"]]
        assert stderr_removed == [], (sc, stderr_removed)

        if sc in {"normal_dated", "normal_fresh_auto", "mdp_skip"}:
            assert file_diffs == [], (sc, file_diffs)
            assert base["health_dms"] == branch["health_dms"] == [], (sc, branch["health_dms"])
            assert len(stderr_added) == 1 and stderr_added[0].startswith("  Morning state check: "), (sc, stderr_added)
        elif sc in {"no_games", "restart_after_delivery"}:
            assert file_diffs == [], (sc, file_diffs)
            assert base["health_dms"] == branch["health_dms"] == [], (sc, branch["health_dms"])
            assert stderr_added == [], (sc, stderr_added)
        elif sc == "thin_slate_1305":
            assert file_diffs == [], (sc, file_diffs)
            assert base["health_dms"] == branch["health_dms"] == [], (sc, branch["health_dms"])
            assert stderr_added == ["  Morning state check skipped: first lineup check is due within 10 minutes."], stderr_added
        elif sc in {"refused_no_preview", "refused_with_preview", "restart_midday_before_delivery"}:
            assert file_diffs == ["health_state/contest_morning_check_dm_delivery_status.json"], (sc, file_diffs)
            assert branch["health_dms"][1:] == base["health_dms"], sc
            assert len(stderr_added) == 1 and stderr_added[0].startswith("  MORNING STATE CHECK — no pick can be made yet: "), stderr_added
        elif sc == "normal_stale_auto":
            assert file_diffs == ["health_state/contest_stale_reading_dm_delivery_status.json"], (sc, file_diffs)
            assert len(base["health_dms"]) == 1 and len(branch["health_dms"]) == 1, sc
            assert branch["health_dms"][0]["at"] == "2027-07-15T10:00:00-04:00"
            assert base["health_dms"][0]["at"] == "2027-07-15T16:10:00-04:00"
            assert len(stderr_added) == 1 and stderr_added[0].startswith("  MORNING STATE CHECK — stale contest reading: "), stderr_added

        results[sc] = {
            "matching_files": sum(1 for k in branch["files_hex"] if base["files_hex"].get(k) == branch["files_hex"].get(k)),
            "file_diffs": file_diffs,
            "base_health_dm_count": len(base["health_dms"]),
            "branch_health_dm_count": len(branch["health_dms"]),
            "pick_dm_count": len(branch["pick_dms"]),
            "check_count": len(branch["check_events"]),
            "heartbeat_count": len(branch["heartbeats"]),
            "stderr_added": stderr_added,
        }

    print("[OK] 10 full-day base-vs-branch byte comparisons verified:", json.dumps(results, indent=2))
    faults = p1.run_fault_injections_and_stall()
    print("[OK] Fault matrix & real send_dm stall verified:", json.dumps({
        "fault_runs": faults["fault_matrix_runs"],
        "status_hazards": faults["status_hazards_verified"],
        "outer_interrupts": faults["outer_interrupts_propagated"],
        "stall_auth_only_sec": faults["stall_auth_only"]["elapsed_sec"],
        "stall_worst_handle_sec": faults["stall_worst_handle"]["elapsed_sec"],
        "stall_worst_did_sec": faults["stall_worst_did"]["elapsed_sec"],
        "remaining_lead_sec": faults["stall_worst_handle"]["remaining_lead_sec"],
    }))

    summary = {"row5_and_scope": row5, "day_comparisons": results, "faults_and_stall": faults}
    (HERE / "probe_day_and_isolation.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
