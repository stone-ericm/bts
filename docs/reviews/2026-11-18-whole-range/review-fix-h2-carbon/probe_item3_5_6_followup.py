"""Clean follow-up probe completing Item 3 (all 12 scenarios + bts run/preview), Item 5 (5.2 & 5.5), Item 6 (1a..c & 4b), and Item 8 extra scratch checks."""
from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import ExitStack, contextmanager, nullcontext, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")


def _write_toml(path: Path, picks_dir: Path, delivery: str = "dm") -> None:
    text = (
        f'[orchestrator]\npicks_dir = "{picks_dir}"\nmodels_dir = "{picks_dir.parent / "models"}"\n'
        f'heartbeat_path = "{picks_dir.parent / ".heartbeat"}"\n\n'
        f'[[tiers]]\nname = "synthetic"\n\n'
        f'[scheduler]\npick_delivery = "{delivery}"\nentry_intent = "enter"\ncontest_state_required = true\n\n'
        f'[health_checks]\ncontest_state_expected = true\n\n'
        f'[bluesky]\ndm_recipient = "synthetic"\n'
    )
    path.write_text(text, encoding="utf-8")


def _write_units_snapshot(picks_dir: Path, season: int, round_id: int, game_pks: list[int]) -> None:
    units_dir = picks_dir.parent / "leaderboard" / "static_snapshots" / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "units": [{"id": 5000 + i, "feedId": int(gpk), "roundId": round_id} for i, gpk in enumerate(game_pks)]
    }
    (units_dir / f"{season}0701T120000Z.json.gz").write_bytes(gzip.compress(json.dumps(payload).encode("utf-8")))


def main() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from click.testing import CliRunner
    from bts import cli, contest_fetch, contest_state, daily_decision, orchestrator, picks, posting, scheduler, strategy, web
    from bts.health import alert as health_alert, contest_state as health_cs, mdp_policy_alignment, streak_validation, tail_policy as health_tp
    from bts.leaderboard import analysis as lb_analysis, auth, models as lb_models
    from scripts.audit.season_walk import walk

    problems: list[str] = []
    notes: list[str] = []
    runner = CliRunner()

    def fail_if(cond: bool, msg: str):
        if cond:
            problems.append(msg)

    # === Part A: Item 3 complete 12 scenarios (58, 100, 400 x dated/automatic x single/double x dm/public) + bts run & preview ===
    scenarios = [
        (58, 58, dt.date(2027, 7, 15), "single", "dm"),
        (58, 60, dt.date(2027, 9, 26), "double", "public"),
        (100, 100, dt.date(2027, 7, 15), "single", "dm"),
        (100, 120, dt.date(2027, 9, 26), "double", "dm"),
        (400, 400, dt.date(2027, 7, 15), "single", "dm"),
        (400, 400, dt.date(2027, 9, 26), "double", "public"),
    ]
    mdp_path = REPO / "data/models/mdp_policy.npz"
    fail_if(
        health_tp.check(base_path=mdp_path, tail_path=REPO / "data/models/mdp_tail_policy.npz", today=dt.date(2027, 7, 15)) != [],
        "tail_policy health check alerted",
    )

    for s_val, b_val, day, want_action, mode in scenarios:
        for kind in ("dated", "automatic"):
            noon = walk.REAL_DATETIME.combine(day, dt.time(12, 0), ET)
            with walk.offline(noon), tempfile.TemporaryDirectory(prefix=f"h2c-f3-{kind}-{s_val}-") as tmp:
                root = Path(tmp)
                picks_dir = root / "picks"
                picks_dir.mkdir()
                toml_path = root / "orchestrator.toml"
                _write_toml(toml_path, picks_dir, delivery=mode)
                cfg = orchestrator.load_config(toml_path)

                fail_if(walk.install_state(picks_dir, day, walk.State(kind, s_val, b_val, "used", 0.96)) != [],
                        f"{kind} {s_val}/{b_val}: install failed")

                r_stat = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks_dir)])
                fail_if(r_stat.exit_code != 0 or f"streak={s_val};" not in r_stat.output,
                        f"{kind} {s_val}/{b_val}: status failed: {r_stat.output}")

                err_buf = io.StringIO()
                dms_sent: list[str] = []
                posts_sent: list[str] = []
                with redirect_stderr(err_buf), patch("bts.dm.send_dm", side_effect=lambda r, m: (dms_sent.append(m) or "dm-m")):
                    scheduler._morning_state_check(
                        cfg, picks_dir, day.isoformat(),
                        walk.REAL_DATETIME.combine(day, dt.time(10, 0), ET),
                        walk.REAL_DATETIME.combine(day, dt.time(16, 5), ET),
                    )
                fail_if(f"Morning state check: {kind} contest state accepted (streak {s_val})." not in err_buf.getvalue() or dms_sent != [],
                        f"{kind} {s_val}/{b_val}: morning check unexpected")

                frame = walk._predictions(day, 0.96).head(2)
                games = [int(g) for g in frame["game_pk"]]
                statuses = {g: {"abstract": "P", "detailed": "Scheduled"} for g in games}
                sched_games = [{"gamePk": g, "status": {"detailedState": "Scheduled"}} for g in games]
                with ExitStack() as st:
                    st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame, "syn")))
                    st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
                    st.enter_context(patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()))
                    st.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *a: 4))
                    st.enter_context(patch.object(scheduler, "_trigger_live_forward_capture_on_lock", Mock()))
                    st.enter_context(patch.object(scheduler, "fetch_schedule", lambda *a: sched_games))
                    sel = scheduler.run_single_check(day.isoformat(), games, set(), cfg, 0.01)["selection"]

                fail_if((sel.action, sel.streak, sel.decision.objective, sel.decision.best_supplied, sel.decision.best_status)
                        != (want_action, s_val, "longest_streak", b_val, "trusted"),
                        f"{kind} {s_val}/{b_val}: selection mismatch: {sel}")

                reloaded = picks.load_pick(day.isoformat(), picks_dir)
                cur_st = contest_state.load_decision_streak_state(picks_dir, require_contest_state=True, now=noon)
                fail_if(picks.saved_pick_refusal(reloaded, cur_st) is not None, f"{kind} {s_val}/{b_val}: refusal on reload")

                with ExitStack() as st:
                    st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame.iloc[:0], "syn")))
                    st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
                    refresh = scheduler._refresh_pick_at_fallback_decision(cfg, day.isoformat(), reloaded, 0.01)
                fail_if(refresh.daily is not reloaded or refresh.selection is not None, "empty fallback refresh failed")

                sched_state = scheduler.SchedulerState(day.isoformat(), noon.isoformat(), [], [], [], False, None, None, None)
                with ExitStack() as st:
                    st.enter_context(patch.object(scheduler, "_trigger_live_forward_capture_on_lock", Mock()))
                    st.enter_context(patch.object(scheduler, "fetch_schedule", lambda *a: sched_games))
                    st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
                    st.enter_context(patch("bts.dm.send_dm", side_effect=lambda r, m: (dms_sent.append(m) or "dm-pick-1")))
                    st.enter_context(patch.object(posting, "post_to_bluesky", side_effect=lambda t: (posts_sent.append(t) or "at://post/1")))
                    ok_del = scheduler._deliver_and_lock_pick(reloaded, cfg, picks_dir, sched_state, day.isoformat(), "lineup lock", selection=sel)
                fail_if(not ok_del, f"{kind} {s_val}/{b_val}: delivery failed")
                sent_text = dms_sent[0] if mode == "dm" else posts_sent[0]
                fail_if(f"Streak: {s_val}" not in sent_text, f"{kind} {s_val}/{b_val}: sent text wrong: {sent_text}")

                dec_rec = daily_decision.load_decision(day.isoformat(), picks_dir)
                fail_if((dec_rec["action"], dec_rec["objective"], dec_rec["streak"], dec_rec["best_streak"], dec_rec["scoreable"])
                        != (want_action, "longest_streak", s_val, b_val, True),
                        f"{kind} {s_val}/{b_val}: decision.json wrong: {dec_rec}")

                with patch.object(orchestrator, "run_cascade", side_effect=AssertionError("no cascade")), \
                        patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()), \
                        patch.object(scheduler, "count_new_confirmations", lambda *a: 0):
                    post_res = scheduler.run_single_check(day.isoformat(), games, set(), cfg, 0.01)
                fail_if(not post_res["pick_result"].locked, "post-commit restart not locked")

                # Entry check
                round_id = 77
                _write_units_snapshot(picks_dir, day.year, round_id, games)
                pending_rows = [{"roundId": round_id, "unitId": 5000, "playerId": 2000, "number": 1, "result": None}]
                if want_action == "double":
                    pending_rows.append({"roundId": round_id, "unitId": 5001, "playerId": 2001, "number": 2, "result": None})
                session = SimpleNamespace(user_id=walk.UID, username=walk.ACCOUNT, xsid="walk_xsid")
                walk._Clock.at = walk.REAL_DATETIME.combine(day, dt.time(18, 35), ET)
                with ExitStack() as st:
                    st.enter_context(patch.object(auth, "load_session_cookies", lambda: {}))
                    st.enter_context(patch.object(auth, "extract_uid", lambda c: walk.UID))
                    st.enter_context(patch.object(auth, "fetch_login_session", lambda **kw: session))
                    st.enter_context(patch.object(contest_fetch, "fetch_profile", lambda *a: {"activeStreak": s_val, "seasonBestStreak": b_val, "predictions": []}))
                    st.enter_context(patch.object(contest_fetch, "fetch_pending_predictions", lambda *a: pending_rows))
                    st.enter_context(patch.object(cli, "_fetch_rounds", lambda: {round_id: day}))
                    st.enter_context(patch.object(cli, "_fetch_bts_to_mlb", lambda: {2000: 100, 2001: 101}))
                    r_ent = runner.invoke(cli.cli, [
                        "check-pick-entered", "--picks-dir", str(picks_dir),
                        "--expected-username", walk.ACCOUNT, "--dm-recipient", "synthetic",
                        "--now-et", f"{day.isoformat()}T18:35:00",
                    ])
                fail_if(r_ent.exit_code != 0 or "pick entered (match)" not in r_ent.output,
                        f"{kind} {s_val}/{b_val}: check-pick-entered failed: {r_ent.output}")

                # Day-outcome
                with patch.object(scheduler, "fetch_schedule", lambda *a: [{"gamePk": g, "status": {"detailedState": "Final"}} for g in games]):
                    r_do = runner.invoke(cli.cli, [
                        "day-outcome", "--date", day.isoformat(), "--config", str(toml_path),
                        "--picks-dir", str(picks_dir), "--expected-username", walk.ACCOUNT,
                    ])
                do_body = json.loads(r_do.output.strip())
                fail_if(r_do.exit_code != 0 or do_body["state"] != "entered" or do_body["ok"] is not True,
                        f"{kind} {s_val}/{b_val}: day-outcome wrong: {do_body}")

                # Dashboard HTML + leaderboard percentile_rank
                snaps_dir = root / "leaderboard" / "leaderboard_snapshots"
                snaps_dir.mkdir(parents=True)
                pq.write_table(
                    pa.Table.from_pylist([{"tab": "active_streak", "streak": x} for x in (15, 57, 90)]),
                    snaps_dir / "2027-07-15.parquet",
                )
                fail_if(lb_analysis.percentile_rank(root / "leaderboard", s_val)["n_above"] != sum(1 for x in (15, 57, 90) if x > s_val),
                        f"percentile_rank wrong at {s_val}")
                lb_row = lb_models.LeaderboardRow(
                    captured_at=walk.REAL_DATETIME(2027, 7, 15, 12, 0), tab="all_season", rank=1,
                    username="u", streak=s_val, season_best_streak=b_val, active_streak=s_val, user_id=1,
                )
                fail_if(lb_row.season_best_streak != b_val, "LeaderboardRow clamped")
                with ExitStack() as st:
                    st.enter_context(patch.object(web, "PICKS_DIR", picks_dir))
                    st.enter_context(patch.object(web, "HEALTH_STATE_DIR", root / "health_state"))
                    st.enter_context(patch.object(web, "VALIDATION_DIR", root / "validation"))
                    st.enter_context(patch.object(web, "fetch_bluesky_posts", lambda limit=5: []))
                    st.enter_context(patch.object(web, "_build_live_game_data", lambda pd: ([], None)))
                    st.enter_context(patch.object(web, "load_skip_policy_shadow_status", lambda: {}))
                    html = web.render_page()
                fail_if(f'<div class="streak-number">{s_val}</div>' not in html or "Streak State Error" in html,
                        f"{kind} {s_val}/{b_val}: dashboard HTML wrong")

                # Grading (check-results) 58->59/60, 100->101/102, 400->401/402
                picks.save_streak(s_val, picks_dir, saver_available=False, season=day.year)
                acct_before = {p.name: p.read_bytes() for p in (picks_dir / "account_state").iterdir() if p.is_file()}
                walk._Clock.at = walk.REAL_DATETIME.combine(day + dt.timedelta(days=1), dt.time(1, 0), ET)
                final_statuses = {g: {"abstract": "F", "detailed": "Final", "code": "F"} for g in games}
                with ExitStack() as st:
                    st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: final_statuses))
                    st.enter_context(patch.object(picks, "check_hit", lambda *a, **kw: "hit"))
                    r_cr = runner.invoke(cli.cli, ["check-results", "--date", day.isoformat(), "--picks-dir", str(picks_dir)])
                inc = 2 if want_action == "double" else 1
                fail_if(r_cr.exit_code != 0 or picks.load_streak(picks_dir, season=day.year) != s_val + inc,
                        f"{kind} {s_val}/{b_val}: check-results wrong: {r_cr.output}")
                acct_after = {p.name: p.read_bytes() for p in (picks_dir / "account_state").iterdir() if p.is_file()}
                fail_if(acct_before != acct_after, "check-results mutated contest state")
                fail_if(streak_validation.check(picks_dir) != [], f"streak_validation alerted at {s_val + inc}")
                fail_if(mdp_policy_alignment.compute_metrics(picks_dir, mdp_path, today=day).primary.n != 0,
                        "mdp_policy_alignment did not skip longest_streak")

    # bts run and bts preview at 58 and 400
    day = dt.date(2027, 7, 15)
    with walk.offline(walk.REAL_DATETIME(2027, 7, 15, 17, 0, tzinfo=ET)), tempfile.TemporaryDirectory(prefix="h2c-f3-cli-") as tmp:
        root = Path(tmp)
        frame = walk._predictions(day, 0.96).head(2)
        statuses = {int(g): {"abstract": "P", "detailed": "Scheduled"} for g in frame["game_pk"]}
        for s_val in (58, 400):
            pdir = root / f"run_{s_val}" / "picks"
            pdir.mkdir(parents=True)
            assert walk.install_state(pdir, day, walk.State("dated", s_val, s_val, "used", 0.96)) == []
            posted = []
            with ExitStack() as st:
                st.enter_context(patch("bts.model.predict.run_pipeline", lambda *a, **kw: frame))
                st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
                st.enter_context(patch.object(posting, "post_to_bluesky", side_effect=lambda t: (posted.append(t) or "at://r/1")))
                r_run = runner.invoke(cli.cli, ["run", "--date", day.isoformat(), "--picks-dir", str(pdir), "--models-dir", str(root / "m")])
            fail_if(r_run.exit_code != 0 or len(posted) != 1 or f"Streak: {s_val}" not in posted[0], f"bts run {s_val} failed")

            pdir_p = root / f"prev_{s_val}" / "picks"
            pdir_p.mkdir(parents=True)
            assert walk.install_state(pdir_p, day, walk.State("dated", s_val, s_val, "used", 0.96)) == []
            with ExitStack() as st:
                st.enter_context(patch("bts.model.predict.run_pipeline", lambda *a, **kw: frame))
                st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
                r_prev = runner.invoke(cli.cli, ["preview", "--date", day.isoformat(), "--picks-dir", str(pdir_p), "--models-dir", str(root / "m")])
            prev_pick = picks.load_pick(day.isoformat(), pdir_p)
            cur_st = contest_state.load_decision_streak_state(pdir_p, require_contest_state=True)
            fail_if(r_prev.exit_code != 0 or prev_pick.chosen_on["required"] is not False or picks.saved_pick_refusal(prev_pick, cur_st) is None,
                    f"bts preview {s_val} failed")
    notes.append("Part A (Item 3): all 12 full-day scenarios + bts run/preview at 58, 100, 400 passed.")

    # === Part B: Item 5 cases 5.1b, 5.2a/b, and 5.5 (clean clock & health_alert.send_dm patch) ===
    r_50 = walk.walk_one(dt.date(2027, 7, 15), walk.State("dated", 50, 57, "used", 0.72))
    r_75_w = walk.walk_one(dt.date(2027, 7, 15), walk.State("dated", 75, 75, "used", 0.72))
    r_57_157 = walk.walk_one(dt.date(2027, 9, 26), walk.State("dated", 57, 157, "used", 0.96))
    r_12_15 = walk.walk_one(dt.date(2027, 9, 25), walk.State("dated", 12, 15, "used", 0.96))
    r_12_157 = walk.walk_one(dt.date(2027, 9, 25), walk.State("dated", 12, 157, "used", 0.96))
    fail_if(r_50["action"] != "skip" or r_75_w["action"] != "single", f"5.1b wrong: {r_50['action']} vs {r_75_w['action']}")
    fail_if(r_57_157["action"] != "double", f"5.2a wrong: {r_57_157}")
    fail_if(r_12_15["action"] == "skip" or r_12_157["action"] != "skip", f"5.2b wrong: {r_12_15['action']} vs {r_12_157['action']}")
    fail_if("season-best 57 can't be beaten" not in (r_12_157["skip_dm"] or ""), f"5.2b skip_dm wrong: {r_12_157['skip_dm']}")

    day_sep25 = dt.date(2027, 9, 25)
    with walk.offline(walk.REAL_DATETIME(2027, 9, 25, 12, 0, tzinfo=ET)), tempfile.TemporaryDirectory(prefix="h2c-f5-") as tmp:
        p5 = Path(tmp) / "picks"
        p5.mkdir()
        assert walk.install_state(p5, day_sep25, walk.State("dated", 58, 157, "used", 0.96)) == []
        frame25 = walk._predictions(day_sep25, 0.96).head(2)
        g25 = [int(g) for g in frame25["game_pk"]]
        st25 = {g: {"abstract": "P", "detailed": "Scheduled"} for g in g25}
        cfg5 = walk._config(p5)
        cfg5["bluesky"] = {"dm_recipient": "synthetic"}
        with ExitStack() as st:
            st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame25, "syn")))
            st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: st25))
            st.enter_context(patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()))
            st.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *a: 4))
            scheduler.run_single_check("2027-09-25", g25, set(), cfg5, 0.01)
        saved5 = picks.load_pick("2027-09-25", p5)
        assert runner.invoke(cli.cli, [
            "set-contest-streak", "--picks-dir", str(p5), "--date", "2027-09-25",
            "--streak", "58", "--best-streak", "58", "--saver", "used",
        ]).exit_code == 0
        alert_dms = []
        with ExitStack() as st:
            st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame25.iloc[:0], "syn")))
            st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: st25))
            st.enter_context(patch.object(health_alert, "send_dm", side_effect=lambda r, m: (alert_dms.append(m) or "dm-a")))
            ref5 = scheduler._refresh_pick_at_fallback_decision(cfg5, "2027-09-25", saved5, 0.01)
            s5 = scheduler.SchedulerState("2027-09-25", None, [], [], [], False, None, None, None)
            ok5 = scheduler._deliver_and_lock_pick(ref5.daily, cfg5, p5, s5, "2027-09-25", "final fallback", selection=ref5.selection)
        fail_if(ok5 is not False or len(alert_dms) != 1 or "best streak 157" not in alert_dms[0] or "best streak 58" not in alert_dms[0],
                f"5.5 alert dm wrong: ok={ok5} dms={alert_dms}")
    notes.append("Part B (Item 5): 5.1b, 5.2a/b, and 5.5 refusal alert DM verified.")

    # === Part C: Item 6 1a..c standalone_mode=False & 4b unknown-outcome alert DM ===
    with walk.offline(walk.REAL_DATETIME(2027, 7, 15, 12, 0, tzinfo=ET)), tempfile.TemporaryDirectory(prefix="h2c-f6-") as tmp:
        root = Path(tmp)
        p_i = root / "int" / "picks"
        p_i.mkdir(parents=True)
        assert walk.install_state(p_i, day, walk.State("automatic", 57, 57, "not_earned", 0.96)) == []
        with walk._fetch_stubs(58, 58, day - dt.timedelta(days=1)), \
                patch.object(cli, "_atomic_write_json", side_effect=KeyboardInterrupt("mid-obs")):
            r_ki = runner.invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(p_i), "--expected-username", walk.ACCOUNT], standalone_mode=False)
        fail_if(not isinstance(r_ki.exception, KeyboardInterrupt), f"6.1a standalone_mode=False exc: {r_ki.exception!r}")

        p_u = root / "unconf" / "picks"
        p_u.mkdir(parents=True)
        assert walk.install_state(p_u, day, walk.State("dated", 58, 58, "used", 0.96)) == []
        frame = walk._predictions(day, 0.96).head(2)
        games = [int(g) for g in frame["game_pk"]]
        statuses = {g: {"abstract": "P", "detailed": "Scheduled"} for g in games}
        cfg_u = walk._config(p_u)
        cfg_u["bluesky"] = {"dm_recipient": "synthetic"}
        with ExitStack() as st:
            st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame, "syn")))
            st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
            st.enter_context(patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()))
            st.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *a: 4))
            scheduler.run_single_check(day.isoformat(), games, set(), cfg_u, 0.01)
        d_pick = picks.load_pick(day.isoformat(), p_u)
        d_pick.delivery_attempted = True
        picks.save_pick(d_pick, p_u)
        unk_dms = []
        s_u = scheduler.SchedulerState(day.isoformat(), None, [], [], [], False, None, None, None)
        with ExitStack() as st:
            st.enter_context(patch("bts.dm.send_dm", side_effect=lambda r, m: (unk_dms.append(m) or "dm-u")))
            st.enter_context(patch.object(health_alert, "send_dm", side_effect=lambda r, m: (unk_dms.append(m) or "dm-u")))
            ok_u = scheduler._deliver_and_lock_pick(picks.load_pick(day.isoformat(), p_u), cfg_u, p_u, s_u, day.isoformat(), "restart", selection=None)
        fail_if(ok_u is not False or len(unk_dms) != 1 or "DELIVERY OUTCOME UNKNOWN" not in unk_dms[0],
                f"6.4b unknown alert wrong: ok={ok_u} dms={unk_dms}")
    notes.append("Part C (Item 6): KeyboardInterrupt propagation and unconfirmed-send alert DM verified.")

    # === Part D: Item 8 targeted check on the 3 surviving scratch changes + s6_outer_drop_dated ===
    scratch = OUT_DIR / "scratch_src_p3"
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(REPO / "src", scratch)
    env = dict(os.environ, PYTHONPATH=f"{scratch}:{REPO}")
    extra_checks = [
        ("policy_effective_best_400_all_tail_tests", "bts/simulate/tail_policy.py",
         "def effective_best(streak: int, best: int | None, target: int = TARGET) -> int:",
         "def effective_best(streak: int, best: int | None, target: int = 400) -> int:",
         ["tests/test_wr_h2_caps.py", "tests/simulate/test_tail_policy.py", "tests/test_tail_policy_strategy.py", "tests/test_scheduler_tail_objective.py"]),
        ("policy_skip_reason_400_all_tail_tests", "bts/strategy.py",
         "min(57, streak + 2 * d_eff)",
         "min(400, streak + 2 * d_eff)",
         ["tests/test_wr_h2_caps.py", "tests/simulate/test_tail_policy.py", "tests/test_tail_policy_strategy.py", "tests/test_scheduler_tail_objective.py"]),
        ("s6_outer_drop_dated", "bts/picks.py",
         'if kind in {"dated", "automatic"} and streak > MAX_SEASON_STREAK:',
         'if kind == "automatic" and streak > MAX_SEASON_STREAK:',
         ["tests/test_wr_h2_caps.py", "tests/test_pick_chosen_on_cs5a.py"]),
    ]
    try:
        for label, rel, target, repl, tfiles in extra_checks:
            fp = scratch / rel
            orig = fp.read_text(encoding="utf-8")
            assert orig.count(target) == 1
            fp.write_text(orig.replace(target, repl, 1), encoding="utf-8")
            try:
                caught = None
                for tf in tfiles:
                    rc = subprocess.run([str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", "-x", tf],
                                        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=30).returncode
                    if rc != 0:
                        caught = tf
                        break
                notes.append(f"Part D ({label}): {'CAUGHT by ' + caught if caught else 'UNNOTICED across ' + ','.join(tfiles)}")
            finally:
                fp.write_text(orig, encoding="utf-8")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    out_lines = [*notes, f"problems={len(problems)}"]
    for p in problems:
        out_lines.append(f"PROBLEM: {p}")
    text = "\n".join(out_lines) + "\n"
    (OUT_DIR / "item3_5_6_followup.log").write_text(text, encoding="utf-8")
    print(text, end="")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
