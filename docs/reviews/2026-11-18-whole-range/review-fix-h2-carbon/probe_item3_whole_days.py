"""Item 3 adversarial probe: whole days at 58, 100, and 400 (M1..M5) across all live paths and displays."""
from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import socket
import sys
import tempfile
from contextlib import ExitStack, contextmanager, redirect_stderr
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
REAL_DATETIME = dt.datetime


def _refuse_net(*args, **kwargs):
    raise OSError("offline probe: network refused")


class _Clock(REAL_DATETIME):
    at: REAL_DATETIME = REAL_DATETIME(2027, 7, 15, 12, 0, tzinfo=ET)

    @classmethod
    def now(cls, tz=None):
        return cls.at.astimezone(tz) if tz else cls.at.astimezone(ET).replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        return cls.at.astimezone(dt.timezone.utc).replace(tzinfo=None)


@contextmanager
def pinned_offline(at: REAL_DATETIME):
    import bts.cli
    import bts.contest_fetch
    import bts.contest_state
    import bts.daily_decision
    import bts.day_outcome
    import bts.entry_receipt
    import bts.health.contest_state
    import bts.health.mdp_policy_alignment
    import bts.health.streak_validation
    import bts.health.tail_policy
    import bts.orchestrator
    import bts.picks
    import bts.posting
    import bts.saver_state
    import bts.scheduler
    import bts.season_contract
    import bts.strategy
    import bts.web

    _Clock.at = at
    with ExitStack() as stack:
        stack.enter_context(patch.object(dt, "datetime", _Clock))
        for name, mod in list(sys.modules.items()):
            if (name == "bts" or name.startswith("bts.")) and getattr(mod, "datetime", None) is REAL_DATETIME:
                stack.enter_context(patch.object(mod, "datetime", _Clock))
        for m in ("connect", "connect_ex", "sendto"):
            stack.enter_context(patch.object(socket.socket, m, _refuse_net))
        stack.enter_context(patch.object(socket, "getaddrinfo", _refuse_net))
        yield


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
        "units": [
            {"id": 5000 + i, "feedId": int(gpk), "roundId": round_id}
            for i, gpk in enumerate(game_pks)
        ]
    }
    raw = gzip.compress(json.dumps(payload).encode("utf-8"))
    (units_dir / f"{season}0701T120000Z.json.gz").write_bytes(raw)


def main() -> int:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from click.testing import CliRunner
    from bts import (
        cli, contest_fetch, contest_state, daily_decision, orchestrator,
        picks, posting, scheduler, strategy, web,
    )
    from bts.health import contest_state as health_cs
    from bts.health import mdp_policy_alignment, streak_validation, tail_policy as health_tp
    from bts.leaderboard import analysis as lb_analysis, auth, models as lb_models, scraper as lb_scraper
    from scripts.audit.season_walk import walk

    problems: list[str] = []
    notes: list[str] = []
    runner = CliRunner()

    def fail_if(cond: bool, msg: str):
        if cond:
            problems.append(msg)

    # Walk full days for (streak, best, day, expected_action) across BOTH dated and automatic, plus dm and public!
    scenarios = [
        (58, 58, dt.date(2027, 7, 15), "single", "dm"),
        (58, 60, dt.date(2027, 9, 26), "double", "public"),
        (100, 100, dt.date(2027, 7, 15), "single", "dm"),
        (100, 120, dt.date(2027, 9, 26), "double", "dm"),
        (400, 400, dt.date(2027, 7, 15), "single", "dm"),
        (400, 400, dt.date(2027, 9, 26), "double", "dm"),
    ]

    for s_val, b_val, day, want_action, mode in scenarios:
        for kind in ("dated", "automatic"):
            noon = REAL_DATETIME.combine(day, dt.time(12, 0), ET)
            with pinned_offline(noon), tempfile.TemporaryDirectory(prefix=f"h2c-i3-{kind}-{s_val}-") as tmp:
                root = Path(tmp)
                picks_dir = root / "picks"
                picks_dir.mkdir()
                toml_path = root / "orchestrator.toml"
                _write_toml(toml_path, picks_dir, delivery=mode)
                cfg = orchestrator.load_config(toml_path)

                # 1. Install state via real CLI / fetch
                setup_errs = walk.install_state(picks_dir, day, walk.State(kind, s_val, b_val, "used", 0.96))
                fail_if(setup_errs != [], f"{kind} {s_val}/{b_val} {day}: setup failed: {setup_errs}")

                # Check season-contract status CLI output
                r_stat = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks_dir)])
                fail_if(r_stat.exit_code != 0, f"{kind} {s_val}/{b_val}: status exit {r_stat.exit_code}")
                fail_if(f"live decision now: accepted; source=contest; streak={s_val};" not in r_stat.output,
                        f"{kind} {s_val}/{b_val}: status missing live streak: {r_stat.output}")
                if kind == "automatic":
                    fail_if(f"automatic path alone: accepted (streak {s_val}," not in r_stat.output,
                            f"auto {s_val}/{b_val}: status missing auto line: {r_stat.output}")

                # 2. Morning state check
                err_buf = io.StringIO()
                dms_sent: list[tuple[str, str]] = []
                posts_sent: list[str] = []
                with redirect_stderr(err_buf), patch("bts.dm.send_dm", side_effect=lambda r, m: (dms_sent.append((r, m)) or "dm-1")):
                    scheduler._morning_state_check(
                        cfg, picks_dir, day.isoformat(),
                        REAL_DATETIME.combine(day, dt.time(10, 0), ET),
                        REAL_DATETIME.combine(day, dt.time(16, 5), ET),
                    )
                m_out = err_buf.getvalue()
                fail_if(f"Morning state check: {kind} contest state accepted (streak {s_val})." not in m_out or dms_sent != [],
                        f"{kind} {s_val}/{b_val}: morning check unexpected: {m_out!r} dms={dms_sent}")

                # 3. Scheduler selection & save
                frame = walk._predictions(day, 0.96).head(2)
                games = [int(g) for g in frame["game_pk"]]
                statuses = {g: {"abstract": "P", "detailed": "Scheduled"} for g in games}
                with ExitStack() as st:
                    st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame, "synthetic")))
                    st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
                    st.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *a: 4))
                    st.enter_context(patch.object(scheduler, "_trigger_live_forward_capture_on_lock", Mock()))
                    st.enter_context(patch.object(scheduler, "fetch_schedule", lambda *a: [
                        {"gamePk": g, "status": {"detailedState": "Scheduled"}} for g in games
                    ]))
                    res = scheduler.run_single_check(day.isoformat(), games, set(), cfg, 0.01)

                sel = res["selection"]
                fail_if(sel.action != want_action or sel.streak != s_val or sel.decision.objective != "longest_streak",
                        f"{kind} {s_val}/{b_val} {day}: selection wrong: {sel}")
                fail_if((sel.decision.best_supplied, sel.decision.best_status) != (b_val, "trusted"),
                        f"{kind} {s_val}/{b_val}: best metadata wrong: {sel.decision}")

                # 4. Restart with saved pick (both normal reload and empty-cascade fallback refresh)
                reloaded = picks.load_pick(day.isoformat(), picks_dir)
                cur_state = contest_state.load_decision_streak_state(picks_dir, require_contest_state=True, now=noon)
                fail_if(picks.saved_pick_refusal(reloaded, cur_state) is not None,
                        f"{kind} {s_val}/{b_val}: saved_pick_refusal rejected reloaded pick")

                with ExitStack() as st:
                    st.enter_context(patch.object(orchestrator, "run_cascade", lambda *a: (frame.iloc[:0], "synthetic")))
                    st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
                    refresh = scheduler._refresh_pick_at_fallback_decision(cfg, day.isoformat(), reloaded, 0.01)
                fail_if(refresh.daily is not reloaded or refresh.selection is not None,
                        f"{kind} {s_val}/{b_val}: empty fallback refresh unexpected: {refresh}")

                # 5. Deliver and commit (using fresh selection on primary pass; also test cached commit separately below)
                sched_state = scheduler.SchedulerState(day.isoformat(), noon.isoformat(), [], [], [], False, None, None, None)
                with ExitStack() as st:
                    st.enter_context(patch.object(scheduler, "_trigger_live_forward_capture_on_lock", Mock()))
                    st.enter_context(patch.object(scheduler, "fetch_schedule", lambda *a: [
                        {"gamePk": g, "status": {"detailedState": "Scheduled"}} for g in games
                    ]))
                    st.enter_context(patch("bts.dm.send_dm", side_effect=lambda r, m: (dms_sent.append((r, m)) or "dm-pick-1")))
                    st.enter_context(patch.object(posting, "post_to_bluesky", side_effect=lambda t: (posts_sent.append(t) or "at://post/1")))
                    delivered = scheduler._deliver_and_lock_pick(
                        reloaded, cfg, picks_dir, sched_state, day.isoformat(), "lineup lock", selection=sel,
                    )
                fail_if(not delivered, f"{kind} {s_val}/{b_val}: delivery failed")
                sent_text = dms_sent[0][1] if mode == "dm" else posts_sent[0]
                fail_if(f"Streak: {s_val}" not in sent_text, f"{kind} {s_val}/{b_val}: sent text wrong streak: {sent_text}")

                dec_rec = daily_decision.load_decision(day.isoformat(), picks_dir)
                fail_if(
                    (dec_rec["action"], dec_rec["objective"], dec_rec["streak"], dec_rec["best_streak"], dec_rec["best_status"], dec_rec["scoreable"])
                    != (want_action, "longest_streak", s_val, b_val, "trusted", True),
                    f"{kind} {s_val}/{b_val}: decision.json wrong: {dec_rec}",
                )

                # Restart AFTER delivery: run_single_check short-circuits without cascade and keeps decision intact
                with patch.object(orchestrator, "run_cascade", side_effect=AssertionError("should not run cascade")), \
                        patch.object(scheduler, "count_new_confirmations", lambda *a: 0):
                    post_res = scheduler.run_single_check(day.isoformat(), games, set(), cfg, 0.01)
                fail_if(not post_res["pick_result"].locked, f"{kind} {s_val}/{b_val}: post-commit restart not locked")

                # 6. Entry check (30 min before 19:05 ET first pitch = 18:35 ET)
                round_id = 77
                _write_units_snapshot(picks_dir, day.year, round_id, games)
                pending_rows = [{"roundId": round_id, "unitId": 5000, "playerId": 2000, "number": 1, "result": None}]
                crosswalk = {2000: 100, 2001: 101}
                if want_action == "double":
                    pending_rows.append({"roundId": round_id, "unitId": 5001, "playerId": 2001, "number": 2, "result": None})
                entry_prof = {"activeStreak": s_val, "seasonBestStreak": b_val, "predictions": []}
                session = SimpleNamespace(user_id=walk.UID, username=walk.ACCOUNT, xsid="walk_xsid")
                _Clock.at = REAL_DATETIME.combine(day, dt.time(18, 35), ET)
                with ExitStack() as st:
                    st.enter_context(patch.object(auth, "load_session_cookies", lambda: {}))
                    st.enter_context(patch.object(auth, "extract_uid", lambda c: walk.UID))
                    st.enter_context(patch.object(auth, "fetch_login_session", lambda **kw: session))
                    st.enter_context(patch.object(contest_fetch, "fetch_profile", lambda *a: entry_prof))
                    st.enter_context(patch.object(contest_fetch, "fetch_pending_predictions", lambda *a: pending_rows))
                    st.enter_context(patch.object(cli, "_fetch_rounds", lambda: {round_id: day}))
                    st.enter_context(patch.object(cli, "_fetch_bts_to_mlb", lambda: crosswalk))
                    r_ent = runner.invoke(cli.cli, [
                        "check-pick-entered", "--picks-dir", str(picks_dir),
                        "--expected-username", walk.ACCOUNT, "--dm-recipient", "synthetic",
                        "--now-et", f"{day.isoformat()}T18:35:00",
                    ])
                fail_if(r_ent.exit_code != 0 or "pick entered (match)" not in r_ent.output,
                        f"{kind} {s_val}/{b_val}: check-pick-entered failed: {r_ent.exit_code} {r_ent.output}")

                # 7. Day-outcome CLI
                with patch.object(scheduler, "fetch_schedule", lambda *a: [
                    {"gamePk": g, "status": {"detailedState": "Final"}} for g in games
                ]):
                    r_do = runner.invoke(cli.cli, [
                        "day-outcome", "--date", day.isoformat(), "--config", str(toml_path),
                        "--picks-dir", str(picks_dir), "--expected-username", walk.ACCOUNT,
                    ])
                fail_if(r_do.exit_code != 0, f"{kind} {s_val}/{b_val}: day-outcome exit {r_do.exit_code}: {r_do.output}")
                do_body = json.loads(r_do.output.strip())
                fail_if(do_body["state"] != "entered" or do_body["ok"] is not True,
                        f"{kind} {s_val}/{b_val}: day-outcome body: {do_body}")
                if want_action == "single":
                    fail_if("+" in (do_body["pick"] or ""), f"single day-outcome printed runner-up: {do_body}")
                else:
                    fail_if(do_body["pick"] != "Batter 0 + Batter 1", f"double day-outcome pick text: {do_body}")

                # 8. Dashboard & leaderboard rendering at high streak
                snaps_dir = root / "leaderboard" / "leaderboard_snapshots"
                snaps_dir.mkdir(parents=True)
                pq.write_table(
                    pa.Table.from_pylist([
                        {"tab": "active_streak", "streak": 15},
                        {"tab": "active_streak", "streak": 57},
                        {"tab": "active_streak", "streak": 90},
                    ]),
                    snaps_dir / "2027-07-15.parquet",
                )
                rank_info = lb_analysis.percentile_rank(root / "leaderboard", s_val)
                want_above = sum(1 for x in (15, 57, 90) if x > s_val)
                fail_if(rank_info["n_above"] != want_above, f"percentile_rank wrong at {s_val}: {rank_info}")

                # Leaderboard pydantic models also accept >57
                lb_row = lb_models.LeaderboardRow(
                    captured_at=REAL_DATETIME(2027, 7, 15, 12, 0),
                    tab="all_season", rank=1, username="u",
                    streak=s_val, season_best_streak=b_val, active_streak=s_val, user_id=1,
                )
                lb_stats = lb_models.SeasonStats(
                    captured_at=REAL_DATETIME(2027, 7, 15, 12, 0),
                    username="u", user_id=1, best_streak=b_val, active_streak=s_val, pick_accuracy_pct=80.0,
                )
                fail_if(lb_row.season_best_streak != b_val or lb_stats.best_streak != b_val, "leaderboard models clamped")

                with ExitStack() as st:
                    st.enter_context(patch.object(web, "PICKS_DIR", picks_dir))
                    st.enter_context(patch.object(web, "HEALTH_STATE_DIR", root / "health_state"))
                    st.enter_context(patch.object(web, "VALIDATION_DIR", root / "validation"))
                    st.enter_context(patch.object(web, "fetch_bluesky_posts", lambda limit=5: []))
                    st.enter_context(patch.object(web, "_build_live_game_data", lambda pd: ([], None)))
                    html = web.render_page()
                fail_if(f'<div class="streak-number">{s_val}</div>' not in html or "Streak State Error" in html,
                        f"{kind} {s_val}/{b_val}: dashboard HTML wrong")

                # 9. Grading via bts check-results & model streak update (58->59/60, 100->101/102, 400->401/402)
                picks.save_streak(s_val, picks_dir, saver_available=False, season=day.year)
                acct_files_before = {
                    p.name: p.read_bytes()
                    for p in (picks_dir / "account_state").iterdir() if p.is_file()
                }
                _Clock.at = REAL_DATETIME.combine(day + dt.timedelta(days=1), dt.time(1, 0), ET)
                final_statuses = {g: {"abstract": "F", "detailed": "Final", "code": "F"} for g in games}
                with ExitStack() as st:
                    st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: final_statuses))
                    st.enter_context(patch.object(picks, "check_hit", lambda *a, **kw: "hit"))
                    r_cr = runner.invoke(cli.cli, ["check-results", "--date", day.isoformat(), "--picks-dir", str(picks_dir)])
                inc = 2 if want_action == "double" else 1
                fail_if(r_cr.exit_code != 0 or f"Streak: {s_val + inc}" not in r_cr.output,
                        f"{kind} {s_val}/{b_val}: check-results failed: {r_cr.exit_code} {r_cr.output}")
                fail_if(picks.load_streak(picks_dir, season=day.year) != s_val + inc,
                        f"{kind} {s_val}/{b_val}: streak.json did not advance to {s_val + inc}")
                acct_files_after = {
                    p.name: p.read_bytes()
                    for p in (picks_dir / "account_state").iterdir() if p.is_file()
                }
                fail_if(acct_files_before != acct_files_after, f"{kind} {s_val}/{b_val}: check-results mutated contest files!")

                # Health checks on the graded state (streak.json even at 401/402!)
                fail_if(streak_validation.check(picks_dir) != [], f"streak_validation failed at {s_val + inc}")
                fail_if(mdp_policy_alignment.compute_metrics(picks_dir, strategy._load_mdp() and Path("data/models/mdp_policy.npz"), today=day).primary.n != 0,
                        "mdp_policy_alignment did not exclude longest_streak")
                fail_if(health_tp.check(
                    base_path=REPO / "data/models/mdp_policy.npz",
                    tail_path=REPO / "data/models/mdp_tail_policy.npz",
                    today=day,
                ) != [], "tail_policy health check alerted")

    notes.append(f"Walked {len(scenarios) * 2} full-day end-to-end scenarios at 58, 100, and 400 (dated+auto, single+double, dm+public).")

    # Also test `bts run` and `bts orchestrate` and `bts preview` at streak 58 and 400!
    day = dt.date(2027, 7, 15)
    noon = REAL_DATETIME(2027, 7, 15, 17, 0, tzinfo=ET)  # within 3h of 19:05 game -> posts!
    with pinned_offline(noon), tempfile.TemporaryDirectory(prefix="h2c-i3-cli-runners-") as tmp:
        root = Path(tmp)
        for s_val in (58, 400):
            pdir = root / f"run_{s_val}" / "picks"
            pdir.mkdir(parents=True)
            assert walk.install_state(pdir, day, walk.State("dated", s_val, s_val, "used", 0.96)) == []
            frame = walk._predictions(day, 0.96).head(2)
            statuses = {int(g): {"abstract": "P", "detailed": "Scheduled"} for g in frame["game_pk"]}
            posted = []
            with ExitStack() as st:
                st.enter_context(patch("bts.model.predict.run_pipeline", lambda *a, **kw: frame))
                st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
                st.enter_context(patch.object(posting, "post_to_bluesky", side_effect=lambda t: (posted.append(t) or "at://run/1")))
                r_run = runner.invoke(cli.cli, ["run", "--date", day.isoformat(), "--picks-dir", str(pdir), "--models-dir", str(root / "m")])
            fail_if(r_run.exit_code != 0 or len(posted) != 1 or f"Streak: {s_val}" not in posted[0],
                    f"bts run at {s_val} failed: {r_run.exit_code} {r_run.output} posted={posted}")

            # bts preview stamps required=False and saved_pick_refusal refuses it on live path
            pdir_prev = root / f"prev_{s_val}" / "picks"
            pdir_prev.mkdir(parents=True)
            assert walk.install_state(pdir_prev, day, walk.State("dated", s_val, s_val, "used", 0.96)) == []
            with ExitStack() as st:
                st.enter_context(patch("bts.model.predict.run_pipeline", lambda *a, **kw: frame))
                st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
                r_prev = runner.invoke(cli.cli, ["preview", "--date", day.isoformat(), "--picks-dir", str(pdir_prev), "--models-dir", str(root / "m")])
            fail_if(r_prev.exit_code != 0, f"bts preview at {s_val} failed: {r_prev.output}")
            prev_pick = picks.load_pick(day.isoformat(), pdir_prev)
            cur_st = contest_state.load_decision_streak_state(pdir_prev, require_contest_state=True, now=noon)
            fail_if(prev_pick.chosen_on["required"] is not False or picks.saved_pick_refusal(prev_pick, cur_st) is None,
                    f"preview at {s_val} was not refused as saved pick!")
    notes.append("Verified bts run and bts preview at 58 and 400.")

    out_lines = [*notes, f"problems={len(problems)}"]
    for p in problems[:50]:
        out_lines.append(f"PROBLEM: {p}")
    text = "\n".join(out_lines) + "\n"
    (OUT_DIR / "item3_whole_days.log").write_text(text, encoding="utf-8")
    print(text, end="")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
