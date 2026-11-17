"""Adversarial Probe 4: Health Sweep, DM Pipeline (Item 6) & Operator Sequences (Item 7)."""
from __future__ import annotations

import datetime as dt
import json
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[3]
REVIEW_DIR = REPO_ROOT / ".codex-review" / "cs4a-review"
BRANCH_SRC = REPO_ROOT / "src"

ET = ZoneInfo("America/New_York")
SCHEMA = "bts_contest_streak_manual_v3"


def install_audit_hooks(flag_reads: list[str]) -> None:
    forbidden = {
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.getnameinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.sendto",
    }

    def hook(event: str, args: tuple) -> None:
        if event in forbidden:
            raise RuntimeError(f"Network access forbidden in probe: {event} {args}")
        if event == "open" and args:
            p = str(args[0])
            if p.endswith("saver_state.json"):
                mode = str(args[1]) if len(args) > 1 else ""
                if "r" in mode or mode == "":
                    flag_reads.append(p)

    sys.addaudithook(hook)


def make_clock(now_val: dt.datetime):
    class FixedClock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return now_val.astimezone(tz) if tz else now_val.replace(tzinfo=None)

    return FixedClock


def run_health_and_dm_checks(flag_reads: list[str]) -> dict:
    from click.testing import CliRunner
    import bts.contest_state as cs
    import bts.health.contest_state as health
    import bts.saver_state as flags
    from bts.cli import cli
    from bts.health.alert import dispatch_dm_for_health_alerts
    from bts.health.attention import apply_warn_attention_policy

    failures: list[str] = []
    now = dt.datetime(2027, 7, 15, 20, 0, tzinfo=ET)
    day = "2027-07-15"
    eod = dt.datetime.combine(dt.date(2027, 7, 15), dt.time.max, ET).isoformat()

    auto_variants = {
        "absent": None,
        "valid_2027": json.dumps({"active_streak": 4, "best_streak": 9, "source_date": "2027-07-14"}),
        "undated": json.dumps({"active_streak": 4}),
        "refused_58": json.dumps({"active_streak": 58, "source_date": "2027-07-14"}),
        "corrupt": "{corrupt",
    }

    sweep_cells = 0
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "health_sweep"
        sdir = pdir / "account_state"
        sdir.mkdir(parents=True)
        # Put 5,000-digit integer in saver_state.json so any read explodes AND triggers audit hook!
        (sdir / "saver_state.json").write_text('{"state":"active","season":' + ("9" * 5000) + "}", encoding="utf-8")
        # Put settled pick up through today (normal EOD state!)
        (pdir / "2027-07-14.json").write_text('{"result":"hit"}', encoding="utf-8")
        (pdir / "2027-07-15.json").write_text('{"result":"hit"}', encoding="utf-8")

        flag_reads.clear()
        for a_name, a_text in auto_variants.items():
            ap = sdir / "contest_streak.json"
            if ap.exists():
                ap.unlink()
            if a_text is not None:
                ap.write_text(a_text, encoding="utf-8")

            for exp in (False, True):
                for sv in ("not_earned", "active", "used"):
                    for streak in range(58):
                        sweep_cells += 1
                        best = max(streak, 12)
                        (sdir / "contest_streak.manual.json").write_text(json.dumps({
                            "schema_version": SCHEMA,
                            "effective_date": day,
                            "season": 2027,
                            "expires_at": eod,
                            "active_streak": streak,
                            "best_streak": best,
                            "saver_state": sv,
                            "recorded_at": "2027-07-15T16:00:00Z",
                            "source": "manual_cli",
                        }), encoding="utf-8")
                        alerts = health.check(pdir, expected=exp, now=now)
                        should_warn = (sv == "not_earned" and 10 <= streak <= 15)
                        if should_warn:
                            if len(alerts) != 1 or alerts[0].level != "WARN" or alerts[0].source != "saver_state":
                                failures.append(f"Expected 1 saver_state WARN at {a_name}/{exp}/{sv}/{streak}, got {alerts}")
                        else:
                            if alerts != []:
                                failures.append(f"Expected [] at {a_name}/{exp}/{sv}/{streak}, got {alerts}")

        if flag_reads:
            failures.append(f"saver_state.json read during health sweep: {flag_reads}")

        # Paste ALL THREE commands from Row 11's warning message into the real CLI!
        paste_dir = Path(tmp) / "dir with 'spaces' & chars"
        clk = make_clock(now)
        runner = CliRunner()
        with patch.object(dt, "datetime", clk), patch.object(cs, "datetime", clk):
            res0 = runner.invoke(cli, [
                "set-contest-streak", "--picks-dir", str(paste_dir),
                "--date", day, "--streak", "12", "--best-streak", "14", "--saver", "not_earned",
                "--reason", "initial",
            ])
            assert res0.exit_code == 0, res0.output
            w_alerts = health.check(paste_dir, expected=True, now=now)
            assert len(w_alerts) == 1 and w_alerts[0].source == "saver_state"
            warn_msg = w_alerts[0].message
            backticked = warn_msg.split("`")[1::2]
            cmds = [c for c in backticked if c.startswith("bts set-contest-streak")]
            if len(cmds) != 3:
                failures.append(f"Expected 3 commands in warning, got {cmds}")

            # Command 1: --saver active
            r1 = runner.invoke(cli, shlex.split(cmds[0])[1:])
            ds1 = cs.load_decision_streak_state(paste_dir, now=now)
            al1 = health.check(paste_dir, expected=True, now=now)
            if r1.exit_code != 0 or ds1.saver_available is not True or ds1.saver_flag_state != "active" or al1 != []:
                failures.append(f"Cmd 1 failed: code={r1.exit_code}, ds={ds1}, alerts={al1}")

            # Reset to not_earned at 12, then run Command 2: --saver used
            runner.invoke(cli, ["set-contest-streak", "--picks-dir", str(paste_dir), "--date", day, "--streak", "12", "--best-streak", "14", "--saver", "not_earned"])
            r2 = runner.invoke(cli, shlex.split(cmds[1])[1:])
            ds2 = cs.load_decision_streak_state(paste_dir, now=now)
            al2 = health.check(paste_dir, expected=True, now=now)
            if r2.exit_code != 0 or ds2.saver_available is not False or ds2.saver_flag_state != "used" or al2 != []:
                failures.append(f"Cmd 2 failed: code={r2.exit_code}, ds={ds2}, alerts={al2}")

            # Reset to not_earned at 12, then run Command 3 with N -> 8 and B -> 9 as MLB shows!
            runner.invoke(cli, ["set-contest-streak", "--picks-dir", str(paste_dir), "--date", day, "--streak", "12", "--best-streak", "14", "--saver", "not_earned"])
            cmd3_concrete = cmds[2].replace("--streak N --best-streak B", "--streak 8 --best-streak 9")
            r3 = runner.invoke(cli, shlex.split(cmd3_concrete)[1:])
            ds3 = cs.load_decision_streak_state(paste_dir, now=now)
            al3 = health.check(paste_dir, expected=True, now=now)
            if r3.exit_code != 0 or ds3.streak != 8 or ds3.saver_available is not True or ds3.saver_flag_state != "not_earned" or al3 != []:
                failures.append(f"Cmd 3 failed: code={r3.exit_code}, ds={ds3}, alerts={al3}")

        # Check coverage-lag behavior if future-dated settled picks exist beside an in-effect dated observation (e.g. clock rollback)
        rollback_dir = Path(tmp) / "rollback_cov"
        (rollback_dir / "account_state").mkdir(parents=True)
        (rollback_dir / "account_state" / "contest_streak.manual.json").write_text(json.dumps({
            "schema_version": SCHEMA, "effective_date": day, "season": 2027, "expires_at": eod,
            "active_streak": 2, "best_streak": 5, "saver_state": "not_earned",
            "recorded_at": "2027-07-15T16:00:00Z", "source": "manual_cli",
        }), encoding="utf-8")
        (rollback_dir / "2027-07-16.json").write_text('{"result":"hit"}', encoding="utf-8")
        (rollback_dir / "2027-07-17.json").write_text('{"result":"miss"}', encoding="utf-8")
        rollback_alerts_exp_true = [
            {"level": a.level, "source": a.source, "message": a.message.replace(str(rollback_dir), "<DIR>")}
            for a in health.check(rollback_dir, expected=True, now=now)
        ]

        # DM pipeline across Day 1 (with observation) -> Day 2 (without observation)
        dm_scenarios = {}
        for scen_name, day1_streak, has_2026_auto in [
            ("opening_healthy_obs_then_missing_next_day_with_2026_auto", 0, True),
            ("opening_healthy_obs_then_missing_next_day_no_auto", 0, False),
            ("contradictory_12_not_earned_then_missing_next_day_with_2026_auto", 12, True),
        ]:
            ddir = Path(tmp) / scen_name / "picks"
            (ddir / "account_state").mkdir(parents=True)
            hstate = ddir.parent / "health_state"
            hstate.mkdir(parents=True)
            if has_2026_auto:
                (ddir / "account_state" / "contest_streak.json").write_text(json.dumps({
                    "schema_version": "bts_contest_streak_auto_v1",
                    "active_streak": 22, "best_streak": 30, "source": "mlb_bts_profile",
                    "source_date": "2026-09-27", "recorded_at": "2026-09-28T10:00:00Z",
                }), encoding="utf-8")
            (ddir / "account_state" / "saver_state.json").write_text(
                json.dumps({"state": "used", "season": 2026}), encoding="utf-8"
            )
            # Day 1: 2027-03-25 with dated observation
            d1 = dt.date(2027, 3, 25)
            now1 = dt.datetime(2027, 3, 25, 23, 30, tzinfo=ET)
            clk1 = make_clock(now1)
            with patch.object(dt, "datetime", clk1), patch.object(cs, "datetime", clk1):
                runner.invoke(cli, [
                    "set-contest-streak", "--picks-dir", str(ddir),
                    "--date", "2027-03-25", "--streak", str(day1_streak),
                    "--best-streak", str(day1_streak), "--saver", "not_earned",
                ])
            a_d1 = health.check(ddir, expected=True, now=now1)
            pc1, wa1 = apply_warn_attention_policy(a_d1, state_path=hstate / "warn_attention.json", today=d1)
            dms_d1: list[str] = []
            with patch("bts.health.alert.send_dm", side_effect=lambda r, b: dms_d1.append(b)):
                dispatch_dm_for_health_alerts(
                    a_d1 + pc1, "operator.bsky.social", warn_attention=wa1,
                    status_path=hstate / "dm_status.json", now_et_date=d1,
                )

            # Day 2: 2027-03-26 WITHOUT new observation
            d2 = dt.date(2027, 3, 26)
            now2 = dt.datetime(2027, 3, 26, 23, 30, tzinfo=ET)
            a_d2 = health.check(ddir, expected=True, now=now2)
            pc2, wa2 = apply_warn_attention_policy(a_d2, state_path=hstate / "warn_attention.json", today=d2)
            dms_d2: list[str] = []
            with patch("bts.health.alert.send_dm", side_effect=lambda r, b: dms_d2.append(b)):
                dispatch_dm_for_health_alerts(
                    a_d2 + pc2, "operator.bsky.social", warn_attention=wa2,
                    status_path=hstate / "dm_status.json", now_et_date=d2,
                )

            dm_scenarios[scen_name] = {
                "day1_dms": [s.replace(str(ddir), "<PICKS>") for s in dms_d1],
                "day2_dms": [s.replace(str(ddir), "<PICKS>") for s in dms_d2],
            }

    return {
        "sweep_cells": sweep_cells,
        "warning_message_sample": warn_msg.replace(str(paste_dir), "<PICKS_WITH_SPACES>"),
        "pasted_commands_count": len(cmds),
        "future_settled_picks_coverage_alerts": rollback_alerts_exp_true,
        "dm_scenarios": dm_scenarios,
        "failures": failures,
    }


def run_operator_sequences() -> dict:
    """Item 7: Real operator sequences on one directory through real CLI, real decision loader, real policy & real fetch."""
    from click.testing import CliRunner
    import bts.contest_state as cs
    import bts.health.contest_state as health
    import bts.saver_state as flags
    import bts.strategy as strat
    from bts.cli import cli

    mdp = strat._load_mdp()
    runner = CliRunner()
    failures: list[str] = []
    steps_log: list[dict] = []

    def eval_day(picks_dir: Path, now_val: dt.datetime, p1: float = 0.84, p2: float = 0.83) -> dict:
        ds = cs.load_decision_streak_state(picks_dir, require_contest_state=True, now=now_val)
        ctx = strat.DecisionContext(
            primary_p=p1, second_p=p2, has_diff_game=True,
            date=now_val.astimezone(ET).date().isoformat(),
            allow_double=ds.allow_double, mdp=mdp,
            best_streak=ds.best_streak, best_status=ds.best_status,
        )
        dec = strat.resolve_policy_decision(ctx, ds.streak, ds.saver_available)
        alerts = health.check(picks_dir, expected=True, now=now_val)
        return {
            "date": now_val.astimezone(ET).date().isoformat(),
            "streak": ds.streak,
            "best_streak": ds.best_streak,
            "best_status": ds.best_status,
            "status": ds.status,
            "saver_flag_state": ds.saver_flag_state,
            "saver_available": ds.saver_available,
            "policy_action": dec.action,
            "alerts": [{"level": a.level, "source": a.source} for a in alerts],
        }

    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "live_box" / "picks"
        sdir = picks_dir / "account_state"
        sdir.mkdir(parents=True)

        # Seed last season's automatic file, streak.json, and 'used' saver flag on disk
        (sdir / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1",
            "active_streak": 22, "best_streak": 30, "source": "mlb_bts_profile",
            "source_date": "2026-09-27", "recorded_at": "2026-09-28T10:00:00Z",
            "user_id": 4242, "username": "operator_bts",
        }), encoding="utf-8")
        (picks_dir / "streak.json").write_text(json.dumps({
            "season": 2026, "streak": 15, "saver_available": False,
        }), encoding="utf-8")
        (sdir / "saver_state.json").write_text(json.dumps({
            "season": 2026, "state": "used", "source": "cli", "updated_at": "2026-09-10T00:00:00+00:00",
        }), encoding="utf-8")

        # Sequence 1: Opening Night (2027-03-25), supply streak 0 not_earned
        t1 = dt.datetime(2027, 3, 25, 11, 0, tzinfo=ET)
        c1 = make_clock(t1)
        with patch.object(dt, "datetime", c1), patch.object(cs, "datetime", c1):
            r1 = runner.invoke(cli, [
                "set-contest-streak", "--picks-dir", str(picks_dir),
                "--date", "2027-03-25", "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
            ])
        assert r1.exit_code == 0, r1.output
        s1_eval = eval_day(picks_dir, t1)
        steps_log.append({"step": "1_opening_night_0_not_earned", **s1_eval})
        if s1_eval["streak"] != 0 or s1_eval["saver_available"] is not True or s1_eval["alerts"] != []:
            failures.append(f"Step 1 wrong: {s1_eval}")
        # Verify 2026 saver flag on disk is untouched
        if json.loads((sdir / "saver_state.json").read_text())["season"] != 2026:
            failures.append("Step 1 touched 2026 saver_state.json!")

        # Sequence 2: Next morning (2027-03-26 09:00 ET) BEFORE any new observation
        t2 = dt.datetime(2027, 3, 26, 9, 0, tzinfo=ET)
        s2_eval = eval_day(picks_dir, t2)
        steps_log.append({"step": "2_next_morning_before_obs", **s2_eval})
        # Falls back to 2026 auto file (streak=22) + uninitialized 2027 saver (saver_available=False) + saver_state WARN
        if s2_eval["streak"] != 22 or s2_eval["saver_flag_state"] != "uninitialized" or s2_eval["saver_available"] is not False:
            failures.append(f"Step 2 wrong: {s2_eval}")

        # Sequence 3: Supply observation on 2027-03-26, then correct streak TWICE in one day (1 -> 2 -> 3) + duplicate no-op
        for sub_idx, (st_val, min_off) in enumerate([(1, 10), (2, 20), (3, 30), (3, 40)]):
            t_sub = dt.datetime(2027, 3, 26, 10, min_off, tzinfo=ET)
            c_sub = make_clock(t_sub)
            with patch.object(dt, "datetime", c_sub), patch.object(cs, "datetime", c_sub):
                r_sub = runner.invoke(cli, [
                    "set-contest-streak", "--picks-dir", str(picks_dir),
                    "--date", "2027-03-26", "--streak", str(st_val), "--best-streak", str(st_val), "--saver", "not_earned",
                ])
            assert r_sub.exit_code == 0, r_sub.output
            if sub_idx == 3 and "unchanged" not in r_sub.output:
                failures.append("Duplicate re-run in Step 3 was not unchanged!")
        s3_eval = eval_day(picks_dir, dt.datetime(2027, 3, 26, 11, 0, tzinfo=ET))
        req_lines = (sdir / "contest_manual_requests.jsonl").read_text(encoding="utf-8").splitlines()
        steps_log.append({"step": "3_corrected_twice_same_day", "request_log_lines": len(req_lines), **s3_eval})
        # 1 from Opening Night + 3 from today (1, 2, 3; 4th was no-op) = 4 lines
        if s3_eval["streak"] != 3 or len(req_lines) != 4:
            failures.append(f"Step 3 wrong: eval={s3_eval}, lines={len(req_lines)}")

        # Sequence 4: Dated observation in effect, then LEGACY-mode set-contest-streak replaces it!
        t4 = dt.datetime(2027, 3, 26, 12, 0, tzinfo=ET)
        c4 = make_clock(t4)
        with patch.object(dt, "datetime", c4), patch.object(cs, "datetime", c4), patch.object(flags, "datetime", c4):
            r4 = runner.invoke(cli, [
                "set-contest-streak", "--picks-dir", str(picks_dir),
                "--streak", "4", "--best-streak", "4",
            ])
        assert r4.exit_code == 0, r4.output
        manual_after_legacy = json.loads((sdir / "contest_streak.manual.json").read_text(encoding="utf-8"))
        flag_after_legacy = json.loads((sdir / "saver_state.json").read_text(encoding="utf-8"))
        s4_eval = eval_day(picks_dir, t4)
        steps_log.append({
            "step": "4_replaced_by_legacy_setter",
            "manual_schema": manual_after_legacy.get("schema_version"),
            "flag_on_disk": flag_after_legacy,
            **s4_eval,
        })
        if manual_after_legacy.get("schema_version") != "bts_contest_streak_manual_v2" or flag_after_legacy.get("season") != 2027 or flag_after_legacy.get("state") != "not_earned":
            failures.append(f"Step 4 legacy replacement unexpected: {manual_after_legacy} / {flag_after_legacy}")

        # Sequence 5: Streak reaching 10 with not_earned -> warning -> corrected to active!
        t5 = dt.datetime(2027, 4, 2, 11, 0, tzinfo=ET)
        c5 = make_clock(t5)
        with patch.object(dt, "datetime", c5), patch.object(cs, "datetime", c5):
            r5a = runner.invoke(cli, [
                "set-contest-streak", "--picks-dir", str(picks_dir),
                "--date", "2027-04-02", "--streak", "10", "--best-streak", "10", "--saver", "not_earned",
            ])
            assert r5a.exit_code == 0
            s5a_eval = eval_day(picks_dir, t5, p1=0.82, p2=0.81)  # at streak=10, p=0.82/0.81: false->double, true->single!
            warn5 = health.check(picks_dir, expected=True, now=t5)
            cmd_active = [c for c in warn5[0].message.split("`")[1::2] if "--saver active" in c][0]
            r5b = runner.invoke(cli, shlex.split(cmd_active)[1:])
            assert r5b.exit_code == 0
            s5b_eval = eval_day(picks_dir, t5, p1=0.82, p2=0.81)
        steps_log.append({"step": "5a_streak10_not_earned_warns", **s5a_eval})
        steps_log.append({"step": "5b_streak10_corrected_to_active", **s5b_eval})
        if s5a_eval["saver_available"] is not False or len(s5a_eval["alerts"]) != 1:
            failures.append(f"Step 5a wrong: {s5a_eval}")
        if s5b_eval["saver_available"] is not True or s5b_eval["alerts"] != [] or s5a_eval["policy_action"] == s5b_eval["policy_action"]:
            failures.append(f"Step 5b wrong or policy action did not change: 5a={s5a_eval}, 5b={s5b_eval}")

        # Sequence 6 & 7: Miss forgiven by contest -> entered next day (2027-04-05) as streak=12 saver=used;
        # THEN fetch-contest-streak runs in between (seeing bestStreak=12 in 2027 and auto-earning saver_state.json not_earned -> active)!
        t6 = dt.datetime(2027, 4, 5, 11, 0, tzinfo=ET)
        c6 = make_clock(t6)
        with patch.object(dt, "datetime", c6), patch.object(cs, "datetime", c6):
            r6 = runner.invoke(cli, [
                "set-contest-streak", "--picks-dir", str(picks_dir),
                "--date", "2027-04-05", "--streak", "12", "--best-streak", "12", "--saver", "used",
            ])
        assert r6.exit_code == 0
        s6_eval = eval_day(picks_dir, t6, p1=0.84, p2=0.83)
        steps_log.append({"step": "6_forgiven_miss_entered_as_used", **s6_eval})

        # Confirm saver_state.json on disk is still 'not_earned' from Step 4 before fetch runs!
        assert json.loads((sdir / "saver_state.json").read_text())["state"] == "not_earned"

        @dataclass
        class FakeSession:
            user_id: int = 4242
            username: str = "operator_bts"
            xsid: str = "fake-xsid"

        fake_profile = {
            "activeStreak": 12,
            "seasonBestStreak": 12,
            "predictions": [{"roundId": 101, "status": "hit"}],
        }
        fake_rounds = [{"id": 101, "date": "2027-04-04"}]

        t7 = dt.datetime(2027, 4, 5, 13, 0, tzinfo=ET)
        c7 = make_clock(t7)
        with (
            patch.object(dt, "datetime", c7),
            patch.object(cs, "datetime", c7),
            patch.object(flags, "datetime", c7),
            patch("bts.leaderboard.auth.load_session_cookies", return_value={"uid": "4242"}),
            patch("bts.leaderboard.auth.extract_uid", return_value=4242),
            patch("bts.leaderboard.auth.fetch_login_session", return_value=FakeSession()),
            patch("bts.contest_fetch.fetch_profile", return_value=fake_profile),
            patch("bts.cli._fetch_rounds", return_value=fake_rounds),
        ):
            r7_fetch = runner.invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks_dir)])
        assert r7_fetch.exit_code == 0, r7_fetch.output

        flag_after_fetch = json.loads((sdir / "saver_state.json").read_text(encoding="utf-8"))
        s7_eval = eval_day(picks_dir, t7, p1=0.84, p2=0.83)
        steps_log.append({
            "step": "7_after_fetch_auto_earns_flag_same_day",
            "flag_on_disk_after_fetch": flag_after_fetch["state"],
            **s7_eval,
        })
        if flag_after_fetch["state"] != "active":
            failures.append(f"Expected fetch to auto-earn flag to active, got {flag_after_fetch}")
        if s7_eval["saver_available"] is not False or s7_eval["saver_flag_state"] != "used" or s7_eval["policy_action"] != s6_eval["policy_action"]:
            failures.append(f"Fetch auto-earning flag changed today's dated decision! s6={s6_eval}, s7={s7_eval}")

    return {"steps": steps_log, "failures": failures}


def main() -> None:
    flag_reads: list[str] = []
    install_audit_hooks(flag_reads)
    sys.path.insert(0, str(BRANCH_SRC))

    health_dm = run_health_and_dm_checks(flag_reads)
    sequences = run_operator_sequences()

    out = {"health_and_dms": health_dm, "operator_sequences": sequences}
    (REVIEW_DIR / "probe4_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if health_dm["failures"] or sequences["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
