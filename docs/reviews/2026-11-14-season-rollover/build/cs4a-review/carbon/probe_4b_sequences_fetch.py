"""Adversarial Probe 4b: Operator Sequences with Real Fetch Auto-Earn Promotion (Item 7)."""
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


def install_no_network_hook() -> None:
    forbidden = {
        "socket.connect", "socket.connect_ex", "socket.getaddrinfo",
        "socket.getnameinfo", "socket.gethostbyname", "socket.gethostbyaddr", "socket.sendto",
    }

    def hook(event: str, args: tuple) -> None:
        if event in forbidden:
            raise RuntimeError(f"Network access forbidden in probe: {event} {args}")

    sys.addaudithook(hook)


def make_clock(now_val: dt.datetime):
    class FixedClock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return now_val.astimezone(tz) if tz else now_val.replace(tzinfo=None)

    return FixedClock


def main() -> None:
    install_no_network_hook()
    sys.path.insert(0, str(BRANCH_SRC))

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

    # Find a probability on 2027-04-02 where streak=10 flips action between saver=False and saver=True!
    flip_p = 0.84
    for cand_i in range(65, 96):
        p = round(cand_i / 100.0, 2)
        c = strat.DecisionContext(
            primary_p=p, second_p=round(p - 0.01, 2), has_diff_game=True,
            date="2027-04-02", allow_double=True, mdp=mdp, best_streak=10, best_status="trusted",
        )
        if strat.resolve_policy_decision(c, 10, False).action != strat.resolve_policy_decision(c, 10, True).action:
            flip_p = p
            break

    def eval_day(picks_dir: Path, now_val: dt.datetime, p1: float = 0.84) -> dict:
        ds = cs.load_decision_streak_state(picks_dir, require_contest_state=True, now=now_val)
        ctx = strat.DecisionContext(
            primary_p=p1, second_p=round(p1 - 0.01, 2), has_diff_game=True,
            date=now_val.astimezone(ET).date().isoformat(),
            allow_double=ds.allow_double, mdp=mdp,
            best_streak=ds.best_streak, best_status=ds.best_status,
        )
        dec = strat.resolve_policy_decision(ctx, ds.streak, ds.saver_available)
        alerts = health.check(picks_dir, expected=True, now=now_val)
        return {
            "date": now_val.astimezone(ET).date().isoformat(),
            "p_game_hit": p1,
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

        # Initial disk state: 2026 auto file, 2026 streak.json, 2026 'used' saver_state.json
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

        # 1. Opening Night (2027-03-25): observation streak 0 not_earned
        t1 = dt.datetime(2027, 3, 25, 11, 0, tzinfo=ET)
        c1 = make_clock(t1)
        with patch.object(dt, "datetime", c1), patch.object(cs, "datetime", c1):
            assert runner.invoke(cli, [
                "set-contest-streak", "--picks-dir", str(picks_dir),
                "--date", "2027-03-25", "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
            ]).exit_code == 0
        s1 = eval_day(picks_dir, t1)
        steps_log.append({"step": "1_opening_night_0_not_earned", **s1})
        if s1["streak"] != 0 or s1["saver_available"] is not True or s1["alerts"] != []:
            failures.append(f"Step 1 failed: {s1}")

        # 2. Next morning (2027-03-26 09:00 ET) before any new observation
        t2 = dt.datetime(2027, 3, 26, 9, 0, tzinfo=ET)
        s2 = eval_day(picks_dir, t2)
        steps_log.append({"step": "2_next_morning_before_obs", **s2})
        if s2["streak"] != 22 or s2["saver_flag_state"] != "uninitialized" or s2["saver_available"] is not False:
            failures.append(f"Step 2 failed: {s2}")

        # 3. Observation on 2027-03-26 corrected twice in one day (1 -> 2 -> 3) + duplicate no-op
        for st_val, m_off in [(1, 10), (2, 20), (3, 30), (3, 40)]:
            ts = dt.datetime(2027, 3, 26, 10, m_off, tzinfo=ET)
            cs_clk = make_clock(ts)
            with patch.object(dt, "datetime", cs_clk), patch.object(cs, "datetime", cs_clk):
                assert runner.invoke(cli, [
                    "set-contest-streak", "--picks-dir", str(picks_dir),
                    "--date", "2027-03-26", "--streak", str(st_val), "--best-streak", str(st_val), "--saver", "not_earned",
                ]).exit_code == 0
        s3 = eval_day(picks_dir, dt.datetime(2027, 3, 26, 11, 0, tzinfo=ET))
        req_count = len((sdir / "contest_manual_requests.jsonl").read_text(encoding="utf-8").splitlines())
        steps_log.append({"step": "3_corrected_twice_same_day", "request_log_lines": req_count, **s3})
        if s3["streak"] != 3 or req_count != 4:
            failures.append(f"Step 3 failed: {s3} req_count={req_count}")

        # 4. Legacy set-contest-streak replaces dated observation and cold-inits 2027 flag to not_earned
        t4 = dt.datetime(2027, 3, 26, 12, 0, tzinfo=ET)
        c4 = make_clock(t4)
        with patch.object(dt, "datetime", c4), patch.object(cs, "datetime", c4), patch.object(flags, "datetime", c4):
            assert runner.invoke(cli, [
                "set-contest-streak", "--picks-dir", str(picks_dir), "--streak", "4", "--best-streak", "4",
            ]).exit_code == 0
        s4 = eval_day(picks_dir, t4)
        steps_log.append({
            "step": "4_replaced_by_legacy_setter",
            "manual_schema": json.loads((sdir / "contest_streak.manual.json").read_text())["schema_version"],
            "flag_state_on_disk": json.loads((sdir / "saver_state.json").read_text())["state"],
            **s4,
        })

        # 5. Streak reaches 10 with not_earned -> warning -> paste --saver active correction!
        t5 = dt.datetime(2027, 4, 2, 11, 0, tzinfo=ET)
        c5 = make_clock(t5)
        with patch.object(dt, "datetime", c5), patch.object(cs, "datetime", c5):
            assert runner.invoke(cli, [
                "set-contest-streak", "--picks-dir", str(picks_dir),
                "--date", "2027-04-02", "--streak", "10", "--best-streak", "10", "--saver", "not_earned",
            ]).exit_code == 0
            s5a = eval_day(picks_dir, t5, p1=flip_p)
            w5 = health.check(picks_dir, expected=True, now=t5)
            cmd_active = [c for c in w5[0].message.split("`")[1::2] if "--saver active" in c][0]
            assert runner.invoke(cli, shlex.split(cmd_active)[1:]).exit_code == 0
            s5b = eval_day(picks_dir, t5, p1=flip_p)
        steps_log.append({"step": "5a_streak10_not_earned_warns", **s5a})
        steps_log.append({"step": "5b_streak10_corrected_to_active", **s5b})
        if s5a["saver_available"] is not False or s5b["saver_available"] is not True or s5b["alerts"] != []:
            failures.append(f"Step 5 failed: 5a={s5a}, 5b={s5b}")

        # 6. Forgiven miss entered next day (2027-04-05) as streak=12 saver=used
        t6 = dt.datetime(2027, 4, 5, 11, 0, tzinfo=ET)
        c6 = make_clock(t6)
        with patch.object(dt, "datetime", c6), patch.object(cs, "datetime", c6):
            assert runner.invoke(cli, [
                "set-contest-streak", "--picks-dir", str(picks_dir),
                "--date", "2027-04-05", "--streak", "12", "--best-streak", "12", "--saver", "used",
            ]).exit_code == 0
        s6 = eval_day(picks_dir, t6, p1=0.84)
        steps_log.append({"step": "6_forgiven_miss_entered_as_used", **s6})

        # 7. Real fetch-contest-streak runs in between on 2027-04-05:
        # On disk saver_state.json is still 'not_earned' (from Step 4); fetch sees bestStreak=12 in 2027 and auto-earns flag -> 'active'!
        assert json.loads((sdir / "saver_state.json").read_text())["state"] == "not_earned"

        @dataclass
        class FakeSession:
            user_id: int = 4242
            username: str = "operator_bts"
            xsid: str = "fake-xsid"

        fake_profile = {
            "activeStreak": 12,
            "seasonBestStreak": 12,
            "predictions": [{"roundId": 101, "result": "hit"}],
        }
        fake_rounds = {101: dt.date(2027, 4, 4)}

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
            r7 = runner.invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks_dir)])
        assert r7.exit_code == 0, r7.output

        flag_after_fetch = json.loads((sdir / "saver_state.json").read_text(encoding="utf-8"))
        s7_same_day = eval_day(picks_dir, t7, p1=0.84)
        # Also check NEXT morning (2027-04-06) before a new observation: THEN the auto-earned 'active' flag WOULD take effect!
        s7_next_day_no_obs = eval_day(picks_dir, dt.datetime(2027, 4, 6, 9, 0, tzinfo=ET), p1=0.84)

        steps_log.append({
            "step": "7a_same_day_after_fetch_auto_earns_flag",
            "flag_on_disk": flag_after_fetch["state"],
            **s7_same_day,
        })
        steps_log.append({
            "step": "7b_next_day_without_obs_sees_auto_earned_flag",
            "flag_on_disk": flag_after_fetch["state"],
            **s7_next_day_no_obs,
        })

        if flag_after_fetch["state"] != "active":
            failures.append(f"Expected fetch to auto-earn flag to active, got {flag_after_fetch}")
        if s7_same_day["saver_available"] is not False or s7_same_day["saver_flag_state"] != "used":
            failures.append(f"Fetch changed same-day dated decision! {s7_same_day}")
        if s7_next_day_no_obs["saver_available"] is not True or s7_next_day_no_obs["saver_flag_state"] != "active":
            failures.append(f"Expected next-day fallback to see auto-earned active flag, got {s7_next_day_no_obs}")

    out = {"steps": steps_log, "failures": failures}
    (REVIEW_DIR / "probe4b_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
