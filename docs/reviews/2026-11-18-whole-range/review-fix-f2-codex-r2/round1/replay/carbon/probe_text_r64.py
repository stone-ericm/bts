"""Item 4 probe: R64 operator-facing text rendering and worst-ordinary-state walkthroughs against B8a and real CLI."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts import cli, contest_state
from bts.health import alert, contest_state as health, runner
from bts.saver_state import load_saver_state, transition_saver_state

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = REAL_DT(2027, 7, 15, 22, 30, tzinfo=ET)
NYE = REAL_DT(2027, 12, 31, 23, 50, tzinfo=ET)
ACT_ID = "b" * 32


def make_clock(fixed_now):
    class PinnedDatetime(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)
    return PinnedDatetime


def invoke_cli(monkeypatch, now: dt.datetime, *args: str):
    clk = make_clock(now)
    with monkeypatch.context() as mp:
        for mod in (dt, cli, contest_state, alert, health, runner):
            mp.setattr(mod, "datetime", clk)
        return CliRunner().invoke(cli.cli, list(args))


def write_switch_and_auto(picks: Path, now: dt.datetime, streak: int = 12, best: int = 15) -> None:
    acct = picks / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    year = now.astimezone(ET).year
    (acct / "season_contract.json").write_text(json.dumps({
        "schema_version": "bts_season_contract_v1",
        "season": year,
        "status": "automatic",
        "activation_id": ACT_ID,
        "switched_at": f"{year}-04-01T12:00:00Z",
        "revoked_at": None,
        "note": None,
    }), encoding="utf-8")
    (acct / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v2",
        "season": year,
        "activation_id": ACT_ID,
        "active_streak": streak,
        "best_streak": best,
        "source": "mlb_bts_profile",
        "source_date": (now.astimezone(ET).date() - dt.timedelta(days=1)).isoformat(),
        "recorded_at": now.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }), encoding="utf-8")


def test_rendered_messages_match_contract_and_grapheme_bounds(tmp_path, monkeypatch):
    picks = tmp_path / "picks with 'quote' and oom"
    assert invoke_cli(
        monkeypatch, DAY,
        "set-contest-streak", "--date", "2027-07-15", "--streak", "14", "--best-streak", "15",
        "--saver", "used", "--picks-dir", str(picks),
    ).exit_code == 0

    for st in ("active", "not_earned"):
        (picks / "account_state" / "saver_state.json").write_text(
            json.dumps({"season": 2027, "state": st}), encoding="utf-8"
        )
        live = health.check(picks, expected=True, now=DAY)
        assert len(live) == 1
        expected_live = (
            f"Streak Saver on 2027-07-15 ET. The dated observation says the saver was used. "
            f"The flag for season 2027 read {st} at this check. A later automatic decision may count "
            f"a saver as available. Verify the current season's saver and prior entries in MLB before "
            f"any correction. Use `bts season-contract status` for the scheduler's picks directory "
            f"and follow checklist B8a for any correction."
        )
        assert live[0].message == expected_live
        assert str(picks) not in live[0].message
        assert "--use" not in live[0].message
        dm_body = alert.format_health_dm_body([], [alert.Alert("WARN", "saver_state", live[0].message + " (1st observed day)")])
        assert len(dm_body) < 600

        hist = health.check(picks, expected=True, now=DAY, historical_day=DAY.date())
        assert len(hist) == 1
        expected_hist = (
            f"Prior slate 2027-07-15 ET. The dated observation recorded the saver as used. "
            f"At this check the flag for season 2027 read {st}. {health.HISTORICAL_POINTER}"
        )
        assert hist[0].message == expected_hist
        assert "`" not in hist[0].message and str(picks) not in hist[0].message


def test_worst_ordinary_states_under_r64_and_b8a(tmp_path, monkeypatch):
    # State 1: Dated observation was the mistaken side (MLB shows saver is 'active', operator mistakenly typed --saver used).
    p1 = tmp_path / "s1" / "picks"
    write_switch_and_auto(p1, DAY, streak=12, best=15)
    assert invoke_cli(monkeypatch, DAY, "saver-state", "--init", "active", "--season", "2027", "--picks-dir", str(p1)).exit_code == 0
    assert invoke_cli(monkeypatch, DAY, "set-contest-streak", "--date", "2027-07-15", "--streak", "12", "--best-streak", "15", "--saver", "used", "--picks-dir", str(p1)).exit_code == 0
    st_out = invoke_cli(monkeypatch, DAY, "season-contract", "status", "--picks-dir", str(p1))
    assert st_out.exit_code == 0
    assert "manual: dated date=2027-07-15 streak=12 best=15 saver=used in effect=True" in st_out.output
    assert "saver: active; season: 2027" in st_out.output
    # B8a says: "if MLB shows this season's saver used, check bts saver-state ...". MLB shows active -> do NOT run --use.
    # Correcting the dated observation on 2027-07-15 (once entries settled) clears the mismatch and preserves active flag:
    assert invoke_cli(monkeypatch, DAY, "set-contest-streak", "--date", "2027-07-15", "--streak", "12", "--best-streak", "15", "--saver", "active", "--picks-dir", str(p1)).exit_code == 0
    assert health.check(p1, expected=True, now=DAY) == []
    assert load_saver_state(p1, season=2027).state == "active"

    # State 2: Observation corrected to 'active' later that day BEFORE operator reads the earlier EOD DM.
    # When operator reads the DM later on 2027-07-15 (or 2027-07-16), `season-contract status` shows both active -> no action taken!
    st_after = invoke_cli(monkeypatch, DAY + dt.timedelta(hours=1), "season-contract", "status", "--picks-dir", str(p1))
    assert "saver=active in effect=True" in st_after.output
    assert "saver: active; season: 2027" in st_after.output

    # State 3: Dashboard button changes the flag between the warning and the operator's action.
    p3 = tmp_path / "s3" / "picks"
    write_switch_and_auto(p3, DAY, streak=14, best=15)
    assert invoke_cli(monkeypatch, DAY, "saver-state", "--init", "active", "--season", "2027", "--picks-dir", str(p3)).exit_code == 0
    assert invoke_cli(monkeypatch, DAY, "set-contest-streak", "--date", "2027-07-15", "--streak", "14", "--best-streak", "15", "--saver", "used", "--picks-dir", str(p3)).exit_code == 0
    assert len(health.check(p3, expected=True, now=DAY)) == 1
    # 3a: Dashboard marks saver used (active -> used) while MLB truly shows used:
    assert transition_saver_state(p3, expected_prior="active", new_state="used", season=2027, source="dashboard", peer="100.64.0.2")
    # Operator follows B8a: checks `bts saver-state --picks-dir DIR`, sees `used` (neither active nor not_earned):
    chk = invoke_cli(monkeypatch, DAY, "saver-state", "--picks-dir", str(p3))
    assert "saver_state: used (season 2027)" in chk.output
    # 3b: Dashboard undoes mark-used (used -> active) because MLB shows active:
    assert transition_saver_state(p3, expected_prior="used", new_state="active", season=2027, source="dashboard", peer="100.64.0.2")
    assert load_saver_state(p3, season=2027).state == "active"

    # State 4: Warning read the next morning (2027-07-16) after the automatic path has already picked, for both active and not_earned flags.
    for init_flag, next_streak in [("active", 14), ("not_earned", 9)]:
        p4 = tmp_path / f"s4_{init_flag}" / "picks"
        assert invoke_cli(monkeypatch, DAY, "saver-state", "--init", init_flag, "--season", "2027", "--picks-dir", str(p4)).exit_code == 0
        assert invoke_cli(monkeypatch, DAY, "set-contest-streak", "--date", "2027-07-15", "--streak", "14", "--best-streak", "15", "--saver", "used", "--picks-dir", str(p4)).exit_code == 0
        morning = REAL_DT(2027, 7, 16, 9, 0, tzinfo=ET)
        write_switch_and_auto(p4, morning, streak=next_streak, best=15)
        # Before operator acts at 09:00 ET, an early 08:30 ET automatic read counted saver_available=True (Known limit #2):
        assert contest_state.load_decision_streak_state(p4, require_contest_state=True, now=morning - dt.timedelta(minutes=30)).saver_available is True
        # At 09:00 ET operator verifies MLB shows 2027 saver used, checks `bts saver-state --picks-dir p4`, runs `bts saver-state --use --picks-dir p4` (no --season needed in 2027 away from midnight per B8a!):
        s_chk = invoke_cli(monkeypatch, morning, "saver-state", "--picks-dir", str(p4))
        assert f"saver_state: {init_flag} (season 2027)" in s_chk.output
        use_res = invoke_cli(monkeypatch, morning, "saver-state", "--use", "--picks-dir", str(p4))
        assert use_res.exit_code == 0
        assert load_saver_state(p4, season=2027).state == "used"
        assert contest_state.load_decision_streak_state(p4, require_contest_state=True, now=morning).saver_available is False

    # State 5: December 31 late evening (23:50 ET warning read at 23:58 ET or 2028-01-01 09:00 ET).
    p5 = tmp_path / "s5" / "picks"
    assert invoke_cli(monkeypatch, NYE, "saver-state", "--init", "active", "--season", "2027", "--picks-dir", str(p5)).exit_code == 0
    assert invoke_cli(monkeypatch, NYE, "set-contest-streak", "--date", "2027-12-31", "--streak", "14", "--best-streak", "15", "--saver", "used", "--picks-dir", str(p5)).exit_code == 0
    nye_warn = health.check(p5, expected=True, now=NYE)
    assert len(nye_warn) == 1 and "Streak Saver on 2027-12-31 ET." in nye_warn[0].message
    # Read on 2028-01-01 09:00 ET: B8a says "In D's ET year away from midnight ...". 2028 != 2027, so operator does NOT run --use/--init used.
    # And in 2028, season-contract status and load_saver_state(season=2028) already treat 2027's active flag as uninitialized (spent)!
    jan1_morning = REAL_DT(2028, 1, 1, 9, 0, tzinfo=ET)
    st_2028 = invoke_cli(monkeypatch, jan1_morning, "season-contract", "status", "--picks-dir", str(p5))
    assert "saver: unavailable (another season); season: 2027; reads uninitialized" in st_2028.output
    assert load_saver_state(p5, season=2028).state == "uninitialized"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "-rA"]))
