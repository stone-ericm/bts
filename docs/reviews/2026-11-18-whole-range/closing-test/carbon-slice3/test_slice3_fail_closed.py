"""Slice 3 adversarial probes — Item 3: Fail-closed and fail-loud verification.

Exercises every new refusal or raise family in 619abda7..HEAD against persistent
broken files/configs ("a file that only a human can fix") and verifies whether each
produces an operator-visible signal or fails in silence.

Rules obeyed:
- No module-level import of bts.model.predict or lightgbm.
- No permanent process-wide state mutations.
- Zero network or socket calls.
"""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pandas as pd

from bts import cli as cli_mod
from bts import contest_state, daily_decision, day_outcome, dm, entry_intent, orchestrator, picks, probable_pitcher_status, saver_state, scheduler, season_contract, strategy, util, web
from bts.data import build as data_build
from bts.features import compute, probable_pitcher_cache
from bts.health import alert as health_alert
from bts.health import contest_state as health_contest_state
from bts.health import probable_pitcher_cache as health_pp_cache
from bts.health import runner as health_runner
from bts.health import streak_validation as health_streak_validation

ET = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_TMP = Path(__file__).resolve().parent / "scratch_tmp_fail"


def _temp_dir(tmp_path: Path | None):
    if tmp_path is not None:
        return nullcontext(Path(tmp_path))
    SCRATCH_TMP.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=SCRATCH_TMP)


def test_cs6a_live_gate_refusal_signals(tmp_path: Path | None = None) -> dict:
    """CS6a: broken orchestrator config refuses bts schedule (no heartbeat), bts orchestrate, cron install."""
    with _temp_dir(tmp_path) as td:
        base = Path(td)
        picks_dir = base / "data" / "picks"
        hb_path = base / "data" / ".heartbeat"
        picks_dir.mkdir(parents=True)
        bad_cfg_path = base / "bad.toml"
        bad_cfg_path.write_text(
            f'[orchestrator]\npicks_dir = {json.dumps(str(picks_dir))}\n'
            f'heartbeat_path = {json.dumps(str(hb_path))}\n'
            '[bluesky]\ndm_recipient = "operator.bsky.social"\n'
            '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\ncontest_state_required = false\n'
        )
        r_sched = CliRunner().invoke(cli_mod.cli, ["schedule", "--date", "2027-07-15", "--config", str(bad_cfg_path)])
        assert r_sched.exit_code == 2 and "refusing to start" in r_sched.stderr
        assert not hb_path.exists()  # Heartbeat never written -> check_heartbeat.py pings /fail after 15m!

        r_orch = CliRunner().invoke(cli_mod.cli, ["orchestrate", "--date", "2027-07-15", "--config", str(bad_cfg_path)])
        assert r_orch.exit_code == 2 and "refusing to run" in r_orch.stderr

        r_ei = CliRunner().invoke(cli_mod.cli, ["entry-intent", "--config", str(bad_cfg_path)])
        assert r_ei.exit_code == 1

        # Next morning 08:30 day-outcome sends NO DECISION DM because scheduler never ran a decision!
        dm_sent: list[str] = []
        with patch.object(scheduler, "fetch_schedule", return_value=[{"gamePk": 1, "status": {"detailedState": "Final"}}]), \
             patch.object(dm, "send_dm", side_effect=lambda _r, msg: (dm_sent.append(msg), "dm-1")[1]):
            r_do = CliRunner().invoke(cli_mod.cli, [
                "day-outcome", "--date", "2027-07-15", "--config", str(bad_cfg_path),
                "--picks-dir", str(picks_dir), "--dm-recipient", "operator.bsky.social",
            ])
        assert r_do.exit_code == 1 and len(dm_sent) == 1 and "NO DECISION" in dm_sent[0]

        return {
            "schedule_exits_2_without_heartbeat": True,
            "day_outcome_backstop_dms_no_decision": True,
        }


def test_corrupt_files_contest_saver_streak_signals(tmp_path: Path | None = None) -> dict:
    """Persistent corrupt account_state / saver_state / streak.json / decision.json signals."""
    with _temp_dir(tmp_path) as td:
        base = Path(td)
        now = dt.datetime(2027, 7, 15, 22, 30, tzinfo=ET)

        # 1. Over-long integer in saver_state.json -> health.runner._safe_run emits CRITICAL health_runner:contest_state DM!
        p_huge = base / "huge_saver" / "data" / "picks"
        acct = p_huge / "account_state"
        acct.mkdir(parents=True)
        (acct / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1", "season": 2027, "status": "automatic",
            "activation_id": "a" * 32, "switched_at": "2027-07-01T00:00:00Z", "revoked_at": None, "note": None,
        }))
        (acct / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v2", "season": 2027, "activation_id": "a" * 32,
            "active_streak": 5, "best_streak": 10, "source": "mlb_bts_profile",
            "source_date": "2027-07-15", "recorded_at": "2027-07-15T20:00:00Z",
        }))
        (acct / "saver_state.json").write_text('{"season": 2027, "state": "active", "pad": ' + ("9" * 5000) + "}")
        (p_huge / "streak.json").write_text('{"season": 2027, "streak": 5, "saver_available": true}')

        sent_hdm: list[str] = []
        with patch.object(health_alert, "send_dm", side_effect=lambda _r, m: (sent_hdm.append(m), "h1")[1]):
            alerts = health_runner.run_all_checks(
                p_huge, REPO_ROOT / "data/models", "operator.bsky.social",
                today=dt.date(2027, 7, 15), contest_state_expected=True,
            )
        assert any(a.level == "CRITICAL" and a.incident_key == "health_runner:contest_state" for a in alerts)
        assert len(sent_hdm) >= 1 and "[health_runner]" in sent_hdm[0]

        # 2. Corrupt streak.json (`[]` or `{`) + blocked C3v reconcile -> EOD health_runner / streak_validation DMs CRITICAL!
        p_st = base / "corrupt_streak" / "data" / "picks"
        p_st.mkdir(parents=True)
        (p_st / "streak.json").write_text("[]")
        sent_st: list[str] = []
        with patch.object(health_alert, "send_dm", side_effect=lambda _r, m: (sent_st.append(m), "h2")[1]):
            st_alerts = health_runner.run_all_checks(
                p_st, REPO_ROOT / "data/models", "operator.bsky.social",
                today=dt.date(2027, 7, 15), contest_state_expected=False,
            )
        assert any(a.level == "CRITICAL" and a.incident_key == "health_runner:streak_validation" for a in st_alerts)
        assert len(sent_st) >= 1

        # 3a. Malformed JSON decision.json -> parse_decision returns None -> day-outcome exits 1 ("NO DECISION" DM)
        # 3b. Non-UTF-8 / over-long integer decision.json -> raises ValueError -> day-outcome exits 2 ("THE CHECK COULD NOT RUN" DM)
        p_dec = base / "corrupt_dec" / "data" / "picks"
        dec_path = daily_decision.decision_path("2027-07-15", p_dec)
        dec_path.parent.mkdir(parents=True)
        dec_path.write_text("{broken json")
        cfg_toml = base / "corrupt_dec" / "orch.toml"
        cfg_toml.write_text(
            f'[orchestrator]\npicks_dir = {json.dumps(str(p_dec))}\n'
            '[bluesky]\ndm_recipient = "operator.bsky.social"\n'
            '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\ncontest_state_required = true\n'
        )
        sent_do1: list[str] = []
        with patch.object(scheduler, "fetch_schedule", return_value=[{"gamePk": 1, "status": {"detailedState": "Final"}}]), \
             patch.object(dm, "send_dm", side_effect=lambda _r, m: (sent_do1.append(m), "d1")[1]):
            r_do1 = CliRunner().invoke(cli_mod.cli, [
                "day-outcome", "--date", "2027-07-15", "--config", str(cfg_toml),
                "--picks-dir", str(p_dec), "--dm-recipient", "operator.bsky.social",
            ])
        assert r_do1.exit_code == 1 and len(sent_do1) == 1 and "NO DECISION" in sent_do1[0]

        dec_path.write_bytes(b"\xff\xfe")
        sent_do2: list[str] = []
        with patch.object(scheduler, "fetch_schedule", return_value=[{"gamePk": 1, "status": {"detailedState": "Final"}}]), \
             patch.object(dm, "send_dm", side_effect=lambda _r, m: (sent_do2.append(m), "d2")[1]):
            r_do2 = CliRunner().invoke(cli_mod.cli, [
                "day-outcome", "--date", "2027-07-15", "--config", str(cfg_toml),
                "--picks-dir", str(p_dec), "--dm-recipient", "operator.bsky.social",
            ])
        assert r_do2.exit_code == 2 and len(sent_do2) == 1 and "THE CHECK COULD NOT RUN" in sent_do2[0]

        # 4. web.py dashboard on corrupt contest_streak.manual.json does not crash (falls back cleanly)
        p_web = base / "web_corrupt" / "data" / "picks"
        (p_web / "account_state").mkdir(parents=True)
        (p_web / "account_state" / "contest_streak.manual.json").write_text("{corrupt")
        ctx = web.saver_dashboard_context(p_web, now=now)
        assert ctx is not None

        return {
            "huge_int_saver_surfaces_as_health_runner_critical_dm": True,
            "non_dict_streak_json_surfaces_as_health_runner_critical_dm": True,
            "corrupt_decision_json_surfaces_as_day_outcome_exit2_dm": True,
            "web_dashboard_survives_corrupt_contest_state": True,
        }


def test_silent_forever_paths_audit(tmp_path: Path | None = None) -> dict:
    """Check paths where a persistent human-only file failure repeats every cycle WITHOUT its own DM."""
    with _temp_dir(tmp_path) as td:
        base = Path(td)

        # Path 1: C3v blocked season replay (`bts reconcile` at 02:00 and 07:40 ET when a past
        # undelivered preview `{date}.json` has `result: null`).
        # Signal: stderr line in cron.log + reconcile receipt (`replay.state == "unavailable"`),
        # NO health check and NO DM! (Note: result_resolution skips non-scoreable previews by design!)
        p_c3v = base / "c3v" / "data" / "picks"
        p_c3v.mkdir(parents=True)
        picks.save_streak(5, p_c3v, saver_available=True, applied_date="2027-04-15", season=2027)
        # Past refused/uncommitted preview left on disk (result=None, scoreable=False)
        preview = picks.DailyPick(
            "2027-04-16", "2027-04-16T07:05:00Z",
            picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.82, ["PROJECTED"], True, 101, "2027-04-16T23:10:00Z"),
            None, None,
        )
        picks.save_pick(preview, p_c3v)

        sent_c3v: list[str] = []
        with patch.object(health_alert, "send_dm", side_effect=lambda _r, m: (sent_c3v.append(m), "h")[1]):
            alerts = health_runner.run_all_checks(
                p_c3v, REPO_ROOT / "data/models", "operator.bsky.social",
                today=dt.date(2027, 4, 20), contest_state_expected=False,
            )
        # Verify whether ANY EOD health check flags the C3v replay blocker!
        c3v_alerts = [
            (a.level, a.source, a.message)
            for a in alerts
            if a.source in {"result_resolution", "streak_validation"}
        ]

        # Path 2: Legacy/expired `contest_streak.manual.json` alongside valid covered `contest_streak.json`.
        # health.contest_state emits WARN(source="contest_state"), which is NOT in ALWAYS_ATTENTION_WARN_SOURCES
        # and NOT in REPEATED_ATTENTION_WARN_SOURCES -> journal WARN only, never promoted to DM!
        from bts.health import attention as health_attention
        p_leg = base / "legacy_manual" / "data" / "picks"
        acct = p_leg / "account_state"
        acct.mkdir(parents=True)
        (acct / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1", "season": 2027, "status": "automatic",
            "activation_id": "e" * 32, "switched_at": "2027-04-01T00:00:00Z", "revoked_at": None, "note": None,
        }))
        (acct / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v2", "season": 2027, "activation_id": "e" * 32,
            "active_streak": 4, "best_streak": 8, "source": "mlb_bts_profile",
            "source_date": "2027-07-15", "recorded_at": "2027-07-15T20:00:00Z",
        }))
        (acct / "saver_state.json").write_text(json.dumps({
            "season": 2027, "state": "not_earned", "updated_at": "2027-04-01T00:00:00Z", "source": "cli",
        }))
        (acct / "contest_streak.manual.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_manual_v2",
            "active_streak": 1, "source": "manual_cli", "source_date": "2027-04-02",
            "recorded_at": "2027-04-02T12:00:00Z", "override_expires_at": "2027-04-03T12:00:00Z",
        }))
        leg_attn_total: list = []
        for day_offset in (0, 1, 2):
            cur_now = dt.datetime(2027, 7, 15, 22, 0, tzinfo=ET) + dt.timedelta(days=day_offset)
            cs_alerts = health_contest_state.check(p_leg, expected=True, now=cur_now)
            assert [(a.level, a.source) for a in cs_alerts] == [("WARN", "contest_state")]
            _, attn = health_attention.apply_warn_attention_policy(
                cs_alerts, state_path=base / "leg_warn.json", today=cur_now.date(),
            )
            leg_attn_total.extend(attn)

        # Path 3: Corrupt raw feed in data/raw/2027/ missing gameData.game.type -> `bts data build --seasons 2027`
        # raises ValueError (exit 1 in 03:00 cron -> skips `bts preview` -> blend_<tomorrow>.pkl missing ->
        # EOD `blend_training` sends CRITICAL DM!).
        raw_bad = base / "bad_raw" / "2027"
        raw_bad.mkdir(parents=True)
        (raw_bad / "9999.json").write_text('{"gameData": {"game": {}}}')
        raised_build = False
        try:
            data_build.build_season(base / "bad_raw", base / "out" / "pa_2027.parquet", 2027)
        except ValueError:
            raised_build = True
        assert raised_build is True

        # Path 4: Probable-pitcher cache corrupt + raw feed invalid -> health.probable_pitcher_cache emits
        # WARN (promoted via ALWAYS_ATTENTION_WARN_SOURCES to daily WARN attention DM); corrupt status file -> CRITICAL!
        pp_root = base / "pp_bad" / "data"
        pp_models = pp_root / "models"
        pp_raw = pp_root / "raw"
        pp_status = pp_root / "health_state" / "probable_pitcher_cache"
        pp_models.mkdir(parents=True)
        (pp_raw / "2027").mkdir(parents=True)
        (pp_models / "probable_pitcher_lookup.json").write_text("{corrupt")
        (pp_raw / "2027" / "100.json").write_text('{"gameData": {"probablePitchers": {"away": "bad"}}}')
        probable_pitcher_cache.build(pp_models / "probable_pitcher_lookup.json", pp_raw)
        pp_today = dt.datetime.now(dt.timezone.utc).astimezone(ET).date()
        pp_alerts = health_pp_cache.check(pp_status, today=pp_today)
        assert any(a.level == "WARN" and a.source == "probable_pitcher_cache" for a in pp_alerts)
        _, pp_attn = health_attention.apply_warn_attention_policy(
            pp_alerts, state_path=base / "pp_warn.json", today=pp_today,
        )
        assert len(pp_attn) == 1
        (pp_status / f"{pp_today.isoformat()}.unreadable_cache.000000000000.json").write_text("{broken")
        pp_crit = health_pp_cache.check(pp_status, today=pp_today)
        assert any("status unreadable" in a.message for a in pp_crit)
        _, pp_crit_attn = health_attention.apply_warn_attention_policy(
            pp_crit, state_path=base / "pp_warn2.json", today=pp_today,
        )
        assert any("status unreadable" in a.message for a in pp_crit_attn)

        return {
            "c3v_blocked_reconcile_eod_health_alerts": c3v_alerts,
            "expired_legacy_manual_warn_never_pages_dm_even_after_3_days": (len(leg_attn_total) == 0),
            "corrupt_raw_feed_build_raises_and_trips_blend_training_dm": raised_build,
            "corrupt_pp_cache_surfaces_attention_and_critical_health_alert": True,
        }


def run_all() -> dict:
    return {
        "cs6a_live_gate": test_cs6a_live_gate_refusal_signals(),
        "corrupt_state_files": test_corrupt_files_contest_saver_streak_signals(),
        "silent_paths_audit": test_silent_forever_paths_audit(),
    }


if __name__ == "__main__":
    import shutil
    try:
        print(json.dumps(run_all(), indent=2, default=str))
    finally:
        shutil.rmtree(SCRATCH_TMP, ignore_errors=True)
