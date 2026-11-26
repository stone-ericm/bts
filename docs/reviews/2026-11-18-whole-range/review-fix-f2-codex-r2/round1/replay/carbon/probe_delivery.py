"""Item 3 & Standing Rows 1/4/5 probe: EOD delivery, two-digest ordering, status files, dedup, consecutive days, chat limit, transport failures, and interrupts."""

from __future__ import annotations

import datetime as dt
import json
import sys
import unicodedata
from collections import namedtuple
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts import cli
from bts.health import alert, contest_state as health, runner

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = REAL_DT(2027, 7, 15, 22, 15, tzinfo=ET)
INCIDENT = "saver_state:dated_used_mismatch"
MAIN_STATUS = "health_dm_delivery_status.json"
MISMATCH_STATUS = MAIN_STATUS + ".dated_used_mismatch.json"


def make_clock(fixed_now):
    class PinnedDatetime(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)
    return PinnedDatetime


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cli_dated(monkeypatch, picks: Path, now: dt.datetime, saver: str = "used", streak: int = 14) -> None:
    clk = make_clock(now)
    with monkeypatch.context() as mp:
        for mod in (dt, cli, alert, health, runner):
            mp.setattr(mod, "datetime", clk)
        res = CliRunner().invoke(cli.cli, [
            "set-contest-streak",
            "--date", now.astimezone(ET).date().isoformat(),
            "--streak", str(streak),
            "--best-streak", str(max(streak, 15)),
            "--saver", saver,
            "--picks-dir", str(picks),
        ])
    assert res.exit_code == 0, res.output


def set_flag(picks: Path, state: str, season: int = 2027) -> Path:
    p = picks / "account_state" / "saver_state.json"
    write_json(p, {"season": season, "state": state})
    return p


def run_real_eod(
    monkeypatch,
    picks: Path,
    *,
    slate: dt.date,
    wall: dt.datetime,
    recipient: str | None = "ops.bsky.social",
    transport=None,
    overrides: dict | None = None,
    extra_alerts: list[alert.Alert] | None = None,
):
    """Run real runner.run_all_checks with real _safe_run across local checks."""
    models = picks.parent / "models"
    models.mkdir(parents=True, exist_ok=True)
    if not (picks / "streak.json").exists():
        write_json(picks / "streak.json", {"season": slate.year, "streak": 6, "saver_available": True})
    (models / f"blend_{(slate + dt.timedelta(days=1)).isoformat()}.pkl").write_text("", encoding="utf-8")

    _Usage = namedtuple("_Usage", ["total", "used", "free"])
    clean_usage = _Usage(total=100 * 1024 ** 3, used=30 * 1024 ** 3, free=70 * 1024 ** 3)

    clk = make_clock(wall)
    sent = []
    if transport is None:
        def default_send(to, body):
            sent.append((to, body))
            return "msg-id-123"
        send_fn = default_send
    else:
        def wrapped_send(to, body):
            sent.append((to, body))
            return transport(to, body)
        send_fn = wrapped_send

    with monkeypatch.context() as mp:
        for mod in (dt, alert, health, runner):
            mp.setattr(mod, "datetime", clk)
        mp.setattr(alert, "send_dm", send_fn)
        mp.setattr("bts.health.disk_fill.shutil.disk_usage", lambda _: clean_usage)
        mp.setattr(runner.realized_calibration, "_current_deploy_iso", lambda _: None)
        mp.setattr(runner.tail_policy, "check", lambda **_: [])
        mp.setattr(runner.park_drag_freshness, "check", lambda *a, **k: [])
        mp.setattr(runner.unit_drift, "check", lambda **_: [])
        mp.setattr(runner.backup_freshness, "check", lambda *a, **k: [])
        if extra_alerts is not None:
            mp.setattr(runner.calibration, "check", lambda *a, **k: list(extra_alerts))
        alerts_out = runner.run_all_checks(
            picks_dir=picks,
            models_dir=models,
            dm_recipient=recipient,
            today=slate,
            thresholds_overrides=overrides,
            contest_state_expected=True,
            live_forward_capture_enabled=False,
            shadow_model_enabled=False,
        )
    return alerts_out, sent


def test_real_run_all_checks_digest_first_then_mismatch_and_custom_status_override(tmp_path, monkeypatch):
    picks = tmp_path / "data" / "picks"
    cli_dated(monkeypatch, picks, DAY, saver="used", streak=14)
    set_flag(picks, "used", 2027)
    crit = alert.Alert("CRITICAL", "pick_entry", "Unentered pick")
    warn_other = alert.Alert("WARN", "backup_freshness", "Backup stale")

    # Baseline with flag=used (no mismatch):
    base_alerts, base_sends = run_real_eod(
        monkeypatch, picks, slate=DAY.date(), wall=DAY, extra_alerts=[crit, warn_other],
    )
    assert len(base_sends) == 1
    base_main_bytes = (tmp_path / "data" / "health_state" / MAIN_STATUS).read_bytes()
    assert not (tmp_path / "data" / "health_state" / MISMATCH_STATUS).exists()

    # Reset state files and flip flag to active so mismatch fires alongside crit + warn_other:
    for f in (tmp_path / "data" / "health_state").iterdir():
        f.unlink()
    set_flag(picks, "active", 2027)
    cand_alerts, cand_sends = run_real_eod(
        monkeypatch, picks, slate=DAY.date(), wall=DAY, extra_alerts=[crit, warn_other],
    )
    assert len(cand_sends) == 2
    # First DM is byte-for-byte identical to base_sends[0], and its status file is byte-for-byte identical!
    assert cand_sends[0] == base_sends[0]
    assert (tmp_path / "data" / "health_state" / MAIN_STATUS).read_bytes() == base_main_bytes
    # Second DM is WARN-only with only the mismatch incident:
    assert cand_sends[1][1].startswith("BTS health WARN attention:\n- [saver_state] Streak Saver on 2027-07-15 ET.")
    assert "CRITICAL" not in cand_sends[1][1]
    mm_receipt = json.loads((tmp_path / "data" / "health_state" / MISMATCH_STATUS).read_text())
    assert mm_receipt["status"] == "sent"
    assert mm_receipt["critical_count"] == 0
    assert mm_receipt["warn_attention_count"] == 1
    assert mm_receipt["sent_sources"] == [f"WARN_ATTENTION:{INCIDENT}"]
    # Raw returned alerts still contain all 3 alerts:
    assert [a for a in cand_alerts if a.incident_key != INCIDENT] == base_alerts
    assert len([a for a in cand_alerts if a.incident_key == INCIDENT]) == 1

    # Custom health_dm_delivery_status override path (as string):
    custom_status = tmp_path / "custom_dir" / "my_status.json"
    _, custom_sends = run_real_eod(
        monkeypatch, picks, slate=DAY.date(), wall=DAY,
        overrides={"health_dm_delivery_status": str(custom_status)},
        extra_alerts=[crit],
    )
    assert len(custom_sends) == 2
    assert custom_status.is_file()
    assert custom_status.with_name("my_status.json.dated_used_mismatch.json").is_file()


def test_consecutive_days_and_independent_attention_streaks(tmp_path, monkeypatch):
    picks = tmp_path / "data" / "picks"
    set_flag(picks, "active", 2027)

    # Day 1, Day 2, Day 3 with dated used mismatch:
    expected_labels = ["(1st observed day)", "(2nd consecutive day)", "(3rd consecutive day)"]
    for offset, exp_label in enumerate(expected_labels):
        d = DAY + dt.timedelta(days=offset)
        cli_dated(monkeypatch, picks, d, saver="used", streak=14)
        _, sends = run_real_eod(monkeypatch, picks, slate=d.date(), wall=d)
        assert len(sends) == 1
        assert exp_label in sends[0][1]

    # Day 4: dated observation is not_earned at streak 14 (existing K1 WARN, source='saver_state', incident_key=None).
    # Its streak must start at (1st observed day), NOT inherit the 3-day mismatch streak!
    d4 = DAY + dt.timedelta(days=3)
    cli_dated(monkeypatch, picks, d4, saver="not_earned", streak=14)
    _, sends_d4 = run_real_eod(monkeypatch, picks, slate=d4.date(), wall=d4)
    assert len(sends_d4) == 1
    assert "(1st observed day)" in sends_d4[0][1]
    assert "contradicts streak 14" in sends_d4[0][1]


def test_same_date_live_then_after_midnight_dedup_and_next_day_budget(tmp_path, monkeypatch):
    picks = tmp_path / "data" / "picks"
    cli_dated(monkeypatch, picks, DAY, saver="used", streak=14)
    set_flag(picks, "not_earned", 2027)

    # 1. After-midnight run for DAY (at D+1 00:10 ET): stamps sent_day = 2027-07-15, NOT 2027-07-16!
    after_midnight = REAL_DT(2027, 7, 16, 0, 10, tzinfo=ET)
    _, sends_hist = run_real_eod(monkeypatch, picks, slate=DAY.date(), wall=after_midnight)
    assert len(sends_hist) == 1
    assert sends_hist[0][1].startswith("BTS health WARN attention:\n- [saver_state] Prior slate 2027-07-15 ET.")

    # Re-running after-midnight for DAY is deduplicated:
    _, sends_hist_rerun = run_real_eod(monkeypatch, picks, slate=DAY.date(), wall=after_midnight + dt.timedelta(minutes=5))
    assert sends_hist_rerun == []

    # 2. On D+1 (2027-07-16 22:15 ET), a new dated used observation for D+1 is NOT suppressed by D's post-midnight run!
    d_next = DAY + dt.timedelta(days=1)
    cli_dated(monkeypatch, picks, d_next, saver="used", streak=15)
    _, sends_d_next = run_real_eod(monkeypatch, picks, slate=d_next.date(), wall=d_next)
    assert len(sends_d_next) == 1
    assert "Streak Saver on 2027-07-16 ET." in sends_d_next[0][1]
    assert "(2nd consecutive day)" in sends_d_next[0][1]


def test_fault_isolation_between_first_and_second_dispatch_and_chat_limit(tmp_path, monkeypatch):
    picks = tmp_path / "data" / "picks"
    cli_dated(monkeypatch, picks, DAY, saver="used", streak=14)
    set_flag(picks, "active", 2027)
    near_limit = alert.Alert("WARN", "pick_entry", "X" * 920)  # ~975 chars in digest, < 1000 alone, > 1400 with F2

    # Case A: First digest fails on transport, second succeeds -> main is 'failed', mismatch is 'sent'!
    calls_a = [0]
    def first_fails(to, body):
        calls_a[0] += 1
        if calls_a[0] == 1:
            raise RuntimeError("transient network failure on digest")
        assert len(body) <= 1000
        return "ok-2"

    _, sends_a = run_real_eod(monkeypatch, picks, slate=DAY.date(), wall=DAY, transport=first_fails, extra_alerts=[near_limit])
    assert len(sends_a) == 2
    main_st = json.loads((tmp_path / "data" / "health_state" / MAIN_STATUS).read_text())
    mm_st = json.loads((tmp_path / "data" / "health_state" / MISMATCH_STATUS).read_text())
    assert main_st["status"] == "failed"
    assert mm_st["status"] == "sent"

    # Rerun on same day: first digest RETRIES and succeeds (<1000 chars), second is deduplicated!
    _, sends_retry = run_real_eod(
        monkeypatch, picks, slate=DAY.date(), wall=DAY,
        transport=lambda to, body: (_ for _ in ()).throw(ValueError("over 1000")) if len(body) > 1000 else "ok-retry",
        extra_alerts=[near_limit],
    )
    assert len(sends_retry) == 1
    assert "X" * 920 in sends_retry[0][1]
    assert json.loads((tmp_path / "data" / "health_state" / MAIN_STATUS).read_text())["status"] == "sent"

    # Case B: Interrupt between first status write and second status write (after second send_dm):
    # Reset status files for clean interrupt test:
    for f in (tmp_path / "data" / "health_state").iterdir():
        f.unlink()
    flag_bytes = (picks / "account_state" / "saver_state.json").read_bytes()
    obs_bytes = (picks / "account_state" / "contest_streak.manual.json").read_bytes()

    orig_persist = alert._persist_health_dm_delivery_status
    persist_count = [0]
    def interrupt_on_second_persist(status_path, **payload):
        persist_count[0] += 1
        if persist_count[0] == 2:
            raise KeyboardInterrupt("killed after second send before status write")
        return orig_persist(status_path, **payload)

    with monkeypatch.context() as mp:
        mp.setattr(alert, "_persist_health_dm_delivery_status", interrupt_on_second_persist)
        with pytest.raises(KeyboardInterrupt):
            run_real_eod(monkeypatch, picks, slate=DAY.date(), wall=DAY, extra_alerts=[near_limit])

    # First digest status was persisted as 'sent', account files untouched, mismatch status absent:
    assert (picks / "account_state" / "saver_state.json").read_bytes() == flag_bytes
    assert (picks / "account_state" / "contest_streak.manual.json").read_bytes() == obs_bytes
    assert json.loads((tmp_path / "data" / "health_state" / MAIN_STATUS).read_text())["status"] == "sent"
    assert not (tmp_path / "data" / "health_state" / MISMATCH_STATUS).exists()

    # Restart / rerun: first digest is deduplicated, mismatch DM sends once and writes its status!
    _, after_restart = run_real_eod(monkeypatch, picks, slate=DAY.date(), wall=DAY, extra_alerts=[near_limit])
    assert len(after_restart) == 1
    assert "Streak Saver on 2027-07-15 ET." in after_restart[0][1]
    assert json.loads((tmp_path / "data" / "health_state" / MISMATCH_STATUS).read_text())["status"] == "sent"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "-rA"]))
