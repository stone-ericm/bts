"""Item 2 & Standing Row 2/3 probe: flag x observation comparison matrix, hostile files, and midnight/NYE clocks."""

from __future__ import annotations

import datetime as dt
import json
import os
import signal
import socket
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pytest

from bts.health import alert, contest_state as health, runner

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = REAL_DT(2027, 7, 15, 22, 30, tzinfo=ET)
NYE_LATE_ET = REAL_DT(2027, 12, 31, 23, 59, 59, 999999, tzinfo=ET)  # 2028-01-01T04:59:59.999999Z
NEW_YEAR_ET = REAL_DT(2028, 1, 1, 0, 0, 0, 0, tzinfo=ET)
NEW_YEAR_0005_ET = REAL_DT(2028, 1, 1, 0, 5, 0, 0, tzinfo=ET)
INCIDENT = "saver_state:dated_used_mismatch"


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_dated_obs(picks: Path, eff_date: dt.date, saver: str, streak: int = 14, best: int | None = None) -> Path:
    best = max(streak, 15) if best is None else best
    exp = REAL_DT.combine(eff_date, dt.time.max, ET)
    p = picks / "account_state" / "contest_streak.manual.json"
    write_json(p, {
        "schema_version": "bts_contest_streak_manual_v3",
        "effective_date": eff_date.isoformat(),
        "season": eff_date.year,
        "expires_at": exp.isoformat(),
        "active_streak": streak,
        "best_streak": best,
        "saver_state": saver,
        "recorded_at": exp.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "cli_dated",
    })
    return p


def plant_flag(picks: Path, flag_case: str, year: int, cleanup_dirs: list[Path], short_tmp: Path) -> None:
    acct = picks / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    target = acct / "saver_state.json"
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            target.rmdir()
        else:
            target.unlink()

    if flag_case == "absent":
        return
    if flag_case in ("active", "not_earned", "used", "uninitialized"):
        write_json(target, {"season": year, "state": flag_case})
    elif flag_case == "another_season":
        write_json(target, {"season": year - 1, "state": "active"})
    elif flag_case == "future_season":
        write_json(target, {"season": year + 1, "state": "active"})
    elif flag_case == "bool_true_season":
        target.write_text('{"season": true, "state": "active"}', encoding="utf-8")
    elif flag_case == "bool_false_season":
        target.write_text('{"season": false, "state": "active"}', encoding="utf-8")
    elif flag_case == "float_season":
        target.write_text(f'{{"season": {year}.0, "state": "active"}}', encoding="utf-8")
    elif flag_case == "unreadable":
        write_json(target, {"season": year, "state": "active"})
        target.chmod(0o000)
        cleanup_dirs.append(target)
    elif flag_case == "invalid_utf8":
        target.write_bytes(b"\xff\xfe\xfd")
    elif flag_case == "utf8_bom":
        target.write_bytes(b"\xef\xbb\xbf" + json.dumps({"season": year, "state": "active"}).encode("utf-8"))
    elif flag_case == "malformed_json":
        target.write_text("{not-json", encoding="utf-8")
    elif flag_case == "json_list":
        target.write_text('["active"]', encoding="utf-8")
    elif flag_case == "json_scalar":
        target.write_text('"active"', encoding="utf-8")
    elif flag_case == "directory":
        target.mkdir()
    elif flag_case == "named_pipe":
        os.mkfifo(target)
    elif flag_case == "symlink_to_fifo":
        fifo = acct / "other.fifo"
        if not fifo.exists():
            os.mkfifo(fifo)
        os.symlink(fifo, target)
    elif flag_case == "unix_socket":
        sock_path = short_tmp / f"s_{os.getpid()}_{len(cleanup_dirs)}.sock"
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.bind(str(sock_path))
            os.symlink(sock_path, target)
        finally:
            s.close()
    elif flag_case == "dev_null_symlink":
        os.symlink("/dev/null", target)
    elif flag_case == "broken_symlink":
        os.symlink(acct / "does_not_exist.json", target)
    elif flag_case == "symlink_loop":
        os.symlink(target, target)
    elif flag_case == "huge_integer_season":
        target.write_bytes(b'{"season": ' + b"9" * 5000 + b', "state": "active"}')
    elif flag_case == "huge_integer_extra":
        target.write_bytes(
            f'{{"season": {year}, "state": "active", "big": '.encode("utf-8") + b"9" * 5000 + b"}"
        )
    elif flag_case == "deep_array_nesting":
        target.write_bytes(b"[" * 2500 + b"]" * 2500)
    elif flag_case == "deep_object_nesting":
        target.write_bytes(b'{"a":' * 2000 + b"1" + b"}" * 2000)
    elif flag_case == "symlink_unsearchable_dir":
        locked_dir = acct / "locked_dir"
        locked_dir.mkdir(exist_ok=True)
        (locked_dir / "inner.json").write_text('{"season": 2027, "state": "active"}', encoding="utf-8")
        locked_dir.chmod(0o000)
        cleanup_dirs.append(locked_dir)
        os.symlink(locked_dir / "inner.json", target)
    elif flag_case == "symlink_enametoolong":
        os.symlink("x" * 300, target)
    else:
        raise ValueError(f"Unknown flag_case: {flag_case}")


STANDARD_FLAG_CASES = [
    "active", "not_earned", "used", "uninitialized", "absent",
    "another_season", "future_season", "bool_true_season", "bool_false_season", "float_season",
    "unreadable", "invalid_utf8", "utf8_bom", "malformed_json", "json_list", "json_scalar",
    "directory", "named_pipe", "symlink_to_fifo", "unix_socket", "dev_null_symlink",
    "broken_symlink", "symlink_loop",
    "huge_integer_season", "huge_integer_extra", "deep_array_nesting", "deep_object_nesting",
]


@pytest.mark.parametrize("flag_case", STANDARD_FLAG_CASES)
@pytest.mark.parametrize("historical", [False, True])
def test_flag_matrix_on_dated_used_with_collected_alert(tmp_path, monkeypatch, flag_case, historical):
    picks = tmp_path / "picks"
    cleanup_perms: list[Path] = []
    # Seed resolved picks so contest_state collects a coverage WARN before reaching the saver branch.
    resolved = [DAY.date() + dt.timedelta(days=1), DAY.date() + dt.timedelta(days=2)]
    monkeypatch.setattr(health, "latest_resolved_pick_date", lambda _: max(resolved))
    monkeypatch.setattr(health, "_scan_resolved_pick_dates", lambda *a, **kw: resolved)
    monkeypatch.setattr(health, "resolved_pick_settlement_gap", lambda *a: 2)

    def alarm_handler(*_):
        raise TimeoutError(f"Blocked reading {flag_case}")

    prev = signal.signal(signal.SIGALRM, alarm_handler)
    with tempfile.TemporaryDirectory(dir="/tmp") as short_dir:
        try:
            for streak in (0, 9, 10, 14, 15, 16, 57, 58, 400):
                write_dated_obs(picks, DAY.date(), "used", streak=streak, best=max(streak, 15))
                plant_flag(picks, flag_case, DAY.year, cleanup_perms, Path(short_dir))
                signal.alarm(5)
                alerts = health.check(
                    picks,
                    expected=True,
                    now=DAY,
                    historical_day=DAY.date() if historical else None,
                )
                signal.alarm(0)
                # Previously collected coverage WARN must ALWAYS remain first and intact!
                assert len(alerts) >= 1
                assert alerts[0].source == "contest_state" and alerts[0].level == "WARN"
                mismatches = [a for a in alerts if a.incident_key == INCIDENT]
                if flag_case in ("active", "not_earned"):
                    assert len(alerts) == 2
                    assert len(mismatches) == 1
                    assert mismatches[0].source == "saver_state"
                    assert mismatches[0].level == "WARN"
                    assert flag_case in mismatches[0].message
                else:
                    assert alerts == [alerts[0]]
                    assert mismatches == []
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev)
            for p in reversed(cleanup_perms):
                try:
                    p.chmod(0o700)
                except OSError:
                    pass


@pytest.mark.parametrize("obs_saver", ["active", "not_earned"])
@pytest.mark.parametrize("streak", [0, 9, 10, 12, 15, 16, 57, 58, 400])
@pytest.mark.parametrize("flag_case", ["active", "not_earned", "used", "uninitialized", "absent", "named_pipe", "huge_integer_season"])
@pytest.mark.parametrize("historical", [False, True])
def test_non_used_dated_observations_never_emit_mismatch(tmp_path, obs_saver, streak, flag_case, historical):
    picks = tmp_path / "picks"
    cleanup_perms: list[Path] = []
    write_dated_obs(picks, DAY.date(), obs_saver, streak=streak, best=max(streak, 15))
    with tempfile.TemporaryDirectory(dir="/tmp") as short_dir:
        plant_flag(picks, flag_case, DAY.year, cleanup_perms, Path(short_dir))
        alerts = health.check(
            picks,
            expected=True,
            now=DAY,
            historical_day=DAY.date() if historical else None,
        )
    assert [a for a in alerts if a.incident_key == INCIDENT] == []
    expected_k1 = (obs_saver == "not_earned" and 10 <= streak <= 15)
    assert len(alerts) == (1 if expected_k1 else 0)


def test_dec31_jan1_and_c1_processed_slate_year_boundaries(tmp_path, monkeypatch):
    picks = tmp_path / "picks"
    # 1. Dec 31 23:59:59.999999 ET (UTC is already 2028-01-01T04:59:59.999999Z!)
    nye_utc = NYE_LATE_ET.astimezone(dt.timezone.utc)
    assert nye_utc.year == 2028
    write_dated_obs(picks, dt.date(2027, 12, 31), "used", streak=14, best=15)
    write_json(picks / "account_state" / "saver_state.json", {"season": 2027, "state": "active"})

    live_nye = health.check(picks, expected=True, now=nye_utc)
    assert len(live_nye) == 1 and live_nye[0].incident_key == INCIDENT
    assert "2027-12-31 ET" in live_nye[0].message and "season 2027 read active" in live_nye[0].message

    # If flag is for 2028 on Dec 31 ET, 2027 reads uninitialized -> no mismatch WARN!
    write_json(picks / "account_state" / "saver_state.json", {"season": 2028, "state": "active"})
    assert health.check(picks, expected=True, now=nye_utc) == []

    # 2. One microsecond later: 2028-01-01 00:00:00 ET, with new 2028-01-01 dated used observation:
    write_dated_obs(picks, dt.date(2028, 1, 1), "used", streak=0, best=0)
    # 2027 active flag must NOT trigger mismatch in 2028!
    write_json(picks / "account_state" / "saver_state.json", {"season": 2027, "state": "active"})
    assert health.check(picks, expected=True, now=NEW_YEAR_ET.astimezone(dt.timezone.utc)) == []
    # 2028 not_earned flag MUST trigger mismatch in 2028!
    write_json(picks / "account_state" / "saver_state.json", {"season": 2028, "state": "not_earned"})
    jan1 = health.check(picks, expected=True, now=NEW_YEAR_ET.astimezone(dt.timezone.utc))
    assert len(jan1) == 1 and "2028-01-01 ET" in jan1[0].message and "season 2028 read not_earned" in jan1[0].message

    # 3. After-midnight EOD run at 2028-01-01 00:05 ET for processed slate 2027-12-31 (C1 clock):
    write_dated_obs(picks, dt.date(2027, 12, 31), "used", streak=14, best=15)
    write_json(picks / "account_state" / "saver_state.json", {"season": 2027, "state": "not_earned"})

    class WallClock(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return NEW_YEAR_0005_ET.astimezone(tz) if tz else NEW_YEAR_0005_ET.replace(tzinfo=None)

    dms = []
    with patch.object(runner, "datetime", WallClock), \
         patch.object(alert, "datetime", WallClock), \
         patch.object(alert, "send_dm", side_effect=lambda r, b: dms.append((r, b)) or "ok"), \
         patch.object(runner, "_safe_run", side_effect=lambda n, fn: runner._safe_run(n, fn) if n == "contest_state" else []):
        eod_alerts = runner.run_all_checks(
            picks_dir=picks,
            models_dir=tmp_path / "models",
            dm_recipient="ops.bsky.social",
            today=dt.date(2027, 12, 31),
            contest_state_expected=True,
        )
    assert len(eod_alerts) == 1 and eod_alerts[0].incident_key == INCIDENT
    assert "Prior slate 2027-12-31 ET. The dated observation recorded the saver as used. At this check the flag for season 2027 read not_earned." in dms[0][1]


@pytest.mark.parametrize("hostile_symlink", ["symlink_unsearchable_dir", "symlink_enametoolong"])
@pytest.mark.parametrize("historical", [False, True])
def test_adversarial_stat_oserror_on_flag_symlink(tmp_path, monkeypatch, hostile_symlink, historical):
    """Adversarial check: a symlink at saver_state.json whose stat() raises OSError (EACCES or ENAMETOOLONG)
    must treat the flag as unusable, must not raise where the base does not, must not drop collected alerts,
    and must not make runner.run_all_checks page a crash CRITICAL (or an analytics_job_oom CRITICAL)."""
    picks = tmp_path / "oom" / "picks"
    cleanup_perms: list[Path] = []
    write_dated_obs(picks, DAY.date(), "used", streak=14, best=15)
    resolved = [DAY.date() + dt.timedelta(days=1), DAY.date() + dt.timedelta(days=2)]
    monkeypatch.setattr(health, "latest_resolved_pick_date", lambda _: max(resolved))
    monkeypatch.setattr(health, "_scan_resolved_pick_dates", lambda *a, **kw: resolved)
    monkeypatch.setattr(health, "resolved_pick_settlement_gap", lambda *a: 2)

    baseline = health.check(
        picks, expected=True, now=DAY, historical_day=DAY.date() if historical else None,
    )
    assert len(baseline) == 1 and baseline[0].source == "contest_state"

    with tempfile.TemporaryDirectory(dir="/tmp") as short_dir:
        try:
            plant_flag(picks, hostile_symlink, DAY.year, cleanup_perms, Path(short_dir))
            # Check what runner.run_all_checks would send before asserting on direct check:
            wall = DAY + dt.timedelta(hours=3) if historical else DAY

            class WallClock(REAL_DT):
                @classmethod
                def now(cls, tz=None):
                    return wall.astimezone(tz) if tz else wall.replace(tzinfo=None)

            dms = []
            real_safe_run = runner._safe_run
            with patch.object(runner, "datetime", WallClock), \
                 patch.object(alert, "datetime", WallClock), \
                 patch.object(alert, "send_dm", side_effect=lambda r, b: dms.append((r, b)) or "ok"), \
                 patch.object(runner, "_safe_run", side_effect=lambda n, fn: real_safe_run(n, fn) if n == "contest_state" else []):
                eod_alerts = runner.run_all_checks(
                    picks_dir=picks,
                    models_dir=tmp_path / "models",
                    dm_recipient="ops.bsky.social",
                    today=DAY.date(),
                    contest_state_expected=True,
                )
            print(f"\n[ADVERSARIAL {hostile_symlink} historical={historical}] eod_alerts={eod_alerts} dms={dms}")
            actual = health.check(
                picks, expected=True, now=DAY, historical_day=DAY.date() if historical else None,
            )
            assert actual == baseline
            assert eod_alerts == baseline
            assert dms == []
        finally:
            for p in reversed(cleanup_perms):
                try:
                    p.chmod(0o700)
                except OSError:
                    pass


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "-rA"]))
