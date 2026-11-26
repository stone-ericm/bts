"""Item 2 probe: exhaustive flag-vs-dated comparison matrix, hostile/non-regular flags, streaks 0..400, and Dec 31 / Jan 1 C1 clocks."""

from __future__ import annotations

import datetime as dt
import json
import os
import signal
import socket
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

import bts
from bts import cli, contest_state
from bts.health import alert, contest_state as health, runner

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = REAL_DT(2027, 7, 15, 22, 0, tzinfo=ET)
DEC31_LAST_US = REAL_DT(2027, 12, 31, 23, 59, 59, 999999, tzinfo=ET)
JAN1_FIRST_US = REAL_DT(2028, 1, 1, 0, 0, 0, 0, tzinfo=ET)
JAN1_0005_ET = REAL_DT(2028, 1, 1, 0, 5, 0, 0, tzinfo=ET)
INCIDENT = "saver_state:dated_used_mismatch"

FLAG_CASES = (
    "active",
    "not_earned",
    "used",
    "uninitialized",
    "absent",
    "prior_season_2026",
    "future_season_2028",
    "bool_true_season",
    "bool_false_season",
    "float_season",
    "nan_season",
    "unreadable_mode_000",
    "invalid_utf8",
    "utf8_bom",
    "malformed_json",
    "json_list",
    "json_scalar_str",
    "json_null",
    "unhashable_state_list",
    "directory",
    "fifo",
    "symlink_to_fifo",
    "unix_socket",
    "dev_null_symlink",
    "dev_zero_symlink",
    "broken_symlink",
    "symlink_loop",
    "stat_permission_error_symlink",
    "stat_enametoolong_symlink",
    "huge_int_5100_digits",
    "deep_nesting_200k",
)

DEEP_200K = b"[" * 200_000 + b"]" * 200_000


class FlagCheckBlocked(BaseException):
    pass


@contextmanager
def per_call_deadline(seconds: int = 5):
    def _on_alarm(*_):
        raise FlagCheckBlocked("contest_state.check blocked on flag read")

    prev = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def _clock(fixed_now: dt.datetime):
    class Clock(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)
    return Clock


def _write_dated(picks: Path, when: dt.datetime, saver: str, streak: int) -> Path:
    clk = _clock(when)
    with patch.object(dt, "datetime", clk), patch.object(cli, "datetime", clk):
        res = CliRunner().invoke(cli.cli, [
            "set-contest-streak",
            "--date", str(when.astimezone(ET).date()),
            "--streak", str(streak),
            "--best-streak", str(max(streak, 15)),
            "--saver", saver,
            "--picks-dir", str(picks),
        ])
    assert res.exit_code == 0, res.output
    return picks / "account_state" / "contest_streak.manual.json"


@contextmanager
def setup_flag_case(picks: Path, case: str, year: int = 2027):
    acct = picks / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    path = acct / "saver_state.json"
    cleanup_chmod: Path | None = None
    sock: socket.socket | None = None
    short_tmp: tempfile.TemporaryDirectory | None = None
    try:
        if case == "absent":
            pass
        elif case in ("active", "not_earned", "used", "uninitialized"):
            path.write_text(json.dumps({"season": year, "state": case}), encoding="utf-8")
        elif case == "prior_season_2026":
            path.write_text(json.dumps({"season": year - 1, "state": "active"}), encoding="utf-8")
        elif case == "future_season_2028":
            path.write_text(json.dumps({"season": year + 1, "state": "active"}), encoding="utf-8")
        elif case == "bool_true_season":
            path.write_text('{"season": true, "state": "active"}', encoding="utf-8")
        elif case == "bool_false_season":
            path.write_text('{"season": false, "state": "active"}', encoding="utf-8")
        elif case == "float_season":
            path.write_text(f'{{"season": {year}.0, "state": "active"}}', encoding="utf-8")
        elif case == "nan_season":
            path.write_text('{"season": NaN, "state": "active"}', encoding="utf-8")
        elif case == "unreadable_mode_000":
            path.write_text(json.dumps({"season": year, "state": "active"}), encoding="utf-8")
            path.chmod(0o000)
            cleanup_chmod = path
        elif case == "invalid_utf8":
            path.write_bytes(b"\xff\xfe\x80")
        elif case == "utf8_bom":
            path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"season": year, "state": "active"}).encode("utf-8"))
        elif case == "malformed_json":
            path.write_bytes(b'{"season": 2027, "state": ')
        elif case == "json_list":
            path.write_bytes(b'[{"season": 2027, "state": "active"}]')
        elif case == "json_scalar_str":
            path.write_bytes(b'"active"')
        elif case == "json_null":
            path.write_bytes(b"null")
        elif case == "unhashable_state_list":
            path.write_bytes(b'{"season": 2027, "state": ["active"]}')
        elif case == "directory":
            path.mkdir()
        elif case == "fifo":
            os.mkfifo(path)
        elif case == "symlink_to_fifo":
            fifo_target = acct / "real_fifo"
            os.mkfifo(fifo_target)
            path.symlink_to(fifo_target)
        elif case == "unix_socket":
            short_tmp = tempfile.TemporaryDirectory(dir="/tmp")
            sock_file = Path(short_tmp.name) / "s.sock"
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(sock_file))
            path.symlink_to(sock_file)
        elif case == "dev_null_symlink":
            path.symlink_to("/dev/null")
        elif case == "dev_zero_symlink":
            path.symlink_to("/dev/zero")
        elif case == "broken_symlink":
            path.symlink_to(acct / "nonexistent_target.json")
        elif case == "symlink_loop":
            loop_b = acct / "loop_b.json"
            path.symlink_to(loop_b)
            loop_b.symlink_to(path)
        elif case == "stat_permission_error_symlink":
            locked = acct / "locked_dir"
            locked.mkdir()
            (locked / "flag.json").write_text(json.dumps({"season": year, "state": "active"}), encoding="utf-8")
            locked.chmod(0o000)
            cleanup_chmod = locked
            path.symlink_to(locked / "flag.json")
        elif case == "stat_enametoolong_symlink":
            path.symlink_to("q" * 300)
        elif case == "huge_int_5100_digits":
            path.write_bytes(b'{"season": ' + b"9" * 5100 + b', "state": "active"}')
        elif case == "deep_nesting_200k":
            path.write_bytes(DEEP_200K)
        else:
            raise ValueError(f"Unhandled flag case: {case}")
        yield path
    finally:
        if sock is not None:
            sock.close()
        if cleanup_chmod is not None:
            cleanup_chmod.chmod(0o700)
        if short_tmp is not None:
            short_tmp.cleanup()


def test_decoder_200k_genuinely_hits_recursion_error():
    with pytest.raises(RecursionError):
        json.loads(DEEP_200K)


def test_full_flag_vs_dated_matrix_and_alert_preservation(tmp_path):
    """Run all 31 flag states x {used, active, not_earned} x streaks (including >57 up to 400) x {live, historical}."""
    streaks = (0, 1, 9, 10, 12, 14, 15, 16, 56, 57, 58, 400)
    total_checks = 0

    for obs_saver in ("used", "active", "not_earned"):
        # Sweep all streaks on core + hostile flags, and all 31 flags at representative streaks (0, 9, 12, 58, 400)
        for streak in streaks:
            cases = FLAG_CASES if streak in (0, 9, 12, 58, 400) else (
                "active", "not_earned", "used", "uninitialized", "absent",
                "fifo", "stat_permission_error_symlink", "deep_nesting_200k",
            )
            for flag_case in cases:
                picks = tmp_path / f"{obs_saver}_{streak}_{flag_case}" / "picks"
                _write_dated(picks, DAY, obs_saver, streak)
                # Add 2 newer settled picks so a coverage-lag WARN is ALREADY collected before the dated branch!
                (picks / "2027-07-16.json").write_text('{"result": "hit"}', encoding="utf-8")
                (picks / "2027-07-17.json").write_text('{"result": "hit"}', encoding="utf-8")

                modes = (None, DAY.date())
                baseline = [health.check(picks, expected=True, now=DAY, historical_day=m) for m in modes]
                assert all(b and b[0].source == "contest_state" and b[0].level == "WARN" for b in baseline)

                with setup_flag_case(picks, flag_case, year=2027):
                    for mode, base_alerts in zip(modes, baseline):
                        with per_call_deadline(5):
                            actual = health.check(picks, expected=True, now=DAY, historical_day=mode)
                        mm = [a for a in actual if a.incident_key == INCIDENT]
                        should_warn = (obs_saver == "used" and flag_case in {"active", "not_earned"})
                        assert len(mm) == int(should_warn), (obs_saver, streak, flag_case, mode, actual)
                        assert [a for a in actual if a.incident_key != INCIDENT] == base_alerts
                        if should_warn:
                            assert actual[-1] == mm[0]
                            assert mm[0].level == "WARN" and mm[0].source == "saver_state"
                            assert flag_case in mm[0].message and "2027" in mm[0].message
                        total_checks += 1

    print(f"\nVerified {total_checks} flag-vs-dated checks across {len(FLAG_CASES)} flag states and streaks up to 400.")


def test_non_in_effect_and_invalid_dated_states_never_emit_mismatch(tmp_path):
    """Expired, future-dated, legacy override, and invalid v3 never emit DATED_USED_MISMATCH even with active flag."""
    for kind in ("expired", "another_date", "legacy", "invalid_v3"):
        picks = tmp_path / kind / "picks"
        acct = picks / "account_state"
        acct.mkdir(parents=True)
        (acct / "saver_state.json").write_text('{"season": 2027, "state": "active"}', encoding="utf-8")
        if kind == "expired":
            _write_dated(picks, DAY - dt.timedelta(days=1), "used", 14)
        elif kind == "another_date":
            _write_dated(picks, DAY + dt.timedelta(days=1), "used", 14)
        elif kind == "legacy":
            (acct / "contest_streak.manual.json").write_text(json.dumps({
                "active_streak": 14, "best_streak": 15, "source_date": "2027-07-15",
                "override_expires_at": (DAY + dt.timedelta(hours=4)).isoformat(),
                "recorded_at": DAY.isoformat(), "source": "manual_cli",
            }), encoding="utf-8")
        elif kind == "invalid_v3":
            p = _write_dated(picks, DAY, "used", 14)
            d = json.loads(p.read_text())
            d["active_streak"] = 401  # exceeds MAX_SEASON_STREAK=400
            p.write_text(json.dumps(d), encoding="utf-8")

        for mode in (None, DAY.date()):
            with per_call_deadline(5):
                out = health.check(picks, expected=True, now=DAY, historical_day=mode)
            assert not [a for a in out if a.incident_key == INCIDENT], (kind, mode, out)
            if kind == "invalid_v3":
                assert len(out) == 1 and out[0].level == "CRITICAL" and out[0].source == "contest_state"


def test_year_boundary_utc_rollover_and_c1_processed_slate_clock(tmp_path):
    """Verify Dec 31 23:59:59.999999 ET (2028 UTC), Jan 1 00:00:00 ET, and Jan 1 00:05 ET C1 EOD for Dec 31."""
    # 1. Dec 31 23:59:59.999999 ET passed in UTC (2028-01-01T04:59:59.999999Z):
    assert DEC31_LAST_US.astimezone(dt.timezone.utc).year == 2028
    p_nye = tmp_path / "nye" / "picks"
    _write_dated(p_nye, DEC31_LAST_US, "used", 14)
    (p_nye / "account_state" / "saver_state.json").write_text('{"season": 2027, "state": "active"}', encoding="utf-8")
    out_utc = health.check(p_nye, expected=True, now=DEC31_LAST_US.astimezone(dt.timezone.utc))
    assert len(out_utc) == 1 and out_utc[0].incident_key == INCIDENT and "season 2027" in out_utc[0].message

    # A 2028 flag at Dec 31 23:59:59.999999 ET does NOT match season 2027 -> no mismatch warning:
    (p_nye / "account_state" / "saver_state.json").write_text('{"season": 2028, "state": "active"}', encoding="utf-8")
    assert health.check(p_nye, expected=True, now=DEC31_LAST_US.astimezone(dt.timezone.utc)) == []

    # 2. Jan 1 00:00:00 ET: dated obs for Jan 1 + 2028 flag warns for season 2028; 2027 flag does not:
    p_jan = tmp_path / "jan1" / "picks"
    _write_dated(p_jan, JAN1_FIRST_US, "used", 14)
    (p_jan / "account_state" / "saver_state.json").write_text('{"season": 2028, "state": "not_earned"}', encoding="utf-8")
    out_jan = health.check(p_jan, expected=True, now=JAN1_FIRST_US)
    assert len(out_jan) == 1 and out_jan[0].incident_key == INCIDENT and "season 2028" in out_jan[0].message
    (p_jan / "account_state" / "saver_state.json").write_text('{"season": 2027, "state": "not_earned"}', encoding="utf-8")
    assert health.check(p_jan, expected=True, now=JAN1_FIRST_US) == []

    # 3. C1 after-midnight EOD at 2028-01-01 00:05 ET for processed slate today=2027-12-31:
    (p_nye / "account_state" / "saver_state.json").write_text('{"season": 2027, "state": "active"}', encoding="utf-8")
    clk = _clock(JAN1_0005_ET)
    dms = []
    orig = runner._safe_run
    with patch.object(dt, "datetime", clk), \
         patch.object(health, "datetime", clk), \
         patch.object(runner, "datetime", clk), \
         patch.object(alert, "datetime", clk), \
         patch.object(runner, "_safe_run", lambda n, fn: orig(n, fn) if n == "contest_state" else []), \
         patch.object(alert, "send_dm", side_effect=lambda r, b: dms.append((r, b)) or "ack-nye"):
        eod_out = runner.run_all_checks(
            p_nye, p_nye.parent / "models", "ops.bsky.social",
            today=dt.date(2027, 12, 31), contest_state_expected=True,
        )
    assert len(eod_out) == 1 and eod_out[0].incident_key == INCIDENT
    assert "Prior slate 2027-12-31 ET. The dated observation recorded the saver as used. At this check the flag for season 2027 read active." in dms[0][1]


if __name__ == "__main__":
    print(f"bts.__file__ = {bts.__file__}")
    sys.exit(pytest.main([__file__, "-v", "-s", "-rA"]))
