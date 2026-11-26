"""Item 5 probe: whole-diff audit, FD leak invariance, non-sent payload parity, multi-alert body binding, and CLAUDE.md check."""

from __future__ import annotations

from contextlib import redirect_stderr
from datetime import date
from io import StringIO
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bts import scheduler
from bts.health import alert

DAY = date(2027, 3, 25)


class ProbeTimeout(BaseException):
    pass


def set_deadline(seconds: int) -> None:
    def on_alarm(*_):
        raise ProbeTimeout(f"probe_5 exceeded {seconds}s")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)


def open_fds() -> set[int]:
    out = set()
    for name in os.listdir("/dev/fd"):
        try:
            out.add(int(name))
        except ValueError:
            pass
    return out


def main() -> None:
    set_deadline(40)
    try:
        # 1. Verify exact changed file sets for 386aa846..HEAD and e897c766..HEAD
        diff_all = subprocess.run(
            ["git", "diff", "--name-only", "386aa846", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True, timeout=10,
        ).stdout.splitlines()
        assert diff_all == [
            "CLAUDE.md",
            "src/bts/health/alert.py",
            "src/bts/scheduler.py",
            "tests/test_alert_status_callers_cs6b2.py",
            "tests/test_wr_c2_carbon_v2.py",
            "tests/test_wr_c2_r2_carbon.py",
            "tests/test_wr_c2_receipt_identity.py",
            "tests/test_wr_c2_recovery.py",
        ], diff_all

        diff_r3 = subprocess.run(
            ["git", "diff", "--name-only", "e897c766", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True, timeout=10,
        ).stdout.splitlines()
        assert diff_r3 == [
            "CLAUDE.md",
            "src/bts/health/alert.py",
            "tests/test_wr_c2_carbon_v2.py",
            "tests/test_wr_c2_r2_carbon.py",
            "tests/test_wr_c2_receipt_identity.py",
            "tests/test_wr_c2_recovery.py",
        ], diff_r3

        # Check no hardcoded calendar dates or season years in added src/ code lines
        src_diff = subprocess.run(
            ["git", "diff", "-U0", "386aa846", "HEAD", "--", "src/"],
            cwd=ROOT, capture_output=True, text=True, check=True, timeout=10,
        ).stdout
        added_code = [
            ln[1:]
            for ln in src_diff.splitlines()
            if ln.startswith("+") and not ln.startswith("+++") and not ln[1:].lstrip().startswith("#")
        ]
        for ln in added_code:
            assert not re.search(r"\b202[5-9]\b", ln), f"Hardcoded year in src diff: {ln}"
            assert not re.search(r"\d{4}-\d{2}-\d{2}", ln), f"Hardcoded ISO date in src diff: {ln}"

        with TemporaryDirectory(prefix="p5_", dir=REVIEW) as td:
            troot = Path(td)

            # 2. Non-sent payload exact shape check (failed & skipped_no_recipient must NOT gain body_sha256)
            fail_path = troot / "failed.json"
            with patch.object(alert, "send_dm", Mock(return_value=None)), redirect_stderr(StringIO()):
                assert alert.dispatch_dm_for_health_alerts(
                    [alert.Alert("CRITICAL", "late_delivery", "msg")],
                    "offline.test", status_path=fail_path, now_et_date=DAY, require_message_id=True,
                ) is True
            assert json.loads(fail_path.read_text(encoding="utf-8")) == {
                "schema_version": alert.HEALTH_DM_STATUS_SCHEMA_VERSION,
                "status": "failed",
                "error": "send_dm returned no usable message id",
            }

            skip_path = troot / "skipped.json"
            with patch.object(alert, "send_dm", Mock()), redirect_stderr(StringIO()):
                assert alert.dispatch_dm_for_health_alerts(
                    [alert.Alert("CRITICAL", "late_delivery", "msg")],
                    None, status_path=skip_path, now_et_date=DAY, require_message_id=True,
                ) is False
            assert json.loads(skip_path.read_text(encoding="utf-8")) == {
                "schema_version": alert.HEALTH_DM_STATUS_SCHEMA_VERSION,
                "status": "skipped_no_recipient",
                "error": "dm_recipient is not configured",
            }

            # 3. Same incident_key + same message, different source -> formatted body differs -> MUST NOT suppress!
            src_path = troot / "same_inc_diff_source.json"
            sender = Mock(return_value="ack-1")
            with patch.object(alert, "send_dm", sender), redirect_stderr(StringIO()):
                assert alert.dispatch_dm_for_health_alerts(
                    [alert.Alert("CRITICAL", "source_one", "identical text", incident_key="shared_inc")],
                    "offline.test", status_path=src_path, now_et_date=DAY, require_message_id=True,
                ) is True
                # Same alert_key ["CRITICAL:shared_inc"] and same message, but source_two changes body "- [source_two] ..."
                assert alert.dispatch_dm_for_health_alerts(
                    [alert.Alert("CRITICAL", "source_two", "identical text", incident_key="shared_inc")],
                    "offline.test", status_path=src_path, now_et_date=DAY, require_message_id=True,
                ) is True
                assert sender.call_count == 2

            # 4. Empty / INFO-only alerts with FIFO at status_path -> returns False without touching FIFO
            fifo_path = troot / "idle.fifo"
            os.mkfifo(fifo_path)
            with patch.object(alert, "send_dm", Mock()), redirect_stderr(StringIO()) as buf:
                assert alert.dispatch_dm_for_health_alerts(
                    [alert.Alert("INFO", "late_delivery", "info only")],
                    "offline.test", status_path=fifo_path, now_et_date=DAY, require_message_id=True,
                ) is False
                assert buf.getvalue() == ""

            # 5. status_path=None on opt-in route: works cleanly without touching filesystem
            with patch.object(alert, "send_dm", Mock(return_value="ack-none-path")), redirect_stderr(StringIO()) as buf:
                assert alert.dispatch_dm_for_health_alerts(
                    [alert.Alert("CRITICAL", "late_delivery", "no status path")],
                    "offline.test", status_path=None, now_et_date=DAY, require_message_id=True,
                ) is True
                assert buf.getvalue() == ""

            # 6. Receipt permissions (0o600) and FD leak invariance across normal, error, and interrupt paths
            perm_path = troot / "perms" / "2027-03-25.json"
            with patch.object(alert, "send_dm", Mock(return_value="ack-perm")), redirect_stderr(StringIO()):
                alert.dispatch_dm_for_health_alerts(
                    [alert.Alert("CRITICAL", "late_delivery", "perm check")],
                    "offline.test", status_path=perm_path, now_et_date=DAY, require_message_id=True,
                )
            st = perm_path.lstat()
            assert stat.S_IMODE(st.st_mode) == 0o600 and st.st_nlink == 1

            fds_before = open_fds()
            dup_path = troot / "fd_dup" / "r.json"
            dup_path.parent.mkdir()
            dup_path.write_text('{"a":1,"a":2}', encoding="utf-8")
            dir_entry = troot / "fd_dir" / "r.json"
            dir_entry.mkdir(parents=True)

            with redirect_stderr(StringIO()):
                for _ in range(5):
                    with patch.object(alert, "send_dm", Mock(return_value="ack")):
                        alert.dispatch_dm_for_health_alerts(
                            [alert.Alert("CRITICAL", "late_delivery", "fd1")],
                            "offline.test", status_path=perm_path, now_et_date=DAY, require_message_id=True,
                        )
                        alert.dispatch_dm_for_health_alerts(
                            [alert.Alert("CRITICAL", "late_delivery", "fd2")],
                            "offline.test", status_path=dup_path, now_et_date=DAY, require_message_id=True,
                        )
                        alert.dispatch_dm_for_health_alerts(
                            [alert.Alert("CRITICAL", "late_delivery", "fd3")],
                            "offline.test", status_path=fifo_path, now_et_date=DAY, require_message_id=True,
                        )
                        alert.dispatch_dm_for_health_alerts(
                            [alert.Alert("CRITICAL", "late_delivery", "fd4")],
                            "offline.test", status_path=dir_entry, now_et_date=DAY, require_message_id=True,
                        )
                    for intr in (KeyboardInterrupt, SystemExit):
                        try:
                            with patch.object(alert, "send_dm", Mock(side_effect=intr("stop"))):
                                alert.dispatch_dm_for_health_alerts(
                                    [alert.Alert("CRITICAL", "late_delivery", "fd_intr")],
                                    "offline.test", status_path=troot / "intr" / "r.json",
                                    now_et_date=DAY, require_message_id=True,
                                )
                        except intr:
                            pass
            fds_after = open_fds()
            assert fds_before == fds_after, f"FD leak detected: before={fds_before}, after={fds_after}"

            # 7. Malformed date / config containment in _alert_delivery_outcome_unknown
            for bad_date in ("not-a-date", "2027-02-30", "\ud800"):
                err = StringIO()
                with redirect_stderr(err):
                    scheduler._alert_delivery_outcome_unknown({"bluesky": {"dm_recipient": "x"}}, troot / "picks", bad_date)
                lines = [ln for ln in err.getvalue().splitlines() if ln.startswith("Unknown-outcome alert failed: ")]
                assert len(lines) == 1 and len(lines[0].removeprefix("Unknown-outcome alert failed: ")) <= 200

            for bad_cfg in ({"bluesky": None}, {"bluesky": 17}, None):
                err = StringIO()
                with redirect_stderr(err):
                    scheduler._alert_delivery_outcome_unknown(bad_cfg, troot / "picks", "2027-03-25")  # type: ignore[arg-type]
                lines = [ln for ln in err.getvalue().splitlines() if ln.startswith("Unknown-outcome alert failed: ")]
                assert len(lines) == 1

        # 8. CLAUDE.md C2 & C2b verification
        claude_md = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        required_phrases = [
            "health_state/delivery_outcome_unknown/<slate>.json",
            "requires a nonblank DM message ID",
            "The recommendation is never resent.",
            "end-of-day second chance",
            "undated scheduler start after midnight selects the new slate",
            "scoreable `delivered` decision with an unconfirmed pick",
            "entry watchdog can still say",
            "C2b requires duplicate-free JSON with the exact sent-receipt fields and a SHA-256 of the full DM body.",
            "five-field C2 receipt from `e897c766` or `dfe774df` permits one repeated alert",
            "byte-copy of a genuine acknowledgement for the same slate still deduplicates",
            "long receipt path can consume the 200-character diagnostic",
        ]
        for phrase in required_phrases:
            assert phrase in claude_md, f"Missing phrase in CLAUDE.md: {phrase}"

        summary = {
            "tracked_files_386aa846_to_HEAD": diff_all,
            "tracked_files_e897c766_to_HEAD": diff_r3,
            "hardcoded_dates_or_years_in_src": 0,
            "failed_and_skipped_payloads_unchanged": True,
            "same_incident_key_different_source_not_suppressed": True,
            "file_mode_0600_and_zero_fd_leaks": True,
            "malformed_date_and_config_contained": True,
            "claude_md_verified": True,
        }
        (REVIEW / "probe_5_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("PROBE 5 OK: diff audit, non-sent payload parity, source-prefix binding, 0600 perms, 0 FD leaks, containment, and CLAUDE.md verified.")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
