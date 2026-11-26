"""Item 1 adversarial probe: exact suppressing receipt set and whole-tree writer audit."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext, redirect_stderr
from datetime import datetime, timedelta
import hashlib
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import unicodedata
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bts import (
    cli,
    contest_state,
    daily_decision,
    dm,
    entry_receipt,
    picks,
    posting,
    reconcile_receipt,
    scheduler,
    strategy,
)
from bts.daily_decision import load_decision, write_decision
from bts.data import backup
from bts.health import alert, attention, memory_growth, restart_spike, runner

ET = ZoneInfo("America/New_York")
DAY = "2027-03-25"
PREV_DAY = "2027-03-24"
NEXT_DAY = "2027-03-26"
NOW = datetime(2027, 3, 25, 18, 25, 0, tzinfo=ET)
KEY = "CRITICAL:late_delivery"


class ProbeTimeout(BaseException):
    pass


def set_deadline(seconds: int) -> None:
    def on_alarm(*_):
        raise ProbeTimeout(f"probe exceeded {seconds}s")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)


def u7_body(day: str = DAY) -> str:
    return (
        "BTS health CRITICAL alert(s):\n"
        f"- [late_delivery] DELIVERY OUTCOME UNKNOWN for {day} ET. A send attempt for that slate "
        "was never confirmed, and the scheduler locked the slate without resending. "
        f"Check the DM or public feed and the MLB entry for {day} ET. Do not enter or change a "
        f"pick based on this alert. If {day} is still today in ET and its submission window is "
        "open, run bts season-contract status with --picks-dir set to the scheduler's picks "
        "directory before making any entry decision for that slate."
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_games(day_str: str) -> list[dict]:
    return [{
        "gamePk": 101,
        "gameDate": f"{day_str}T23:10:00Z",
        "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
        "teams": {"away": {"team": {"name": "Away"}}, "home": {"team": {"name": "Home"}}},
    }]


def make_cfg(pdir: Path, recipient: str | None = "offline.test", contest_req: bool = False) -> dict:
    bsky = {"dm_recipient": recipient} if recipient is not None else {}
    return {
        "orchestrator": {"picks_dir": str(pdir), "heartbeat_path": str(pdir.parent / "hb")},
        "bluesky": bsky,
        "scheduler": {
            "pick_delivery": "dm",
            "live_forward_capture_on_lock": False,
            "contest_state_required": contest_req,
        },
        "health_checks": {"enabled": False},
        "tiers": [],
    }


def stage_attempted_pick(pdir: Path, day_str: str = DAY, moment: datetime = NOW):
    p1 = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.82, [], False, 101, f"{day_str}T23:10:00Z")
    dp = picks.DailyPick(day_str, moment.isoformat(), p1, None, None, delivery_attempted=True)
    picks.save_pick(dp, pdir)
    return dp


def enter_world(stack: ExitStack, now_box: list[datetime], alert_sender: Mock, err_buf: StringIO, allow_pick_dm: bool = False):
    class PinnedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now_box[0].astimezone(tz) if tz else now_box[0].replace(tzinfo=None)

    pick_dm_mock = Mock(return_value="pick-dm-id") if allow_pick_dm else Mock(side_effect=AssertionError("unexpected pick resend"))
    for ctx in (
        patch.object(contest_state, "datetime", PinnedClock),
        patch.object(daily_decision, "datetime", PinnedClock),
        patch.object(alert, "datetime", PinnedClock),
        patch.object(scheduler, "_now_et", lambda: now_box[0]),
        patch.object(scheduler, "fetch_schedule", side_effect=lambda d: make_games(d)),
        patch.object(scheduler, "check_confirmed_lineups", lambda *_: {}),
        patch.object(scheduler, "_idle_until_next_wakeup", lambda *_: None),
        patch.object(scheduler, "_watchdog_ping_sleep", lambda *_: None),
        patch.object(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext()),
        patch.object(scheduler, "notify_ready", Mock()),
        patch.object(scheduler, "notify_watchdog", Mock()),
        patch.object(scheduler, "run_result_polling", Mock(return_value="final")),
        patch.object(dm, "send_dm", pick_dm_mock),
        patch.object(alert, "send_dm", alert_sender),
        patch.object(posting, "post_to_bluesky", Mock(side_effect=AssertionError("unexpected public post"))),
        patch.object(strategy, "_mdp_cache", None),
        redirect_stderr(err_buf),
    ):
        stack.enter_context(ctx)


def unicode_escape_json_str(s: str) -> str:
    return '"' + "".join(f"\\u{ord(ch):04x}" for ch in s) + '"'


def run_encounter_pair(base_dir: Path, route: str, seeded_bytes: bytes) -> tuple[int, bytes, str]:
    pdir = base_dir / "picks"
    pdir.mkdir(parents=True)
    now_box = [NOW]
    dp = stage_attempted_pick(pdir, DAY, NOW)
    own_path = base_dir / "health_state" / "delivery_outcome_unknown" / f"{DAY}.json"
    own_path.parent.mkdir(parents=True)
    own_path.write_bytes(seeded_bytes)
    sender = Mock(return_value="fresh-ack-id")
    err = StringIO()
    with ExitStack() as stack:
        enter_world(stack, now_box, sender, err)
        stack.enter_context(patch.object(contest_state, "load_decision_streak_state", return_value=Mock(streak=0)))
        cfg = make_cfg(pdir)
        if route == "direct":
            st = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
            assert scheduler._deliver_and_lock_pick(dp, cfg, pdir, st, DAY, "item1") is False
        else:
            assert scheduler.run_day(DAY, cfg) is None
        after_first = own_path.read_bytes()
        assert scheduler.run_day(DAY, cfg) is None
        after_second = own_path.read_bytes()
        assert after_second == after_first
        assert scheduler.load_state(DAY, pdir).pick_locked is True
        assert load_decision(DAY, pdir)["delivery_status"] == "locked_unconfirmed"
    return sender.call_count, after_first, err.getvalue()


def main() -> None:
    set_deadline(50)
    try:
        body = u7_body(DAY)
        body_hash = sha256_hex(body)
        canonical_dict = {
            "body_sha256": body_hash,
            "message_id": "seed-ack",
            "schema_version": alert.HEALTH_DM_STATUS_SCHEMA_VERSION,
            "sent_day": DAY,
            "sent_sources": [KEY],
            "status": "sent",
        }
        canonical_bytes = (json.dumps(canonical_dict, indent=2, sort_keys=True) + "\n").encode("utf-8")

        with TemporaryDirectory(prefix="p1_", dir=REVIEW) as tmp_str:
            troot = Path(tmp_str)

            # 1. Construct real outputs from every writer in the repository
            gen_root = troot / "writers"
            gen_picks = gen_root / "picks"
            gen_picks.mkdir(parents=True)
            now_box = [NOW]
            gen_sender = Mock(return_value="writer-ack")
            gen_err = StringIO()

            with ExitStack() as stack:
                enter_world(stack, now_box, gen_sender, gen_err, allow_pick_dm=True)
                cfg = make_cfg(gen_picks)

                # (W0) Own writer, acknowledged for SAME slate DAY
                own_same_dir = troot / "w_own_same" / "picks"
                own_same_dir.mkdir(parents=True)
                scheduler._alert_delivery_outcome_unknown(make_cfg(own_same_dir), own_same_dir, DAY)
                real_own_same_bytes = (own_same_dir.parent / "health_state/delivery_outcome_unknown" / f"{DAY}.json").read_bytes()

                # (W1) Own writer, acknowledged for OTHER slate PREV_DAY
                own_prev_dir = troot / "w_own_prev" / "picks"
                own_prev_dir.mkdir(parents=True)
                scheduler._alert_delivery_outcome_unknown(make_cfg(own_prev_dir), own_prev_dir, PREV_DAY)
                real_own_prev_bytes = (own_prev_dir.parent / "health_state/delivery_outcome_unknown" / f"{PREV_DAY}.json").read_bytes()

                # (W2) Own writer, failed (missing message_id)
                own_fail_dir = troot / "w_own_fail" / "picks"
                own_fail_dir.mkdir(parents=True)
                gen_sender.return_value = None
                scheduler._alert_delivery_outcome_unknown(make_cfg(own_fail_dir), own_fail_dir, DAY)
                real_own_fail_bytes = (own_fail_dir.parent / "health_state/delivery_outcome_unknown" / f"{DAY}.json").read_bytes()
                gen_sender.return_value = "writer-ack"

                # (W3) Own writer, skipped_no_recipient
                own_skip_dir = troot / "w_own_skip" / "picks"
                own_skip_dir.mkdir(parents=True)
                scheduler._alert_delivery_outcome_unknown(make_cfg(own_skip_dir, recipient=None), own_skip_dir, DAY)
                real_own_skip_bytes = (own_skip_dir.parent / "health_state/delivery_outcome_unknown" / f"{DAY}.json").read_bytes()

                # (W4) Legacy cutoff refusal (_refuse_late_delivery -> health_dm_delivery_status.json)
                cut_dir = troot / "w_cutoff" / "picks"
                cut_dir.mkdir(parents=True)
                p1 = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Cole", 543037, 0.82, [], False, 101, f"{DAY}T22:20:00Z")
                dp_cut = picks.DailyPick(DAY, NOW.isoformat(), p1, None, None)
                picks.save_pick(dp_cut, cut_dir)
                st_cut = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
                scheduler._refuse_late_delivery(dp_cut, make_cfg(cut_dir), cut_dir, st_cut, DAY, "lineup", NOW - timedelta(minutes=5), NOW)
                cutoff_dm_body = gen_sender.call_args.args[1]
                cutoff_alert_msg = cutoff_dm_body.split("[late_delivery] ", 1)[1]
                real_legacy_cutoff_bytes = (cut_dir.parent / "health_state/health_dm_delivery_status.json").read_bytes()

                # (W5) Direct opt-in dispatcher call for another alert (same slate, same CRITICAL:late_delivery, cutoff body)
                foreign_optin_path = troot / "w_foreign_optin.json"
                assert alert.dispatch_dm_for_health_alerts(
                    [alert.Alert("CRITICAL", "late_delivery", cutoff_alert_msg)],
                    "offline.test",
                    status_path=foreign_optin_path,
                    now_et_date=NOW.date(),
                    require_message_id=True,
                ) is True
                real_foreign_optin_bytes = foreign_optin_path.read_bytes()

                # (W6) Legacy void game refusal
                void_dir = troot / "w_void" / "picks"
                void_dir.mkdir(parents=True)
                picks.save_pick(dp_cut, void_dir)
                st_void = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
                scheduler._refuse_void_game_delivery(dp_cut, make_cfg(void_dir), void_dir, st_void, DAY, "lineup", {101: "Postponed"})
                real_legacy_void_bytes = (void_dir.parent / "health_state/health_dm_delivery_status.json").read_bytes()

                # (W7) Legacy contest_state failure
                csf_dir = troot / "w_csf" / "picks"
                csf_dir.mkdir(parents=True)
                scheduler._alert_contest_state_failure(make_cfg(csf_dir), ValueError("contest state broken"))
                real_legacy_csf_bytes = (csf_dir.parent / "health_state/health_dm_delivery_status.json").read_bytes()

                # (W8) Legacy missed_pick alert
                mp_dir = troot / "w_mp" / "picks"
                mp_dir.mkdir(parents=True)
                scheduler._alert_missed_pick(make_cfg(mp_dir), dp_cut, 9.0)
                real_legacy_missed_bytes = (mp_dir.parent / "health_state/health_dm_delivery_status.json").read_bytes()

                # (W9) Legacy EOD run_all_checks + warn_attention_state.json
                eod_dir = troot / "w_eod" / "picks"
                eod_dir.mkdir(parents=True)
                stage_attempted_pick(eod_dir, DAY, NOW)
                with patch.object(
                    runner,
                    "_safe_run",
                    lambda n, fn: fn() if n == "post_failure" else ([alert.Alert("WARN", "saver_state", "warn")] if n == "disk_fill" else []),
                ), patch.object(runner.realized_calibration, "_current_deploy_iso", lambda _: None):
                    runner.run_all_checks(eod_dir, troot / "models", "offline.test", today=NOW.date())
                real_eod_shared_bytes = (eod_dir.parent / "health_state/health_dm_delivery_status.json").read_bytes()
                real_warn_state_bytes = (eod_dir.parent / "health_state/warn_attention_state.json").read_bytes()

                # (W10) Legacy 4c stale contest reading
                s4c_dir = troot / "w_4c" / "picks"
                s4c_dir.mkdir(parents=True)
                scheduler._warn_stale_contest_reading(
                    {"stale": True, "reason": "old", "recorded_at": "2027-03-23T12:00:00-04:00", "age_hours": 54.0},
                    make_cfg(s4c_dir),
                    s4c_dir,
                )
                real_4c_bytes = (s4c_dir.parent / "health_state/contest_stale_reading_dm_delivery_status.json").read_bytes()

                # (W11) Legacy 6b-1 morning state check (refused)
                m6b_dir = troot / "w_6b1" / "picks"
                m6b_dir.mkdir(parents=True)
                with patch.object(
                    contest_state,
                    "load_decision_streak_state",
                    side_effect=contest_state.ContestStateError("no automatic state"),
                ):
                    scheduler._morning_state_check(
                        make_cfg(m6b_dir, contest_req=True),
                        m6b_dir,
                        DAY,
                        NOW.replace(hour=9, minute=0),
                        NOW.replace(hour=14, minute=0),
                    )
                real_6b1_bytes = (m6b_dir.parent / "health_state/contest_morning_check_dm_delivery_status.json").read_bytes()

                # (W12) memory_growth_history.jsonl (single-line JSON object)
                mem_path = troot / "w_mem" / "memory_growth_history.jsonl"
                memory_growth._append_history(mem_path, NOW.date(), 142.3)
                real_mem_bytes = mem_path.read_bytes()

                # (W13) restart_spike .nrestarts_checkpoint
                rs_dir = troot / "w_rs"
                rs_dir.mkdir(parents=True)
                restart_spike.check(rs_dir, current_nrestarts=1, today=NOW.date())
                real_rs_bytes = (rs_dir / ".nrestarts_checkpoint").read_bytes()

                # (W14) EntryReceipt.publish
                er_dir = troot / "w_er" / "picks"
                er_dir.mkdir(parents=True)
                er = entry_receipt.EntryReceipt(
                    et_date=DAY, started_at=NOW, clock=lambda: NOW, expected_username="offline", picks_dir=er_dir
                )
                er.set_outcome("outside_window")
                er_pub = er.publish()
                assert er_pub is not None
                real_entry_receipt_bytes = er_pub.read_bytes()

                # (W15) ReconcileReceipt.publish
                rr_dir = troot / "w_rr" / "picks"
                rr_dir.mkdir(parents=True)
                rr = reconcile_receipt.ReconcileReceipt(clock=lambda: NOW, lookback_days=7)
                rr.finish([])
                rr_pub = rr.publish(rr_dir)
                assert rr_pub is not None
                real_reconcile_receipt_bytes = rr_pub.read_bytes()

                # (W16) cli._contest_fetch_alert -> contest_streak_fetch_status.json
                cfs_path = troot / "w_cfs" / "contest_streak_fetch_status.json"
                cfs_path.parent.mkdir(parents=True)
                cli._contest_fetch_alert(cfs_path, "offline.test", "upstream 503", category="transient")
                real_cfs_bytes = cfs_path.read_bytes()

                # (W17) backup._write_status_entry -> data/health_state/backup_status.json
                bk_repo = troot / "w_bk"
                backup._write_status_entry(bk_repo, "ops", {"ok": True, "snapshot_id": "abc12345"})
                real_backup_bytes = (bk_repo / "data/health_state/backup_status.json").read_bytes()

                # (W18) scripts/check_heartbeat.py save_churn_samples -> scheduler_churn.json
                spec = importlib.util.spec_from_file_location("check_heartbeat", ROOT / "scripts/check_heartbeat.py")
                assert spec and spec.loader
                hb_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(hb_mod)
                churn_path = troot / "w_churn" / "scheduler_churn.json"
                hb_mod.save_churn_samples(churn_path, "bts-scheduler", [{"at": NOW.isoformat(), "nrestarts": 0}])
                real_churn_bytes = churn_path.read_bytes()

                # (W19) daily_decision.write_decision -> decision.json & delivered pick file
                dec_dir = troot / "w_dec" / "picks"
                dec_dir.mkdir(parents=True)
                write_decision(
                    DAY, dec_dir, action="single", source="mdp", delivery_status="delivered",
                    primary={"batter_id": 680757, "batter_name": "Steven Kwan"}, scoreable=True,
                )
                real_decision_bytes = (dec_dir / DAY / "decision.json").read_bytes()
                dp_deliv = picks.DailyPick(
                    DAY, NOW.isoformat(), p1, None, None,
                    delivery_attempted=True, delivered_at=NOW.isoformat(),
                    notification_sent=True, notification_channel="bluesky_dm", notification_id="dm-1",
                )
                picks.save_pick(dp_deliv, dec_dir)
                real_pick_bytes = (dec_dir / f"{DAY}.json").read_bytes()

            # 2. Verify all SUPPRESSING byte patterns (must send 0 DMs and preserve seeded bytes)
            escaped_json = (
                "{"
                + ",".join([
                    f"{unicode_escape_json_str('schema_version')}:{unicode_escape_json_str(alert.HEALTH_DM_STATUS_SCHEMA_VERSION)}",
                    f"{unicode_escape_json_str('status')}:{unicode_escape_json_str('sent')}",
                    f"{unicode_escape_json_str('sent_day')}:{unicode_escape_json_str(DAY)}",
                    f"{unicode_escape_json_str('sent_sources')}:[{unicode_escape_json_str(KEY)}]",
                    f"{unicode_escape_json_str('message_id')}:{unicode_escape_json_str('seed-ack')}",
                    f"{unicode_escape_json_str('body_sha256')}:{unicode_escape_json_str(body_hash)}",
                ])
                + "}"
            ).encode("utf-8")

            suppressing_cases: dict[str, bytes] = {
                "own_writer_exact_bytes": real_own_same_bytes,
                "compact_reordered_keys": json.dumps(
                    dict(reversed(list(canonical_dict.items()))), separators=(",", ":")
                ).encode("utf-8"),
                "rfc8259_outer_inner_whitespace": b" \t\r\n" + canonical_bytes + b"\r\n\t ",
                "unicode_escaped_keys_and_values": escaped_json,
                "padded_message_id": json.dumps(canonical_dict | {"message_id": "  ack-with-spaces  "}).encode("utf-8"),
                "zwsp_message_id_non_isspace": json.dumps(canonical_dict | {"message_id": "\u200b"}).encode("utf-8"),
                "surrogate_escape_in_message_id": (
                    json.dumps(canonical_dict | {"message_id": "PLACEHOLDER"})
                    .replace('"PLACEHOLDER"', r'"\ud800"')
                    .encode("utf-8")
                ),
            }

            suppress_results = {}
            for idx, (label, raw_bytes) in enumerate(suppressing_cases.items()):
                for route in ("direct", "classification"):
                    sends, after_bytes, stderr_text = run_encounter_pair(troot / f"sup_{idx}_{route}", route, raw_bytes)
                    assert sends == 0, f"Expected suppression for {label} ({route}), got {sends} sends"
                    assert after_bytes == raw_bytes, f"Suppressing receipt mutated on {label} ({route})"
                    assert "Unknown-outcome alert failed:" not in stderr_text
                    suppress_results[f"{label}:{route}"] = sends

            # 3. Verify all NON-SUPPRESSING hostile encodings, numbers, normalizations, duplicates, and foreign writers
            nfkc_sent = "\uff53\uff45\uff4e\uff54"  # fullwidth "sent", NFKC == "sent"
            assert unicodedata.normalize("NFKC", nfkc_sent) == "sent" and nfkc_sent != "sent"
            nfkc_day = "2027-\uff10\uff13-25"
            assert unicodedata.normalize("NFKC", nfkc_day) == DAY and nfkc_day != DAY

            non_suppressing_cases: dict[str, bytes] = {
                # Encoding / BOM / whitespace
                "utf8_bom_prefix": b"\xef\xbb\xbf" + canonical_bytes,
                "utf16_le_encoded": canonical_bytes.decode("utf-8").encode("utf-16-le"),
                "utf16_be_bom": canonical_bytes.decode("utf-8").encode("utf-16"),
                "invalid_utf8_bytes": b"\xff\xfe" + canonical_bytes,
                "nbsp_outer_whitespace": "\u00a0".encode("utf-8") + canonical_bytes,
                "formfeed_outer_whitespace": b"\x0c" + canonical_bytes,
                "nul_prefix": b"\x00" + canonical_bytes,
                "trailing_garbage": canonical_bytes + b"extra",
                # Duplicate members (top-level, identical, unicode-escaped alias, nested)
                "dup_failed_then_sent": b'{"status":"failed",' + json.dumps(canonical_dict).encode("utf-8")[1:],
                "dup_identical_status_sent": b'{"status":"sent",' + json.dumps(canonical_dict).encode("utf-8")[1:],
                "dup_unicode_escaped_status": (
                    b'{"\\u0073tatus":"failed",' + json.dumps(canonical_dict).encode("utf-8")[1:]
                ),
                "dup_nested_in_sent_sources": (
                    b'{"body_sha256":"' + body_hash.encode() + b'","message_id":"id","schema_version":"'
                    + alert.HEALTH_DM_STATUS_SCHEMA_VERSION.encode()
                    + b'","sent_day":"' + DAY.encode() + b'","sent_sources":[{"a":1,"a":2}],"status":"sent"}'
                ),
                "dup_nested_in_extra_object": (
                    b'{"nested":{"a":1,"a":2},' + json.dumps(canonical_dict).encode("utf-8")[1:]
                ),
                # Key-set violations
                "five_member_r2_receipt": json.dumps(
                    {k: v for k, v in canonical_dict.items() if k != "body_sha256"}
                ).encode("utf-8"),
                "extra_error_member": json.dumps(
                    canonical_dict | {"error": "send_dm returned no usable message id"}
                ).encode("utf-8"),
                "extra_null_member": json.dumps(canonical_dict | {"extra": None}).encode("utf-8"),
                # NaN, Infinity, huge float, 4000-digit int, 5000-digit int (>4300 str limit), deep recursion
                "nan_in_message_id": json.dumps(canonical_dict | {"message_id": float("nan")}, allow_nan=True).encode("utf-8"),
                "nan_in_body_sha256": json.dumps(canonical_dict | {"body_sha256": float("nan")}, allow_nan=True).encode("utf-8"),
                "inf_in_message_id": json.dumps(canonical_dict | {"message_id": float("inf")}, allow_nan=True).encode("utf-8"),
                "float_overflow_1e500": (
                    json.dumps(canonical_dict | {"message_id": "NUM"}).replace('"NUM"', "1e500").encode("utf-8")
                ),
                "int_4000_digits": (
                    json.dumps(canonical_dict | {"message_id": "NUM"}).replace('"NUM"', "9" * 4000).encode("utf-8")
                ),
                "int_5000_digits_over_limit": (
                    json.dumps(canonical_dict | {"message_id": "NUM"}).replace('"NUM"', "9" * 5000).encode("utf-8")
                ),
                "deep_nesting_1500": (b"[" * 1500 + b"]" * 1500),
                # Normalization / case / whitespace in string values
                "nfkc_fullwidth_status": json.dumps(canonical_dict | {"status": nfkc_sent}, ensure_ascii=False).encode("utf-8"),
                "nfkc_fullwidth_sent_day": json.dumps(canonical_dict | {"sent_day": nfkc_day}, ensure_ascii=False).encode("utf-8"),
                "nfd_combining_in_schema": json.dumps(
                    canonical_dict | {"schema_version": alert.HEALTH_DM_STATUS_SCHEMA_VERSION + "\u0301"}
                ).encode("utf-8"),
                "uppercase_body_sha256": json.dumps(canonical_dict | {"body_sha256": body_hash.upper()}).encode("utf-8"),
                "whitespace_padded_body_sha256": json.dumps(canonical_dict | {"body_sha256": f" {body_hash} "}).encode("utf-8"),
                "bool_true_schema_with_hash": json.dumps(canonical_dict | {"schema_version": True}).encode("utf-8"),
                "unicode_isspace_message_id_with_hash": json.dumps(
                    canonical_dict | {"message_id": "\u00a0\u2000\u2003\u2028\u2029\u3000"}
                ).encode("utf-8"),
                # Real outputs from every writer in the codebase
                "writer_own_other_slate": real_own_prev_bytes,
                "writer_own_failed": real_own_fail_bytes,
                "writer_own_skipped_no_recipient": real_own_skip_bytes,
                "writer_foreign_optin_same_key_and_day": real_foreign_optin_bytes,
                "writer_legacy_cutoff_shared": real_legacy_cutoff_bytes,
                "writer_legacy_cutoff_plus_message_id": json.dumps(
                    json.loads(real_legacy_cutoff_bytes) | {"message_id": "cutoff-ack"}
                ).encode("utf-8"),
                "writer_legacy_void_shared": real_legacy_void_bytes,
                "writer_legacy_contest_state_failure": real_legacy_csf_bytes,
                "writer_legacy_missed_pick": real_legacy_missed_bytes,
                "writer_legacy_eod_shared": real_eod_shared_bytes,
                "writer_legacy_4c_stale": real_4c_bytes,
                "writer_legacy_6b1_morning": real_6b1_bytes,
                "writer_warn_attention_state": real_warn_state_bytes,
                "writer_memory_growth_jsonl": real_mem_bytes,
                "writer_restart_spike_checkpoint": real_rs_bytes,
                "writer_entry_receipt": real_entry_receipt_bytes,
                "writer_reconcile_receipt": real_reconcile_receipt_bytes,
                "writer_contest_fetch_status": real_cfs_bytes,
                "writer_backup_status": real_backup_bytes,
                "writer_scheduler_churn": real_churn_bytes,
                "writer_decision_json": real_decision_bytes,
                "writer_delivered_pick_json": real_pick_bytes,
                "writer_pick_entry_check_marker": json.dumps(
                    {"date": DAY, "status": "confirmed", "reason": "match", "checked_at": NOW.isoformat(), "receipt": "r1"}
                ).encode("utf-8"),
                "writer_slate_auc_status": json.dumps(
                    {"computed_at": DAY, "window_days": 30, "n_rows": 100, "auc": 0.61, "reason": None}
                ).encode("utf-8"),
            }

            expected_replaced = canonical_dict | {"message_id": "fresh-ack-id"}
            non_suppress_results = {}
            for idx, (label, raw_bytes) in enumerate(non_suppressing_cases.items()):
                for route in ("direct", "classification"):
                    sends, after_bytes, stderr_text = run_encounter_pair(troot / f"nonsup_{idx}_{route}", route, raw_bytes)
                    assert sends == 1, f"FALSE SUPPRESSION on {label} ({route}): got {sends} sends!"
                    assert json.loads(after_bytes) == expected_replaced, (label, route, json.loads(after_bytes))
                    assert "Unknown-outcome alert failed:" not in stderr_text, (label, route, stderr_text)
                    non_suppress_results[f"{label}:{route}"] = sends

        report = {
            "suppressing_patterns_verified": len(suppressing_cases),
            "suppressing_route_runs": len(suppress_results),
            "non_suppressing_patterns_verified": len(non_suppressing_cases),
            "non_suppressing_route_runs": len(non_suppress_results),
            "false_suppressions": 0,
            "canonical_receipt_bytes": canonical_bytes.decode("utf-8"),
        }
        (REVIEW / "probe_1_results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"PROBE 1 OK: {len(suppressing_cases)} suppressing patterns x 2 routes ({len(suppress_results)} runs -> 0 sends); "
            f"{len(non_suppressing_cases)} hostile/foreign-writer patterns x 2 routes ({len(non_suppress_results)} runs -> 1 send then dedup); "
            "0 false suppressions."
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
