"""Item 1 adversarial probe: B1 and every way an own receipt could wrongly suppress C2, plus reverse dedup."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext, redirect_stderr
from datetime import datetime, timedelta
from io import StringIO
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

ROOT = Path("/Users/eric/projects/bts-work/wr-c2-r2-carbon")
REVIEW_DIR = ROOT / ".codex-review" / "wr-c2-r2"
R1_SRC = REVIEW_DIR / "r1_dfe774df" / "src"
HEAD_SRC = ROOT / "src"
PYTHON = ROOT / ".venv" / "bin" / "python"
ET = ZoneInfo("America/New_York")
DAY = "2027-03-25"
PREV_DAY = "2027-03-24"
NEXT_DAY = "2027-03-26"
PINNED_NOW = datetime(2027, 3, 25, 18, 35, 0, tzinfo=ET)


def expected_u7_copy(day: str = DAY) -> str:
    return (
        "BTS health CRITICAL alert(s):\n"
        f"- [late_delivery] DELIVERY OUTCOME UNKNOWN for {day} ET. A send attempt for that slate "
        "was never confirmed, and the scheduler locked the slate without resending. "
        f"Check the DM or public feed and the MLB entry for {day} ET. Do not enter or change a "
        f"pick based on this alert. If {day} is still today in ET and its submission window is "
        "open, run bts season-contract status with --picks-dir set to the scheduler's picks "
        "directory before making any entry decision for that slate."
    )


def _worker(mode: str) -> dict:
    from bts import contest_state, daily_decision, dm, picks, posting, scheduler, strategy
    from bts.daily_decision import write_decision
    from bts.health import alert, attention, runner

    now_box = [PINNED_NOW]

    class PinnedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now_box[0].astimezone(tz) if tz else now_box[0].replace(tzinfo=None)

    def make_games(day_str: str) -> list[dict]:
        return [{
            "gamePk": 101,
            "gameDate": f"{day_str}T23:10:00Z",
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
            "teams": {"away": {"team": {"name": "Away"}}, "home": {"team": {"name": "Home"}}},
        }]

    def enter_world(stack: ExitStack, sent_mock: Mock, err_buf: StringIO):
        for p in (
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
            patch.object(scheduler, "_contest_state_required", lambda _: False),
            patch.object(contest_state, "load_decision_streak_state", return_value=Mock(streak=0)),
            patch.object(dm, "send_dm", Mock(side_effect=AssertionError("unexpected pick resend"))),
            patch.object(alert, "send_dm", sent_mock),
            patch.object(posting, "post_to_bluesky", Mock(side_effect=AssertionError("unexpected post"))),
            patch.object(strategy, "_mdp_cache", None),
            redirect_stderr(err_buf),
        ):
            stack.enter_context(p)

    def make_attempted_pick(pdir: Path, day_str: str = DAY):
        p1 = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Cole", 543037, 0.82, [], False, 101, f"{day_str}T23:10:00Z")
        dp = picks.DailyPick(day_str, now_box[0].isoformat(), p1, None, None, delivery_attempted=True)
        picks.save_pick(dp, pdir)
        return dp

    def make_cfg(pdir: Path, recipient: str | None = "offline.test") -> dict:
        bsky = {"dm_recipient": recipient} if recipient is not None else {}
        return {
            "orchestrator": {"picks_dir": str(pdir), "heartbeat_path": str(pdir.parent / "hb")},
            "bluesky": bsky,
            "scheduler": {"pick_delivery": "dm", "live_forward_capture_on_lock": False, "contest_state_required": False},
            "health_checks": {"enabled": False},
            "tiers": [],
        }

    valid_sent = {
        "schema_version": alert.HEALTH_DM_STATUS_SCHEMA_VERSION,
        "status": "sent",
        "sent_day": DAY,
        "sent_sources": ["CRITICAL:late_delivery"],
        "message_id": "prior-ack",
    }

    def raw_json(d: dict) -> bytes:
        return (json.dumps(d, allow_nan=True) + "\n").encode("utf-8")

    # Build exhaustive candidate matrix
    schema_cases = {
        "schema_missing": raw_json({k: v for k, v in valid_sent.items() if k != "schema_version"}),
        "schema_null": raw_json(valid_sent | {"schema_version": None}),
        "schema_empty": raw_json(valid_sent | {"schema_version": ""}),
        "schema_v0": raw_json(valid_sent | {"schema_version": "bts_health_dm_delivery_status_v0"}),
        "schema_v2": raw_json(valid_sent | {"schema_version": "bts_health_dm_delivery_status_v2"}),
        "schema_trailing_space": raw_json(valid_sent | {"schema_version": "bts_health_dm_delivery_status_v1 "}),
        "schema_uppercase": raw_json(valid_sent | {"schema_version": "BTS_HEALTH_DM_DELIVERY_STATUS_V1"}),
        "schema_list_damaged": raw_json(valid_sent | {"schema_version": ["damaged"]}),
        "schema_list_valid_str": raw_json(valid_sent | {"schema_version": [alert.HEALTH_DM_STATUS_SCHEMA_VERSION]}),
        "schema_dict": raw_json(valid_sent | {"schema_version": {"v": alert.HEALTH_DM_STATUS_SCHEMA_VERSION}}),
        "schema_int_0": raw_json(valid_sent | {"schema_version": 0}),
        "schema_int_1": raw_json(valid_sent | {"schema_version": 1}),
        "schema_bool_true": raw_json(valid_sent | {"schema_version": True}),
        "schema_bool_false": raw_json(valid_sent | {"schema_version": False}),
        "schema_nan": raw_json(valid_sent | {"schema_version": float("nan")}),
        "schema_inf": raw_json(valid_sent | {"schema_version": float("inf")}),
    }

    if mode == "r1_schema_only":
        out_r1 = {}
        with TemporaryDirectory() as tmp:
            troot = Path(tmp)
            for cname, payload_bytes in schema_cases.items():
                for route in ("direct", "classification"):
                    cdir = troot / f"{cname}_{route}"
                    pdir = cdir / "picks"
                    pdir.mkdir(parents=True)
                    dp = make_attempted_pick(pdir)
                    own = cdir / "health_state" / "delivery_outcome_unknown" / f"{DAY}.json"
                    own.parent.mkdir(parents=True)
                    own.write_bytes(payload_bytes)
                    sent = Mock(return_value="new-ack")
                    with ExitStack() as stack:
                        enter_world(stack, sent, StringIO())
                        cfg = make_cfg(pdir)
                        if route == "direct":
                            st = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
                            scheduler._deliver_and_lock_pick(dp, cfg, pdir, st, DAY, "r1")
                        else:
                            scheduler.run_day(DAY, cfg)
                    out_r1[f"{cname}:{route}"] = sent.call_count
        return {"r1_schema_sends": out_r1}

    other_field_cases = {
        # status wrong type/value
        "status_missing": raw_json({k: v for k, v in valid_sent.items() if k != "status"}),
        "status_null": raw_json(valid_sent | {"status": None}),
        "status_empty": raw_json(valid_sent | {"status": ""}),
        "status_failed": raw_json(valid_sent | {"status": "failed"}),
        "status_skipped": raw_json(valid_sent | {"status": "skipped_no_recipient"}),
        "status_upper": raw_json(valid_sent | {"status": "SENT"}),
        "status_padded": raw_json(valid_sent | {"status": "sent "}),
        "status_list": raw_json(valid_sent | {"status": ["sent"]}),
        "status_dict": raw_json(valid_sent | {"status": {"status": "sent"}}),
        "status_int": raw_json(valid_sent | {"status": 1}),
        "status_bool": raw_json(valid_sent | {"status": True}),
        "status_nan": raw_json(valid_sent | {"status": float("nan")}),
        # sent_day wrong type/value
        "day_missing": raw_json({k: v for k, v in valid_sent.items() if k != "sent_day"}),
        "day_null": raw_json(valid_sent | {"sent_day": None}),
        "day_empty": raw_json(valid_sent | {"sent_day": ""}),
        "day_prev": raw_json(valid_sent | {"sent_day": PREV_DAY}),
        "day_next": raw_json(valid_sent | {"sent_day": NEXT_DAY}),
        "day_unpadded": raw_json(valid_sent | {"sent_day": "2027-3-25"}),
        "day_compact": raw_json(valid_sent | {"sent_day": "20270325"}),
        "day_padded_ws": raw_json(valid_sent | {"sent_day": f"{DAY} "}),
        "day_datetime": raw_json(valid_sent | {"sent_day": f"{DAY}T00:00:00"}),
        "day_list": raw_json(valid_sent | {"sent_day": [DAY]}),
        "day_int": raw_json(valid_sent | {"sent_day": 20270325}),
        "day_bool": raw_json(valid_sent | {"sent_day": True}),
        # sent_sources wrong type/value/superset/duplicate
        "src_missing": raw_json({k: v for k, v in valid_sent.items() if k != "sent_sources"}),
        "src_null": raw_json(valid_sent | {"sent_sources": None}),
        "src_empty_list": raw_json(valid_sent | {"sent_sources": []}),
        "src_str": raw_json(valid_sent | {"sent_sources": "CRITICAL:late_delivery"}),
        "src_bare_name": raw_json(valid_sent | {"sent_sources": ["late_delivery"]}),
        "src_warn_level": raw_json(valid_sent | {"sent_sources": ["WARN_ATTENTION:late_delivery"]}),
        "src_other_source": raw_json(valid_sent | {"sent_sources": ["CRITICAL:pick_delivery"]}),
        "src_superset": raw_json(valid_sent | {"sent_sources": ["CRITICAL:late_delivery", "CRITICAL:pick_delivery"]}),
        "src_duplicate_elem": raw_json(valid_sent | {"sent_sources": ["CRITICAL:late_delivery", "CRITICAL:late_delivery"]}),
        "src_mixed_int": raw_json(valid_sent | {"sent_sources": ["CRITICAL:late_delivery", 1]}),
        "src_dict": raw_json(valid_sent | {"sent_sources": {"CRITICAL:late_delivery": True}}),
        "src_bool": raw_json(valid_sent | {"sent_sources": True}),
        # message_id wrong type/value/blank
        "id_missing": raw_json({k: v for k, v in valid_sent.items() if k != "message_id"}),
        "id_null": raw_json(valid_sent | {"message_id": None}),
        "id_empty": raw_json(valid_sent | {"message_id": ""}),
        "id_spaces": raw_json(valid_sent | {"message_id": "   "}),
        "id_tabs_newlines": raw_json(valid_sent | {"message_id": "\t\r\n"}),
        "id_nbsp": raw_json(valid_sent | {"message_id": "\u00a0"}),
        "id_unicode_spaces": raw_json(valid_sent | {"message_id": "\u2000\u3000"}),
        "id_zero": raw_json(valid_sent | {"message_id": 0}),
        "id_int": raw_json(valid_sent | {"message_id": 17}),
        "id_float": raw_json(valid_sent | {"message_id": 1.5}),
        "id_bool_true": raw_json(valid_sent | {"message_id": True}),
        "id_bool_false": raw_json(valid_sent | {"message_id": False}),
        "id_list": raw_json(valid_sent | {"message_id": ["prior-ack"]}),
        "id_dict": raw_json(valid_sent | {"message_id": {"id": "prior-ack"}}),
        "id_nan": raw_json(valid_sent | {"message_id": float("nan")}),
    }

    with TemporaryDirectory() as tmp:
        troot = Path(tmp)

        # Generate real foreign/other-path receipts using real writers!
        gen_dir = troot / "generators"
        gen_picks = gen_dir / "picks"
        gen_picks.mkdir(parents=True)
        with ExitStack() as stack:
            gen_sent = Mock(return_value="gen-ack")
            enter_world(stack, gen_sent, StringIO())
            gen_cfg = make_cfg(gen_picks)

            # 1. Real C2 receipt for PREV_DAY (another slate copied in)
            scheduler._alert_delivery_outcome_unknown(gen_cfg, gen_picks, PREV_DAY)
            copied_other_slate = (gen_dir / "health_state/delivery_outcome_unknown" / f"{PREV_DAY}.json").read_bytes()

            # 2. Real same-day cutoff refusal in health_dm_delivery_status.json
            p_cut = picks.Pick("Kwan", 680757, "CLE", 1, "Cole", 543037, 0.82, [], False, 101, f"{DAY}T22:30:00Z")
            dp_cut = picks.DailyPick(DAY, PINNED_NOW.isoformat(), p_cut, None, None)
            picks.save_pick(dp_cut, gen_picks)
            st_cut = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
            scheduler._refuse_late_delivery(dp_cut, gen_cfg, gen_picks, st_cut, DAY, "lineup",
                                            PINNED_NOW - timedelta(minutes=5), PINNED_NOW)
            cutoff_shared_bytes = (gen_dir / "health_state/health_dm_delivery_status.json").read_bytes()

            # 3. Real 4c stale reading status
            scheduler._warn_stale_contest_reading(
                {"stale": True, "reason": "old", "recorded_at": "2027-03-23T12:00:00-04:00", "age_hours": 54.0},
                gen_cfg, gen_picks,
            )
            stale_4c_bytes = (gen_dir / "health_state/contest_stale_reading_dm_delivery_status.json").read_bytes()

            # 4. Real EOD run_all_checks shared status
            eod_sub = troot / "gen_eod" / "picks"
            eod_sub.mkdir(parents=True)
            make_attempted_pick(eod_sub)
            with patch.object(runner, "_safe_run", lambda n, fn: fn() if n == "post_failure" else []), \
                    patch.object(runner.realized_calibration, "_current_deploy_iso", lambda _: None):
                runner.run_all_checks(eod_sub, troot / "models", "offline.test", today=PINNED_NOW.date())
            eod_shared_bytes = (eod_sub.parent / "health_state/health_dm_delivery_status.json").read_bytes()

            # 5. Real C2 failed receipt
            fail_sub = troot / "gen_fail" / "picks"
            fail_sub.mkdir(parents=True)
            gen_sent.return_value = None
            scheduler._alert_delivery_outcome_unknown(make_cfg(fail_sub), fail_sub, DAY)
            c2_failed_bytes = (fail_sub.parent / "health_state/delivery_outcome_unknown" / f"{DAY}.json").read_bytes()
            gen_sent.return_value = "gen-ack"

            # 6. Real C2 skipped_no_recipient receipt
            skip_sub = troot / "gen_skip" / "picks"
            skip_sub.mkdir(parents=True)
            scheduler._alert_delivery_outcome_unknown(make_cfg(skip_sub, recipient=None), skip_sub, DAY)
            c2_skipped_bytes = (skip_sub.parent / "health_state/delivery_outcome_unknown" / f"{DAY}.json").read_bytes()

            # 7. Real decision.json
            dec_sub = troot / "gen_dec" / "picks"
            dec_sub.mkdir(parents=True)
            write_decision(DAY, dec_sub, action="single", source="mdp", delivery_status="delivered",
                           primary={"batter_id": 680757, "batter_name": "Kwan"}, scoreable=True)
            decision_bytes = (dec_sub / DAY / "decision.json").read_bytes()

            # 8. Real delivered pick file with notification_id
            dp_deliv = picks.DailyPick(
                DAY, PINNED_NOW.isoformat(), p_cut, None, None,
                delivery_attempted=True, delivered_at=PINNED_NOW.isoformat(),
                notification_sent=True, notification_channel="bluesky_dm", notification_id="msg-real",
            )
            deliv_sub = troot / "gen_deliv" / "picks"
            picks.save_pick(dp_deliv, deliv_sub)
            pick_file_bytes = (deliv_sub / f"{DAY}.json").read_bytes()

            # 9. Real warn_attention_state.json
            warn_path = troot / "gen_warn" / "warn_attention_state.json"
            attention.update_warn_streaks([alert.Alert("WARN", "saver_state", "msg")],
                                          state_path=warn_path, today=PINNED_NOW.date())
            warn_state_bytes = warn_path.read_bytes()

        foreign_cases = {
            "copied_prev_slate_c2": copied_other_slate,
            "foreign_same_day_cutoff_shared": cutoff_shared_bytes,
            "foreign_4c_stale_status": stale_4c_bytes,
            "foreign_eod_shared": eod_shared_bytes,
            "own_c2_failed": c2_failed_bytes,
            "own_c2_skipped_no_recipient": c2_skipped_bytes,
            "tampered_failed_with_sent_fields": raw_json(valid_sent | {"status": "failed", "error": "offline"}),
            "foreign_decision_json": decision_bytes,
            "foreign_delivered_pick_json": pick_file_bytes,
            "foreign_warn_attention_json": warn_state_bytes,
            "foreign_entry_receipt_shape": raw_json({
                "schema_version": "bts_pick_entry_receipt_v1", "status": "sent",
                "sent_day": DAY, "sent_sources": ["CRITICAL:late_delivery"],
                "date": DAY, "outcome": "entered",
            }),
            "foreign_reconcile_receipt_shape": raw_json({
                "schema_version": "bts_reconcile_receipt_v1", "status": "sent",
                "sent_day": DAY, "outcome": "complete",
            }),
        }

        all_must_send = {}
        all_must_send.update(schema_cases)
        all_must_send.update(other_field_cases)
        all_must_send.update(foreign_cases)

        checked = 0
        for cname, payload_bytes in all_must_send.items():
            for route in ("direct", "classification"):
                cdir = troot / f"head_{cname}_{route}"
                pdir = cdir / "picks"
                pdir.mkdir(parents=True)
                dp = make_attempted_pick(pdir)
                own = cdir / "health_state" / "delivery_outcome_unknown" / f"{DAY}.json"
                own.parent.mkdir(parents=True)
                own.write_bytes(payload_bytes)
                sent = Mock(return_value="fresh-ack")
                err = StringIO()
                with ExitStack() as stack:
                    enter_world(stack, sent, err)
                    cfg = make_cfg(pdir)
                    if route == "direct":
                        st = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
                        assert scheduler._deliver_and_lock_pick(dp, cfg, pdir, st, DAY, cname) is False
                    else:
                        assert scheduler.run_day(DAY, cfg) is None
                    assert sent.call_count == 1, f"SUPPRESSED FALSELY on {cname} ({route})"
                    assert sent.call_args.args == ("offline.test", expected_u7_copy(DAY))
                    replaced = json.loads(own.read_bytes())
                    assert replaced == valid_sent | {"message_id": "fresh-ack"}, (cname, replaced)
                    assert err.getvalue().count("Unknown-outcome alert failed:") == 0, (cname, err.getvalue())
                    # Immediately verify reverse dedup on second encounter!
                    assert scheduler.run_day(DAY, cfg) is None
                    assert sent.call_count == 1, f"Failed to dedup after replacing {cname} ({route})"
                checked += 1

        # Multi-source order sensitivity check in dispatch_dm_for_health_alerts(require_message_id=True)
        order_dir = troot / "source_order"
        order_dir.mkdir()
        order_receipt = order_dir / f"{DAY}.json"
        a_b = [alert.Alert("CRITICAL", "b_src", "m_b"), alert.Alert("CRITICAL", "a_src", "m_a")]
        b_a = [alert.Alert("CRITICAL", "a_src", "m_a"), alert.Alert("CRITICAL", "b_src", "m_b")]
        # Seed unsorted ["CRITICAL:b_src", "CRITICAL:a_src"] -> must NOT suppress!
        order_receipt.write_bytes(raw_json({
            "schema_version": alert.HEALTH_DM_STATUS_SCHEMA_VERSION,
            "status": "sent",
            "sent_day": DAY,
            "sent_sources": ["CRITICAL:b_src", "CRITICAL:a_src"],
            "message_id": "unsorted-ack",
        }))
        sent_ord = Mock(return_value="sorted-ack")
        with patch.object(alert, "send_dm", sent_ord), redirect_stderr(StringIO()):
            assert alert.dispatch_dm_for_health_alerts(
                a_b, "offline.test", status_path=order_receipt, now_et_date=PINNED_NOW.date(), require_message_id=True
            ) is True
            assert sent_ord.call_count == 1
            saved_ord = json.loads(order_receipt.read_bytes())
            assert saved_ord["sent_sources"] == ["CRITICAL:a_src", "CRITICAL:b_src"]
            # Reverse order of Alert inputs still deduplicates against canonical sorted receipt!
            assert alert.dispatch_dm_for_health_alerts(
                b_a, "offline.test", status_path=order_receipt, now_et_date=PINNED_NOW.date(), require_message_id=True
            ) is False
            assert sent_ord.call_count == 1

        # Extra keys alongside all 5 valid sent fields: record exact behavior
        extra_observations = {}
        for ename, extra_dict in (
            ("extra_benign_key", valid_sent | {"extra_note": " benign "}),
            ("extra_error_key", valid_sent | {"error": "stale error"}),
        ):
            epath = troot / f"{ename}.json"
            epath.write_bytes(raw_json(extra_dict))
            sent_ex = Mock(return_value="ex-ack")
            with patch.object(alert, "send_dm", sent_ex), redirect_stderr(StringIO()):
                attempted = alert.dispatch_dm_for_health_alerts(
                    [alert.Alert("CRITICAL", "late_delivery", "m")],
                    "offline.test", status_path=epath, now_et_date=PINNED_NOW.date(), require_message_id=True,
                )
            extra_observations[ename] = {"attempted": attempted, "sends": sent_ex.call_count}

        # Reverse check: genuine acknowledged slate deduplicates across cycles, routes, midnight, and no-recipient
        dedup_dir = troot / "reverse_dedup" / "picks"
        dedup_dir.mkdir(parents=True)
        dp_dedup = make_attempted_pick(dedup_dir)
        sent_dedup = Mock(return_value="  exact-ack-id  ")
        with ExitStack() as stack:
            enter_world(stack, sent_dedup, StringIO())
            cfg_dedup = make_cfg(dedup_dir)
            # Cycle 1: classification sends
            scheduler.run_day(DAY, cfg_dedup)
            assert sent_dedup.call_count == 1
            own_dedup = dedup_dir.parent / "health_state/delivery_outcome_unknown" / f"{DAY}.json"
            snap_bytes = own_dedup.read_bytes()
            assert json.loads(snap_bytes)["message_id"] == "  exact-ack-id  "
            # Cycle 2: direct dedups
            st_d = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
            assert scheduler._deliver_and_lock_pick(dp_dedup, cfg_dedup, dedup_dir, st_d, DAY, "c2") is False
            # Cycle 3: post-midnight classification dedups
            now_box[0] = datetime(2027, 3, 26, 0, 5, 0, tzinfo=ET)
            scheduler.run_day(DAY, cfg_dedup)
            # Cycle 4: no-recipient encounter does NOT overwrite acknowledged receipt
            scheduler.run_day(DAY, make_cfg(dedup_dir, recipient=None))
            assert sent_dedup.call_count == 1
            assert own_dedup.read_bytes() == snap_bytes

    return {
        "total_must_send_cases_per_route": len(all_must_send),
        "total_route_executions_verified": checked,
        "unsorted_sent_sources_sends_and_rewrites": True,
        "swapped_alert_inputs_dedup_against_sorted_receipt": True,
        "extra_keys_with_all_5_valid_sent_fields": extra_observations,
        "reverse_dedup_across_4_cycles_preserved_bytes": True,
    }


def run_worker(src: Path, mode: str) -> dict:
    env = dict(os.environ, PYTHONPATH=str(src), PYTHONDONTWRITEBYTECODE="1", TZ="America/New_York")
    proc = subprocess.run(
        [str(PYTHON), __file__, "--worker", mode],
        env=env, capture_output=True, text=True, timeout=30, check=True,
    )
    return json.loads(proc.stdout)


def main() -> None:
    signal.alarm(60)
    try:
        r1 = run_worker(R1_SRC, "r1_schema_only")
        # Confirm B1 reproduced on dfe774df: every non-conforming schema_version falsely suppressed (0 sends)!
        assert all(v == 0 for v in r1["r1_schema_sends"].values()), r1
        head = run_worker(HEAD_SRC, "full_head")
        out_path = REVIEW_DIR / "item1_receipt_hunt.json"
        summary = {"r1_dfe774df_schema_sends": r1["r1_schema_sends"], "head_e897c766": head}
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(
            f"ITEM 1 PROBE OK: reproduced B1 on dfe774df ({len(r1['r1_schema_sends'])} schema cases sent 0 DMs); "
            f"on HEAD e897c766 all {head['total_must_send_cases_per_route']} hostile/foreign receipt shapes x 2 routes "
            f"({head['total_route_executions_verified']} runs) sent 1 DM, replaced receipt, and deduped on retry; "
            f"unsorted multi-source sent & canonicalized; reverse dedup held across 4 cycles."
        )
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        print(json.dumps(_worker(sys.argv[2])))
    else:
        main()
