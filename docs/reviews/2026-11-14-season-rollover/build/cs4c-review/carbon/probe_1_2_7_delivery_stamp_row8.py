"""Adversarial Probe 1/2/7: Delivery invariance, fault injection, stamp semantics, and Row 8.

Runs isolated worker subprocesses against archived base (1c6cbc77) and branch (9fc2ee6d)
with all DNS and outward socket connections forbidden.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_ROOT = REVIEW_DIR / "scratch" / "base_1c6cbc77"
ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime(2027, 7, 15, 18, 35, 0, tzinfo=ET)
SCHEMA = "bts_pick_automatic_reading_v1"
KEYS = ("schema", "recorded_at", "age_hours", "stale", "reason")


def _expected_stale_message(warning_day_iso: str, detail: str) -> str:
    return (
        f"The pick was delivered using an automatic contest reading flagged stale ({detail}). "
        "Check MLB. If the streak or saver differs, use bts set-contest-streak "
        f"--date {warning_day_iso} --streak N --best-streak B --saver S with the "
        "scheduler's picks directory and the values shown in MLB. "
        "Use the current ET date when you run it. Check the latest fetch status."
    )


def _seed_reading(picks_dir: Path, kind: str, *, now: datetime = NOW, streak: int = 12) -> str | None:
    acct = picks_dir / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    if kind in {"fresh_auto", "old_auto", "future_auto", "unreadable_auto", "uncovered_auto"}:
        if kind == "fresh_auto":
            rec = (now - timedelta(hours=3)).isoformat()
        elif kind == "old_auto":
            rec = (now - timedelta(hours=20)).isoformat()
        elif kind == "future_auto":
            rec = (now + timedelta(minutes=10)).isoformat()
        elif kind == "unreadable_auto":
            rec = "not-an-iso-timestamp"
        else:
            rec = (now.replace(year=now.year - 1)).isoformat()
        year = now.year if kind != "uncovered_auto" else now.year - 1
        (acct / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v2",
            "season": year,
            "activation_id": "a" * 32,
            "active_streak": streak,
            "best_streak": max(streak, 15),
            "source": "mlb_bts_profile",
            "source_date": now.date().replace(year=year).isoformat(),
            "recorded_at": rec,
        }))
        if kind != "uncovered_auto":
            (acct / "season_contract.json").write_text(json.dumps({
                "schema_version": "bts_season_contract_v1",
                "season": now.year,
                "status": "automatic",
                "activation_id": "a" * 32,
                "switched_at": now.isoformat(),
                "revoked_at": None,
                "note": None,
            }))
        return rec
    if kind == "dated":
        rec = (now - timedelta(hours=1)).isoformat()
        d_iso = now.date().isoformat()
        (acct / "contest_streak.manual.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_manual_v3",
            "season": now.year,
            "effective_date": d_iso,
            "active_streak": streak,
            "best_streak": max(streak, 15),
            "saver_state": "used",
            "recorded_at": rec,
            "expires_at": f"{d_iso}T23:59:59.999999-04:00",
            "source": "manual_cli",
        }))
        return rec
    if kind == "legacy_manual":
        rec = "legacy-literal-ts"
        (acct / "contest_streak.manual.json").write_text(json.dumps({
            "active_streak": streak,
            "source_date": now.date().isoformat(),
            "recorded_at": rec,
        }))
        return rec
    if kind == "corrupt_state":
        (acct / "contest_streak.json").write_text("{corrupt json")
        return None
    if kind == "missing_state":
        return None
    assert kind == "model", f"unknown kind {kind}"
    (picks_dir / "streak.json").write_text(json.dumps({
        "season": now.year, "streak": streak, "saver_available": True,
    }))
    return None


def _run_matrix_worker(tree: str) -> dict:
    sys.path.insert(0, str(REVIEW_DIR))
    from netguard import forbid_network

    with forbid_network(allow_loopback=False):
        from bts import contest_state, daily_decision, dm, picks, posting, scheduler, strategy
        from bts.health import alert

        has_auto_field = "automatic_reading" in picks.DailyPick.__dataclass_fields__

        def make_pick(*, double: bool = False) -> picks.DailyPick:
            p1 = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                            0.82, [], False, 101, f"{DAY}T23:10:00Z")
            p2 = (picks.Pick("Juan Soto", 665742, "NYM", 2, "Blake Snell", 605483,
                             0.81, [], False, 102, f"{DAY}T23:40:00Z") if double else None)
            return picks.DailyPick(DAY, NOW.isoformat(), p1, p2, None)

        def make_fresh_sel(daily_obj):
            pol = strategy.PolicyDecision(
                "double" if daily_obj.double_down else "single",
                "double" if daily_obj.double_down else "single",
                "heuristic", "reach57", 12, 80, None, "missing",
                None, None, None, None, None,
            )
            return strategy.SelectionResult(
                strategy.PickResult(daily_obj),
                "double" if daily_obj.double_down else "single",
                "heuristic",
                scheduler._row_from_daily(daily_obj.pick),
                scheduler._row_from_daily(daily_obj.double_down),
                None,
                streak=12,
                saver_available=False,
                state_source="contest",
                state_status="fresh",
                allow_double=True,
                contest_source_date=DAY,
                decision=pol,
            )

        def make_reuse_sel(daily_obj):
            return strategy.SelectionResult(
                strategy.PickResult(daily_obj),
                "single", "reuse",
                scheduler._row_from_daily(daily_obj.pick),
                None, None, streak=12, decision=None,
            )

        scenarios: list[dict] = []

        # 1. Confirmed deliveries across modes, readings, and selection shapes
        for mode in ("dm", "public"):
            for reading_kind in ("fresh_auto", "old_auto", "future_auto", "unreadable_auto", "dated"):
                for sel_kind in ("fresh_choice", "matching_saved_reuse", "matching_saved_none"):
                    scenarios.append({
                        "id": f"confirmed:{mode}:{reading_kind}:{sel_kind}",
                        "mode": mode, "required": True, "reading": reading_kind,
                        "sel_kind": sel_kind, "outcome": "ok",
                    })
            # Saved pick fresh at choice (12h) -> stale at send (20h) with selection=None and reuse
            for sel_kind in ("matching_saved_none", "matching_saved_reuse"):
                scenarios.append({
                    "id": f"aged_saved:{mode}:{sel_kind}",
                    "mode": mode, "required": True, "reading": "old_auto",
                    "sel_kind": sel_kind, "outcome": "aged_saved",
                })
            # Send that raises
            for reading_kind in ("old_auto", "dated"):
                scenarios.append({
                    "id": f"send_raises:{mode}:{reading_kind}",
                    "mode": mode, "required": True, "reading": reading_kind,
                    "sel_kind": "matching_saved_none", "outcome": "send_raises",
                })
            # Send that returns no id (None and "") + re-entry check
            for empty_id in (None, ""):
                for reading_kind in ("old_auto", "fresh_auto"):
                    scenarios.append({
                        "id": f"send_no_id:{mode}:{reading_kind}:{repr(empty_id)}",
                        "mode": mode, "required": True, "reading": reading_kind,
                        "sel_kind": "fresh_choice", "outcome": "send_no_id",
                        "empty_id": empty_id,
                    })
            # Final cutoff refusal
            scenarios.append({
                "id": f"final_cutoff:{mode}",
                "mode": mode, "required": True, "reading": "old_auto",
                "sel_kind": "fresh_choice", "outcome": "final_cutoff",
            })
            # Initial void guards: single void, double both void, double one void leg (delivers!)
            for void_case in ("single_void", "double_both_void", "double_one_void"):
                scenarios.append({
                    "id": f"void:{mode}:{void_case}",
                    "mode": mode, "required": True, "reading": "old_auto",
                    "sel_kind": "fresh_choice", "outcome": void_case,
                })
            # Initial late guard
            scenarios.append({
                "id": f"initial_late:{mode}",
                "mode": mode, "required": True, "reading": "old_auto",
                "sel_kind": "fresh_choice", "outcome": "initial_late",
            })
            # Already delivered & crash-attempt guard
            for guard in ("already_delivered", "crash_attempt"):
                scenarios.append({
                    "id": f"guard:{mode}:{guard}",
                    "mode": mode, "required": True, "reading": "old_auto",
                    "sel_kind": "matching_saved_none", "outcome": guard,
                })
            # Not required across all 4 state kinds
            for reading_kind in ("model", "legacy_manual", "old_auto", "dated"):
                scenarios.append({
                    "id": f"not_required:{mode}:{reading_kind}",
                    "mode": mode, "required": False, "reading": reading_kind,
                    "sel_kind": "matching_saved_none", "outcome": "ok",
                })
            # Failed state read
            for err_kind in ("missing_state", "corrupt_state", "uncovered_auto"):
                scenarios.append({
                    "id": f"state_err:{mode}:{err_kind}",
                    "mode": mode, "required": True, "reading": err_kind,
                    "sel_kind": "fresh_choice", "outcome": "state_err",
                })
            # 5a refusal
            for refusal_kind in ("no_chosen_on", "preview_chosen_on", "dated_mismatch"):
                scenarios.append({
                    "id": f"cs5a_refusal:{mode}:{refusal_kind}",
                    "mode": mode, "required": True,
                    "reading": "dated" if refusal_kind == "dated_mismatch" else "old_auto",
                    "sel_kind": "matching_saved_none", "outcome": refusal_kind,
                })
            # Missing DM recipient in DM and public modes (missing key, None, "")
            for recip_val in ("MISSING", None, ""):
                scenarios.append({
                    "id": f"no_recipient:{mode}:{repr(recip_val)}",
                    "mode": mode, "required": True, "reading": "old_auto",
                    "sel_kind": "fresh_choice", "outcome": "no_recipient",
                    "recip_val": recip_val,
                })

        # Private mode (required True and False, normal and initial_late)
        for req in (True, False):
            for reading_kind in ("old_auto", "dated"):
                scenarios.append({
                    "id": f"private:req={req}:{reading_kind}",
                    "mode": "private", "required": req, "reading": reading_kind,
                    "sel_kind": "fresh_choice", "outcome": "ok",
                })

        results = {}
        for sc in scenarios:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                picks_dir = root / "picks"
                picks_dir.mkdir(parents=True, exist_ok=True)
                now_box = [NOW]

                class PinnedClock(datetime):
                    @classmethod
                    def now(cls, tz=None):
                        return now_box[0].astimezone(tz) if tz else now_box[0].replace(tzinfo=None)

                rec_str = _seed_reading(picks_dir, sc["reading"], now=NOW)
                is_double = sc["outcome"] in {"double_both_void", "double_one_void"}
                daily = make_pick(double=is_double)

                # Seed chosen_on where appropriate
                if sc["sel_kind"] in {"matching_saved_reuse", "matching_saved_none"}:
                    if sc["outcome"] == "no_chosen_on":
                        daily.chosen_on = None
                    elif sc["outcome"] == "preview_chosen_on":
                        st = contest_state.load_decision_streak_state(
                            picks_dir, now=NOW, require_contest_state=True
                        )
                        daily.chosen_on = picks.chosen_on_record(st, required=False)
                    elif sc["outcome"] == "dated_mismatch":
                        st = contest_state.load_decision_streak_state(
                            picks_dir, now=NOW, require_contest_state=True
                        )
                        bad = picks.chosen_on_record(st, required=True)
                        bad["streak"] = 13
                        bad["dated"]["active_streak"] = 13
                        daily.chosen_on = bad
                    elif sc["outcome"] == "aged_saved":
                        chosen_at = NOW - timedelta(hours=8)
                        st = contest_state.load_decision_streak_state(
                            picks_dir, now=chosen_at, require_contest_state=True
                        )
                        daily.chosen_on = picks.chosen_on_record(st, required=True)
                        if has_auto_field:
                            daily.automatic_reading = {
                                "schema": SCHEMA,
                                "recorded_at": rec_str,
                                "age_hours": 12.0,
                                "stale": False,
                                "reason": None,
                            }
                    elif sc["required"] and sc["reading"] in {
                        "fresh_auto", "old_auto", "future_auto", "unreadable_auto", "dated"
                    }:
                        st = contest_state.load_decision_streak_state(
                            picks_dir, now=NOW, require_contest_state=True
                        )
                        daily.chosen_on = picks.chosen_on_record(st, required=True)

                # Seed markers for guard outcomes
                if sc["outcome"] == "already_delivered":
                    if sc["mode"] == "dm":
                        daily.notification_sent = True
                        daily.notification_channel = "bluesky_dm"
                        daily.notification_id = "prior-dm-id"
                    else:
                        daily.bluesky_posted = True
                        daily.bluesky_uri = "at://prior-post"
                    daily.delivered_at = NOW.isoformat()
                elif sc["outcome"] == "crash_attempt":
                    daily.delivery_attempted = True

                # Seed a prior valid automatic_reading on disk for Row-6 preservation checks
                seeded_prior_reading = {
                    "schema": SCHEMA,
                    "recorded_at": "2027-07-15T12:00:00-04:00",
                    "age_hours": 6.58,
                    "stale": False,
                    "reason": None,
                }
                if has_auto_field and sc["outcome"] in {
                    "already_delivered", "crash_attempt", "no_chosen_on",
                    "preview_chosen_on", "dated_mismatch", "state_err", "no_recipient",
                } and daily.automatic_reading is None:
                    daily.automatic_reading = dict(seeded_prior_reading)
                if has_auto_field and (not sc["required"] or sc["mode"] == "private"):
                    daily.automatic_reading = dict(seeded_prior_reading)

                with patch.object(picks, "datetime", PinnedClock):
                    pick_path = picks.save_pick(daily, picks_dir)
                daily = picks.load_pick(DAY, picks_dir)

                if sc["sel_kind"] == "fresh_choice":
                    sel = make_fresh_sel(daily)
                elif sc["sel_kind"] == "matching_saved_reuse":
                    sel = make_reuse_sel(daily)
                else:
                    sel = None

                bluesky_cfg: dict = {"dm_recipient": "operator.test"}
                if sc["outcome"] == "no_recipient":
                    if sc["recip_val"] == "MISSING":
                        bluesky_cfg = {}
                    else:
                        bluesky_cfg = {"dm_recipient": sc["recip_val"]}

                config = {
                    "orchestrator": {
                        "picks_dir": str(picks_dir),
                        "heartbeat_path": str(root / "hb"),
                        "models_dir": str(root / "models"),
                    },
                    "bluesky": bluesky_cfg,
                    "scheduler": {
                        "contest_state_required": sc["required"],
                        "pick_delivery": sc["mode"],
                        "live_forward_capture_on_lock": True,
                    },
                }

                statuses = {
                    101: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
                    102: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
                }
                if sc["outcome"] == "single_void":
                    statuses[101] = {"abstract": "F", "detailed": "Postponed", "code": "PO"}
                elif sc["outcome"] == "double_both_void":
                    statuses[101] = {"abstract": "F", "detailed": "Cancelled", "code": "CO"}
                    statuses[102] = {"abstract": "F", "detailed": "Postponed", "code": "PO"}
                elif sc["outcome"] == "double_one_void":
                    statuses[101] = {"abstract": "F", "detailed": "Postponed", "code": "PO"}

                cutoff = picks.submission_cutoff_et(daily)
                if sc["outcome"] == "initial_late":
                    now_box[0] = cutoff
                    now_fn = lambda: now_box[0]
                elif sc["outcome"] == "final_cutoff":
                    ticks = iter([NOW, cutoff, cutoff, cutoff])
                    def now_fn():
                        val = next(ticks, cutoff)
                        now_box[0] = val
                        return val
                else:
                    now_fn = lambda: now_box[0]

                dm_calls: list[tuple[str, str]] = []
                post_calls: list[str] = []
                alert_dm_calls: list[tuple[str, str]] = []
                capture_calls: list[str] = []
                attempt_snap: list[dict | None] = []
                auto_fn_calls: list[int] = []

                def fake_dm(rec, txt):
                    on_disk = json.loads((picks_dir / f"{DAY}.json").read_text())
                    attempt_snap.append(on_disk)
                    dm_calls.append((rec, txt))
                    if sc["outcome"] == "send_raises":
                        raise RuntimeError("simulated dm failure")
                    if sc["outcome"] == "send_no_id":
                        return sc["empty_id"]
                    return "dm-msg-101"

                def fake_post(txt):
                    on_disk = json.loads((picks_dir / f"{DAY}.json").read_text())
                    attempt_snap.append(on_disk)
                    post_calls.append(txt)
                    if sc["outcome"] == "send_raises":
                        raise RuntimeError("simulated post failure")
                    if sc["outcome"] == "send_no_id":
                        return sc["empty_id"]
                    return "at://did:plc:bot/post/101"

                def fake_alert_dm(rec, txt):
                    alert_dm_calls.append((rec, txt))
                    return "alert-dm-202"

                sched_state = scheduler.SchedulerState(
                    date=DAY, schedule_fetched_at=NOW.isoformat(), games=[],
                    confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                    pick_locked_at=None, result_status=None, next_wakeup=None,
                )

                err_buf = io.StringIO()
                stack = contextlib.ExitStack()
                stack.enter_context(patch.object(contest_state, "datetime", PinnedClock))
                stack.enter_context(patch.object(daily_decision, "datetime", PinnedClock))
                stack.enter_context(patch.object(picks, "datetime", PinnedClock))
                stack.enter_context(patch.object(alert, "datetime", PinnedClock))
                stack.enter_context(patch.object(scheduler, "_now_et", now_fn))
                stack.enter_context(patch.object(scheduler, "_delivery_game_statuses",
                                                 lambda _d: statuses))
                stack.enter_context(patch.object(
                    scheduler, "_trigger_live_forward_capture_on_lock",
                    lambda _cfg, d: capture_calls.append(d),
                ))
                stack.enter_context(patch.object(dm, "send_dm", fake_dm))
                stack.enter_context(patch.object(posting, "post_to_bluesky", fake_post))
                stack.enter_context(patch.object(alert, "send_dm", fake_alert_dm))
                if hasattr(contest_state, "automatic_reading_record"):
                    real_arr = contest_state.automatic_reading_record
                    def spy_arr(*a, **kw):
                        auto_fn_calls.append(1)
                        return real_arr(*a, **kw)
                    stack.enter_context(patch.object(contest_state, "automatic_reading_record", spy_arr))
                stack.enter_context(contextlib.redirect_stderr(err_buf))

                with stack:
                    ret = scheduler._deliver_and_lock_pick(
                        daily, config, picks_dir, sched_state, DAY, "probe", selection=sel,
                    )
                    reentry_ret = None
                    if sc["outcome"] == "send_no_id":
                        reloaded = picks.load_pick(DAY, picks_dir)
                        reentry_state = scheduler.SchedulerState(
                            date=DAY, schedule_fetched_at=NOW.isoformat(), games=[],
                            confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                            pick_locked_at=None, result_status=None, next_wakeup=None,
                        )
                        reentry_ret = scheduler._deliver_and_lock_pick(
                            reloaded, config, picks_dir, reentry_state, DAY, "reentry",
                        )

                pick_json = (
                    json.loads(pick_path.read_text()) if pick_path.exists() else None
                )
                archives = {}
                if (picks_dir / DAY).exists():
                    for ap in sorted((picks_dir / DAY).glob("*.json")):
                        archives[ap.name] = json.loads(ap.read_text())
                sched_disk = (
                    json.loads((picks_dir / f"scheduler_{DAY}.json").read_text())
                    if (picks_dir / f"scheduler_{DAY}.json").exists() else None
                )
                dec_disk = (
                    json.loads((picks_dir / f"{DAY}.decision.json").read_text())
                    if (picks_dir / f"{DAY}.decision.json").exists() else None
                )
                evo_lines = (
                    (picks_dir / f"lineup_evolution_{DAY}.jsonl").read_text().splitlines()
                    if (picks_dir / f"lineup_evolution_{DAY}.jsonl").exists() else []
                )
                health_dir = picks_dir.parent / "health_state"
                crit_bytes = (
                    (health_dir / "health_dm_delivery_status.json").read_bytes().decode("utf-8")
                    if (health_dir / "health_dm_delivery_status.json").exists() else None
                )
                warn_status = (
                    json.loads((health_dir / "contest_stale_reading_dm_delivery_status.json").read_text())
                    if (health_dir / "contest_stale_reading_dm_delivery_status.json").exists() else None
                )
                # Normalize temp path embedded in ContestStateError messages
                norm_alerts = [
                    (r, t.replace(str(picks_dir), "<PICKS_DIR>")) for r, t in alert_dm_calls
                ]
                norm_stderr = [
                    line.replace(str(picks_dir), "<PICKS_DIR>")
                    for line in err_buf.getvalue().splitlines()
                ]
                if crit_bytes is not None:
                    # Normalize body_sha256 if body contained temp path
                    crit_obj = json.loads(crit_bytes)
                    if sc["outcome"] == "state_err":
                        crit_obj["body_sha256"] = "<PATH_DEPENDENT_SHA>"
                    crit_norm = json.dumps(crit_obj, sort_keys=True)
                else:
                    crit_norm = None

                results[sc["id"]] = {
                    "ret": ret,
                    "reentry_ret": reentry_ret,
                    "dm_calls": dm_calls,
                    "post_calls": post_calls,
                    "alert_dm_calls": norm_alerts,
                    "capture_calls": capture_calls,
                    "attempt_snap": attempt_snap,
                    "auto_fn_calls": len(auto_fn_calls),
                    "pick_json": pick_json,
                    "archives": archives,
                    "sched_disk": sched_disk,
                    "in_mem_locked": sched_state.pick_locked,
                    "dec_disk": dec_disk,
                    "evo_lines": evo_lines,
                    "crit_bytes_exact": crit_bytes if sc["outcome"] != "state_err" else crit_norm,
                    "warn_status": warn_status,
                    "stderr_lines": norm_stderr,
                }

        # Suite 4 part B: load_decision_streak_state matrix
        loader_results = {}
        idx = 0
        for req in (False, True):
            for switch_mode in ("none", "active", "revoked", "wrong_id"):
                for manual_mode in ("none", "dated_today", "dated_yesterday", "legacy_unexpired", "legacy_expired"):
                    for auto_mode in ("none", "v2_valid", "v2_naive_time", "v2_nonstr_time", "v1_legacy"):
                        for gap_mode in ("none", "hit_1", "miss_1", "hit_2"):
                            idx += 1
                            with tempfile.TemporaryDirectory() as td:
                                pd_dir = Path(td) / "picks"
                                acct = pd_dir / "account_state"
                                acct.mkdir(parents=True, exist_ok=True)
                                (pd_dir / "streak.json").write_text(json.dumps({
                                    "season": 2027, "streak": 4, "saver_available": True,
                                }))
                                if switch_mode != "none":
                                    (acct / "season_contract.json").write_text(json.dumps({
                                        "schema_version": "bts_season_contract_v1",
                                        "season": 2027,
                                        "status": "revoked" if switch_mode == "revoked" else "automatic",
                                        "activation_id": ("b" * 32 if switch_mode == "wrong_id" else "a" * 32),
                                        "switched_at": NOW.isoformat(),
                                        "revoked_at": NOW.isoformat() if switch_mode == "revoked" else None,
                                        "note": "revoked" if switch_mode == "revoked" else None,
                                    }))
                                if manual_mode == "dated_today":
                                    (acct / "contest_streak.manual.json").write_text(json.dumps({
                                        "schema_version": "bts_contest_streak_manual_v3",
                                        "season": 2027, "effective_date": DAY,
                                        "active_streak": 11, "best_streak": 15, "saver_state": "active",
                                        "recorded_at": "2027-07-15T10:00:00Z",
                                        "expires_at": f"{DAY}T23:59:59.999999-04:00",
                                        "source": "manual_cli",
                                    }))
                                elif manual_mode == "dated_yesterday":
                                    (acct / "contest_streak.manual.json").write_text(json.dumps({
                                        "schema_version": "bts_contest_streak_manual_v3",
                                        "season": 2027, "effective_date": "2027-07-14",
                                        "active_streak": 11, "best_streak": 15, "saver_state": "active",
                                        "recorded_at": "2027-07-14T10:00:00Z",
                                        "expires_at": "2027-07-14T23:59:59.999999-04:00",
                                        "source": "manual_cli",
                                    }))
                                elif manual_mode in {"legacy_unexpired", "legacy_expired"}:
                                    exp = (NOW + timedelta(hours=2) if manual_mode == "legacy_unexpired"
                                           else NOW - timedelta(hours=2)).isoformat()
                                    (acct / "contest_streak.manual.json").write_text(json.dumps({
                                        "schema_version": "bts_contest_streak_manual_v1",
                                        "active_streak": 9, "best_streak": 14,
                                        "source_date": "2027-07-13",
                                        "recorded_at": "2027-07-13T08:00:00",
                                        "override_expires_at": exp,
                                    }))
                                if auto_mode != "none":
                                    rec_val: object = "2027-07-13T13:30:00-04:00"
                                    if auto_mode == "v2_naive_time":
                                        rec_val = "2027-07-13T13:30:00"
                                    elif auto_mode == "v2_nonstr_time":
                                        rec_val = 12345
                                    auto_body = {
                                        "schema_version": ("bts_contest_streak_auto_v1"
                                                           if auto_mode == "v1_legacy"
                                                           else "bts_contest_streak_auto_v2"),
                                        "active_streak": 8, "best_streak": 12,
                                        "source": "mlb_bts_profile",
                                        "source_date": "2027-07-13",
                                        "recorded_at": rec_val,
                                    }
                                    if auto_mode != "v1_legacy":
                                        auto_body["season"] = 2027
                                        auto_body["activation_id"] = "a" * 32
                                    (acct / "contest_streak.json").write_text(json.dumps(auto_body))
                                if gap_mode in {"hit_1", "hit_2"}:
                                    (pd_dir / "2027-07-14.json").write_text(json.dumps({"result": "hit"}))
                                if gap_mode == "hit_2":
                                    (pd_dir / "2027-07-15.json").write_text(json.dumps({"result": "hit"}))
                                if gap_mode == "miss_1":
                                    (pd_dir / "2027-07-14.json").write_text(json.dumps({"result": "miss"}))

                                key = f"{req}:{switch_mode}:{manual_mode}:{auto_mode}:{gap_mode}"
                                try:
                                    st = contest_state.load_decision_streak_state(
                                        pd_dir, require_contest_state=req, now=NOW,
                                    )
                                    d = asdict(st)
                                    if d.get("contest_source_date") is not None:
                                        d["contest_source_date"] = d["contest_source_date"].isoformat()
                                    selected = contest_state.load_contest_streak_state(
                                        pd_dir, now=NOW, require_dated_auto=req,
                                    )
                                    loader_results[key] = {
                                        "ok": True,
                                        "state": d,
                                        "selected_recorded_at": selected.recorded_at if selected else None,
                                    }
                                except contest_state.ContestStateError as exc:
                                    loader_results[key] = {
                                        "ok": False,
                                        "error": str(exc).replace(str(pd_dir), "<PICKS_DIR>"),
                                    }

        return {"scenarios": results, "loader": loader_results}


def _compare_base_and_branch(base_data: dict, branch_data: dict) -> None:
    base_sc = base_data["scenarios"]
    branch_sc = branch_data["scenarios"]
    assert set(base_sc) == set(branch_sc)

    stale_warn_prefix = "  STALE CONTEST READING — "
    stale_cells = 0
    fresh_stamped_cells = 0
    dated_stamped_cells = 0
    control_cells = 0

    for sid in sorted(base_sc):
        b0 = base_sc[sid]
        b1 = branch_sc[sid]

        # 1. Exact pick delivery invariants
        assert b0["ret"] == b1["ret"], f"{sid}: return value mismatch {b0['ret']} vs {b1['ret']}"
        assert b0["reentry_ret"] == b1["reentry_ret"], f"{sid}: reentry mismatch"
        assert b0["dm_calls"] == b1["dm_calls"], f"{sid}: dm_calls mismatch"
        assert b0["post_calls"] == b1["post_calls"], f"{sid}: post_calls mismatch"
        assert b0["capture_calls"] == b1["capture_calls"], f"{sid}: capture mismatch"
        assert b0["in_mem_locked"] == b1["in_mem_locked"], f"{sid}: in_mem_locked mismatch"
        assert b0["sched_disk"] == b1["sched_disk"], f"{sid}: scheduler_state.json mismatch"
        assert b0["dec_disk"] == b1["dec_disk"], f"{sid}: decision.json mismatch"
        assert b0["evo_lines"] == b1["evo_lines"], f"{sid}: lineup_evolution mismatch"
        assert b0["crit_bytes_exact"] == b1["crit_bytes_exact"], (
            f"{sid}: health_dm_delivery_status.json bytes changed!"
        )

        # 2. Pick JSON identical apart from trailing automatic_reading
        if b0["pick_json"] is None:
            assert b1["pick_json"] is None, f"{sid}: pick_json existence mismatch"
        else:
            p1 = dict(b1["pick_json"])
            assert list(p1)[-2:] == ["chosen_on", "automatic_reading"], f"{sid}: wrong trailing keys {list(p1)[-2:]}"
            auto_rec = p1.pop("automatic_reading")
            assert p1 == b0["pick_json"], f"{sid}: non-automatic_reading pick fields differ"

        # 3. Archives identical apart from automatic_reading second-to-last
        assert set(b0["archives"]) == set(b1["archives"]), f"{sid}: archive names differ"
        for aname in b0["archives"]:
            a1 = dict(b1["archives"][aname])
            keys = list(a1)
            assert keys[-3:-1] == ["chosen_on", "automatic_reading"], f"{sid}: archive key order {keys[-3:]}"
            a1.pop("automatic_reading")
            assert a1 == b0["archives"][aname], f"{sid}: archive payload differs"

        # 4. Determine whether this scenario is a qualifying attempt & confirmed stale send
        is_qualifying_attempt = (
            len(b1["attempt_snap"]) == 1
            and not sid.startswith("not_required:")
            and not sid.startswith("private:")
        )
        if not is_qualifying_attempt:
            control_cells += 1
            assert b1["auto_fn_calls"] == 0, f"{sid}: unexpected automatic_reading_record call!"
            assert b0["alert_dm_calls"] == b1["alert_dm_calls"], f"{sid}: unexpected alert DM"
            assert b0["stderr_lines"] == b1["stderr_lines"], f"{sid}: unexpected stderr diff"
            assert b1["warn_status"] is None, f"{sid}: unexpected warn status file"
        else:
            assert b1["auto_fn_calls"] == 1, f"{sid}: expected 1 automatic_reading_record call"
            att_auto = b1["attempt_snap"][0]["automatic_reading"]
            fin_auto = b1["pick_json"]["automatic_reading"]
            assert att_auto == fin_auto, f"{sid}: attempt vs final automatic_reading mismatch"

            confirmed = b1["ret"] is True
            if ":dated" in sid:
                dated_stamped_cells += 1
                assert fin_auto is None, f"{sid}: dated should stamp None"
                assert b0["alert_dm_calls"] == b1["alert_dm_calls"]
                assert b0["stderr_lines"] == b1["stderr_lines"]
                assert b1["warn_status"] is None
            elif ":fresh_auto" in sid:
                fresh_stamped_cells += 1
                assert fin_auto is not None and fin_auto["stale"] is False and fin_auto["reason"] is None
                assert b0["alert_dm_calls"] == b1["alert_dm_calls"]
                assert b0["stderr_lines"] == b1["stderr_lines"]
                assert b1["warn_status"] is None
            else:
                # Stale automatic reading (old_auto, future_auto, unreadable_auto)
                assert fin_auto is not None and fin_auto["stale"] is True
                if not confirmed:
                    # e.g. send_raises: stamped and kept on disk, no warning!
                    assert b1["pick_json"]["delivery_attempted"] is False
                    assert b0["alert_dm_calls"] == b1["alert_dm_calls"]
                    assert b0["stderr_lines"] == b1["stderr_lines"]
                    assert b1["warn_status"] is None
                else:
                    stale_cells += 1
                    reason = fin_auto["reason"]
                    if reason == "old":
                        detail = f"last accepted fetch {fin_auto['recorded_at']}, about {fin_auto['age_hours']:.2f} hours ago"
                    elif reason == "future":
                        detail = f"its recorded_at {fin_auto['recorded_at']} is in the future"
                    else:
                        detail = "it has no usable recorded_at"
                    msg = _expected_stale_message(DAY, detail)
                    expected_line = stale_warn_prefix + msg
                    b1_non_warn = [ln for ln in b1["stderr_lines"] if ln != expected_line]
                    assert b1["stderr_lines"].count(expected_line) == 1, f"{sid}: missing/dup stderr warning"
                    assert b1_non_warn == b0["stderr_lines"], f"{sid}: other stderr lines changed"
                    if sid.startswith("no_recipient:public:"):
                        assert b1["alert_dm_calls"] == b0["alert_dm_calls"] == []
                        assert b1["warn_status"] is None, f"{sid}: Amendment 1 violated (status file written)"
                    else:
                        expected_env = ("operator.test", f"BTS health WARN attention:\n- [contest_state] {msg}")
                        # In send_no_id:dm, reentry adds a CRITICAL late_delivery alert AFTER the warning!
                        extra = [c for c in b1["alert_dm_calls"] if c not in b0["alert_dm_calls"]]
                        assert extra == [expected_env], f"{sid}: unexpected warning DM {extra}"
                        assert b1["warn_status"] is not None
                        assert b1["warn_status"]["status"] == "sent"
                        assert b1["warn_status"]["sent_day"] == DAY
                        assert b1["warn_status"]["sent_sources"] == ["WARN_ATTENTION:contest_state:stale_reading"]

    # Compare load_decision_streak_state matrix (320 cells)
    b0_ld = base_data["loader"]
    b1_ld = branch_data["loader"]
    assert set(b0_ld) == set(b1_ld)
    for k, v0 in b0_ld.items():
        v1 = b1_ld[k]
        assert v0["ok"] == v1["ok"], f"loader {k} ok mismatch"
        if not v0["ok"]:
            assert v0["error"] == v1["error"], f"loader {k} error mismatch"
        else:
            s1 = dict(v1["state"])
            assert list(s1)[-1] == "contest_recorded_at"
            rec_at = s1.pop("contest_recorded_at")
            assert s1 == v0["state"], f"loader {k} fields changed"
            if s1["observation_kind"] == "model":
                assert rec_at is None
            else:
                assert rec_at == v1["selected_recorded_at"], f"loader {k} contest_recorded_at mismatch"

    print(
        f"[PASS] Base vs Branch parity: {len(base_sc)} delivery scenarios "
        f"(controls={control_cells}, dated_stamped={dated_stamped_cells}, "
        f"fresh_stamped={fresh_stamped_cells}, confirmed_stale={stale_cells}) "
        f"and {len(b0_ld)} loader cells matched 100%."
    )


def _run_branch_adversarial_faults_and_stamp_precision() -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REVIEW_DIR))
    from netguard import forbid_network

    with forbid_network(allow_loopback=False):
        from bts import contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
        from bts.health import alert

        class UnprintableError(RuntimeError):
            def __str__(self):
                raise RuntimeError("cannot stringify error")

        def make_env(td: str, mode: str = "dm"):
            root = Path(td)
            picks_dir = root / "picks"
            picks_dir.mkdir(parents=True, exist_ok=True)
            rec = _seed_reading(picks_dir, "old_auto", now=NOW)
            p1 = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                            0.82, [], False, 101, f"{DAY}T23:10:00Z")
            daily = picks.DailyPick(DAY, NOW.isoformat(), p1, None, None)
            pol = strategy.PolicyDecision(
                "single", "single", "heuristic", "reach57", 12, 80, None, "missing",
                None, None, None, None, None,
            )
            sel = strategy.SelectionResult(
                strategy.PickResult(daily), "single", "heuristic",
                scheduler._row_from_daily(p1), None, None, streak=12, decision=pol,
            )
            cfg = {
                "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(root / "hb"),
                                 "models_dir": str(root / "models")},
                "bluesky": {"dm_recipient": "operator.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": mode,
                              "live_forward_capture_on_lock": True},
                "health_checks": {"enabled": False},
            }
            st = scheduler.SchedulerState(
                date=DAY, schedule_fetched_at=NOW.isoformat(), games=[],
                confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            return root, picks_dir, rec, daily, sel, cfg, st

        # 1. Non-BaseException faults: NONE may change return True, markers, lock, decision, or capture!
        fault_cases = [
            "calc_runtime", "calc_unprintable",
            "disp_runtime", "disp_unprintable",
            "transport_runtime",
            "status_is_dir", "health_parent_is_file", "health_chmod_ro",
            "status_json_list", "status_json_corrupt",
            "record_str", "record_missing_keys", "record_bad_age_type", "record_unknown_reason",
            "stderr_closed", "stderr_broken_pipe",
            "warn_clock_raises", "bluesky_cfg_none_public",
        ]
        for mode in ("dm", "public"):
            for fcase in fault_cases:
                if fcase == "bluesky_cfg_none_public" and mode != "public":
                    continue
                with tempfile.TemporaryDirectory() as td:
                    root, picks_dir, rec, daily, sel, cfg, st = make_env(td, mode=mode)
                    if fcase == "bluesky_cfg_none_public":
                        cfg["bluesky"] = None
                    health_dir = picks_dir.parent / "health_state"
                    warn_file = health_dir / "contest_stale_reading_dm_delivery_status.json"
                    chmod_target = None
                    if fcase == "status_is_dir":
                        warn_file.mkdir(parents=True)
                    elif fcase == "health_parent_is_file":
                        health_dir.write_text("not a directory")
                    elif fcase == "health_chmod_ro":
                        health_dir.mkdir(parents=True)
                        health_dir.chmod(0o500)
                        chmod_target = health_dir
                    elif fcase == "status_json_list":
                        health_dir.mkdir(parents=True)
                        warn_file.write_text("[]")
                    elif fcase == "status_json_corrupt":
                        health_dir.mkdir(parents=True)
                        warn_file.write_text("{bad")

                    captures: list[str] = []
                    pick_sends: list[str] = []
                    warn_sends: list[str] = []

                    def on_capture(_c, d):
                        captures.append(d)
                        if fcase == "record_str":
                            daily.automatic_reading = "corrupted"  # type: ignore[assignment]
                        elif fcase == "record_missing_keys":
                            daily.automatic_reading = {"stale": True}
                        elif fcase == "record_bad_age_type":
                            daily.automatic_reading = {
                                "stale": True, "reason": "old", "recorded_at": rec, "age_hours": "bad"
                            }
                        elif fcase == "record_unknown_reason":
                            daily.automatic_reading = {
                                "stale": True, "reason": "mystery", "recorded_at": rec, "age_hours": None
                            }

                    now_calls = 0
                    def now_fn():
                        nonlocal now_calls
                        now_calls += 1
                        if fcase == "warn_clock_raises" and now_calls >= 5:
                            raise RuntimeError("warning clock failed")
                        return NOW

                    err_stream: object = io.StringIO()
                    if fcase == "stderr_closed":
                        closed = io.StringIO()
                        closed.close()
                        err_stream = closed
                    elif fcase == "stderr_broken_pipe":
                        class BrokenStderr:
                            def write(self, _s):
                                raise BrokenPipeError("broken pipe")
                            def flush(self):
                                pass
                        err_stream = BrokenStderr()

                    try:
                        with contextlib.ExitStack() as stack:
                            stack.enter_context(patch.object(scheduler, "_now_et", now_fn))
                            stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                            stack.enter_context(patch.object(
                                scheduler, "_trigger_live_forward_capture_on_lock", on_capture
                            ))
                            stack.enter_context(patch.object(
                                dm, "send_dm", lambda r, t: (pick_sends.append(t), "dm-1")[1]
                            ))
                            stack.enter_context(patch.object(
                                posting, "post_to_bluesky", lambda t: (pick_sends.append(t), "at://1")[1]
                            ))
                            if fcase == "transport_runtime":
                                stack.enter_context(patch.object(
                                    alert, "send_dm", Mock(side_effect=RuntimeError("warn dm down"))
                                ))
                            else:
                                stack.enter_context(patch.object(
                                    alert, "send_dm", lambda r, t: (warn_sends.append(t), "wdm-1")[1]
                                ))
                            if fcase == "calc_runtime":
                                stack.enter_context(patch.object(
                                    contest_state, "automatic_reading_record",
                                    Mock(side_effect=RuntimeError("calc boom")),
                                ))
                            elif fcase == "calc_unprintable":
                                stack.enter_context(patch.object(
                                    contest_state, "automatic_reading_record",
                                    Mock(side_effect=UnprintableError()),
                                ))
                            elif fcase == "disp_runtime":
                                stack.enter_context(patch.object(
                                    alert, "dispatch_dm_for_health_alerts",
                                    Mock(side_effect=RuntimeError("disp boom")),
                                ))
                            elif fcase == "disp_unprintable":
                                stack.enter_context(patch.object(
                                    alert, "dispatch_dm_for_health_alerts",
                                    Mock(side_effect=UnprintableError()),
                                ))
                            stack.enter_context(patch.object(sys, "stderr", err_stream))

                            ok = scheduler._deliver_and_lock_pick(
                                daily, cfg, picks_dir, st, DAY, "fault", selection=sel,
                            )
                        assert ok is True, f"{mode}:{fcase} did not return True"
                        assert len(pick_sends) == 1, f"{mode}:{fcase} wrong pick send count"
                        assert captures == [DAY], f"{mode}:{fcase} capture skipped"
                        saved = picks.load_pick(DAY, picks_dir)
                        assert saved is not None and saved.delivery_attempted is True
                        assert picks.pick_was_delivered(saved) is True
                        assert st.pick_locked is True
                        assert scheduler.load_state(DAY, picks_dir).pick_locked is True
                        dec = daily_decision.load_decision(DAY, picks_dir)
                        assert dec is not None and dec["delivery_status"] == "delivered"
                        if fcase in {"stderr_closed", "stderr_broken_pipe"}:
                            assert len(warn_sends) == 1, f"{mode}:{fcase} dispatch was blocked by stderr failure!"
                    finally:
                        if chmod_target is not None:
                            chmod_target.chmod(0o755)

        # 2. BaseException subclasses (KeyboardInterrupt, SystemExit) inside warning vs stamp
        for bexc_cls in (KeyboardInterrupt, SystemExit):
            for mode in ("dm", "public"):
                # (a) Inside warning: pick is ALREADY durably delivered, locked, captured, and committed!
                with tempfile.TemporaryDirectory() as td:
                    root, picks_dir, rec, daily, sel, cfg, st = make_env(td, mode=mode)
                    captures = []
                    pick_sends = []
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(patch.object(scheduler, "_now_et", lambda: NOW))
                        stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                        stack.enter_context(patch.object(
                            scheduler, "_trigger_live_forward_capture_on_lock",
                            lambda _c, d: captures.append(d),
                        ))
                        stack.enter_context(patch.object(
                            dm, "send_dm", lambda r, t: (pick_sends.append(t), "dm-be")[1]
                        ))
                        stack.enter_context(patch.object(
                            posting, "post_to_bluesky", lambda t: (pick_sends.append(t), "at://be")[1]
                        ))
                        stack.enter_context(patch.object(
                            alert, "dispatch_dm_for_health_alerts", Mock(side_effect=bexc_cls("sig")),
                        ))
                        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                        raised = False
                        try:
                            scheduler._deliver_and_lock_pick(
                                daily, cfg, picks_dir, st, DAY, "bexc_warn", selection=sel,
                            )
                        except bexc_cls:
                            raised = True
                        assert raised
                        # All delivery side effects completed BEFORE BaseException propagated
                        saved = picks.load_pick(DAY, picks_dir)
                        assert saved is not None and saved.delivery_attempted is True
                        assert picks.pick_was_delivered(saved) is True
                        assert saved.automatic_reading is not None and saved.automatic_reading["stale"] is True
                        assert st.pick_locked is True
                        assert scheduler.load_state(DAY, picks_dir).pick_locked is True
                        assert captures == [DAY]
                        dec = daily_decision.load_decision(DAY, picks_dir)
                        assert dec is not None and dec["delivery_status"] == "delivered"

                    # Restart recovery after BaseException during warning: no re-send, no crash-guard, no warning!
                    alert_dm_after = Mock(return_value="a")
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(patch.object(scheduler, "_now_et", lambda: NOW))
                        stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                        stack.enter_context(patch.object(
                            scheduler, "_trigger_live_forward_capture_on_lock", lambda *_: None
                        ))
                        stack.enter_context(patch.object(dm, "send_dm", lambda r, t: pick_sends.append(t)))
                        stack.enter_context(patch.object(posting, "post_to_bluesky", lambda t: pick_sends.append(t)))
                        stack.enter_context(patch.object(alert, "send_dm", alert_dm_after))
                        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                        st2 = scheduler.SchedulerState(
                            date=DAY, schedule_fetched_at=NOW.isoformat(), games=[],
                            confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                            pick_locked_at=None, result_status=None, next_wakeup=None,
                        )
                        ok2 = scheduler._deliver_and_lock_pick(
                            picks.load_pick(DAY, picks_dir), cfg, picks_dir, st2, DAY, "restart",
                        )
                        assert ok2 is True
                        assert len(pick_sends) == 1
                        assert not alert_dm_after.called

                # (b) Inside _stamp_reading: before attempt save, so disk stays untouched and 0 sends
                with tempfile.TemporaryDirectory() as td:
                    root, picks_dir, rec, daily, sel, cfg, st = make_env(td, mode=mode)
                    path = picks.save_pick(daily, picks_dir)
                    raw_before = path.read_bytes()
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(patch.object(scheduler, "_now_et", lambda: NOW))
                        stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                        stack.enter_context(patch.object(
                            contest_state, "automatic_reading_record", Mock(side_effect=bexc_cls("pre")),
                        ))
                        raised = False
                        try:
                            scheduler._deliver_and_lock_pick(
                                daily, cfg, picks_dir, st, DAY, "bexc_stamp", selection=sel,
                            )
                        except bexc_cls:
                            raised = True
                        assert raised
                        assert path.read_bytes() == raw_before

        # 3. Item 2 Stamp Precision: distinct _now_et() samples, single state read, exact save_pick count
        for mode in ("dm", "public"):
            with tempfile.TemporaryDirectory() as td:
                root, picks_dir, _, daily, sel, cfg, st = make_env(td, mode=mode)
                # Game at 2027-07-16T05:00:00Z (01:00 ET July 16, cutoff 00:55 ET July 16)
                daily.pick.game_time = "2027-07-16T05:00:00Z"
                rec_time = datetime(2027, 7, 15, 7, 0, 0, tzinfo=ET)
                (picks_dir / "account_state" / "contest_streak.json").write_text(json.dumps({
                    "schema_version": "bts_contest_streak_auto_v2",
                    "season": 2027, "activation_id": "a" * 32,
                    "active_streak": 12, "best_streak": 15,
                    "source": "mlb_bts_profile", "source_date": DAY,
                    "recorded_at": rec_time.isoformat(),
                }))
                t0_init = datetime(2027, 7, 15, 20, 0, 0, tzinfo=ET)     # 13.0h (fresh!)
                t1_final = datetime(2027, 7, 15, 23, 0, 0, tzinfo=ET)    # 16.0h (stale!)
                t2_deliv = datetime(2027, 7, 15, 23, 1, 0, tzinfo=ET)    # 16.02h
                t3_lock = datetime(2027, 7, 15, 23, 2, 0, tzinfo=ET)     # 16.03h
                t4_warn = datetime(2027, 7, 16, 0, 10, 0, tzinfo=ET)     # next ET day!
                ticks = iter([t0_init, t1_final, t2_deliv, t3_lock, t4_warn])

                real_load_ds = contest_state.load_decision_streak_state
                ds_reads = []

                def spy_load_ds(*a, **kw):
                    res = real_load_ds(*a, **kw)
                    ds_reads.append(1)
                    # Delete on-disk file right after delivery's single read to prove no second read!
                    (picks_dir / "account_state" / "contest_streak.json").unlink()
                    return res

                real_save = picks.save_pick
                saves = []

                def spy_save(d_obj, p_dir):
                    res = real_save(d_obj, p_dir)
                    saves.append(json.loads(res.read_text()))
                    return res

                warn_bodies = []
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(contest_state, "datetime", datetime))
                    stack.enter_context(patch.object(
                        contest_state, "load_decision_streak_state", spy_load_ds
                    ))
                    # Pin now inside load_decision_streak_state via wrapper
                    stack.enter_context(patch.object(scheduler, "_now_et", lambda: next(ticks)))
                    stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                    stack.enter_context(patch.object(
                        scheduler, "_trigger_live_forward_capture_on_lock", lambda *_: None
                    ))
                    stack.enter_context(patch.object(picks, "save_pick", spy_save))
                    stack.enter_context(patch.object(dm, "send_dm", lambda *_: "dm-p"))
                    stack.enter_context(patch.object(posting, "post_to_bluesky", lambda *_: "at://p"))
                    stack.enter_context(patch.object(
                        alert, "send_dm", lambda r, b: (warn_bodies.append(b), "w-1")[1]
                    ))
                    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

                    # Patch contest_state.datetime so real_load_ds sees DAY
                    class DayClock(datetime):
                        @classmethod
                        def now(cls, tz=None):
                            return t0_init.astimezone(tz) if tz else t0_init.replace(tzinfo=None)
                    stack.enter_context(patch.object(contest_state, "datetime", DayClock))

                    assert scheduler._deliver_and_lock_pick(
                        daily, cfg, picks_dir, st, DAY, "precision", selection=sel,
                    ) is True

                assert len(ds_reads) == 1, f"{mode}: expected 1 state read, got {len(ds_reads)}"
                assert len(saves) == 2, f"{mode}: expected 2 save_pick calls (attempt + delivered), got {len(saves)}"
                assert saves[0]["delivery_attempted"] is True
                assert saves[0]["automatic_reading"]["age_hours"] == 16.0
                assert saves[0]["automatic_reading"]["reason"] == "old"
                assert saves[1]["delivered_at"] == t2_deliv.isoformat()
                assert saves[1]["automatic_reading"] == saves[0]["automatic_reading"]
                assert st.pick_locked_at == t3_lock.isoformat()
                assert len(warn_bodies) == 1
                assert "--date 2027-07-16" in warn_bodies[0]
                assert "about 16.00 hours ago" in warn_bodies[0]

    print("[PASS] Fault injection (35 non-BaseException + 8 BaseException + restart cases) and stamp precision verified.")


def _check_row8_source_invariance() -> None:
    unchanged_whole_files = [
        "src/bts/season_contract.py",
        "src/bts/contest_fetch.py",
        "src/bts/daily_decision.py",
        "src/bts/strategy.py",
        "src/bts/orchestrator.py",
        "src/bts/cli.py",
        "src/bts/health/contest_state.py",
        "src/bts/health/attention.py",
        "src/bts/health/alert.py",
    ]
    for rel in unchanged_whole_files:
        b0 = (BASE_ROOT / rel).read_bytes()
        b1 = (REPO_ROOT / rel).read_bytes()
        assert b0 == b1, f"Row 8 violation: {rel} changed!"

    def func_dump(path: Path, names: set[str]) -> dict[str, str]:
        tree = ast.parse(path.read_text())
        out = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                out[node.name] = ast.dump(node)
        assert set(out) == names, f"missing functions {names - set(out)} in {path}"
        return out

    picks_fns = {"_valid_chosen_on", "chosen_on_record", "saved_pick_refusal"}
    assert func_dump(BASE_ROOT / "src/bts/picks.py", picks_fns) == func_dump(
        REPO_ROOT / "src/bts/picks.py", picks_fns
    )

    sched_fns = {
        "run_single_check",
        "_refresh_pick_at_fallback_decision",
        "_refresh_pick_at_fallback",
        "plan_fallback_action",
        "_should_defer_at_fallback",
        "_alert_contest_state_failure",
        "_alert_missed_pick",
    }
    assert func_dump(BASE_ROOT / "src/bts/scheduler.py", sched_fns) == func_dump(
        REPO_ROOT / "src/bts/scheduler.py", sched_fns
    )
    print("[PASS] Row 8 whole-file byte invariance and AST function invariance verified.")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        tree = sys.argv[2]
        out = _run_matrix_worker(tree)
        sys.stdout.write(json.dumps(out))
        return

    _check_row8_source_invariance()

    def run_worker(tree: str, src_path: Path) -> dict:
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{src_path}:{REPO_ROOT}"
        env["TZ"] = "America/New_York"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", tree],
            env=env, capture_output=True, text=True, check=True,
        )
        return json.loads(proc.stdout)

    base_out = run_worker("base", BASE_ROOT / "src")
    branch_out = run_worker("branch", REPO_ROOT / "src")
    (REVIEW_DIR / "probe_1_base.json").write_text(json.dumps(base_out["scenarios"], indent=2))
    (REVIEW_DIR / "probe_1_branch.json").write_text(json.dumps(branch_out["scenarios"], indent=2))
    _compare_base_and_branch(base_out, branch_out)
    _run_branch_adversarial_faults_and_stamp_precision()
    print("PROBE_1_2_7_ALL_PASSED")


if __name__ == "__main__":
    main()
