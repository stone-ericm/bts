"""Adversarial Probe 3: Complete verification of Probe 1b + Probe 2 Part B + Item 5 (Warning) + Item 6 (run_day)."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_ROOT = REVIEW_DIR / "scratch" / "base_1c6cbc77"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REVIEW_DIR))

from netguard import forbid_network

ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime(2027, 7, 15, 18, 35, 0, tzinfo=ET)
SCHEMA = "bts_pick_automatic_reading_v1"
WARN_FILE = "contest_stale_reading_dm_delivery_status.json"
CRIT_FILE = "health_dm_delivery_status.json"
WARN_KEY = "WARN_ATTENTION:contest_state:stale_reading"
WARN_PREFIX = "  STALE CONTEST READING — "


def _msg(day_iso: str, detail: str) -> str:
    return (
        f"The pick was delivered using an automatic contest reading flagged stale ({detail}). "
        "Check MLB. If the streak or saver differs, use bts set-contest-streak "
        f"--date {day_iso} --streak N --best-streak B --saver S with the "
        "scheduler's picks directory and the values shown in MLB. "
        "Use the current ET date when you run it. Check the latest fetch status."
    )


def _env(day_iso: str, detail: str) -> str:
    return f"BTS health WARN attention:\n- [contest_state] {_msg(day_iso, detail)}"


def _finish_probe_1_verification() -> None:
    import probe_1_2_7_delivery_stamp_row8 as p1
    import probe_1b_verify_and_faults as p1b

    base_sc = json.loads((REVIEW_DIR / "probe_1_base.json").read_text())
    branch_sc = json.loads((REVIEW_DIR / "probe_1_branch.json").read_text())
    assert set(base_sc) == set(branch_sc)

    stale_cells = fresh_cells = dated_cells = control_cells = 0
    for sid in sorted(base_sc):
        b0, b1 = base_sc[sid], branch_sc[sid]
        for field in (
            "ret", "reentry_ret", "dm_calls", "post_calls", "capture_calls",
            "in_mem_locked", "evo_lines", "crit_bytes_exact",
        ):
            assert b0[field] == b1[field], f"{sid}: {field} mismatch"

        if b0["pick_json"] is None:
            assert b1["pick_json"] is None
        else:
            p_copy = dict(b1["pick_json"])
            assert list(p_copy)[-2:] == ["chosen_on", "automatic_reading"]
            p_copy.pop("automatic_reading")
            assert p_copy == b0["pick_json"]

        assert set(b0["archives"]) == set(b1["archives"])
        for aname in b0["archives"]:
            if aname in {"decision.json", "scheduler_state.json"}:
                assert "automatic_reading" not in b1["archives"][aname]
                assert b0["archives"][aname] == b1["archives"][aname]
            else:
                acopy = dict(b1["archives"][aname])
                assert list(acopy)[-3:-1] == ["chosen_on", "automatic_reading"]
                acopy.pop("automatic_reading")
                assert acopy == b0["archives"][aname]

        qualifying = (
            len(b1["attempt_snap"]) == 1
            and not sid.startswith("not_required:")
            and not sid.startswith("private:")
        )
        if not qualifying:
            control_cells += 1
            assert b1["auto_fn_calls"] == 0
            assert b0["alert_dm_calls"] == b1["alert_dm_calls"]
            assert b0["stderr_lines"] == b1["stderr_lines"]
            assert b1["warn_status"] is None
        else:
            assert b1["auto_fn_calls"] == 1
            att_auto = b1["attempt_snap"][0]["automatic_reading"]
            fin_auto = b1["pick_json"]["automatic_reading"]
            assert att_auto == fin_auto
            if ":dated" in sid:
                dated_cells += 1
                assert fin_auto is None
                assert b0["alert_dm_calls"] == b1["alert_dm_calls"]
                assert b0["stderr_lines"] == b1["stderr_lines"]
                assert b1["warn_status"] is None
            elif ":fresh_auto" in sid:
                fresh_cells += 1
                assert fin_auto is not None and fin_auto["stale"] is False
                assert b0["alert_dm_calls"] == b1["alert_dm_calls"]
                assert b0["stderr_lines"] == b1["stderr_lines"]
                assert b1["warn_status"] is None
            elif not b1["ret"]:
                assert b1["pick_json"]["delivery_attempted"] is False
                assert fin_auto is not None and fin_auto["stale"] is True
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
                expected_line = WARN_PREFIX + _msg(DAY, detail)
                assert b1["stderr_lines"].count(expected_line) == 1
                assert [ln for ln in b1["stderr_lines"] if ln != expected_line] == b0["stderr_lines"]
                if sid.startswith("no_recipient:public:"):
                    assert b1["alert_dm_calls"] == b0["alert_dm_calls"] == []
                    assert b1["warn_status"] is None
                else:
                    expected_dm = ["operator.test", _env(DAY, detail)]
                    extra = [c for c in b1["alert_dm_calls"] if c not in b0["alert_dm_calls"]]
                    assert extra == [expected_dm], f"{sid}: {extra}"
                    assert b1["warn_status"]["status"] == "sent"
                    assert b1["warn_status"]["sent_day"] == DAY
                    assert b1["warn_status"]["sent_sources"] == [WARN_KEY]

    print(
        f"[PASS] Item 1 Base vs Branch delivery parity: {len(base_sc)} scenarios "
        f"(controls={control_cells}, dated={dated_cells}, fresh={fresh_cells}, stale={stale_cells}) verified."
    )

    def run_loader(src_path: Path) -> dict:
        env = dict(os.environ, PYTHONPATH=f"{src_path}:{REPO_ROOT}", TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, str(REVIEW_DIR / "probe_1b_verify_and_faults.py"), "--loader-worker"],
            env=env, capture_output=True, text=True, check=True,
        )
        return json.loads(proc.stdout)

    b0_ld = run_loader(BASE_ROOT / "src")
    b1_ld = run_loader(REPO_ROOT / "src")
    assert set(b0_ld) == set(b1_ld)
    for k, v0 in b0_ld.items():
        v1 = b1_ld[k]
        assert v0["ok"] == v1["ok"]
        if not v0["ok"]:
            assert v0["error"] == v1["error"]
        else:
            s1 = dict(v1["state"])
            assert list(s1)[-1] == "contest_recorded_at"
            rec_at = s1.pop("contest_recorded_at")
            assert s1 == v0["state"]
            assert rec_at == (None if s1["observation_kind"] == "model" else v1["selected_recorded_at"])
    print(f"[PASS] Row 1 & Row 8 load_decision_streak_state parity: {len(b0_ld)} cells matched 100%.")

    p1._check_row8_source_invariance()
    p1._run_branch_adversarial_faults_and_stamp_precision()


def _finish_probe_2_resaves_and_readers() -> None:
    from click.testing import CliRunner
    from bts import cli, contest_state, daily_decision, entry_receipt, picks, scheduler, web
    from bts.experiment import artifacts as exp_artifacts
    from bts.health import (
        alert, fallback_defer, late_delivery, mdp_policy_alignment,
        pitcher_sparsity, post_failure, postponed_pick, predicted_vs_realized,
        projected_lineup, same_team_corr,
    )
    from bts.state import export as state_export, regenerate as state_regen
    from scripts import live_forward_capture_once as lf_capture
    from scripts import live_forward_provenance_inventory as lf_inventory
    from scripts.audit.season_ledger.sources import pick_files as ledger_picks

    old_rec = {
        "schema": SCHEMA,
        "recorded_at": "2027-07-14T22:35:00-04:00",
        "age_hours": 20.0,
        "stale": True,
        "reason": "old",
    }
    fresh_rec = {
        "schema": SCHEMA,
        "recorded_at": "2027-07-15T16:35:00-04:00",
        "age_hours": 2.0,
        "stale": False,
        "reason": None,
    }

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pdir = root / "picks"
        pdir.mkdir(parents=True)
        p1 = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                        0.82, [], False, 101, f"{DAY}T23:10:00Z")
        dp = picks.attach_provenance(picks.DailyPick(DAY, NOW.isoformat(), p1, None, None))
        dp.notification_sent = True
        dp.notification_channel = "bluesky_dm"
        dp.notification_id = "dm-1"
        dp.delivered_at = NOW.isoformat()
        dp.automatic_reading = old_rec
        picks.save_pick(dp, pdir)
        sh = picks.DailyPick(DAY, NOW.isoformat(), p1, None, None, automatic_reading=old_rec)
        picks.save_shadow_pick(sh, pdir)
        daily_decision.write_decision(
            DAY, pdir, action="single", source="heuristic",
            primary=scheduler._row_from_daily(p1),
            delivery_status="delivered", scoreable=True, now=NOW,
        )

        # CLI check-results grading both production and shadow pick files
        with patch.object(picks, "resolve_daily_slot_results", lambda *_: {"primary": "hit"}), patch(
            "bts.posting.reply_to_bluesky", lambda *_: "at://reply"
        ):
            cr = CliRunner().invoke(
                cli.cli, ["check-results", "--date", DAY, "--picks-dir", str(pdir)],
                catch_exceptions=False,
            )
            assert cr.exit_code == 0, cr.output
        assert picks.load_pick(DAY, pdir).result == "hit"
        assert picks.load_pick(DAY, pdir).automatic_reading == old_rec
        assert picks.load_shadow_pick(DAY, pdir).result == "hit"
        assert picks.load_shadow_pick(DAY, pdir).automatic_reading == old_rec

        # reconcile_results
        with patch.object(picks, "resolve_daily_slot_results", lambda *_: {"primary": "miss"}):
            flips = picks.reconcile_results(
                pdir, lookback_days=2, clock=lambda: NOW + timedelta(hours=8)
            )
            assert len(flips) == 1
        graded = picks.load_pick(DAY, pdir)
        assert graded.result == "miss" and graded.automatic_reading == old_rec

        # All 4 archive saves
        s_state = scheduler.SchedulerState(
            date=DAY, schedule_fetched_at=NOW.isoformat(), games=[],
            confirmed_game_pks=[], runs_completed=[], pick_locked=False,
            pick_locked_at=None, result_status=None, next_wakeup=None,
        )
        cfg_live = {"orchestrator": {"picks_dir": str(pdir)}, "bluesky": {"dm_recipient": "op.test"}}
        with patch.object(scheduler, "_now_et", lambda: NOW), patch.object(
            alert, "send_dm", lambda *_: "a"
        ), contextlib.redirect_stderr(io.StringIO()):
            picks.save_pick(graded, pdir)
            a_def = scheduler._defer_pick_at_fallback(pdir, DAY, graded, reason="test")
            picks.save_pick(graded, pdir)
            scheduler._refuse_late_delivery(
                graded, cfg_live, pdir, s_state, DAY, "late", NOW - timedelta(minutes=5), NOW
            )
            a_late, = (pdir / DAY).glob("refused_delivery_*.json")
            a_late_data = json.loads(a_late.read_text())
            a_late.unlink()
            picks.save_pick(graded, pdir)
            scheduler._refuse_void_game_delivery(
                graded, cfg_live, pdir, s_state, DAY, "void", {101: "Postponed"}
            )
            a_void, = (pdir / DAY).glob("refused_delivery_*.json")
            a_void_data = json.loads(a_void.read_text())
            picks.save_pick(graded, pdir)
            a_stale = scheduler._archive_and_remove_pick(
                pdir, DAY, graded, prefix="stale_pick", key="stale_pick",
                at_key="staled_at", reason="committed_game_past_cutoff",
            )
        for arch_data, expected_last in [
            (json.loads(a_def.read_text()), "deferred_fallback"),
            (a_late_data, "refused_delivery"),
            (a_void_data, "refused_delivery"),
            (json.loads(a_stale.read_text()), "stale_pick"),
        ]:
            assert list(arch_data)[-3:] == ["chosen_on", "automatic_reading", expected_last]
            assert arch_data["automatic_reading"] == old_rec

        # Both live-forward scripts
        base_doc = {k: v for k, v in asdict(graded).items() if k != "automatic_reading"}
        null_doc = dict(base_doc, automatic_reading=None)
        old_doc = dict(base_doc, automatic_reading=old_rec)
        fresh_doc = dict(base_doc, automatic_reading=fresh_rec)
        changed_old_doc = dict(base_doc, automatic_reading=dict(old_rec, age_hours=20.01))

        lf_dir = root / "lf_picks"
        lf_dir.mkdir()
        manifest_path = root / "manifest.json"
        pick_file = lf_dir / f"{DAY}.json"
        for d_before, d_after, expect_drift in [
            (base_doc, null_doc, False),
            (null_doc, base_doc, False),
            (base_doc, old_doc, True),
            (null_doc, old_doc, True),
            (old_doc, base_doc, True),
            (old_doc, null_doc, True),
            (old_doc, fresh_doc, True),
            (old_doc, changed_old_doc, True),
        ]:
            assert (lf_capture.decision_snapshot(d_before) != lf_capture.decision_snapshot(d_after)) is expect_drift
            assert (lf_inventory._decision_snapshot(d_before) != lf_inventory._decision_snapshot(d_after)) is expect_drift
            raw_before = json.dumps(d_before, indent=2).encode("utf-8")
            raw_after = json.dumps(d_after, indent=2).encode("utf-8")
            sha_before = hashlib.sha256(raw_before).hexdigest()
            sha_after = hashlib.sha256(raw_after).hexdigest()
            assert sha_before != sha_after
            pick_file.write_bytes(raw_after)
            snap = {"present": True, "source_sha256": sha_before, "production_pick_json": d_before}
            manifest_path.write_text(json.dumps({"production_pick_snapshot": snap}))
            cap_state = lf_capture.existing_snapshot_state(
                manifest_path=manifest_path, pick_path=pick_file, expected_date=DAY,
            )
            inv_state = lf_inventory._current_pick_snapshot_summary(
                snapshot=snap, date_key=DAY, picks_dir=lf_dir,
            )
            assert cap_state["current_pick_sha256"] == inv_state["current_pick_sha256"] == sha_after
            assert cap_state["snapshot_matches_current_pick"] is False
            assert inv_state["matches_current_pick"] is False
            assert cap_state["snapshot_decision_matches_current_pick"] is (not expect_drift)
            assert inv_state["decision_matches_current_pick"] is (not expect_drift)

        # Readers across 4 variants
        policy_npz = REPO_ROOT / "data" / "models" / "mdp_policy.npz"
        projected_outputs = []
        for vname, vdoc in [("absent", base_doc), ("null", null_doc), ("fresh", fresh_doc), ("stale", old_doc)]:
            vdir = root / f"readers_{vname}"
            vdir.mkdir()
            raw_v = json.dumps(vdoc, indent=2).encode("utf-8")
            vpath = vdir / f"{DAY}.json"
            vpath.write_bytes(raw_v)
            (vdir / f"{DAY}.shadow.json").write_bytes(raw_v)
            daily_decision.write_decision(
                DAY, vdir, action="single", source="heuristic",
                primary=scheduler._row_from_daily(graded.pick),
                delivery_status="delivered", scoreable=True, now=NOW,
            )
            r_dates = contest_state._resolved_pick_dates(vdir)
            r_latest = contest_state.latest_resolved_pick_date(vdir)
            r_gap = contest_state.resolved_pick_settlement_gap(vdir, date(2027, 7, 14))
            r_miss = contest_state._has_unconfirmed_miss(vdir, date(2027, 7, 14))
            r_replay = picks._replay_season_streak(vdir, 2027)
            r_hist = state_export._pick_to_historical(json.loads(raw_v))
            r_regen = state_regen._hist_to_pick_file(r_hist)
            assert "automatic_reading" not in r_hist and "automatic_reading" not in r_regen

            dp_loaded, held = picks.load_pick_bytes(DAY, vdir)
            rec_obj = entry_receipt.EntryReceipt(
                et_date=DAY, started_at=NOW, clock=lambda: NOW,
                expected_username=None, picks_dir=vdir,
            )
            rec_obj.selection(dp_loaded, held, None, None)
            rec_sel = dict(rec_obj.record()["selection"])
            assert rec_sel.pop("pick_file_sha256") == hashlib.sha256(raw_v).hexdigest()

            led_row, = ledger_picks.parse_pick_file(vpath.name, raw_v).rows
            assert led_row["content_sha256"] == hashlib.sha256(raw_v).hexdigest()
            assert json.loads(led_row["record_raw_json"]) == vdoc
            led_proj = {k: v for k, v in led_row.items() if k not in {"obs_id", "content_sha256", "record_raw_json"}}

            exp_snap = exp_artifacts.load_production_pick_snapshot(vpath, expected_date=DAY)
            assert exp_snap["source_sha256"] == hashlib.sha256(raw_v).hexdigest()
            exp_proj = {k: v for k, v in exp_snap.items() if k not in {"source_sha256", "production_pick_json"}}

            web_picks = web.load_all_picks(vdir)
            web_proj = {k: v for k, v in web_picks[0].items() if k != "automatic_reading"}

            today_d = date(2027, 7, 15)
            h_checks = {
                "post_failure": post_failure.check(vdir, today=today_d, now=NOW + timedelta(hours=5)),
                "late_delivery": late_delivery.check(vdir, today=today_d),
                "fallback_defer": fallback_defer.check(vdir, today=today_d),
                "postponed_pick": postponed_pick.check(vdir, today=today_d),
                "projected_lineup": projected_lineup.check(vdir, today=today_d),
                "same_team_corr": same_team_corr.check(vdir, today=today_d),
                "pitcher_sparsity": pitcher_sparsity.check(vdir, today=today_d),
                "predicted_vs_realized": predicted_vs_realized.check(vdir, today=today_d),
                "mdp_policy_alignment": mdp_policy_alignment.check(
                    vdir, policy_path=policy_npz, today=today_d
                ),
            }
            projected_outputs.append({
                "r_dates": r_dates, "r_latest": r_latest, "r_gap": r_gap, "r_miss": r_miss,
                "r_replay": r_replay, "r_hist": r_hist, "r_regen": r_regen,
                "rec_sel": rec_sel, "led_proj": led_proj, "exp_proj": exp_proj,
                "web_proj": web_proj, "h_checks": h_checks,
            })

        for other in projected_outputs[1:]:
            assert other == projected_outputs[0]

    print("[PASS] Item 4 grading, reconcile, all 4 archives, both live-forward scripts, and 20+ readers verified.")


def _test_item_5_warning_and_item_6_run_day() -> None:
    import pandas as pd
    from click.testing import CliRunner
    from bts import cli, contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
    from bts.health import alert
    from bts.model import predict
    import probe_1_2_7_delivery_stamp_row8 as p1

    # 1. Item 5: .2f formatting edges (14.00, 20.50, 20.12), future, unreadable,
    #    decoupled alert.datetime vs warning_day dedup, retry without resurface,
    #    critical status independence, legacy fresh+reuse no-warn on 20h old auto!
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # (a) .2f formatting check on 14.0h + 1us, 20.5h, 20.12h, future, unreadable
        for label, delta, raw_override, exp_detail_fn in [
            ("old_14_00", timedelta(hours=14, microseconds=1), None,
             lambda r: f"last accepted fetch {r}, about 14.00 hours ago"),
            ("old_20_50", timedelta(hours=20, minutes=30), None,
             lambda r: f"last accepted fetch {r}, about 20.50 hours ago"),
            ("old_20_12", timedelta(seconds=int(20.12 * 3600)), None,
             lambda r: f"last accepted fetch {r}, about 20.12 hours ago"),
            ("future", timedelta(minutes=-10), None,
             lambda r: f"its recorded_at {r} is in the future"),
            ("unreadable", None, "unparseable-time",
             lambda _r: "it has no usable recorded_at"),
        ]:
            sub = root / label / "picks"
            sub.mkdir(parents=True)
            rec = raw_override if raw_override is not None else (NOW - delta).isoformat()
            p1._seed_reading(sub, "old_auto", now=NOW)
            (sub / "account_state" / "contest_streak.json").write_text(json.dumps({
                "schema_version": "bts_contest_streak_auto_v2",
                "season": 2027, "activation_id": "a" * 32,
                "active_streak": 10, "best_streak": 15,
                "source": "mlb_bts_profile", "source_date": DAY,
                "recorded_at": rec,
            }))
            pk = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                            0.82, [], False, 101, f"{DAY}T23:10:00Z")
            dp = picks.DailyPick(DAY, NOW.isoformat(), pk, None, None)
            pol = strategy.PolicyDecision(
                "single", "single", "heuristic", "reach57", 10, 80, None, "missing",
                None, None, None, None, None,
            )
            sel = strategy.SelectionResult(strategy.PickResult(dp), "single", "heuristic",
                                           None, None, None, streak=10, decision=pol)
            cfg = {
                "orchestrator": {"picks_dir": str(sub)},
                "bluesky": {"dm_recipient": "op.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm",
                              "live_forward_capture_on_lock": False},
            }
            st = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
            sent_warns = []
            err = io.StringIO()

            class DayClock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

            # Deliberately pin alert.datetime to a DIFFERENT year (2028) to prove now_et_date=warning_day!
            class WrongAlertClock(datetime):
                @classmethod
                def now(cls, tz=None):
                    wrong = datetime(2028, 1, 1, 12, 0, tzinfo=ET)
                    return wrong.astimezone(tz) if tz else wrong.replace(tzinfo=None)

            with patch.object(contest_state, "datetime", DayClock), patch.object(
                alert, "datetime", WrongAlertClock
            ), patch.object(
                scheduler, "_now_et", lambda: NOW
            ), patch.object(
                scheduler, "_delivery_game_statuses", lambda _d: {}
            ), patch.object(
                dm, "send_dm", lambda *_: "dm-ok"
            ), patch.object(
                alert, "send_dm", lambda r, b: (sent_warns.append((r, b)), "w-ok")[1]
            ), contextlib.redirect_stderr(err):
                assert scheduler._deliver_and_lock_pick(dp, cfg, sub, st, DAY, "fmt", selection=sel) is True

            exp_detail = exp_detail_fn(rec)
            assert sent_warns == [("op.test", _env(DAY, exp_detail))], f"{label}: {sent_warns}"
            assert err.getvalue().splitlines().count(WARN_PREFIX + _msg(DAY, exp_detail)) == 1
            st_doc = json.loads((sub.parent / "health_state" / WARN_FILE).read_text())
            assert st_doc["sent_day"] == DAY, f"now_et_date not honored: {st_doc['sent_day']}"
            assert st_doc["sent_sources"] == [WARN_KEY]

        # (b) Legacy commands (bts preview, bts run, bts orchestrate) on a 20h-old stale automatic reading
        #     making FRESH choices: neither stamp automatic_reading nor warn!
        leg_dir = root / "legacy_fresh" / "picks"
        leg_dir.mkdir(parents=True)
        p1._seed_reading(leg_dir, "old_auto", now=NOW, streak=10)
        preds = pd.DataFrame([{
            "batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
            "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
            "p_hit_pa": 0.24, "p_game_hit": 0.99, "flags": "", "projected_lineup": False,
            "game_pk": 101, "game_time": f"{DAY}T23:10:00Z",
        }])
        toml_leg = root / "leg.toml"
        toml_leg.write_text(
            'tiers = []\n[orchestrator]\n'
            f'picks_dir = {json.dumps(str(leg_dir))}\n'
            f'models_dir = {json.dumps(str(root / "models"))}\n'
            '[bluesky]\ndm_recipient = "op.test"\n'
            '[scheduler]\ncontest_state_required = true\npick_delivery = "public"\n'
        )
        for cmd in ("preview", "run", "orchestrate"):
            if (leg_dir / f"{DAY}.json").exists():
                (leg_dir / f"{DAY}.json").unlink()
            warn_mock = Mock(return_value="w")
            arr_spy = Mock(side_effect=AssertionError("legacy command called automatic_reading_record"))
            with patch.object(contest_state, "datetime", DayClock), patch.object(
                cli, "_now_et", lambda: NOW
            ), patch.object(
                predict, "run_pipeline", Mock(return_value=preds)
            ), patch.object(
                orchestrator, "run_cascade", Mock(return_value=(preds, "offline"))
            ), patch.object(
                picks, "get_game_statuses_detailed",
                lambda *_: {101: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}
            ), patch.object(
                strategy, "get_game_statuses_detailed",
                lambda *_: {101: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}
            ), patch.object(
                posting, "post_to_bluesky", Mock(return_value="at://leg")
            ), patch.object(
                posting, "should_post_now", Mock(return_value=True)
            ), patch.object(
                alert, "send_dm", warn_mock
            ), patch.object(
                contest_state, "automatic_reading_record", arr_spy
            ):
                args = (
                    ["orchestrate", "--date", DAY, "--config", str(toml_leg)]
                    if cmd == "orchestrate"
                    else [cmd, "--date", DAY, "--picks-dir", str(leg_dir), "--models-dir", str(root / "models")]
                )
                r_cli = CliRunner().invoke(cli.cli, args, catch_exceptions=False)
                assert r_cli.exit_code == 0, r_cli.output
                assert WARN_PREFIX not in r_cli.stderr
            assert not warn_mock.called
            assert not arr_spy.called
            assert picks.load_pick(DAY, leg_dir).automatic_reading is None

    # 2. Item 6: All 5 scenarios through the real scheduler.run_day!
    def run_real_day_scenario(mode_case: str) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdir = root / "picks"
            pdir.mkdir(parents=True)

            # One game at 23:10Z (19:10 ET) and a second game at 23:55Z (19:55 ET)
            games = [
                {
                    "gamePk": 101, "gameDate": f"{DAY}T23:10:00Z",
                    "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                    "teams": {"away": {"team": {"name": "Away 101"}}, "home": {"team": {"name": "Home 101"}}},
                },
                {
                    "gamePk": 102, "gameDate": f"{DAY}T23:55:00Z",
                    "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                    "teams": {"away": {"team": {"name": "Away 102"}}, "home": {"team": {"name": "Home 102"}}},
                },
            ]

            now_box = [datetime(2027, 7, 15, 16, 10, 0, tzinfo=ET)]  # Thin-slate early check 16:10 ET!
            if mode_case != "turns_stale_at_fallback":
                now_box[0] = NOW  # 18:35 ET

            class SimClock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return now_box[0].astimezone(tz) if tz else now_box[0].replace(tzinfo=None)

            if mode_case in {"old_20h", "restart_after_delivery"}:
                rec = p1._seed_reading(pdir, "old_auto", now=NOW, streak=30)
            elif mode_case == "fresh_2h":
                rec = p1._seed_reading(pdir, "fresh_auto", now=NOW, streak=30)
            elif mode_case == "dated_obs":
                rec = p1._seed_reading(pdir, "dated", now=NOW, streak=30)
            elif mode_case in {"turns_stale_at_fallback", "turns_stale_reuse_at_fallback"}:
                # Recorded at 04:10 ET: at 16:10 ET choice it is 12.0h old (FRESH!);
                # at 18:35 ET fallback delivery it is 14.42h old (STALE!)!
                rec = datetime(2027, 7, 15, 4, 10, 0, tzinfo=ET).isoformat()
                p1._seed_reading(pdir, "fresh_auto", now=now_box[0], streak=30)
                (pdir / "account_state" / "contest_streak.json").write_text(json.dumps({
                    "schema_version": "bts_contest_streak_auto_v2",
                    "season": 2027, "activation_id": "a" * 32,
                    "active_streak": 30, "best_streak": 30,
                    "source": "mlb_bts_profile", "source_date": DAY,
                    "recorded_at": rec,
                }))
            else:
                raise ValueError(mode_case)

            if mode_case in {"turns_stale_at_fallback", "turns_stale_reuse_at_fallback"}:
                # At 16:10 ET check: game 101 confirmed (0.98), game 102 PROJECTED contender (0.97 within 0.03 gap!)
                # so run_single_check chooses Steven Kwan & stamps chosen_on, saves pick, should_lock=False!
                early_preds = pd.DataFrame([
                    {"batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
                     "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
                     "p_hit_pa": 0.28, "p_game_hit": 0.98, "flags": "", "projected_lineup": False,
                     "game_pk": 101, "game_time": f"{DAY}T23:10:00Z"},
                    {"batter_id": 665742, "batter_name": "Juan Soto", "team": "NYM", "lineup": 2,
                     "pitcher_id": 605483, "pitcher_name": "Blake Snell", "opp": "LAD",
                     "p_hit_pa": 0.27, "p_game_hit": 0.97, "flags": "PROJECTED", "projected_lineup": True,
                     "game_pk": 102, "game_time": f"{DAY}T23:55:00Z"},
                ])
                cascade_calls = 0

                def cascade_fn(*_a, **_kw):
                    nonlocal cascade_calls
                    cascade_calls += 1
                    if cascade_calls == 1:
                        return early_preds, "offline"
                    if mode_case == "turns_stale_at_fallback":
                        raise RuntimeError("fallback cascade failed -> cached pick with selection=None")
                    return early_preds, "offline"

                def fake_sleep(secs):
                    if secs > 0:
                        now_box[0] = now_box[0] + timedelta(seconds=secs)
            else:
                preds = pd.DataFrame([
                    {"batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
                     "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
                     "p_hit_pa": 0.28, "p_game_hit": 0.99, "flags": "", "projected_lineup": False,
                     "game_pk": 101, "game_time": f"{DAY}T23:10:00Z"},
                ])
                cascade_fn = lambda *_a, **_kw: (preds, "offline")
                fake_sleep = lambda _s: None

            cfg = {
                "orchestrator": {
                    "picks_dir": str(pdir),
                    "heartbeat_path": str(root / "hb"),
                    "models_dir": str(REPO_ROOT / "data" / "models"),
                },
                "bluesky": {"dm_recipient": "op.test"},
                "scheduler": {
                    "contest_state_required": True,
                    "pick_delivery": "dm",
                    "entry_intent": "enter",
                    "live_forward_capture_on_lock": False,
                },
                "health_checks": {"enabled": False},
                "tiers": [],
            }

            pick_dms = []
            warn_dms = []
            err_buf = io.StringIO()
            statuses = {pk: {"abstract": "P", "detailed": "Scheduled", "code": "S"} for pk in (101, 102)}
            if mode_case == "turns_stale_reuse_at_fallback":
                # Keep Scheduled on call 1, switch to Warmup after sleep so fallback select_pick reuses!
                pass

            def status_detailed_fn(*_a):
                if mode_case == "turns_stale_reuse_at_fallback" and now_box[0] > datetime(2027, 7, 15, 18, 0, tzinfo=ET):
                    return {pk: {"abstract": "P", "detailed": "Warmup", "code": "PW"} for pk in (101, 102)}
                return statuses

            with patch.object(contest_state, "datetime", SimClock), patch.object(
                daily_decision, "datetime", SimClock
            ), patch.object(
                alert, "datetime", SimClock
            ), patch.object(
                scheduler, "_now_et", lambda: now_box[0]
            ), patch.object(
                scheduler, "fetch_schedule", lambda d: games if d == DAY else []
            ), patch.object(
                scheduler, "check_confirmed_lineups", lambda *_: {101: {"away", "home"}}
            ), patch.object(
                scheduler, "_idle_until_next_wakeup", lambda *_: None
            ), patch.object(
                scheduler, "_watchdog_ping_sleep", fake_sleep
            ), patch.object(
                scheduler, "heartbeat_watchdog", lambda *a, **k: contextlib.nullcontext()
            ), patch.object(
                scheduler, "run_result_polling", lambda *a, **k: "final"
            ), patch.object(
                orchestrator, "run_cascade", cascade_fn
            ), patch.object(
                picks, "get_game_statuses_detailed", status_detailed_fn
            ), patch.object(
                strategy, "get_game_statuses_detailed", status_detailed_fn
            ), patch.object(
                picks, "get_game_statuses", lambda *_: {101: "P", 102: "P"}
            ), patch.object(
                dm, "send_dm", lambda r, b: (pick_dms.append((r, b)), "pick-dm-1")[1]
            ), patch.object(
                posting, "post_to_bluesky", Mock(side_effect=AssertionError("unexpected public post"))
            ), patch.object(
                alert, "send_dm", lambda r, b: (warn_dms.append((r, b)), "warn-dm-1")[1]
            ), contextlib.redirect_stderr(err_buf):
                scheduler.run_day(date=DAY, config=cfg)

                if mode_case == "restart_after_delivery":
                    # Run run_day a SECOND time on the same picks_dir to verify restart behavior!
                    err_restart = io.StringIO()
                    with contextlib.redirect_stderr(err_restart):
                        scheduler.run_day(date=DAY, config=cfg)
                    assert WARN_PREFIX not in err_restart.getvalue()

            final_pick = picks.load_pick(DAY, pdir)
            assert final_pick is not None and final_pick.notification_sent is True
            assert scheduler.load_state(DAY, pdir).pick_locked is True
            assert daily_decision.load_decision(DAY, pdir)["delivery_status"] == "delivered"
            assert len(pick_dms) == 1, f"{mode_case}: expected 1 pick DM, got {len(pick_dms)}"

            warn_status_file = pdir.parent / "health_state" / WARN_FILE
            if mode_case in {"old_20h", "restart_after_delivery"}:
                assert final_pick.automatic_reading == {
                    "schema": SCHEMA, "recorded_at": rec, "age_hours": 20.0, "stale": True, "reason": "old"
                }
                assert len(warn_dms) == 1
                assert WARN_PREFIX in err_buf.getvalue()
                assert warn_status_file.exists()
            elif mode_case == "fresh_2h":
                assert final_pick.automatic_reading == {
                    "schema": SCHEMA, "recorded_at": rec, "age_hours": 3.0, "stale": False, "reason": None
                }
                assert warn_dms == []
                assert WARN_PREFIX not in err_buf.getvalue()
                assert not warn_status_file.exists()
            elif mode_case == "dated_obs":
                assert final_pick.automatic_reading is None
                assert warn_dms == []
                assert WARN_PREFIX not in err_buf.getvalue()
                assert not warn_status_file.exists()
            elif mode_case in {"turns_stale_at_fallback", "turns_stale_reuse_at_fallback"}:
                assert final_pick.chosen_on is not None and final_pick.chosen_on["kind"] == "automatic"
                assert final_pick.automatic_reading is not None
                assert final_pick.automatic_reading["stale"] is True
                assert final_pick.automatic_reading["reason"] == "old"
                assert final_pick.automatic_reading["age_hours"] > 14.0
                assert len(warn_dms) == 1
                assert f"about {final_pick.automatic_reading['age_hours']:.2f} hours ago" in warn_dms[0][1]
                assert WARN_PREFIX in err_buf.getvalue()

    for sc_name in (
        "old_20h", "fresh_2h", "dated_obs",
        "turns_stale_at_fallback", "turns_stale_reuse_at_fallback",
        "restart_after_delivery",
    ):
        run_real_day_scenario(sc_name)

    print("[PASS] Item 5 (exact warning copy, .2f edges, decoupled clock dedup, legacy commands) and Item 6 (6 real run_day scenarios) verified.")


def main() -> None:
    with forbid_network(allow_loopback=False):
        _finish_probe_1_verification()
        _finish_probe_2_resaves_and_readers()
        _test_item_5_warning_and_item_6_run_day()
    print("PROBE_3_ALL_PASSED")


if __name__ == "__main__":
    main()
