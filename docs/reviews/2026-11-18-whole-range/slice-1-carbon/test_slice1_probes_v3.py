"""Slice 1 adversarial counterexample probes v3 (ready for adoption into tests/).

Each ``test_probe_*`` sets up a concrete, realistic state and input sequence on
the real scheduler / CLI / health / scorer code, captures all transports and
state files, and records exact evidence of the defect.

Regression-test hygiene:
- No module-level import of ``bts.model.predict`` or ``lightgbm``.
- No process-wide state left modified on exit.
- No network sockets permitted while active.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch


def _load_v3_harness():
    harness_path = Path(__file__).resolve().with_name("rehearsal_v3_harness.py")
    spec = importlib.util.spec_from_file_location("_wr_slice1_v3_harness_probes_v3", harness_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wr_slice1_v3_harness_probes_v3"] = mod
    spec.loader.exec_module(mod)
    return mod


_H = _load_v3_harness()
RehearsalWorld = _H.RehearsalWorld
SimulatedBoxRestart = _H.SimulatedBoxRestart
et_dt = _H.et_dt

EVIDENCE: dict[str, Any] = {}


def _is_pick_dm(dm: dict[str, Any]) -> bool:
    txt = dm["text"]
    return (
        not txt.startswith("BTS health ")
        and not txt.startswith("BTS day check ")
        and not txt.startswith("BTS fetch-contest-streak ")
        and not txt.startswith("⚠️")
        and not txt.startswith("⏰")
        and not txt.startswith("✅")
        and "No pick" not in txt
    )


def _setup_revoked_box(w) -> None:
    w.configure_live_dm(contest_state_required=True, contest_state_expected=True)
    r = w.invoke_cli([
        "season-contract", "revoke",
        "--season", "2027",
        "--reason", "restored from the offseason snapshot",
        "--picks-dir", str(w.picks_dir),
    ])
    assert r.exit_code == 0, r.stderr


def test_probe_1_post_midnight_eod_false_critical_and_harmful_advice(tmp_path: Path) -> None:
    """Opening Night game ending at 00:15 ET triggers false CRITICAL + harmful instructions."""
    sched = _H.default_opening_days_schedules()
    g = sched["2027-03-24"][0]
    g.start_et = et_dt(2027, 3, 24, 21, 5)
    g.away_confirm_et = et_dt(2027, 3, 24, 18, 45)
    g.home_confirm_et = et_dt(2027, 3, 24, 19, 45)
    g.final_et = et_dt(2027, 3, 25, 0, 15)
    g.away_hits = 0

    with RehearsalWorld(tmp_path / "p1", initial_et=et_dt(2027, 3, 24, 9, 0), schedules=sched) as w:
        _setup_revoked_box(w)
        r_sv = w.invoke_cli([
            "saver-state", "--init", "not_earned", "--season", "2027", "--picks-dir", str(w.picks_dir),
        ])
        assert r_sv.exit_code == 0
        (w.park_dir / "park_drag_manifest.json").write_text(json.dumps({
            "max_source_game_date": "2027-03-24",
            "generated_at": "2027-03-25T00:10:00",
            "export_rows": 2,
        }))

        w.clock.set_et(et_dt(2027, 3, 24, 9, 30))
        assert w.invoke_cli([
            "set-contest-streak", "--date", "2027-03-24",
            "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
            "--picks-dir", str(w.picks_dir),
        ]).exit_code == 0

        w.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        w.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 25, 1, 0))

        eod_dms = [m for m in w.dms if m["text"].startswith("BTS health ")]
        assert len(eod_dms) == 1
        dm_body = eod_dms[0]["text"]
        assert "[contest_state] Live decision refused: season 2027 switch is revoked." in dm_body
        assert "--date 2027-03-25" in dm_body
        assert "contradicts streak 12" in dm_body
        assert "correct the contest streak with bts set-contest-streak" in dm_body

        w.clock.set_et(et_dt(2027, 3, 25, 0, 25))
        r_legacy = w.invoke_cli([
            "set-contest-streak", "--streak", "0", "--picks-dir", str(w.picks_dir),
        ])
        assert r_legacy.exit_code == 0
        w.invoke_cli(["season-contract", "enable", "--season", "2027", "--picks-dir", str(w.picks_dir)])
        w.invoke_cli(["fetch-contest-streak", "--picks-dir", str(w.picks_dir), "--expected-username", "stonehengee"])
        r_status = w.invoke_cli(["season-contract", "status", "--picks-dir", str(w.picks_dir)])
        assert "the selected legacy manual override is no longer accepted for a live pick" in r_status.stdout

        EVIDENCE["probe_1_post_midnight_eod"] = {
            "eod_dm_at_et": eod_dms[0]["at_et"],
            "eod_dm_body": dm_body,
            "status_after_following_advice": r_status.stdout,
        }


def test_probe_2_midsend_restart_silently_locks_without_critical_dm(tmp_path: Path) -> None:
    """Crash mid-send before send_dm reaches Bluesky leaves 0 DMs all day on pre-22:00 ET games."""
    sched = _H.default_opening_days_schedules()
    g = sched["2027-03-24"][0]
    g.start_et = et_dt(2027, 3, 24, 16, 10)
    g.away_confirm_et = et_dt(2027, 3, 24, 13, 50)
    g.home_confirm_et = et_dt(2027, 3, 24, 14, 50)
    g.final_et = et_dt(2027, 3, 24, 19, 0)

    with RehearsalWorld(tmp_path / "p2", initial_et=et_dt(2027, 3, 24, 9, 30), schedules=sched) as w:
        _setup_revoked_box(w)
        (w.park_dir / "park_drag_manifest.json").write_text(json.dumps({
            "max_source_game_date": "2027-03-24",
            "generated_at": "2027-03-24T07:45:10",
            "export_rows": 2,
        }))
        w.invoke_cli([
            "set-contest-streak", "--date", "2027-03-24",
            "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
            "--picks-dir", str(w.picks_dir),
        ])

        crashed_once = {"done": False}

        def _crash_before_wire(recipient: str, text: str) -> str | None:
            if not crashed_once["done"] and not text.startswith("BTS "):
                crashed_once["done"] = True
                raise SimulatedBoxRestart()
            return None

        w.dm_hook = _crash_before_wire
        w.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        try:
            w.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 24, 23, 30))
            raise AssertionError("Expected SimulatedBoxRestart during pick send")
        except SimulatedBoxRestart:
            pass

        mid_pick = w.read_pick_json("2027-03-24")
        assert mid_pick["delivery_attempted"] is True and mid_pick["notification_sent"] is False
        assert len(w.dms) == 0

        w.clock.advance_seconds(30)
        w.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 24, 23, 30))

        dec = w.read_decision_json("2027-03-24")
        st = w.read_streak_json()
        assert len(w.dms) == 0
        assert dec["delivery_status"] == "locked_unconfirmed" and dec["scoreable"] is True
        assert st["streak"] == 1 and st["season"] == 2027

        EVIDENCE["probe_2_midsend_restart"] = {
            "dms_sent_all_day": len(w.dms),
            "decision_json": dec,
            "streak_json": st,
        }


def test_probe_3_undelivered_preview_poisons_reconcile_season_replay(tmp_path: Path) -> None:
    """Undelivered 2027-03-24.json (result=None) causes _replay_season_streak to return None forever."""
    with RehearsalWorld(tmp_path / "p3", initial_et=et_dt(2027, 3, 24, 3, 0)) as w:
        _setup_revoked_box(w)
        assert w.invoke_cli([
            "preview", "--date", "2027-03-24",
            "--picks-dir", str(w.picks_dir), "--models-dir", str(w.models_dir),
        ]).exit_code == 0

        w.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        w.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 25, 0, 30))
        w.clock.set_et(et_dt(2027, 3, 25, 1, 0))
        w.invoke_cli(["check-results", "--date", "2027-03-24", "--picks-dir", str(w.picks_dir)])
        assert w.read_pick_json("2027-03-24")["result"] is None

        for g in w.schedules["2027-03-25"][1:]:
            g.away_p_hit = 0.60
            g.home_p_hit = 0.58
        w.clock.set_et(et_dt(2027, 3, 25, 9, 30))
        w.invoke_cli([
            "set-contest-streak", "--date", "2027-03-25",
            "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
            "--picks-dir", str(w.picks_dir),
        ])
        w.clock.set_et(et_dt(2027, 3, 25, 10, 0))
        w.run_scheduler_day("2027-03-25", stop_at_et=et_dt(2027, 3, 26, 0, 30))
        assert w.read_pick_json("2027-03-25")["result"] == "hit"
        assert w.read_streak_json()["streak"] == 2

        w.schedules["2027-03-25"][0].away_hits = 0
        w.schedules["2027-03-25"][0].away_at_bats = 4
        w.clock.set_et(et_dt(2027, 3, 26, 7, 40))
        r_rec = w.invoke_cli(["reconcile", "--picks-dir", str(w.picks_dir)])
        assert r_rec.exit_code == 0
        assert "CORRECTIONS FOUND (1)" in r_rec.stdout
        assert w.read_pick_json("2027-03-25")["result"] == "miss"
        assert w.read_streak_json()["streak"] == 2

        (w.picks_dir / "streak.json").write_text("{corrupt")
        r_rec2 = w.invoke_cli(["reconcile", "--picks-dir", str(w.picks_dir)], catch_exceptions=True)
        assert r_rec2.exit_code != 0
        assert (w.picks_dir / "streak.json").read_text() == "{corrupt"

        EVIDENCE["probe_3_reconcile_poisoned"] = {
            "reconcile_stdout": r_rec.stdout,
            "day1_result": w.read_pick_json("2027-03-24")["result"],
            "day2_result_after_reconcile": w.read_pick_json("2027-03-25")["result"],
            "streak_json_stays_wrong": 2,
            "corrupt_reconcile_exception": type(r_rec2.exception).__name__,
            "corrupt_streak_not_rebuilt": True,
        }


def test_probe_4_missing_lock_decision_double_cascades_past_cutoff(tmp_path: Path) -> None:
    """run_single_check omits lock_decision, so both normal overrun and late restart run 2 cascades past cutoff."""
    sched_a = _H.default_opening_days_schedules()
    sched_a["2027-03-24"][0].away_confirm_et = None
    sched_a["2027-03-24"][0].home_confirm_et = None

    with RehearsalWorld(tmp_path / "p4a", initial_et=et_dt(2027, 3, 24, 9, 30), schedules=sched_a) as wa:
        _setup_revoked_box(wa)
        wa.invoke_cli([
            "set-contest-streak", "--date", "2027-03-24",
            "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
            "--picks-dir", str(wa.picks_dir),
        ])
        wa.clock.schedule_at(
            et_dt(2027, 3, 24, 19, 0), "slow_cascade_window",
            lambda: setattr(wa, "cascade_duration_sec", 28 * 60),
        )
        wa.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        wa.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 24, 23, 30))
        assert len([m for m in wa.dms if _is_pick_dm(m)]) == 0
        assert any("LATE DELIVERY REFUSED" in m["text"] for m in wa.dms)

    sched_b = _H.default_opening_days_schedules()
    sched_b["2027-03-24"][0].away_confirm_et = None
    sched_b["2027-03-24"][0].home_confirm_et = None

    with RehearsalWorld(tmp_path / "p4b", initial_et=et_dt(2027, 3, 24, 9, 30), schedules=sched_b) as wb:
        _setup_revoked_box(wb)
        wb.cascade_duration_sec = 16 * 60
        wb.invoke_cli([
            "set-contest-streak", "--date", "2027-03-24",
            "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
            "--picks-dir", str(wb.picks_dir),
        ])
        wb.clock.set_et(et_dt(2027, 3, 24, 19, 32))
        wb.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 24, 23, 30))
        assert len(wb.cascade_calls) == 2
        assert len([m for m in wb.dms if _is_pick_dm(m)]) == 0
        assert any("LATE DELIVERY REFUSED" in m["text"] for m in wb.dms)

    EVIDENCE["probe_4_double_cascade_cutoff_miss"] = {
        "no_restart_overrun_cascades": wa.cascade_calls,
        "no_restart_dms": [m["text"] for m in wa.dms],
        "restart_cascades": wb.cascade_calls,
        "restart_dms": [m["text"] for m in wb.dms],
    }


def test_probe_5_status_masks_unfetched_switch_when_dated_observation_present(tmp_path: Path) -> None:
    """season-contract status prints 'live decision now: accepted; source=contest' while auto_v1 still on disk."""
    with RehearsalWorld(tmp_path / "p5", initial_et=et_dt(2027, 3, 26, 9, 30)) as w:
        _setup_revoked_box(w)
        w.invoke_cli([
            "set-contest-streak", "--date", "2027-03-26",
            "--streak", "3", "--best-streak", "3", "--saver", "not_earned",
            "--picks-dir", str(w.picks_dir),
        ])
        w.clock.set_et(et_dt(2027, 3, 26, 11, 0))
        w.invoke_cli(["season-contract", "enable", "--season", "2027", "--picks-dir", str(w.picks_dir)])
        r_status_day3 = w.invoke_cli(["season-contract", "status", "--picks-dir", str(w.picks_dir)])
        assert "live decision now: accepted; source=contest; streak=3;" in r_status_day3.stdout
        assert "identifier matches switch" not in r_status_day3.stdout

        w.clock.set_et(et_dt(2027, 3, 27, 9, 0))
        r_status_day4 = w.invoke_cli(["season-contract", "status", "--picks-dir", str(w.picks_dir)])
        assert "live decision now: refused; Live decision refused: schema_version is 'bts_contest_streak_auto_v1'" in r_status_day4.stdout

        EVIDENCE["probe_5_status_masked_by_dated"] = {
            "day3_status": r_status_day3.stdout,
            "day4_status": r_status_day4.stdout,
        }


def test_probe_6_early_contest_state_dm_suppresses_later_cs5a_saved_pick_refusal_dm(tmp_path: Path) -> None:
    """17:05 missing-state CRITICAL consumes CRITICAL:contest_state, suppressing 19:30 CS5a refusal DM."""
    sched = _H.default_opening_days_schedules()
    sched["2027-03-24"][0].away_confirm_et = None
    sched["2027-03-24"][0].home_confirm_et = None

    with RehearsalWorld(tmp_path / "p6", initial_et=et_dt(2027, 3, 24, 10, 0), schedules=sched) as w:
        _setup_revoked_box(w)
        w.clock.schedule_at(
            et_dt(2027, 3, 24, 17, 15), "enter_wrong_streak_5",
            lambda: w.invoke_cli(["set-contest-streak", "--date", "2027-03-24", "--streak", "5", "--best-streak", "5", "--saver", "not_earned", "--picks-dir", str(w.picks_dir)]),
        )

        def _correct_and_fail() -> None:
            w.invoke_cli(["set-contest-streak", "--date", "2027-03-24", "--streak", "0", "--best-streak", "0", "--saver", "not_earned", "--picks-dir", str(w.picks_dir)])
            w.prediction_override = lambda _d: None

        w.clock.schedule_at(et_dt(2027, 3, 24, 18, 20), "correct_to_0_and_fail_cascade", _correct_and_fail)
        w.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 24, 21, 0))

        assert len([m for m in w.dms if _is_pick_dm(m)]) == 0
        assert not any("Saved pick for 2027-03-24 was not delivered" in m["text"] for m in w.dms)

        EVIDENCE["probe_6_cs5a_dm_suppressed"] = {
            "dms": [m["text"] for m in w.dms],
        }


def test_probe_7_orchestrate_straddling_midnight_raises_uncaught_contest_state_error(tmp_path: Path) -> None:
    """bts orchestrate catches ContestStateError on run_and_pick but NOT on post-save load_decision_streak_state."""
    sched = _H.default_opening_days_schedules()
    # Late rain-delayed / West-Coast game on 2027-03-24 slate starting at 00:35 ET (still Scheduled at 23:59:50 ET)
    sched["2027-03-24"][0].start_et = et_dt(2027, 3, 25, 0, 35)
    sched["2027-03-24"][0].final_et = et_dt(2027, 3, 25, 3, 0)

    with RehearsalWorld(tmp_path / "p7", initial_et=et_dt(2027, 3, 24, 23, 59, 50), schedules=sched) as w:
        _setup_revoked_box(w)
        w.invoke_cli([
            "set-contest-streak", "--date", "2027-03-24",
            "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
            "--picks-dir", str(w.picks_dir),
        ])
        import bts.picks as picks_mod
        real_save_pick = picks_mod.save_pick

        def _save_and_tick_past_midnight(daily, picks_dir):
            out = real_save_pick(daily, picks_dir)
            w.clock.set_et(et_dt(2027, 3, 25, 0, 0, 2))
            return out

        with patch.object(picks_mod, "save_pick", _save_and_tick_past_midnight):
            r = w.invoke_cli(
                ["orchestrate", "--date", "2027-03-24", "--config", str(w.toml_path)],
                catch_exceptions=True,
            )
        assert r.exit_code != 0
        assert r.exception is not None and type(r.exception).__name__ == "ContestStateError"
        assert w.read_pick_json("2027-03-24") is not None

        EVIDENCE["probe_7_orchestrate_midnight"] = {
            "exception_type": type(r.exception).__name__,
            "pick_saved_unposted": w.read_pick_json("2027-03-24")["bluesky_posted"] is False,
        }


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    runs_root = out_dir / "_runs_probes_v3"
    runs_root.mkdir(parents=True, exist_ok=True)
    probes = [
        ("probe_1", test_probe_1_post_midnight_eod_false_critical_and_harmful_advice),
        ("probe_2", test_probe_2_midsend_restart_silently_locks_without_critical_dm),
        ("probe_3", test_probe_3_undelivered_preview_poisons_reconcile_season_replay),
        ("probe_4", test_probe_4_missing_lock_decision_double_cascades_past_cutoff),
        ("probe_5", test_probe_5_status_masks_unfetched_switch_when_dated_observation_present),
        ("probe_6", test_probe_6_early_contest_state_dm_suppresses_later_cs5a_saved_pick_refusal_dm),
        ("probe_7", test_probe_7_orchestrate_straddling_midnight_raises_uncaught_contest_state_error),
    ]
    with tempfile.TemporaryDirectory(dir=runs_root) as td:
        base = Path(td)
        for idx, (name, fn) in enumerate(probes, start=1):
            sub = base / f"{idx:02d}_{name}"
            sub.mkdir(parents=True, exist_ok=True)
            print(f"[RUN] {name} ...", flush=True)
            fn(sub)
            print(f"[PASS] {name}", flush=True)
    ev_path = out_dir / "adversarial_probes_evidence.json"
    ev_path.write_text(json.dumps(EVIDENCE, indent=2) + "\n", encoding="utf-8")
    print(f"ALL ADVERSARIAL PROBES VERIFIED -> {ev_path}")
