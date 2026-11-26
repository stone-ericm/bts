# VARIANT (closing test re-run, FINAL2): the two skip-notice wording asserts follow Ea's text; nothing else changed.
"""Slice 1 end-to-end rehearsal suite (ready for adoption into tests/).

Exercises all 2027 season-start scenarios (a)-(h) and checklist rows A5, B7, B8a
starting from the BASESHA (619abda7) restored snapshot state, using the
self-contained ``rehearsal_v3_harness.py``.

Regression-test hygiene:
- No module-level import of ``bts.model.predict`` or ``lightgbm``.
- No process-wide state left modified on exit.
- No network sockets permitted while active.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any


def _load_v3_harness():
    harness_path = Path(__file__).resolve().with_name("rehearsal_v3_harness.py")
    spec = importlib.util.spec_from_file_location("_wr_slice1_v3_harness_final", harness_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wr_slice1_v3_harness_final"] = mod
    spec.loader.exec_module(mod)
    return mod


_H = _load_v3_harness()
RehearsalWorld = _H.RehearsalWorld
SimulatedBoxRestart = _H.SimulatedBoxRestart
et_dt = _H.et_dt
install_cron_shims = _H.install_cron_shims
write_units_snapshot = _H.write_units_snapshot

TELEMETRY_LOG: dict[str, Any] = {}


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


def _prepare_revoked_live_box(w) -> dict[str, Any]:
    r_init = w.invoke_cli(["entry-intent", "--config", str(w.toml_path)])
    assert r_init.exit_code == 0 and r_init.stdout.strip() == "research"

    w.configure_live_dm(contest_state_required=None, contest_state_expected=None)
    r_bad_intent = w.invoke_cli(["entry-intent", "--config", str(w.toml_path)])
    assert r_bad_intent.exit_code == 1
    assert "require contest state" in r_bad_intent.stderr

    r_bad_sched = w.invoke_cli(["schedule", "--config", str(w.toml_path), "--date", "2027-03-24"])
    assert r_bad_sched.exit_code == 2
    assert "refusing to start" in r_bad_sched.stderr

    w.configure_live_dm(contest_state_required=True, contest_state_expected=True)
    r_ok_intent = w.invoke_cli(["entry-intent", "--config", str(w.toml_path)])
    assert r_ok_intent.exit_code == 0 and r_ok_intent.stdout.strip() == "enter"

    shims = install_cron_shims(w.home_dir)
    cron_env = dict(os.environ)
    cron_env["HOME"] = str(w.home_dir)
    cron_env["PATH"] = f"{shims['shim_bin']}:{cron_env.get('PATH', '')}"
    cron_env["HEALTHCHECKS_PING_URL"] = "https://hc-ping.example.invalid/0000"
    cron_script = _H.REPO_ROOT / "scripts" / "cron-setup-hetzner.sh"
    r_cron_install = subprocess.run(
        ["bash", str(cron_script), "install"],
        env=cron_env, capture_output=True, text=True, check=False,
    )
    assert r_cron_install.returncode == 0, r_cron_install.stderr
    installed_cron = shims["crontab_store"].read_text()
    assert "check-pick-entered" in installed_cron
    assert "bts day-outcome --date" in installed_cron
    assert "--seasons $(date +\\%Y)" in installed_cron
    assert "fetch-contest-streak --picks-dir data/picks --expected-username stonehengee --dm-recipient stonehengee.bsky.social" in installed_cron

    r_revoke = w.invoke_cli([
        "season-contract", "revoke",
        "--season", "2027",
        "--reason", "restored from the offseason snapshot",
        "--picks-dir", str(w.picks_dir),
    ])
    assert r_revoke.exit_code == 0, r_revoke.stderr
    revoked_savers = sorted((w.picks_dir / "account_state").glob("saver_state.revoked-*.json"))
    assert len(revoked_savers) == 1
    assert not (w.picks_dir / "account_state" / "saver_state.json").exists()
    assert (w.picks_dir / "account_state" / "contest_streak.json").exists()

    r_status = w.invoke_cli(["season-contract", "status", "--picks-dir", str(w.picks_dir)])
    assert r_status.exit_code == 0
    assert "switch: revoked" in r_status.stdout
    assert "live decision now: refused; Live decision refused: season 2027 switch is revoked." in r_status.stdout

    return {"cron_installed": True, "status_after_revoke": r_status.stdout}


def test_a_operator_does_everything_right_days_1_to_3(tmp_path: Path) -> None:
    with RehearsalWorld(tmp_path / "box_a", initial_et=et_dt(2027, 3, 24, 8, 50)) as w:
        prep = _prepare_revoked_live_box(w)
        write_units_snapshot(w.box_root, w.schedules, w.contest)

        assert w.read_streak_json()["streak"] == 4
        assert "season" not in w.read_streak_json()

        r_build = w.invoke_cli([
            "data", "build", "--seasons", "2027",
            "--raw-dir", str(w.box_root / "data" / "raw"),
            "--out-dir", str(w.proc_dir),
        ])
        assert r_build.exit_code == 0
        assert not (w.proc_dir / "pa_2027.parquet").exists()

        r_pd = w.invoke_cli(["park-drag-refresh", "--root", str(w.park_dir)])
        assert r_pd.exit_code == 1
        assert "2026 is behind 2027" in r_pd.stdout

        w.clock.set_et(et_dt(2027, 3, 24, 9, 30))
        r_set1 = w.invoke_cli([
            "set-contest-streak", "--date", "2027-03-24",
            "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
            "--reason", "Opening Night MLB check",
            "--picks-dir", str(w.picks_dir),
        ])
        assert r_set1.exit_code == 0, r_set1.stderr

        def _cron_1030() -> None:
            r = w.invoke_cli([
                "fetch-contest-streak", "--picks-dir", str(w.picks_dir),
                "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
            ])
            assert r.exit_code == 0
            assert "not switched to automatic" in r.stdout

        def _cron_1915_before_entry() -> None:
            r = w.invoke_cli([
                "check-pick-entered", "--picks-dir", str(w.picks_dir),
                "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
            ])
            assert r.exit_code == 1

        def _operator_enter_1920() -> None:
            p = w.read_pick_json("2027-03-24")
            assert p is not None and p["notification_sent"] is True
            w.contest.enter_pick(date(2027, 3, 24), [(p["pick"]["batter_id"], p["pick"]["game_pk"])])

        def _cron_1930_after_entry() -> None:
            r = w.invoke_cli([
                "check-pick-entered", "--picks-dir", str(w.picks_dir),
                "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
            ])
            assert r.exit_code == 0

        w.clock.schedule_at(et_dt(2027, 3, 24, 10, 30), "cron_fetch_1030", _cron_1030)
        w.clock.schedule_at(et_dt(2027, 3, 24, 13, 30), "cron_fetch_1330", _cron_1030)
        w.clock.schedule_at(et_dt(2027, 3, 24, 19, 15), "check_entered_1915", _cron_1915_before_entry)
        w.clock.schedule_at(et_dt(2027, 3, 24, 19, 20), "operator_enter_1920", _operator_enter_1920)
        w.clock.schedule_at(et_dt(2027, 3, 24, 19, 30), "check_entered_1930", _cron_1930_after_entry)

        w.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        w.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 25, 0, 30))

        p1 = w.read_pick_json("2027-03-24")
        d1 = w.read_decision_json("2027-03-24")
        s1 = w.read_streak_json()
        assert p1 is not None and p1["pick"]["batter_name"] == "Aaron Judge"
        assert p1["double_down"] is None
        assert p1["chosen_on"]["kind"] == "dated" and p1["chosen_on"]["streak"] == 0
        assert p1["chosen_on"]["saver_available"] is True
        assert p1["automatic_reading"] is None
        assert p1["result"] == "hit"
        assert d1 is not None and d1["action"] == "single" and d1["scoreable"] is True
        assert d1["delivery_status"] == "delivered" and d1["streak"] == 0
        assert s1["streak"] == 1 and s1["season"] == 2027 and s1["saver_available"] is True
        assert s1["applied_dates"] == ["2027-03-24"]
        assert len(w.posts) == 0

        day1_pick_dms = [m for m in w.dms if _is_pick_dm(m)]
        assert len(day1_pick_dms) == 1
        assert "Aaron Judge" in day1_pick_dms[0]["text"]
        assert "Streak: 0" in day1_pick_dms[0]["text"]

        auto_disk = json.loads((w.picks_dir / "account_state" / "contest_streak.json").read_text())
        assert auto_disk["schema_version"] == "bts_contest_streak_auto_v1"
        assert auto_disk["active_streak"] == 12

        w.contest.settle_day(date(2027, 3, 24), "hit", new_active=1, new_best=1)
        w.clock.set_et(et_dt(2027, 3, 25, 1, 0))
        assert w.invoke_cli(["check-results", "--date", "2027-03-24", "--picks-dir", str(w.picks_dir)]).exit_code == 0
        assert w.read_streak_json()["streak"] == 1

        w.clock.set_et(et_dt(2027, 3, 25, 1, 10))
        assert w.invoke_cli(["fetch-contest-streak", "--picks-dir", str(w.picks_dir), "--expected-username", "stonehengee"]).exit_code == 0

        w.clock.set_et(et_dt(2027, 3, 25, 2, 0))
        r_rec1 = w.invoke_cli(["reconcile", "--picks-dir", str(w.picks_dir)])
        assert r_rec1.exit_code == 0 and "Streak: 1" in r_rec1.stdout

        w.clock.set_et(et_dt(2027, 3, 25, 3, 0))
        r_prev2 = w.invoke_cli([
            "preview", "--date", "2027-03-25",
            "--picks-dir", str(w.picks_dir), "--models-dir", str(w.models_dir),
        ])
        assert r_prev2.exit_code == 0
        assert w.read_pick_json("2027-03-25")["chosen_on"]["required"] is False

        (w.park_dir / "park_drag_manifest.json").write_text(json.dumps({
            "max_source_game_date": "2027-03-24",
            "generated_at": "2027-03-25T07:45:10",
            "export_rows": 2,
        }))

        w.clock.set_et(et_dt(2027, 3, 25, 8, 30))
        dms_before_do1 = len(w.dms)
        r_do1 = w.invoke_cli([
            "day-outcome", "--date", "2027-03-24",
            "--config", str(w.toml_path), "--picks-dir", str(w.picks_dir),
            "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
        ])
        assert r_do1.exit_code == 0, r_do1.stdout
        assert json.loads(r_do1.stdout)["state"] == "entered"
        assert len(w.dms) == dms_before_do1

        w.clock.set_et(et_dt(2027, 3, 25, 9, 30))
        assert w.invoke_cli([
            "set-contest-streak", "--date", "2027-03-25",
            "--streak", "1", "--best-streak", "1", "--saver", "not_earned",
            "--picks-dir", str(w.picks_dir),
        ]).exit_code == 0

        def _op_enter_day2() -> None:
            p = w.read_pick_json("2027-03-25")
            assert p is not None and p["notification_sent"] is True
            slots = [(p["pick"]["batter_id"], p["pick"]["game_pk"])]
            if p["double_down"]:
                slots.append((p["double_down"]["batter_id"], p["double_down"]["game_pk"]))
            w.contest.enter_pick(date(2027, 3, 25), slots)
            r = w.invoke_cli([
                "check-pick-entered", "--picks-dir", str(w.picks_dir),
                "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
            ])
            assert r.exit_code == 0

        w.clock.schedule_at(et_dt(2027, 3, 25, 15, 30), "enter_and_verify_day2", _op_enter_day2)
        w.clock.set_et(et_dt(2027, 3, 25, 10, 0))
        dms_before_day2 = len(w.dms)
        w.run_scheduler_day("2027-03-25", stop_at_et=et_dt(2027, 3, 26, 0, 30))

        p2 = w.read_pick_json("2027-03-25")
        d2 = w.read_decision_json("2027-03-25")
        s2 = w.read_streak_json()
        assert p2["chosen_on"]["required"] is True and p2["chosen_on"]["streak"] == 1
        assert p2["double_down"] is not None
        assert d2["action"] == "double" and d2["scoreable"] is True and d2["streak"] == 1
        assert s2["streak"] == 3 and s2["season"] == 2027
        day2_new_dms = w.dms[dms_before_day2:]
        assert len([m for m in day2_new_dms if _is_pick_dm(m)]) == 1

        w.contest.settle_day(date(2027, 3, 25), "hit", new_active=3, new_best=3)
        w.clock.set_et(et_dt(2027, 3, 26, 8, 30))
        r_do2 = w.invoke_cli([
            "day-outcome", "--date", "2027-03-25",
            "--config", str(w.toml_path), "--picks-dir", str(w.picks_dir),
            "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
        ])
        assert r_do2.exit_code == 0 and json.loads(r_do2.stdout)["state"] == "entered"

        w.clock.set_et(et_dt(2027, 3, 26, 9, 30))
        assert w.invoke_cli([
            "set-contest-streak", "--date", "2027-03-26",
            "--streak", "3", "--best-streak", "3", "--saver", "not_earned",
            "--picks-dir", str(w.picks_dir),
        ]).exit_code == 0

        def _op_enter_day3() -> None:
            p = w.read_pick_json("2027-03-26")
            assert p is not None and p["notification_sent"] is True
            slots = [(p["pick"]["batter_id"], p["pick"]["game_pk"])]
            if p["double_down"]:
                slots.append((p["double_down"]["batter_id"], p["double_down"]["game_pk"]))
            w.contest.enter_pick(date(2027, 3, 26), slots)
            assert w.invoke_cli([
                "check-pick-entered", "--picks-dir", str(w.picks_dir),
                "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
            ]).exit_code == 0

        w.clock.schedule_at(et_dt(2027, 3, 26, 18, 30), "enter_and_verify_day3", _op_enter_day3)
        w.clock.set_et(et_dt(2027, 3, 26, 10, 0))
        dms_before_day3 = len(w.dms)
        w.run_scheduler_day("2027-03-26", stop_at_et=et_dt(2027, 3, 26, 23, 50))

        p3 = w.read_pick_json("2027-03-26")
        d3 = w.read_decision_json("2027-03-26")
        s3 = w.read_streak_json()
        assert p3["chosen_on"]["streak"] == 3 and d3["scoreable"] is True
        expected_day3_streak = 3 + (2 if d3["action"] == "double" else 1)
        assert s3["streak"] == expected_day3_streak and s3["season"] == 2027
        assert len([m for m in w.dms[dms_before_day3:] if _is_pick_dm(m)]) == 1

        TELEMETRY_LOG["1a"] = {
            "used_basesha_writers": w.paths["used_basesha_writers"],
            "cron_installed": prep["cron_installed"],
            "total_dms": len(w.dms),
            "pick_dms": len([m for m in w.dms if _is_pick_dm(m)]),
            "dms": w.dms,
            "day1_decision": d1,
            "day2_decision": d2,
            "day3_decision": d3,
            "final_streak_json": s3,
        }


def test_b_forgets_dated_observation_day1(tmp_path: Path) -> None:
    with RehearsalWorld(tmp_path / "box_b1", initial_et=et_dt(2027, 3, 24, 9, 0)) as w1:
        _prepare_revoked_live_box(w1)
        w1.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        w1.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 25, 0, 30))

        assert w1.read_pick_json("2027-03-24") is None
        assert w1.read_decision_json("2027-03-24") is None
        assert w1.read_streak_json()["streak"] == 4
        assert len([m for m in w1.dms if _is_pick_dm(m)]) == 0
        assert any("Morning check for 2027-03-24" in m["text"] and "17:05 ET" in m["text"] for m in w1.dms)
        assert any("Live decision refused: season 2027 switch is revoked" in m["text"] for m in w1.dms)

        w1.clock.set_et(et_dt(2027, 3, 25, 8, 30))
        r_do = w1.invoke_cli([
            "day-outcome", "--date", "2027-03-24",
            "--config", str(w1.toml_path), "--picks-dir", str(w1.picks_dir),
            "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
        ])
        assert r_do.exit_code == 1
        assert json.loads(r_do.stdout)["state"] == "no_decision"
        assert any("NO DECISION" in m["text"] for m in w1.dms)

    with RehearsalWorld(tmp_path / "box_b2", initial_et=et_dt(2027, 3, 24, 3, 0)) as w2:
        _prepare_revoked_live_box(w2)
        assert w2.invoke_cli([
            "preview", "--date", "2027-03-24",
            "--picks-dir", str(w2.picks_dir), "--models-dir", str(w2.models_dir),
        ]).exit_code == 0
        prev_pick = w2.read_pick_json("2027-03-24")
        assert prev_pick["chosen_on"]["streak"] == 12 and prev_pick["chosen_on"]["required"] is False

        w2.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        w2.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 25, 0, 30))

        assert len([m for m in w2.dms if _is_pick_dm(m)]) == 0
        assert w2.read_pick_json("2027-03-24")["notification_sent"] is False
        assert w2.read_decision_json("2027-03-24") is None
        missed = [m for m in w2.dms if "NO PICK DELIVERED" in m["text"]]
        assert len(missed) == 1
        assert "Aaron Judge" not in missed[0]["text"]

        w2.clock.set_et(et_dt(2027, 3, 25, 1, 0))
        r_cr = w2.invoke_cli(["check-results", "--date", "2027-03-24", "--picks-dir", str(w2.picks_dir)])
        assert r_cr.exit_code == 0 and "not a committed pick" in r_cr.stdout
        assert w2.read_streak_json()["streak"] == 4

        TELEMETRY_LOG["1b"] = {"b1_dms": w1.dms, "b2_dms": w2.dms}


def test_c_enters_dated_observation_between_morning_and_first_check(tmp_path: Path) -> None:
    with RehearsalWorld(tmp_path / "box_c", initial_et=et_dt(2027, 3, 24, 9, 0)) as w:
        _prepare_revoked_live_box(w)

        def _enter_at_noon() -> None:
            assert any("Morning check for 2027-03-24" in m["text"] for m in w.dms)
            r = w.invoke_cli([
                "set-contest-streak", "--date", "2027-03-24",
                "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
                "--picks-dir", str(w.picks_dir),
            ])
            assert r.exit_code == 0

        w.clock.schedule_at(et_dt(2027, 3, 24, 12, 0), "operator_noon_observation", _enter_at_noon)
        w.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        w.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 24, 23, 30))

        pick_dms = [m for m in w.dms if _is_pick_dm(m)]
        assert len(pick_dms) == 1
        assert "Streak: 0" in pick_dms[0]["text"]
        assert w.read_decision_json("2027-03-24")["scoreable"] is True
        assert w.read_streak_json()["streak"] == 1
        TELEMETRY_LOG["1c"] = {"dms": w.dms}


def test_d_enters_dated_observation_after_last_lineup_check(tmp_path: Path) -> None:
    with RehearsalWorld(tmp_path / "box_d1", initial_et=et_dt(2027, 3, 24, 9, 0)) as w1:
        _prepare_revoked_live_box(w1)
        w1.clock.schedule_at(
            et_dt(2027, 3, 24, 19, 15),
            "late_obs_1915",
            lambda: w1.invoke_cli([
                "set-contest-streak", "--date", "2027-03-24",
                "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
                "--picks-dir", str(w1.picks_dir),
            ]),
        )
        w1.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        w1.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 24, 23, 30))
        assert len([m for m in w1.dms if _is_pick_dm(m)]) == 0
        assert w1.read_decision_json("2027-03-24") is None

    with RehearsalWorld(tmp_path / "box_d2", initial_et=et_dt(2027, 3, 24, 3, 0)) as w2:
        _prepare_revoked_live_box(w2)
        w2.invoke_cli(["preview", "--date", "2027-03-24", "--picks-dir", str(w2.picks_dir), "--models-dir", str(w2.models_dir)])
        w2.clock.schedule_at(
            et_dt(2027, 3, 24, 19, 15),
            "late_obs_before_fallback",
            lambda: w2.invoke_cli([
                "set-contest-streak", "--date", "2027-03-24",
                "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
                "--picks-dir", str(w2.picks_dir),
            ]),
        )
        w2.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        w2.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 24, 23, 30))
        pick_dms_d2 = [m for m in w2.dms if _is_pick_dm(m)]
        assert len(pick_dms_d2) == 1 and "Streak: 0" in pick_dms_d2[0]["text"]
        assert w2.read_pick_json("2027-03-24")["chosen_on"]["required"] is True

    with RehearsalWorld(tmp_path / "box_d3", initial_et=et_dt(2027, 3, 24, 3, 0)) as w3:
        _prepare_revoked_live_box(w3)
        w3.invoke_cli(["preview", "--date", "2027-03-24", "--picks-dir", str(w3.picks_dir), "--models-dir", str(w3.models_dir)])

        def _obs_and_break_cascade() -> None:
            w3.invoke_cli([
                "set-contest-streak", "--date", "2027-03-24",
                "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
                "--picks-dir", str(w3.picks_dir),
            ])
            w3.prediction_override = lambda _d: None

        w3.clock.schedule_at(et_dt(2027, 3, 24, 19, 15), "obs_with_dead_cascade", _obs_and_break_cascade)
        w3.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        w3.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 24, 23, 30))
        assert len([m for m in w3.dms if _is_pick_dm(m)]) == 0
        assert any("NO PICK DELIVERED" in m["text"] for m in w3.dms)

    TELEMETRY_LOG["1d"] = {"d1_dms": w1.dms, "d2_dms": w2.dms, "d3_dms": w3.dms}


def test_e_wrong_streak_then_corrected(tmp_path: Path) -> None:
    with RehearsalWorld(tmp_path / "box_e1", initial_et=et_dt(2027, 3, 24, 9, 30)) as w1:
        _prepare_revoked_live_box(w1)
        w1.invoke_cli([
            "set-contest-streak", "--date", "2027-03-24",
            "--streak", "12", "--best-streak", "18", "--saver", "used",
            "--picks-dir", str(w1.picks_dir),
        ])
        w1.clock.schedule_at(
            et_dt(2027, 3, 24, 17, 30),
            "correct_to_0",
            lambda: w1.invoke_cli([
                "set-contest-streak", "--date", "2027-03-24",
                "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
                "--picks-dir", str(w1.picks_dir),
            ]),
        )
        w1.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        w1.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 24, 23, 30))
        p_dms = [m for m in w1.dms if _is_pick_dm(m)]
        assert len(p_dms) == 1 and "Streak: 0" in p_dms[0]["text"]
        assert w1.read_pick_json("2027-03-24")["chosen_on"]["streak"] == 0

    sched_low = _H.default_opening_days_schedules()
    sched_low["2027-03-24"][0].away_p_hit = 0.770
    sched_low["2027-03-24"][0].home_p_hit = 0.750
    with RehearsalWorld(tmp_path / "box_e2", initial_et=et_dt(2027, 3, 24, 9, 30), schedules=sched_low) as w2:
        _prepare_revoked_live_box(w2)
        w2.invoke_cli([
            "set-contest-streak", "--date", "2027-03-24",
            "--streak", "12", "--best-streak", "18", "--saver", "used",
            "--picks-dir", str(w2.picks_dir),
        ])
        w2.clock.schedule_at(
            et_dt(2027, 3, 24, 17, 30),
            "correct_skip_to_0",
            lambda: w2.invoke_cli([
                "set-contest-streak", "--date", "2027-03-24",
                "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
                "--picks-dir", str(w2.picks_dir),
            ]),
        )
        w2.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        w2.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 24, 23, 30))
        assert any("No pick at this check" in m["text"] and "Streak used at this check: 12" in m["text"] for m in w2.dms)
        p_dms2 = [m for m in w2.dms if _is_pick_dm(m)]
        assert len(p_dms2) == 1 and "Streak: 0" in p_dms2[0]["text"]
        assert w2.read_decision_json("2027-03-24")["action"] == "single"

    sched_unconf = _H.default_opening_days_schedules()
    sched_unconf["2027-03-24"][0].away_confirm_et = None
    sched_unconf["2027-03-24"][0].home_confirm_et = None
    with RehearsalWorld(tmp_path / "box_e3", initial_et=et_dt(2027, 3, 24, 9, 30), schedules=sched_unconf) as w3:
        _prepare_revoked_live_box(w3)
        w3.invoke_cli([
            "set-contest-streak", "--date", "2027-03-24",
            "--streak", "5", "--best-streak", "5", "--saver", "not_earned",
            "--picks-dir", str(w3.picks_dir),
        ])

        def _correct_and_fail_cascades() -> None:
            w3.invoke_cli([
                "set-contest-streak", "--date", "2027-03-24",
                "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
                "--picks-dir", str(w3.picks_dir),
            ])
            w3.prediction_override = lambda _d: None

        w3.clock.schedule_at(et_dt(2027, 3, 24, 17, 30), "correct_and_break_cascade", _correct_and_fail_cascades)
        w3.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        w3.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 24, 23, 30))
        assert len([m for m in w3.dms if _is_pick_dm(m)]) == 0
        assert any("Saved pick for 2027-03-24 was not delivered" in m["text"] for m in w3.dms)
        assert any("NO PICK DELIVERED" in m["text"] for m in w3.dms)

    with RehearsalWorld(tmp_path / "box_e4", initial_et=et_dt(2027, 3, 24, 9, 30)) as w4:
        _prepare_revoked_live_box(w4)
        w4.invoke_cli([
            "set-contest-streak", "--date", "2027-03-24",
            "--streak", "5", "--best-streak", "5", "--saver", "not_earned",
            "--picks-dir", str(w4.picks_dir),
        ])
        w4.clock.schedule_at(
            et_dt(2027, 3, 24, 18, 30),
            "late_correction_after_lock",
            lambda: w4.invoke_cli([
                "set-contest-streak", "--date", "2027-03-24",
                "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
                "--picks-dir", str(w4.picks_dir),
            ]),
        )
        w4.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        w4.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 24, 23, 30))
        p_dms4 = [m for m in w4.dms if _is_pick_dm(m)]
        assert len(p_dms4) == 1 and "Streak: 5" in p_dms4[0]["text"]

    TELEMETRY_LOG["1e"] = {"e1_dms": w1.dms, "e2_dms": w2.dms, "e3_dms": w3.dms, "e4_dms": w4.dms}


def test_f_switches_to_automatic_day3_with_matching_reading(tmp_path: Path) -> None:
    with RehearsalWorld(tmp_path / "box_f1", initial_et=et_dt(2027, 3, 26, 9, 0)) as w1:
        _prepare_revoked_live_box(w1)
        w1.contest.enter_pick(date(2027, 3, 25), [(683002, 777101)])
        w1.contest.settle_day(date(2027, 3, 25), "hit", new_active=3, new_best=3)

        assert w1.invoke_cli(["season-contract", "enable", "--season", "2027", "--picks-dir", str(w1.picks_dir)]).exit_code == 0
        assert w1.invoke_cli([
            "fetch-contest-streak", "--picks-dir", str(w1.picks_dir), "--expected-username", "stonehengee",
        ]).exit_code == 0
        r_st = w1.invoke_cli(["season-contract", "status", "--picks-dir", str(w1.picks_dir)])
        assert "live decision now: accepted; source=contest; streak=3;" in r_st.stdout
        assert "saver: not_earned; season: 2027" in r_st.stdout

        w1.clock.set_et(et_dt(2027, 3, 26, 10, 0))
        w1.run_scheduler_day("2027-03-26", stop_at_et=et_dt(2027, 3, 26, 23, 30))
        p = w1.read_pick_json("2027-03-26")
        assert p["chosen_on"]["kind"] == "automatic" and p["chosen_on"]["streak"] == 3
        assert p["automatic_reading"]["stale"] is False and p["automatic_reading"]["reason"] is None
        p_dms = [m for m in w1.dms if _is_pick_dm(m)]
        assert len(p_dms) == 1 and "Streak: 3" in p_dms[0]["text"]
        assert not any("stale" in m["text"].lower() for m in w1.dms)

    with RehearsalWorld(tmp_path / "box_f2", initial_et=et_dt(2027, 3, 25, 18, 0)) as w2:
        _prepare_revoked_live_box(w2)
        w2.contest.enter_pick(date(2027, 3, 24), [(592450, 777001)])
        w2.contest.settle_day(date(2027, 3, 24), "hit", new_active=1, new_best=1)
        w2.invoke_cli(["season-contract", "enable", "--season", "2027", "--picks-dir", str(w2.picks_dir)])
        w2.invoke_cli(["fetch-contest-streak", "--picks-dir", str(w2.picks_dir), "--expected-username", "stonehengee"])

        w2.clock.set_et(et_dt(2027, 3, 26, 10, 0))
        w2.run_scheduler_day("2027-03-26", stop_at_et=et_dt(2027, 3, 26, 23, 30))
        p2 = w2.read_pick_json("2027-03-26")
        assert p2["automatic_reading"]["stale"] is True and p2["automatic_reading"]["reason"] == "old"
        stale_dms = [m for m in w2.dms if "stale" in m["text"].lower()]
        assert len(stale_dms) == 1
        assert "Morning check for 2027-03-26" in stale_dms[0]["text"]
        assert len([m for m in w2.dms if _is_pick_dm(m)]) == 1

    TELEMETRY_LOG["1f"] = {"f1_dms": w1.dms, "f2_dms": w2.dms}


def test_g_switches_with_last_season_automatic_file_on_disk(tmp_path: Path) -> None:
    with RehearsalWorld(tmp_path / "box_g1", initial_et=et_dt(2027, 3, 26, 3, 0)) as w1:
        _prepare_revoked_live_box(w1)
        w1.invoke_cli(["preview", "--date", "2027-03-26", "--picks-dir", str(w1.picks_dir), "--models-dir", str(w1.models_dir)])

        w1.clock.set_et(et_dt(2027, 3, 26, 9, 0))
        assert w1.invoke_cli(["season-contract", "enable", "--season", "2027", "--picks-dir", str(w1.picks_dir)]).exit_code == 0
        r_st = w1.invoke_cli(["season-contract", "status", "--picks-dir", str(w1.picks_dir)])
        assert "schema_version is 'bts_contest_streak_auto_v1', not 'bts_contest_streak_auto_v2'" in r_st.stdout

        w1.clock.set_et(et_dt(2027, 3, 26, 10, 0))
        w1.run_scheduler_day("2027-03-26", stop_at_et=et_dt(2027, 3, 26, 23, 30))
        assert len([m for m in w1.dms if _is_pick_dm(m)]) == 0
        assert w1.read_pick_json("2027-03-26") is None
        archived = list((w1.picks_dir / "2027-03-26").glob("stale_pick_*.json"))
        assert len(archived) == 1
        assert w1.read_decision_json("2027-03-26") is None
        assert any("Morning check for 2027-03-26" in m["text"] for m in w1.dms)
        assert any("bts_contest_streak_auto_v1" in m["text"] for m in w1.dms)
        assert not any("NO PICK DELIVERED" in m["text"] for m in w1.dms)

        w1.clock.set_et(et_dt(2027, 3, 27, 8, 30))
        r_do = w1.invoke_cli([
            "day-outcome", "--date", "2027-03-26",
            "--config", str(w1.toml_path), "--picks-dir", str(w1.picks_dir),
            "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
        ])
        assert r_do.exit_code == 1 and json.loads(r_do.stdout)["state"] == "no_decision"

    with RehearsalWorld(tmp_path / "box_g2", initial_et=et_dt(2027, 3, 24, 3, 0)) as w2:
        _prepare_revoked_live_box(w2)
        w2.invoke_cli(["preview", "--date", "2027-03-24", "--picks-dir", str(w2.picks_dir), "--models-dir", str(w2.models_dir)])
        w2.clock.set_et(et_dt(2027, 3, 24, 9, 0))
        w2.invoke_cli(["season-contract", "enable", "--season", "2027", "--picks-dir", str(w2.picks_dir)])
        w2.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        w2.run_scheduler_day("2027-03-24", stop_at_et=et_dt(2027, 3, 24, 23, 30))
        assert len([m for m in w2.dms if _is_pick_dm(m)]) == 0
        assert any("NO PICK DELIVERED" in m["text"] for m in w2.dms)

    TELEMETRY_LOG["1g"] = {"g1_dms": w1.dms, "g2_dms": w2.dms}


def test_h_midday_restarts_across_scenarios(tmp_path: Path) -> None:
    def _run_with_restart_at(w, date_str: str, restart_et, stop_et) -> None:
        def _boom() -> None:
            raise SimulatedBoxRestart()

        w.clock.schedule_at(restart_et, "simulated_box_restart", _boom)
        try:
            w.run_scheduler_day(date_str, stop_at_et=stop_et)
            raise AssertionError("Expected SimulatedBoxRestart")
        except SimulatedBoxRestart:
            pass
        w.run_scheduler_day(date_str, stop_at_et=stop_et)

    with RehearsalWorld(tmp_path / "box_ha1", initial_et=et_dt(2027, 3, 24, 9, 30)) as w:
        _prepare_revoked_live_box(w)
        w.invoke_cli(["set-contest-streak", "--date", "2027-03-24", "--streak", "0", "--best-streak", "0", "--saver", "not_earned", "--picks-dir", str(w.picks_dir)])
        w.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        _run_with_restart_at(w, "2027-03-24", et_dt(2027, 3, 24, 14, 0), et_dt(2027, 3, 24, 23, 30))
        assert len([m for m in w.dms if _is_pick_dm(m)]) == 1
        assert w.read_streak_json()["streak"] == 1

    with RehearsalWorld(tmp_path / "box_ha2", initial_et=et_dt(2027, 3, 24, 9, 30)) as w:
        _prepare_revoked_live_box(w)
        w.invoke_cli(["set-contest-streak", "--date", "2027-03-24", "--streak", "0", "--best-streak", "0", "--saver", "not_earned", "--picks-dir", str(w.picks_dir)])
        w.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        _run_with_restart_at(w, "2027-03-24", et_dt(2027, 3, 24, 18, 30), et_dt(2027, 3, 24, 23, 30))
        assert len([m for m in w.dms if _is_pick_dm(m)]) == 1
        assert w.read_streak_json()["streak"] == 1

    with RehearsalWorld(tmp_path / "box_hb", initial_et=et_dt(2027, 3, 24, 10, 0)) as w:
        _prepare_revoked_live_box(w)
        _run_with_restart_at(w, "2027-03-24", et_dt(2027, 3, 24, 17, 30), et_dt(2027, 3, 24, 21, 0))
        # Exactly 3 DMs: 10:00 morning check, 17:05 check 1 refusal (18:05 & 19:05 suppressed by dedup!),
        # and 19:05 EOD combined health digest (which escalates because park_drag_freshness + saver_state join).
        assert len([m for m in w.dms if "Morning check for 2027-03-24" in m["text"]]) == 1
        assert len(w.dms) == 3
        assert "park_drag_freshness" not in w.dms[1]["text"]
        assert "park_drag_freshness" in w.dms[2]["text"]
        assert len([m for m in w.dms if _is_pick_dm(m)]) == 0

    with RehearsalWorld(tmp_path / "box_hc", initial_et=et_dt(2027, 3, 24, 10, 0)) as w:
        _prepare_revoked_live_box(w)
        w.clock.schedule_at(
            et_dt(2027, 3, 24, 12, 0), "obs_1200",
            lambda: w.invoke_cli(["set-contest-streak", "--date", "2027-03-24", "--streak", "0", "--best-streak", "0", "--saver", "not_earned", "--picks-dir", str(w.picks_dir)]),
        )
        _run_with_restart_at(w, "2027-03-24", et_dt(2027, 3, 24, 13, 0), et_dt(2027, 3, 24, 23, 30))
        assert len([m for m in w.dms if _is_pick_dm(m)]) == 1

    with RehearsalWorld(tmp_path / "box_hd", initial_et=et_dt(2027, 3, 24, 3, 0)) as w:
        _prepare_revoked_live_box(w)
        w.invoke_cli(["preview", "--date", "2027-03-24", "--picks-dir", str(w.picks_dir), "--models-dir", str(w.models_dir)])
        w.clock.schedule_at(
            et_dt(2027, 3, 24, 19, 15), "obs_1915_hd",
            lambda: w.invoke_cli(["set-contest-streak", "--date", "2027-03-24", "--streak", "0", "--best-streak", "0", "--saver", "not_earned", "--picks-dir", str(w.picks_dir)]),
        )
        w.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        _run_with_restart_at(w, "2027-03-24", et_dt(2027, 3, 24, 19, 20), et_dt(2027, 3, 24, 23, 30))
        assert len([m for m in w.dms if _is_pick_dm(m)]) == 1

    sched_low = _H.default_opening_days_schedules()
    sched_low["2027-03-24"][0].away_p_hit = 0.770
    sched_low["2027-03-24"][0].home_p_hit = 0.750
    with RehearsalWorld(tmp_path / "box_he", initial_et=et_dt(2027, 3, 24, 9, 30), schedules=sched_low) as w:
        _prepare_revoked_live_box(w)
        w.invoke_cli(["set-contest-streak", "--date", "2027-03-24", "--streak", "12", "--best-streak", "18", "--saver", "used", "--picks-dir", str(w.picks_dir)])
        w.clock.set_et(et_dt(2027, 3, 24, 10, 0))
        _run_with_restart_at(w, "2027-03-24", et_dt(2027, 3, 24, 17, 20), et_dt(2027, 3, 24, 23, 30))
        assert len([m for m in w.dms if "No pick at this check" in m["text"]]) == 1
        assert w.read_decision_json("2027-03-24")["action"] == "skip"

    with RehearsalWorld(tmp_path / "box_hf", initial_et=et_dt(2027, 3, 25, 18, 0)) as w:
        _prepare_revoked_live_box(w)
        w.contest.enter_pick(date(2027, 3, 24), [(592450, 777001)])
        w.contest.settle_day(date(2027, 3, 24), "hit", new_active=1, new_best=1)
        w.invoke_cli(["season-contract", "enable", "--season", "2027", "--picks-dir", str(w.picks_dir)])
        w.invoke_cli(["fetch-contest-streak", "--picks-dir", str(w.picks_dir), "--expected-username", "stonehengee"])
        w.clock.set_et(et_dt(2027, 3, 26, 10, 0))
        _run_with_restart_at(w, "2027-03-26", et_dt(2027, 3, 26, 14, 0), et_dt(2027, 3, 26, 23, 30))
        assert len([m for m in w.dms if "stale" in m["text"].lower()]) == 1
        assert len([m for m in w.dms if _is_pick_dm(m)]) == 1

    with RehearsalWorld(tmp_path / "box_hg", initial_et=et_dt(2027, 3, 26, 9, 0)) as w:
        _prepare_revoked_live_box(w)
        w.invoke_cli(["season-contract", "enable", "--season", "2027", "--picks-dir", str(w.picks_dir)])
        w.clock.set_et(et_dt(2027, 3, 26, 10, 0))
        _run_with_restart_at(w, "2027-03-26", et_dt(2027, 3, 26, 14, 0), et_dt(2027, 3, 26, 23, 30))
        assert len([m for m in w.dms if "Morning check for 2027-03-26" in m["text"]]) == 1
        assert len([m for m in w.dms if _is_pick_dm(m)]) == 0

    TELEMETRY_LOG["1h"] = {"restarts_verified": ["ha1", "ha2", "hb", "hc", "hd", "he", "hf", "hg"]}


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    runs_root = out_dir / "_runs_rehearsal_final"
    runs_root.mkdir(parents=True, exist_ok=True)
    tests = [
        ("test_a", test_a_operator_does_everything_right_days_1_to_3),
        ("test_b", test_b_forgets_dated_observation_day1),
        ("test_c", test_c_enters_dated_observation_between_morning_and_first_check),
        ("test_d", test_d_enters_dated_observation_after_last_lineup_check),
        ("test_e", test_e_wrong_streak_then_corrected),
        ("test_f", test_f_switches_to_automatic_day3_with_matching_reading),
        ("test_g", test_g_switches_with_last_season_automatic_file_on_disk),
        ("test_h", test_h_midday_restarts_across_scenarios),
    ]
    with tempfile.TemporaryDirectory(dir=runs_root) as td:
        base = Path(td)
        for idx, (name, fn) in enumerate(tests, start=1):
            sub = base / f"{idx:02d}_{name}"
            sub.mkdir(parents=True, exist_ok=True)
            print(f"[RUN] {name} ...", flush=True)
            fn(sub)
            print(f"[PASS] {name}", flush=True)
    summary_path = out_dir / "rehearsal_final_telemetry.json"
    summary_path.write_text(json.dumps(TELEMETRY_LOG, indent=2) + "\n", encoding="utf-8")
    print(f"ALL REHEARSAL FINAL CHECKS PASSED -> {summary_path}")
