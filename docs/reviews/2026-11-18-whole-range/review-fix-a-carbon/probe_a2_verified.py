"""Adversarial probe for A2 (all outward paths, observation transitions, hostile inputs, cost side)."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner
import pytest

from bts import cli, contest_state, day_outcome, orchestrator, picks, scheduler, strategy
from bts.daily_decision import load_decision

ROOT = Path(__file__).resolve().parents[2]
DAY = "2027-03-24"


def _load_offline(tmp_root: Path, mp: pytest.MonkeyPatch):
    mpl_dir = tmp_root / "_mpl"
    mpl_dir.mkdir(exist_ok=True)
    mp.setenv("MPLCONFIGDIR", str(mpl_dir))
    fixture = ROOT / "docs/reviews/2026-11-18-whole-range/slice-3-codex/offline.py"
    spec = spec_from_file_location("_wr_a_probe_offline", fixture)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    mp.setattr(mod, "ROOT", ROOT)
    mp.setattr(mod, "HERE", tmp_root)
    return mod


def _scheduler_state(day: str = DAY) -> scheduler.SchedulerState:
    return scheduler.SchedulerState(
        date=day,
        schedule_fetched_at=f"{day}T10:00:00-04:00",
        games=[],
        confirmed_game_pks=[],
        runs_completed=[],
        pick_locked=False,
        pick_locked_at=None,
        result_status=None,
        next_wakeup=None,
    )


CASES = (
    "equal_dated",
    "dated_streak",
    "dated_best",
    "dated_saver",
    "dated_to_automatic",
    "automatic_to_dated",
    "automatic_to_automatic",
    "midnight_to_automatic",
    "midnight_no_automatic",
    "corrupt_final",
    "missing_final",
    "interrupt_keyboard",
    "interrupt_exit",
)

PATHS = (
    "scheduler_dm",
    "scheduler_public",
    "scheduler_fallback",
    "bts_run",
    "bts_orchestrate",
    "bts_run_catchup",
    "bts_orchestrate_catchup",
)


def _run_one_matrix_cell(tmp_root: Path, path: str, case: str) -> dict:
    with pytest.MonkeyPatch.context() as outer_mp:
        offline_mod = _load_offline(tmp_root, outer_mp)
        is_midnight = case in {"midnight_to_automatic", "midnight_no_automatic"}
        start_at = (
            offline_mod.instant(DAY, 23, 58)
            if is_midnight
            else offline_mod.instant(DAY, 18, 25)
        )
        with offline_mod.offline(start_at) as (world, patch):
            if is_midnight:
                world.schedule[DAY] = [
                    offline_mod.game(101, offline_mod.instant("2027-03-25", 0, 15)),
                    offline_mod.game(102, offline_mod.instant("2027-03-25", 0, 20)),
                ]
            world.predictions = lambda: world.frame((0.99, 0.98))

            starts_auto = case in {"automatic_to_dated", "automatic_to_automatic"}
            if starts_auto or case in {"dated_to_automatic", "midnight_to_automatic"}:
                world.automatic(0)
                acc = world.account / "contest_streak.json"
                data = json.loads(acc.read_text())
                data["source"] = "mlb_bts_profile"
                acc.write_text(json.dumps(data))
            if not starts_auto:
                world.dated(0, best=0, saver="used")

            manual_path = world.account / "contest_streak.manual.json"
            is_interrupt = case in {"interrupt_keyboard", "interrupt_exit"}
            interrupt_fired = []

            if is_interrupt:
                def boom(*a, **kw):
                    interrupt_fired.append(True)
                    if case == "interrupt_keyboard":
                        raise KeyboardInterrupt("simulated sigint")
                    raise SystemExit(130)

                patch.setattr(picks, "saved_pick_refusal", boom)

            def apply_mutation():
                if case in {"equal_dated", "interrupt_keyboard", "interrupt_exit"}:
                    return
                if case == "dated_streak":
                    world.dated(30, best=30, saver="used")
                elif case == "dated_best":
                    world.dated(0, best=5, saver="used")
                elif case == "dated_saver":
                    world.dated(0, best=0, saver="not_earned")
                elif case == "dated_to_automatic":
                    manual_path.unlink()
                elif case == "automatic_to_dated":
                    world.dated(0, best=0, saver="used")
                elif case == "automatic_to_automatic":
                    world.automatic(8)
                    acc_p = world.account / "contest_streak.json"
                    d = json.loads(acc_p.read_text())
                    d["best_streak"] = 12
                    d["source"] = "mlb_bts_profile"
                    acc_p.write_text(json.dumps(d))
                elif is_midnight:
                    world.now = offline_mod.instant("2027-03-25", 0, 1)
                elif case == "corrupt_final":
                    manual_path.write_text("{broken json")
                elif case == "missing_final":
                    manual_path.unlink()

            should_deliver = case in {"equal_dated", "automatic_to_automatic"}
            is_final_reader_error = case in {"midnight_no_automatic", "corrupt_final", "missing_final"}

            outcome = {
                "path": path,
                "case": case,
                "sent": 0,
                "marked_attempted": None,
                "marked_delivered": None,
                "pick_on_disk": False,
                "stamp_preserved": False,
                "locked": None,
                "decision_recorded": None,
                "contest_alerts": 0,
                "exit_or_return": None,
            }

            if path in {"scheduler_dm", "scheduler_public", "scheduler_fallback"}:
                mode = "public" if path == "scheduler_public" else "dm"
                world.config["scheduler"]["pick_delivery"] = mode
                state = _scheduler_state(DAY)
                if path == "scheduler_fallback":
                    refresh = scheduler._refresh_pick_at_fallback_decision(
                        world.config, DAY, None, 0.03
                    )
                    daily = refresh.daily
                    sel = refresh.selection
                else:
                    res = scheduler.run_single_check(
                        DAY, [101, 102], set(), world.config, 0.03
                    )
                    daily = res["pick_result"].daily
                    sel = res["selection"]
                first_stamp = dict(daily.chosen_on)
                apply_mutation()

                label = "fallback" if path == "scheduler_fallback" else "lineup"
                if is_interrupt:
                    expected_exc = KeyboardInterrupt if case == "interrupt_keyboard" else SystemExit
                    with pytest.raises(expected_exc):
                        scheduler._deliver_and_lock_pick(
                            daily, world.config, world.picks, state, DAY, label, selection=sel
                        )
                    assert interrupt_fired == [True]
                    ret = "interrupted"
                else:
                    ret = scheduler._deliver_and_lock_pick(
                        daily, world.config, world.picks, state, DAY, label, selection=sel
                    )
                    assert ret is should_deliver, (path, case, ret, should_deliver)
                outcome["exit_or_return"] = ret
                outcome["locked"] = state.pick_locked
                assert state.pick_locked is should_deliver

            elif path in {"bts_run", "bts_orchestrate"}:
                cmd = "run" if path == "bts_run" else "orchestrate"
                save_orig = picks.save_pick
                mutated = False
                first_stamp_holder = []

                def save_then_mutate(daily_obj, directory):
                    nonlocal mutated
                    p = save_orig(daily_obj, directory)
                    if not mutated:
                        mutated = True
                        first_stamp_holder.append(dict(daily_obj.chosen_on))
                        apply_mutation()
                    return p

                patch.setattr(picks, "save_pick", save_then_mutate)
                args = (
                    ["run", "--date", DAY, "--picks-dir", str(world.picks), "--models-dir", str(world.models)]
                    if cmd == "run"
                    else ["orchestrate", "--date", DAY, "--config", str(world.config_path)]
                )
                if is_interrupt:
                    r = CliRunner().invoke(cli.cli, args, catch_exceptions=False)
                    assert interrupt_fired == [True], (path, case, r.output)
                    expected_code = 1 if case == "interrupt_keyboard" else 130
                    assert r.exit_code == expected_code, (path, case, r.exit_code)
                    outcome["exit_or_return"] = f"interrupted:{r.exit_code}"
                elif cmd == "orchestrate" and is_final_reader_error:
                    with pytest.raises(contest_state.ContestStateError):
                        CliRunner().invoke(cli.cli, args, catch_exceptions=False)
                    outcome["exit_or_return"] = "uncaught_ContestStateError"
                else:
                    r = CliRunner().invoke(cli.cli, args, catch_exceptions=False)
                    expected_code = 0 if should_deliver else 1
                    assert r.exit_code == expected_code, (path, case, r.exit_code, r.output)
                    if not should_deliver:
                        assert f"bts {cmd}: no pick:" in r.stderr
                    outcome["exit_or_return"] = r.exit_code
                first_stamp = first_stamp_holder[0]

            else:  # catch-up paths: bts_run_catchup, bts_orchestrate_catchup
                cmd = "run" if path == "bts_run_catchup" else "orchestrate"
                _, sel, _ = orchestrator.run_and_pick(world.config, DAY)
                daily = sel.pick_result.daily
                first_stamp = dict(daily.chosen_on)
                picks.save_pick(daily, world.picks)
                world.schedule[DAY][0]["status"] = {
                    "abstractGameCode": "L",
                    "detailedState": "In Progress",
                    "statusCode": "I",
                }
                orig_select = strategy.select_pick

                def select_then_mutate(*a, **kw):
                    out_sel = orig_select(*a, **kw)
                    assert out_sel.pick_result is not None and out_sel.pick_result.locked is True
                    apply_mutation()
                    return out_sel

                patch.setattr(strategy, "select_pick", select_then_mutate)
                args = (
                    ["run", "--date", DAY, "--picks-dir", str(world.picks), "--models-dir", str(world.models)]
                    if cmd == "run"
                    else ["orchestrate", "--date", DAY, "--config", str(world.config_path)]
                )
                if is_interrupt:
                    r = CliRunner().invoke(cli.cli, args, catch_exceptions=False)
                    assert interrupt_fired == [True], (path, case, r.output)
                    expected_code = 1 if case == "interrupt_keyboard" else 130
                    assert r.exit_code == expected_code, (path, case, r.exit_code)
                    outcome["exit_or_return"] = f"interrupted:{r.exit_code}"
                elif cmd == "orchestrate" and is_final_reader_error:
                    with pytest.raises(contest_state.ContestStateError):
                        CliRunner().invoke(cli.cli, args, catch_exceptions=False)
                    outcome["exit_or_return"] = "uncaught_ContestStateError"
                else:
                    r = CliRunner().invoke(cli.cli, args, catch_exceptions=False)
                    expected_code = 0 if should_deliver else 1
                    assert r.exit_code == expected_code, (path, case, r.exit_code, r.output)
                    if not should_deliver:
                        assert f"bts {cmd}: no pick:" in r.stderr
                    outcome["exit_or_return"] = r.exit_code

            persisted = picks.load_pick(DAY, world.picks)
            outcome["pick_on_disk"] = persisted is not None
            assert persisted is not None
            outcome["marked_attempted"] = persisted.delivery_attempted
            outcome["marked_delivered"] = picks.pick_was_delivered(persisted)
            outcome["stamp_preserved"] = persisted.chosen_on == first_stamp
            outcome["sent"] = len(world.sends("dm")) + len(world.sends("post"))
            dec = load_decision(DAY, world.picks)
            outcome["decision_recorded"] = dec["delivery_status"] if dec else None
            outcome["contest_alerts"] = sum(
                "contest_state" in b for b in world.sends("health-dm")
            )

            expected_attempt = should_deliver and path.startswith("scheduler_")
            assert outcome["sent"] == int(should_deliver), outcome
            assert outcome["marked_attempted"] is expected_attempt, outcome
            assert outcome["marked_delivered"] is should_deliver, outcome
            assert outcome["stamp_preserved"] is True, outcome
            if should_deliver and path.startswith("scheduler_"):
                assert outcome["decision_recorded"] == "delivered", outcome
            else:
                assert outcome["decision_recorded"] is None, outcome
            if not should_deliver:
                if path.startswith("scheduler_"):
                    expected_alerts = 0 if is_interrupt else 1
                    assert outcome["contest_alerts"] == expected_alerts, outcome
                else:
                    assert outcome["contest_alerts"] == 0, outcome
            return outcome


def _run_cost_and_stuck_probes(tmp_root: Path) -> dict:
    results = {}
    with pytest.MonkeyPatch.context() as outer_mp:
        offline_mod = _load_offline(tmp_root, outer_mp)

        # 1. True optional / model-only cold start (NO contest file at all, both flags False)
        with offline_mod.offline(offline_mod.instant(DAY, 18, 25)) as (world, _):
            world.config["scheduler"]["contest_state_required"] = False
            world.config["health_checks"]["contest_state_expected"] = False
            assert scheduler._contest_state_required(world.config) is False
            scheduler.run_day(DAY, world.config)
            dec = load_decision(DAY, world.picks)
            assert dec is not None and dec["delivery_status"] == "delivered"
            assert len(world.sends("dm")) == 1
            results["optional_cold_start_model_only"] = "delivered"

        # 2. Optional mode WITH dated observation changing mid-cycle: still delivers!
        with offline_mod.offline(offline_mod.instant(DAY, 18, 25)) as (world, patch):
            world.config["scheduler"]["contest_state_required"] = False
            world.config["health_checks"]["contest_state_expected"] = False
            world.dated(0)
            save_orig = picks.save_pick
            changed = False

            def save_and_change(daily_obj, directory):
                nonlocal changed
                p = save_orig(daily_obj, directory)
                if not changed:
                    changed = True
                    world.dated(30)
                return p

            patch.setattr(picks, "save_pick", save_and_change)
            scheduler.run_day(DAY, world.config)
            dec = load_decision(DAY, world.picks)
            assert dec is not None and dec["delivery_status"] == "delivered"
            assert len(world.sends("dm")) == 1
            results["optional_midcycle_dated_change"] = "delivered"

        # 3. Private / research mode WITH dated observation changing mid-cycle: locks private_locked!
        with offline_mod.offline(offline_mod.instant(DAY, 18, 25)) as (world, patch):
            world.config["scheduler"]["pick_delivery"] = "private"
            world.config["scheduler"]["entry_intent"] = "research"
            world.dated(0)
            save_orig = picks.save_pick
            changed = False

            def save_and_change_priv(daily_obj, directory):
                nonlocal changed
                p = save_orig(daily_obj, directory)
                if not changed:
                    changed = True
                    world.dated(30)
                return p

            patch.setattr(picks, "save_pick", save_and_change_priv)
            scheduler.run_day(DAY, world.config)
            dec = load_decision(DAY, world.picks)
            assert dec is not None and dec["delivery_status"] == "private_locked"
            assert len(world.sends("dm")) == 0
            results["private_midcycle_dated_change"] = "private_locked"

        # 4a. One-game slate Warmup-stuck after refusal: fallback + final fallback both refuse,
        # missed_pick DM fires at 19:00, and day_outcome classifies no_decision!
        with offline_mod.offline(offline_mod.instant(DAY, 18, 25)) as (world, patch):
            world.schedule[DAY] = [offline_mod.game(101, offline_mod.instant(DAY, 19, 10))]
            world.predictions = lambda: world.frame((0.99,))
            world.dated(0)
            save_orig = picks.save_pick
            changed = False

            def save_and_warmup_single(daily_obj, directory):
                nonlocal changed
                p = save_orig(daily_obj, directory)
                if not changed:
                    changed = True
                    world.dated(30)
                    world.schedule[DAY][0]["status"] = {
                        "abstractGameCode": "L",
                        "detailedState": "Warmup",
                        "statusCode": "PW",
                    }
                return p

            patch.setattr(picks, "save_pick", save_and_warmup_single)
            scheduler.run_day(DAY, world.config)
            assert not any("Today's pick" in b for b in world.sends("dm"))
            assert load_decision(DAY, world.picks) is None
            health_dms = world.sends("health-dm")
            assert any("contest_state" in b for b in health_dms)
            assert any("NO PICK DELIVERED" in b for b in health_dms)
            out = day_outcome.gather(DAY, world.picks, world.config, expected_username="u")
            assert out.state == "no_decision" and not out.ok
            results["one_game_warmup_stuck"] = {
                "pick_dms": sum("Today's pick" in b for b in world.sends("dm")),
                "has_contest_state_dm": any("contest_state" in b for b in health_dms),
                "has_missed_pick_dm": any("NO PICK DELIVERED" in b for b in health_dms),
                "day_outcome": out.state,
            }

        # 5. CLI replacement after refusal: first `bts run` refuses on mid-cycle change, second `bts run` replaces & posts!
        with offline_mod.offline(offline_mod.instant(DAY, 18, 25)) as (world, patch):
            world.predictions = lambda: world.frame((0.99, 0.98))
            world.dated(0)
            save_orig = picks.save_pick
            changed = False

            def save_and_change_once(daily_obj, directory):
                nonlocal changed
                p = save_orig(daily_obj, directory)
                if not changed:
                    changed = True
                    world.dated(30)
                return p

            patch.setattr(picks, "save_pick", save_and_change_once)
            args = ["run", "--date", DAY, "--picks-dir", str(world.picks), "--models-dir", str(world.models)]
            r1 = CliRunner().invoke(cli.cli, args, catch_exceptions=False)
            assert r1.exit_code == 1 and len(world.sends("post")) == 0
            assert picks.load_pick(DAY, world.picks).double_down is not None
            r2 = CliRunner().invoke(cli.cli, args, catch_exceptions=False)
            assert r2.exit_code == 0 and len(world.sends("post")) == 1
            replaced = picks.load_pick(DAY, world.picks)
            assert replaced.chosen_on["streak"] == 30 and replaced.double_down is None
            results["cli_run_recovery_after_refusal"] = "replaced_and_posted"

    return results


def run_all_a2_probes() -> dict:
    with TemporaryDirectory(prefix="wr_a2_") as tmp:
        tmp_path = Path(tmp)
        matrix = []
        for p in PATHS:
            for c in CASES:
                cell_dir = tmp_path / f"{p}__{c}"
                cell_dir.mkdir()
                matrix.append(_run_one_matrix_cell(cell_dir, p, c))
        cost_dir = tmp_path / "cost_and_stuck"
        cost_dir.mkdir()
        cost = _run_cost_and_stuck_probes(cost_dir)
    return {"matrix_count": len(matrix), "matrix": matrix, "cost_and_stuck": cost}


def test_probe_a2_verified():
    res = run_all_a2_probes()
    assert res["matrix_count"] == len(PATHS) * len(CASES)


if __name__ == "__main__":
    out = run_all_a2_probes()
    summary_path = Path(__file__).resolve().parent / "probe_a2_summary.json"
    summary_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "matrix_count": out["matrix_count"],
        "cost_and_stuck": out["cost_and_stuck"],
    }, indent=2))
