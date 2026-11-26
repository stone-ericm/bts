"""Adversarial review regressions for whole-range fix set A (A1, A2, A5)."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path

from click.testing import CliRunner
import pytest

from bts import cli, day_outcome, orchestrator, picks, scheduler, strategy
from bts.daily_decision import load_decision

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src" / "bts").exists():
    ROOT = Path(__file__).resolve().parents[3]
DAY = "2027-03-24"


@pytest.fixture
def review_world(tmp_path, monkeypatch):
    mpl_dir = tmp_path / "_mpl"
    mpl_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_dir))
    fixture = ROOT / "docs/reviews/2026-11-18-whole-range/slice-3-codex/offline.py"
    spec = spec_from_file_location("_wr_a_carbon_offline", fixture)
    assert spec is not None and spec.loader is not None
    offline = module_from_spec(spec)
    spec.loader.exec_module(offline)
    monkeypatch.setattr(offline, "ROOT", ROOT)
    monkeypatch.setattr(offline, "HERE", tmp_path)
    return offline


@pytest.mark.parametrize("mode", ["optional_changed", "optional_cold_start", "private_changed"])
def test_true_optional_and_private_paths_deliver_across_midcycle_change(review_world, mode):
    """Verify optional mode (both config flags False) and private mode under actual mid-cycle state change."""
    with review_world.offline(review_world.instant(DAY, 18, 25)) as (world, patch):
        world.predictions = lambda: world.frame((0.99, 0.98))
        if mode == "private_changed":
            world.config["scheduler"]["pick_delivery"] = "private"
            world.config["scheduler"]["entry_intent"] = "research"
            world.dated(0)
        elif mode == "optional_changed":
            world.config["scheduler"]["contest_state_required"] = False
            world.config["health_checks"]["contest_state_expected"] = False
            assert scheduler._contest_state_required(world.config) is False
            world.dated(0)
        else:
            world.config["scheduler"]["contest_state_required"] = False
            world.config["health_checks"]["contest_state_expected"] = False
            assert scheduler._contest_state_required(world.config) is False

        if mode != "optional_cold_start":
            save_orig = picks.save_pick
            changed = False

            def save_and_change(daily, directory):
                nonlocal changed
                saved = save_orig(daily, directory)
                if not changed:
                    changed = True
                    world.dated(30)
                return saved

            patch.setattr(picks, "save_pick", save_and_change)

        scheduler.run_day(DAY, world.config)
        decision = load_decision(DAY, world.picks)
        assert decision is not None
        expected_status = "private_locked" if mode == "private_changed" else "delivered"
        assert decision["delivery_status"] == expected_status
        assert len(world.sends("dm")) == (0 if mode == "private_changed" else 1)


@pytest.mark.parametrize("command", ["run", "orchestrate"])
@pytest.mark.parametrize(
    "change,should_post",
    [
        ("equal_dated", True),
        ("dated_streak", False),
        ("dated_best", False),
        ("dated_saver", False),
        ("dated_to_automatic", False),
        ("automatic_to_dated", False),
        ("automatic_to_automatic", True),
    ],
)
def test_cli_catchup_compares_saved_stamp_against_final_read(
    review_world, command, change, should_post
):
    """Catch-up branch (result.locked and not bluesky_posted) enforces saved_pick_refusal."""
    with review_world.offline(review_world.instant(DAY, 18, 25)) as (world, patch):
        world.predictions = lambda: world.frame((0.99, 0.98))
        starts_auto = change in {"automatic_to_dated", "automatic_to_automatic"}
        if starts_auto or change == "dated_to_automatic":
            world.automatic(0)
            acc = world.account / "contest_streak.json"
            acc.write_text(json.dumps({**json.loads(acc.read_text()), "source": "mlb_bts_profile"}))
        if not starts_auto:
            world.dated(0, best=0, saver="used")

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
            out = orig_select(*a, **kw)
            assert out.pick_result is not None and out.pick_result.locked is True
            if change == "dated_streak":
                world.dated(30, best=30, saver="used")
            elif change == "dated_best":
                world.dated(0, best=5, saver="used")
            elif change == "dated_saver":
                world.dated(0, best=0, saver="not_earned")
            elif change == "dated_to_automatic":
                (world.account / "contest_streak.manual.json").unlink()
            elif change == "automatic_to_dated":
                world.dated(0, best=0, saver="used")
            elif change == "automatic_to_automatic":
                world.automatic(8)
                acc_p = world.account / "contest_streak.json"
                acc_p.write_text(
                    json.dumps({**json.loads(acc_p.read_text()), "source": "mlb_bts_profile"})
                )
            return out

        patch.setattr(strategy, "select_pick", select_then_mutate)
        args = (
            ["run", "--date", DAY, "--picks-dir", str(world.picks), "--models-dir", str(world.models)]
            if command == "run"
            else ["orchestrate", "--date", DAY, "--config", str(world.config_path)]
        )
        res = CliRunner().invoke(cli.cli, args, catch_exceptions=False)
        assert res.exit_code == (0 if should_post else 1)
        assert len(world.sends("post")) == int(should_post)
        persisted = picks.load_pick(DAY, world.picks)
        assert persisted.chosen_on == first_stamp
        assert persisted.bluesky_posted is should_post


def test_final_fallback_fresh_choice_refuses_midcycle_dated_change_and_alerts(review_world):
    """Exercise run_day step 5 ('final fallback') when its fresh refresh races a dated correction."""
    with review_world.offline(review_world.instant(DAY, 15, 0)) as (world, patch):
        world.schedule[DAY] = [review_world.game(101, review_world.instant(DAY, 19, 10))]
        world.dated(0)
        calls = 0

        def staged_predictions():
            nonlocal calls
            calls += 1
            # 16:10 early check: projected candidate saved (should_lock=False).
            # 17:10 and 18:25 checks: empty predictions so loop ends unlocked and enters step 5 at 18:35.
            # 18:35 final fallback refresh: fresh confirmed candidate, racing dated(30) on save.
            if calls == 1:
                return world.frame((0.99,), projected=(True,))
            if calls in {2, 3}:
                return world.frame((0.99,)).iloc[0:0]
            return world.frame((0.99,), projected=(False,))

        world.predictions = staged_predictions
        save_orig = picks.save_pick

        def save_and_race_on_final_fallback(daily, directory):
            saved = save_orig(daily, directory)
            if calls == 4:
                world.dated(30)
            return saved

        patch.setattr(picks, "save_pick", save_and_race_on_final_fallback)
        scheduler.run_day(DAY, world.config)
        assert not any("Today's pick" in b for b in world.sends("dm"))
        assert load_decision(DAY, world.picks) is None
        health_dms = world.sends("health-dm")
        assert any("contest_state" in b for b in health_dms)
        assert any("NO PICK DELIVERED" in b for b in health_dms)
        out = day_outcome.gather(DAY, world.picks, world.config, expected_username="u")
        assert out.state == "no_decision" and not out.ok


@pytest.mark.parametrize("path", ["scheduler", "run", "orchestrate"])
def test_real_midnight_rollover_on_late_slate_refuses_expired_dated_stamp(review_world, path):
    """Selection at 23:58 ET followed by 00:01 ET final read on a 00:15 ET game refuses without reader hooks."""
    with review_world.offline(review_world.instant(DAY, 23, 58)) as (world, patch):
        world.schedule[DAY] = [
            review_world.game(101, review_world.instant("2027-03-25", 0, 15)),
            review_world.game(102, review_world.instant("2027-03-25", 0, 20)),
        ]
        world.predictions = lambda: world.frame((0.99, 0.98))
        world.automatic(0)
        acc = world.account / "contest_streak.json"
        acc.write_text(json.dumps({**json.loads(acc.read_text()), "source": "mlb_bts_profile"}))
        world.dated(0)

        save_orig = picks.save_pick
        rolled = False

        def save_and_roll_midnight(daily, directory):
            nonlocal rolled
            saved = save_orig(daily, directory)
            if not rolled:
                rolled = True
                world.now = review_world.instant("2027-03-25", 0, 1)
            return saved

        patch.setattr(picks, "save_pick", save_and_roll_midnight)
        if path == "scheduler":
            res = scheduler.run_single_check(DAY, [101, 102], set(), world.config, 0.03)
            state = scheduler.SchedulerState(DAY, world.now.isoformat(), [], [], [], False, None, None, None)
            assert (
                scheduler._deliver_and_lock_pick(
                    res["pick_result"].daily,
                    world.config,
                    world.picks,
                    state,
                    DAY,
                    "lineup",
                    selection=res["selection"],
                )
                is False
            )
            assert not state.pick_locked and not world.sends("dm")
            assert sum("contest_state" in b for b in world.sends("health-dm")) == 1
        else:
            args = (
                ["run", "--date", DAY, "--picks-dir", str(world.picks), "--models-dir", str(world.models)]
                if path == "run"
                else ["orchestrate", "--date", DAY, "--config", str(world.config_path)]
            )
            r = CliRunner().invoke(cli.cli, args, catch_exceptions=False)
            assert r.exit_code == 1 and f"bts {path}: no pick:" in r.stderr
            assert not world.sends("post")


@pytest.mark.parametrize(
    "payload,match",
    [
        (b"", "JSON"),
        (b"[" * 5000 + b"]" * 5000, "JSON"),
        (b'{"copyright":"MLB"}', "dates"),
        (b'{"dates":"2027-03-24"}', "dates"),
    ],
)
def test_fetch_schedule_empty_deep_and_missing_dates_raise_value_error(monkeypatch, payload, match):
    monkeypatch.setattr(scheduler, "retry_urlopen", lambda *a, **k: BytesIO(payload))
    with pytest.raises(ValueError, match=match):
        scheduler.fetch_schedule(DAY)
