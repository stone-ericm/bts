"""Offline regressions for the whole-range pick-path contract."""

from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from click.testing import CliRunner

from bts import cli, contest_state, picks, posting, scheduler
from bts.daily_decision import load_decision
from tests.test_pick_chosen_on_cs5a import daily_pick
from tests.test_saved_pick_paths_cs5a import scheduler_state


ROOT = Path(__file__).resolve().parents[1]
DAY = "2027-03-24"


@pytest.fixture
def review_world(tmp_path, monkeypatch):
    fixture = ROOT / "docs/reviews/2026-11-18-whole-range/slice-3-codex/offline.py"
    spec = spec_from_file_location("wr_a_offline", fixture)
    offline = module_from_spec(spec)
    spec.loader.exec_module(offline)
    monkeypatch.setattr(offline, "ROOT", ROOT)
    monkeypatch.setattr(offline, "HERE", tmp_path)
    return offline


def invoke(world, command, *extra):
    args = ["--config", str(world.config_path)] if command == "orchestrate" else [
        "--picks-dir", str(world.picks), "--models-dir", str(world.models),
    ]
    return CliRunner().invoke(cli.cli, [command, "--date", world.day, *args, *extra],
                              catch_exceptions=False)


def assert_uncommitted(world):
    daily = picks.load_pick(world.day, world.picks)
    assert not daily.delivery_attempted and not daily.notification_sent and not daily.bluesky_posted
    assert not world.sends("dm") and not world.sends("post")
    assert load_decision(world.day, world.picks) is None
    return daily


def capture_first_scheduler_refusal(world, patch, after_refusal):
    deliver = scheduler._deliver_and_lock_pick
    refusals = []

    def delivery(daily, config, directory, state, day, label, *, selection=None):
        delivered = deliver(daily, config, directory, state, day, label, selection=selection)
        if not refusals:
            assert delivered is False and not state.pick_locked
            saved = assert_uncommitted(world)
            assert saved.chosen_on["streak"] == 0 and saved.double_down
            refusals.append(label)
            after_refusal()
        return delivered

    patch.setattr(scheduler, "_deliver_and_lock_pick", delivery)
    return refusals


@pytest.mark.parametrize("path", ["dm", "public", "run", "orchestrate", "fallback"])
@pytest.mark.parametrize("early", [False, True])
def test_live_fresh_pick_refuses_observation_changed_after_selection(review_world, path, early):
    with review_world.offline(review_world.instant(DAY, 18, 25)) as (world, patch):
        world.predictions = lambda: world.frame((0.99, 0.98))
        world.dated(30 if early else 0)
        save = picks.save_pick
        corrected = False

        def save_and_correct(daily, directory):
            nonlocal corrected
            saved = save(daily, directory)
            if not early and not corrected:
                corrected = True
                assert daily.chosen_on["streak"] == 0 and daily.double_down
                world.dated(30)
            return saved

        patch.setattr(picks, "save_pick", save_and_correct)
        if path == "fallback":
            state = scheduler_state()
            refresh = scheduler._refresh_pick_at_fallback_decision(world.config, DAY, None, .03)
            assert refresh.selection.decision is not None
            assert scheduler._deliver_and_lock_pick(
                refresh.daily, world.config, world.picks, state, DAY, "fallback",
                selection=refresh.selection,
            ) is early
            assert state.pick_locked is early
        elif path in {"dm", "public"}:
            world.config["scheduler"]["pick_delivery"] = path
            if not early:
                refusals = capture_first_scheduler_refusal(world, patch, lambda: None)
            scheduler.run_day(DAY, world.config)
            if not early:
                assert refusals == ["lineup"]
        else:
            result = invoke(world, path)
            assert result.exit_code == int(not early), result.output
            if not early:
                assert f"bts {path}: no pick:" in result.stderr

        if early or path in {"dm", "public"}:
            daily = picks.load_pick(DAY, world.picks)
            assert daily.chosen_on["streak"] == 30 and daily.double_down is None
            assert len(world.sends("dm") + world.sends("post")) == 1
        elif not early:
            assert assert_uncommitted(world).chosen_on["streak"] == 0
        assert sum("contest_state" in body for body in world.sends("health-dm")) == int(
            not early and path not in {"run", "orchestrate"}
        )


@pytest.mark.parametrize("early", [False, True])
def test_correction_during_delivery_status_request(review_world, early):
    """Slice 1 timing, with a second real check after the first refusal."""
    day = "2027-03-25"
    with review_world.offline(review_world.instant(day, 18, 25)) as (world, patch):
        world.dated(30 if early else 0)
        world.predictions = lambda: world.frame((0.99, 0.98))
        world.config["scheduler"]["cluster_min"] = 0
        world.schedule[day][1]["gameDate"] = review_world.instant(day, 19, 15).isoformat()
        statuses = scheduler._delivery_game_statuses
        request = picks.retry_urlopen
        corrected = False

        def during_request(url, **kwargs):
            nonlocal corrected
            if not early and not corrected:
                corrected = True
                assert picks.load_pick(day, world.picks).chosen_on["streak"] == 0
                world.dated(30)
            return request(url, **kwargs)

        def delivery_statuses(date):
            with patch.context() as status_patch:
                status_patch.setattr(picks, "retry_urlopen", during_request)
                return statuses(date)

        patch.setattr(scheduler, "_delivery_game_statuses", delivery_statuses)
        if not early:
            def change_predictions():
                world.predictions = lambda: world.frame((0.98, 0.99))

            refusals = capture_first_scheduler_refusal(world, patch, change_predictions)
        scheduler.run_day(day, world.config)
        if not early:
            assert refusals == ["lineup"]
            sent_at, = [when for kind, when, _ in world.events if kind == "dm"]
            assert sent_at == review_world.instant(day, 18, 30)
        assert len(world.sends("dm")) == 1
        daily = picks.load_pick(day, world.picks)
        assert daily.chosen_on["streak"] == 30 and daily.double_down is None
        assert daily.pick.p_game_hit == (.99 if early else .98)
        assert load_decision(day, world.picks)["delivery_status"] == "delivered"


@pytest.mark.parametrize("change", [
    "equal", "best", "saver", "automatic-to-dated", "dated-to-automatic",
    "automatic-to-automatic", "missing", "invalid", "non-required", "malformed",
    "directory", "midnight", "year", "keyboard", "exit",
])
def test_final_read_and_selection_stamp_edges(review_world, change):
    day = "2027-12-31" if change == "year" else DAY
    with review_world.offline(review_world.instant(day, 18, 25)) as (world, patch):
        world.predictions = lambda: world.frame((0.99, 0.98))
        world.automatic(0)
        if not change.startswith("automatic"):
            world.dated(0)
        if change == "year":
            # There is no regular-season selection on New Year's Eve.
            daily = daily_pick()
            daily.date = day
            daily.pick.game_time = review_world.instant(day, 19, 10).isoformat()
            chosen = contest_state.load_decision_streak_state(world.picks, require_contest_state=True)
            daily.chosen_on = picks.chosen_on_record(chosen, required=True)
            picks.save_pick(daily, world.picks)
            selection = None
        else:
            refresh = scheduler._refresh_pick_at_fallback_decision(world.config, day, None, .03)
            daily = refresh.daily
            selection = refresh.selection
        state = scheduler_state()
        state.date = day
        manual = world.account / "contest_streak.manual.json"
        if change == "best":
            world.dated(0, best=1)
        elif change == "saver":
            world.dated(0, saver="not_earned")
        elif change == "automatic-to-dated":
            world.dated(0)
        elif change == "dated-to-automatic":
            manual.unlink()
        elif change == "automatic-to-automatic":
            world.automatic(4)
        elif change == "missing":
            daily.chosen_on = None
        elif change == "invalid":
            daily.chosen_on["schema"] = "invalid"
        elif change == "non-required":
            daily.chosen_on["required"] = False
        elif change == "malformed":
            manual.write_text("{")
        elif change == "directory":
            manual.unlink()
            manual.mkdir()
        elif change in {"midnight", "year"}:
            world.now = world.now.replace(hour=0, minute=0) + timedelta(days=1)
            # Isolate the reader's date boundary from the earlier cutoff guard.
            load = contest_state.load_decision_streak_state
            after = world.now
            world.now = review_world.instant(day, 18, 25)

            def load_after_midnight(*args, **kwargs):
                world.now = after
                return load(*args, **kwargs)

            patch.setattr(contest_state, "load_decision_streak_state", load_after_midnight)
        elif change in {"keyboard", "exit"}:
            def interrupt(*args):
                raise KeyboardInterrupt if change == "keyboard" else SystemExit

            patch.setattr(picks, "saved_pick_refusal", interrupt)

        def deliver():
            return scheduler._deliver_and_lock_pick(
                daily, world.config, world.picks, state, day, "fallback",
                selection=selection,
            )

        if change in {"keyboard", "exit"}:
            with pytest.raises(KeyboardInterrupt if change == "keyboard" else SystemExit):
                deliver()
            expected = False
        else:
            expected = change in {"equal", "automatic-to-automatic"}
            assert deliver() is expected
        assert state.pick_locked is expected
        if expected:
            assert len(world.sends("dm")) == 1
        else:
            assert_uncommitted(world)


@pytest.mark.parametrize("mode", ["private", "optional", "dry-run", "closed"])
def test_non_posting_and_optional_paths(review_world, mode):
    with review_world.offline(review_world.instant(DAY, 18, 25)) as (world, patch):
        world.dated(0)
        if mode in {"dry-run", "closed"}:
            patch.setattr(posting, "should_post_now", lambda *args: False)
            save = picks.save_pick

            def save_and_correct(daily, directory):
                saved = save(daily, directory)
                world.dated(30)
                return saved

            patch.setattr(picks, "save_pick", save_and_correct)
            result = invoke(world, "run", *(["--dry-run"] if mode == "dry-run" else []))
            assert result.exit_code == 0 and not world.sends("post")
            assert (picks.load_pick(DAY, world.picks) is None) is (mode == "dry-run")
        else:
            if mode == "private":
                world.config["scheduler"]["pick_delivery"] = "private"
            else:
                world.config["scheduler"]["contest_state_required"] = False
            scheduler.run_day(DAY, world.config)
            status = load_decision(DAY, world.picks)["delivery_status"]
            assert status == ("private_locked" if mode == "private" else "delivered")
