"""Entered-pick counterexamples from both C3 review rounds, using the real writers."""

from dataclasses import replace
from datetime import datetime
import json
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bts import picks
from tests.test_wr_c3v_corpus import load, pick_data, seed_streak, stagger
from tests.test_wr_c3v_reconcile import (
    ET, PENDING, RUN, base_modules, compare_reconcile, deadline, only_receipt, warning,
)


DAY = "2027-04-19"

ENTERED = "2027-04-18"
DELIVERY = datetime(2027, 4, 18, 18, 50, tzinfo=ET)


def check_cli(root, day, now=RUN):
    from bts import cli
    from click.testing import CliRunner

    with patch.object(cli, "_now_et", return_value=now):
        return CliRunner().invoke(cli.cli, ["check-results", "--date", day, "--picks-dir", str(root)])


def scheduler_commit(root, scenario):
    from bts import contest_state, daily_decision, dm, picks, scheduler

    evidence = []
    local = picks.parse_daily_pick(json.dumps(pick_data(ENTERED)))
    state = scheduler.SchedulerState(ENTERED, DELIVERY.isoformat(), [], [], [], False, None, None, None)
    model = contest_state.DecisionStreakState(9, True, True, "model", "fresh", 0, True)
    private = scenario == "private_crash"
    config = {"scheduler": {"pick_delivery": "private" if private else "dm"},
              "bluesky": {"dm_recipient": "synthetic.invalid"}}

    def send(recipient, body):
        evidence.append({"remote_accepted": True, "operator_entered": True, "synthetic_settled_result": "hit"})
        if scenario == "lost_ack":
            raise TimeoutError("the transport lost its response after accepting the DM")
        return "synthetic-delivery-id"

    def capture(*args):
        if private:
            # The dashboard can show the canonical local selection once save_state marks it locked.
            evidence.append({"operator_entered": True, "synthetic_settled_result": "hit"})
            raise KeyboardInterrupt()

    if scenario == "skip_conflict":
        # A provisional MDP skip, then a later pre-cutoff DM on the same date.
        state.final_skip_candidate = {"source": "mdp", "primary": {"batter_id": 17}}

    with patch.object(scheduler, "_now_et", return_value=DELIVERY), \
         patch.object(scheduler, "_void_committed_games", return_value={}), \
         patch.object(scheduler, "_trigger_live_forward_capture_on_lock", capture), \
         patch.object(contest_state, "load_decision_streak_state", return_value=model), \
         patch.object(dm, "send_dm", send), \
         patch.object(daily_decision, "write_decision", return_value=None):
        try:
            outcome = scheduler._deliver_and_lock_pick(local, config, root, state, ENTERED, "fallback")
            assert outcome == (scenario == "skip_conflict")
        except KeyboardInterrupt:
            assert private
    if scenario == "skip_conflict":
        scheduler._write_endofday_skip(root, ENTERED, state)
        assert daily_decision.load_decision(ENTERED, root)["scoreable"] is False
    elif private:
        assert scheduler.load_state(ENTERED, root).pick_locked
    else:
        assert scheduler.load_state(ENTERED, root) is None
    return evidence


def manual_run_commit(root):
    import pandas as pd
    from bts import cli, contest_state, picks, posting, strategy
    from bts.model import predict
    from click.testing import CliRunner

    model = contest_state.DecisionStreakState(
        9, True, True, "contest", "fresh", 9, True, observation_kind="automatic",
    )
    daily = picks.parse_daily_pick(json.dumps(pick_data(ENTERED)))
    selected = SimpleNamespace(pick_result=SimpleNamespace(daily=daily, locked=False), decision=True)
    predictions = pd.DataFrame([{**pick_data(ENTERED)["pick"], "lineup": 1}])
    evidence = []

    def send(body):
        evidence.append({"remote_accepted": True, "operator_entered": True, "synthetic_settled_result": "hit"})
        raise KeyboardInterrupt()

    with patch.object(contest_state, "DecisionStreakState", return_value=model), \
         patch.object(predict, "run_pipeline", return_value=predictions), \
         patch.object(contest_state, "load_decision_streak_state", return_value=model), \
         patch.object(picks, "get_game_statuses_detailed", return_value={}), \
         patch.object(picks, "attach_provenance"), patch.object(strategy, "select_pick", return_value=selected), \
         patch.object(posting, "should_post_now", return_value=True), patch.object(posting, "post_to_bluesky", send):
        result = CliRunner().invoke(cli.cli, ["run", "--date", ENTERED, "--picks-dir", str(root)])
    assert result.exit_code == 1 and len(evidence) == 1, result.output
    return evidence


def plant(root, day, result=None, **extra):
    from bts import picks

    daily = picks.parse_daily_pick(json.dumps(pick_data(day, result, **extra)))
    return picks.save_pick(daily, root)


def inputs():
    import pandas as pd
    from bts.contest_state import DecisionStreakState
    model = DecisionStreakState(9, True, True, "model", "fresh", 0, True)
    frame = pd.DataFrame([{**pick_data(ENTERED)["pick"], "lineup": 1}])
    return model, frame


def preview(root):
    from bts import cli, contest_state, picks, strategy
    from bts.model import predict
    from click.testing import CliRunner

    model, frame = inputs()
    candidate = picks.parse_daily_pick(json.dumps(pick_data(ENTERED)))
    selected = SimpleNamespace(pick_result=SimpleNamespace(daily=candidate), decision=True)
    with patch.object(predict, "run_pipeline", return_value=frame), \
         patch.object(contest_state, "load_decision_streak_state", return_value=model), \
         patch.object(picks, "get_game_statuses_detailed", return_value={}), \
         patch.object(picks, "attach_provenance"), patch.object(strategy, "select_pick", return_value=selected):
        result = CliRunner().invoke(cli.cli, ["preview", "--date", ENTERED, "--picks-dir", str(root)])
    assert result.exit_code == 0 and "PROJECTED" in result.stdout, result.output
    return picks.load_pick(ENTERED, root)


def scheduler_reviewer_commit(root, scenario):
    from bts import contest_state, daily_decision, dm, picks, scheduler, web

    model, _ = inputs()
    state = scheduler.SchedulerState(ENTERED, DELIVERY.isoformat(), [], [], [], False, None, None, None)
    private = scenario.startswith("private-preview-")
    local = preview(root) if private else picks.load_pick(ENTERED, root)
    if scenario == "lost-ack-eod":
        state.final_skip_candidate = {"source": "mdp", "primary": {"batter_id": 17}}
        scheduler.save_state(state, root)
    config = {"scheduler": {"pick_delivery": "private" if private else "dm"},
              "bluesky": {"dm_recipient": "synthetic.invalid"}}
    events = []

    def send(*_):
        events.append("remote accepted; human entered eventual hit")
        raise TimeoutError("response lost after upstream accepted")

    def capture(*_):
        with patch.object(web, "PICKS_DIR", root):
            assert web.load_scheduler_state(ENTERED)["pick_locked"]
        events.append("operator saw private lock and entered eventual hit")
        raise KeyboardInterrupt()

    with patch.object(scheduler, "_now_et", return_value=DELIVERY), \
         patch.object(scheduler, "_void_committed_games", return_value={}), \
         patch.object(scheduler, "_trigger_live_forward_capture_on_lock", capture), \
         patch.object(contest_state, "load_decision_streak_state", return_value=model), \
         patch.object(dm, "send_dm", send):
        try:
            assert not scheduler._deliver_and_lock_pick(local, config, root, state, ENTERED, "fallback")
        except KeyboardInterrupt:
            assert private

    if scenario == "lost-ack-eod":
        scheduler._write_endofday_skip(root, ENTERED, state)
        assert daily_decision.load_decision(ENTERED, root)["scoreable"] is False
    if scenario == "private-preview-quarantine":
        # Simulate a restored/truncated lock after the private entry and interrupt.
        lock = root / ENTERED / "scheduler_state.json"
        assert load(lock)["pick_locked"]
        lock.write_text("{")
        assert scheduler.load_state(ENTERED, root) is None
        assert list(lock.parent.glob("scheduler_state.json.corrupt-*")) and not lock.exists()
    if scenario == "private-preview-restart":
        real_save_state = scheduler.save_state
        lock = root / ENTERED / "scheduler_state.json"
        assert load(lock)["pick_locked"]
        config["orchestrator"] = {"picks_dir": str(root), "heartbeat_path": str(root / "heartbeat")}

        def stop_after_startup_write(rebuilt, directory):
            real_save_state(rebuilt, directory)
            raise KeyboardInterrupt()

        with patch.object(scheduler, "_now_et", return_value=DELIVERY), \
             patch.object(scheduler, "notify_ready"), patch.object(scheduler, "notify_watchdog"), \
             patch.object(scheduler, "fetch_schedule", return_value=[{"gamePk": 101, "gameDate": f"{ENTERED}T23:00:00Z"}]), \
             patch.object(scheduler, "detect_doubleheader_game2s", return_value=[]), \
             patch.object(scheduler, "plan_run_times", return_value=[]), \
             patch.object(scheduler, "save_state", stop_after_startup_write):
            try:
                scheduler.run_day(ENTERED, config)
            except KeyboardInterrupt:
                pass
        assert load(lock)["pick_locked"] is False
    assert len(events) == 1
    return events


def orchestrator_commit(root):
    from bts import contest_state, orchestrator, picks, posting, strategy

    model, frame = inputs()
    frame["p_game_hit"] = 0.99
    config = {"orchestrator": {"picks_dir": str(root)}, "tiers": [],
              "scheduler": {"contest_state_required": False}, "bluesky": {"dm_recipient": "unused"}}
    events = []
    resolve = strategy.resolve_policy_decision

    def remote_accept(*_):
        events.append("post accepted; human entered eventual hit")
        raise KeyboardInterrupt()

    with patch.object(orchestrator, "run_cascade", return_value=(frame, "synthetic")), \
         patch.object(contest_state, "load_decision_streak_state", return_value=model), \
         patch.object(picks, "get_game_statuses_detailed", return_value={101: {"abstract": "P", "detailed": "Scheduled"}}), \
         patch.object(strategy, "resolve_policy_decision", side_effect=lambda *args: replace(resolve(*args), action="single")), \
         patch.object(picks, "attach_provenance"), \
         patch.object(posting, "should_post_now", return_value=True), \
         patch.object(posting, "post_to_bluesky", remote_accept):
        try:
            orchestrator.orchestrate(root / "unused", ENTERED, config=config)
        except KeyboardInterrupt:
            assert len(events) == 1
    assert picks.load_pick(ENTERED, root).chosen_on["required"] is False
    return events


@pytest.mark.parametrize("scenario", [
    "lost_ack", "skip_conflict", "private_crash", "manual_run_crash",
    "lost-ack-eod", "private-preview-restart", "private-preview-quarantine",
    "scheduler-preview-ack", "orchestrator-post-interrupted",
])
def test_entered_pick_holds_replay(tmp_path, monkeypatch, base_modules, scenario):
    root = tmp_path / "current" / "picks"

    def unexpected_network(*args, **kwargs):
        pytest.fail("Unexpected network call")

    monkeypatch.setattr(socket.socket, "connect", unexpected_network)
    monkeypatch.setattr(socket, "getaddrinfo", unexpected_network)
    for offset in range(9):
        plant(root, stagger("2027-04-01", offset), "hit", bluesky_posted=True)
    target = seed_streak(root, streak=9, saver=True)

    if scenario == "manual_run_crash":
        evidence = manual_run_commit(root)
    elif scenario in ("lost_ack", "skip_conflict", "private_crash"):
        evidence = scheduler_commit(root, scenario)
    elif scenario == "orchestrator-post-interrupted":
        evidence = orchestrator_commit(root)
    else:
        if scenario == "scheduler-preview-ack":
            preview(root)
        elif scenario == "lost-ack-eod":
            plant(root, ENTERED)
        evidence = scheduler_reviewer_commit(root, scenario)
    assert len(evidence) == 1
    assert picks.load_pick(ENTERED, root).result is None

    monkeypatch.setattr(picks, "get_game_statuses_detailed", lambda *args: {})
    monkeypatch.setattr(picks, "check_hit", unexpected_network)
    if scenario in ("lost_ack", "skip_conflict", "private_crash", "manual_run_crash"):
        refused = check_cli(root, ENTERED)
        assert refused.exit_code == 0 and "not scoring" in refused.stdout, refused.output

    plant(root, DAY, bluesky_posted=True)
    monkeypatch.setattr(picks, "check_hit", lambda *args, **kwargs: "hit")
    scored = check_cli(root, DAY)
    assert scored.exit_code == 0 and "Streak: 10" in scored.stdout, scored.output
    before = target.read_bytes()
    result, fold = compare_reconcile(
        root, monkeypatch, base_modules, lookback=1,
        result_feed=lambda *args, **kwargs: {"pick": "miss"},
    )
    assert fold is None
    assert result.exit_code == 0, result.output
    assert result.stderr == warning(ENTERED, PENDING)
    assert result.stdout.splitlines() == [
        "Reconciling last 1 days of picks...",
        "CORRECTIONS FOUND (1):",
        f"  {DAY}: Synthetic batter — hit -> miss",
        "Streak retained: 10",
    ]
    assert only_receipt(root)["replay"] == {
        "state": "unavailable", "blocked_date": ENTERED, "reason": PENDING,
    }
    assert load(root / f"{DAY}.json")["result"] == "miss"
    assert target.read_bytes() == before
