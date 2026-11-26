"""Probe stranded real-writer states with synthetic transport and score answers."""

from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

from compare_probes import BASE, HEAD, archive, offline, python, scratch
from delivery_probe import DAY, DELIVERY, ENTERED, check_cli
from probe_support import dump, load, plant, run_cli, seed_streak, stagger
from reconcile_probe import menu, receipts


def inputs(root):
    import pandas as pd
    from bts.contest_state import DecisionStreakState
    from probe_support import pick_data

    model = DecisionStreakState(9, True, True, "model", "fresh", 0, True)
    frame = pd.DataFrame([{**pick_data(ENTERED)["pick"], "lineup": 1}])
    return model, frame


def preview(root):
    from bts import cli, contest_state, picks, strategy
    from bts.model import predict
    from click.testing import CliRunner
    from types import SimpleNamespace
    from probe_support import pick_data

    model, frame = inputs(root)
    candidate = picks.parse_daily_pick(json.dumps(pick_data(ENTERED)))
    selected = SimpleNamespace(pick_result=SimpleNamespace(daily=candidate), decision=True)
    with patch.object(predict, "run_pipeline", return_value=frame), \
         patch.object(contest_state, "load_decision_streak_state", return_value=model), \
         patch.object(picks, "get_game_statuses_detailed", return_value={}), \
         patch.object(picks, "attach_provenance"), patch.object(strategy, "select_pick", return_value=selected):
        result = CliRunner().invoke(cli.cli, ["preview", "--date", ENTERED, "--picks-dir", str(root)])
    assert result.exit_code == 0 and "PROJECTED" in result.stdout, result.output
    return picks.load_pick(ENTERED, root)


def scheduler_commit(root, scenario, preserve_attempt=False):
    from bts import contest_state, daily_decision, dm, picks, scheduler, web

    model, _ = inputs(root)
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

    real_save = picks.save_pick

    def keep_unknown(daily, directory):
        if events and not private:
            daily.delivery_attempted = True
        return real_save(daily, directory)

    with patch.object(scheduler, "_now_et", return_value=DELIVERY), \
         patch.object(scheduler, "_void_committed_games", return_value={}), \
         patch.object(scheduler, "_trigger_live_forward_capture_on_lock", capture), \
         patch.object(contest_state, "load_decision_streak_state", return_value=model), \
         patch.object(dm, "send_dm", send), \
         (patch.object(picks, "save_pick", keep_unknown) if preserve_attempt else nullcontext()):
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
            if preserve_attempt:
                rebuilt.pick_locked = True
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
        assert load(lock)["pick_locked"] == preserve_attempt
    assert len(events) == 1
    return events


def orchestrator_commit(root, preserve_attempt=False):
    from bts import contest_state, orchestrator, picks, posting, strategy

    model, frame = inputs(root)
    frame["p_game_hit"] = 0.99
    config = {"orchestrator": {"picks_dir": str(root)}, "tiers": [],
              "scheduler": {"contest_state_required": False}, "bluesky": {"dm_recipient": "unused"}}
    events = []
    resolve = strategy.resolve_policy_decision

    def remote_accept(*_):
        if preserve_attempt:
            daily = picks.load_pick(ENTERED, root)
            daily.delivery_attempted = True
            picks.save_pick(daily, root)
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


def manual_preview(root):
    from bts import cli, contest_state, picks, posting, strategy
    from bts.model import predict
    from click.testing import CliRunner

    daily = preview(root)
    model, frame = inputs(root)
    live = contest_state.DecisionStreakState(9, True, True, "contest", "fresh", 9, True,
                                           observation_kind="automatic")
    with patch.object(predict, "run_pipeline", return_value=frame), \
         patch.object(contest_state, "load_decision_streak_state", return_value=live), \
         patch.object(picks, "get_game_statuses_detailed", return_value={}), \
         patch.object(picks, "attach_provenance"), \
         patch.object(strategy, "classify_pick_lock_state", return_value=picks.PickLockState(False)), \
         patch.object(posting, "should_post_now", return_value=True), \
         patch.object(posting, "post_to_bluesky", side_effect=AssertionError("unexpected post")) as post:
        result = CliRunner().invoke(cli.cli, ["run", "--date", ENTERED, "--picks-dir", str(root)])
    assert daily.chosen_on["required"] is False and not post.called
    assert result.exit_code == 1 and "chosen without the live check" in result.output, result.output
    print(json.dumps(dict(case="manual-preview", exit=result.exit_code, output=result.output)))


def run_case(root, scenario, control=False):
    from bts import picks

    for index in range(9):
        plant(root, stagger("2027-04-01", index), "hit", bluesky_posted=True)
    target = seed_streak(root, streak=9, saver=True)
    if scenario in ("lost-ack-eod", "private-preview-quarantine", "private-preview-restart", "scheduler-preview-ack"):
        if not scenario.startswith("private-preview-"):
            if scenario == "scheduler-preview-ack":
                preview(root)
            else:
                plant(root, ENTERED)
        events = scheduler_commit(root, scenario, preserve_attempt=control)
    else:
        events = orchestrator_commit(root, preserve_attempt=control)
    assert len(events) == 1
    stranded = load(root / f"{ENTERED}.json")
    assert stranded["result"] is None
    if not control:
        assert not any(stranded.get(field) for field in (
            "bluesky_posted", "bluesky_uri", "delivery_attempted", "delivered_at", "notification_sent", "notification_id"))

    plant(root, DAY, bluesky_posted=True)
    with menu("hit"):
        scored = check_cli(root, DAY)
    assert scored.exit_code == 0 and "Streak: 10" in scored.stdout, scored.output
    before = target.read_bytes()
    guard = nullcontext()
    if control and scenario == "private-preview-quarantine":
        original = picks._past_pick_blocker

        def conservative(directory, day, daily, decision):
            reason = original(directory, day, daily, decision)
            if reason is None and any((directory / day).glob("scheduler_state.json.corrupt-*")):
                return "scheduler state was quarantined"
            return reason

        guard = patch.object(picks, "_past_pick_blocker", conservative)
    with menu("miss"), guard:
        result = run_cli(root)
    assert result.exit_code == 0 and "hit -> miss" in result.stdout, result.output
    state = load(target)
    replay_receipts = receipts(root)
    assert len(replay_receipts) == 1
    replay = replay_receipts[0]["replay"]
    assert replay["state"] == ("unavailable" if before == target.read_bytes() else "saved")
    print(json.dumps(dict(case=scenario, control=control, evidence=events, correct_fold=[10, False],
                          actual=[state["streak"], state["saver_available"]], retained=before == target.read_bytes(),
                          chosen_on=stranded["chosen_on"], receipt=replay,
                          stdout=result.stdout, stderr=result.stderr)))


def numeric_update(root, control=False):
    from bts import picks
    from probe_support import pick_data

    for index in range(9):
        plant(root, stagger("2027-04-01", index), "hit")
    data = pick_data(DAY, "hit", bluesky_posted=True)
    data["pick"]["batter_id"] = str(data["pick"]["batter_id"])
    path = dump(root / f"{DAY}.json", data)
    target = seed_streak(root, streak=10)

    def compatible(directory, day):
        data = picks._read_reconcile_pick_data(directory, day)
        return None if data is None else picks._daily_pick_from_data(data)

    def feed(url, **kwargs):
        assert "/game/101/feed/live" in url
        player = {"stats": {"batting": {"hits": 0, "atBats": 4}}}
        teams = {"away": {"players": {"ID100001": player}}, "home": {"players": {}}}
        return {"gameData": {"status": {"abstractGameCode": "F"}}, "liveData": {"boxscore": {"teams": teams}}}

    guard = patch.object(picks, "_read_reconcile_pick", compatible) if control else nullcontext()
    with guard, patch.object(picks, "get_game_statuses_detailed", return_value={}), patch.object(picks, "_fetch_json", feed):
        result = run_cli(root)
    state = load(target)
    receipt = receipts(root)[0]
    print(json.dumps(dict(case="numeric-correction", control=control, actual=[state["streak"], state["saver_available"]],
                          oracle=[0, True], result=load(path)["result"], stdout=result.stdout, stderr=result.stderr,
                          exit=result.exit_code, receipt=receipt["replay"])))


def grading_resave(root, old_source):
    from bts import picks, scheduler

    preview(root)
    assert scheduler.save_nonterminal_result(ENTERED, root, "unresolved").result == "unresolved"
    before = picks.load_pick(ENTERED, root)
    assert before.chosen_on is not None and not before.chosen_on["required"]
    python(old_source, __file__, "old-resave", root, timeout=15)
    after = picks.load_pick(ENTERED, root)
    assert after.chosen_on is None and after.result == "suspended"
    replay = picks._replay_season_streak(root, 2027, DAY)
    assert replay is None
    print(json.dumps(dict(case="grading-resave", current_stamp=before.chosen_on,
                          old_result=after.result, old_stamp=after.chosen_on, replay=replay)))


def worker(root, revision):
    with offline():
        manual_preview(root / "manual-preview")
        for name in ("lost-ack-eod", "private-preview-restart", "private-preview-quarantine", "scheduler-preview-ack", "orchestrator-post-interrupted"):
            run_case(root / name / "picks", name)
            if revision == HEAD:
                run_case(root / (name + "-control") / "picks", name, control=True)
        numeric_update(root / "numeric" / "picks")
        if revision == HEAD:
            numeric_update(root / "numeric-control" / "picks", control=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "old-resave":
        with offline():
            from bts import picks

            old_root = Path(sys.argv[2])
            daily = picks.load_pick(ENTERED, old_root)
            daily.result = "suspended"
            picks.save_pick(daily, old_root)
    elif len(sys.argv) > 1 and sys.argv[1] == "grading-resave":
        with offline(), scratch("old-grading") as tmp:
            root = Path(tmp)
            source = archive("30956cdd^", root / "old", "src") / "src"
            grading_resave(root / "picks", source)
    elif len(sys.argv) > 1:
        worker(Path(sys.argv[1]), sys.argv[2])
    else:
        with scratch("writers") as tmp:
            for revision in (BASE, HEAD):
                root = Path(tmp)
                source = archive(revision, root / revision, "src") / "src"
                try:
                    proc = python(source, __file__, root / (revision + "-fixtures"), revision, timeout=45)
                except Exception as error:
                    print(getattr(error, "stdout", ""), getattr(error, "stderr", ""), file=sys.stderr)
                    raise
                print(revision, proc.stdout, sep="\n", end="", flush=True)
