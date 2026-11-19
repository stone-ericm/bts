"""Exercise real delivery/scoring entry points around a crash or lost send response."""

from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from probe_support import BASE, DAY, ET, HEAD, RUN, STREAK, archive, decision, load, offline
from probe_support import pick_data, plant, python, run_cli, scratch, seed_streak, stagger
from reconcile_probe import menu, receipts


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
    return evidence, daily_decision.load_decision(ENTERED, root)


def manual_run_commit(root):
    import pandas as pd
    from bts import cli, contest_state, picks, posting, strategy
    from bts.model import predict
    from click.testing import CliRunner

    model = contest_state.DecisionStreakState(9, True, True, "model", "fresh", 0, True)
    daily = picks.parse_daily_pick(json.dumps(pick_data(ENTERED)))
    selected = SimpleNamespace(pick_result=SimpleNamespace(daily=daily, locked=False), decision=True)
    predictions = pd.DataFrame([{**pick_data(ENTERED)["pick"], "lineup": 1}])
    evidence = []

    def send(body):
        evidence.append({"remote_accepted": True, "operator_entered": True, "synthetic_settled_result": "hit"})
        raise KeyboardInterrupt()

    with patch.object(predict, "run_pipeline", return_value=predictions), \
         patch.object(contest_state, "load_decision_streak_state", return_value=model), \
         patch.object(picks, "get_game_statuses_detailed", return_value={}), \
         patch.object(picks, "attach_provenance"), patch.object(strategy, "select_pick", return_value=selected), \
         patch.object(posting, "should_post_now", return_value=True), patch.object(posting, "post_to_bluesky", send):
        result = CliRunner().invoke(cli.cli, ["run", "--date", ENTERED, "--picks-dir", str(root)])
    assert result.exit_code == 1 and len(evidence) == 1, result.output
    return evidence, None


def scenario_runs(work):
    from bts import picks

    print(json.dumps({"module": picks.__file__}))
    for name in ("lost_ack", "skip_conflict", "private_crash", "manual_run_crash"):
        root = work / name / "picks"
        for offset in range(9):
            day = stagger("2027-04-01", offset)
            plant(root, day, "hit", bluesky_posted=True)
        target = seed_streak(root, streak=9, saver=True)
        evidence, decision_record = manual_run_commit(root) if name == "manual_run_crash" else scheduler_commit(root, name)
        stranded = load(root / f"{ENTERED}.json")
        assert stranded["result"] is None
        with menu():
            refused = check_cli(root, ENTERED)
            assert refused.exit_code == 0 and "not scoring" in refused.stdout, refused.output
        plant(root, DAY, bluesky_posted=True)
        with menu("hit"):
            scored = check_cli(root, DAY)
            assert scored.exit_code == 0 and "Streak: 10" in scored.stdout, scored.output
        before = target.read_text()
        with menu("miss"):
            result = run_cli(root)
        assert result.exit_code == 0 and "hit -> miss" in result.stdout, result.output
        replay = receipts(root)[0]["replay"]
        updated = load(target)
        observation = {
            "scenario": name, "synthetic_external_evidence": evidence, "decision": decision_record,
            "past_file": stranded, "correct_fold": [10, False], "actual": [updated["streak"], updated["saver_available"]],
            "before": before, "after": target.read_text(), "stdout": result.stdout, "stderr": result.stderr, "receipt": replay,
        }
        print(json.dumps(observation, sort_keys=True), flush=True)


def crash_marker(work):
    from bts import picks

    root = work / "scorer-crash" / "picks"
    target = seed_streak(root, streak=0, saver=True)
    plant(root, DAY, bluesky_posted=True)
    with menu("hit"), patch.object(picks, "save_pick", side_effect=KeyboardInterrupt):
        result = check_cli(root, DAY, now=datetime(2027, 4, 19, 23, 59, tzinfo=ET))
    assert result.exit_code == 1 and load(root / f"{DAY}.json")["result"] is None
    partial = load(target)
    assert partial["streak"] == 1 and partial["applied_dates"] == [DAY]
    with menu():
        held = run_cli(root, datetime(2027, 4, 20, 0, 1, tzinfo=ET))
    assert "applied-date marker" in held.stderr and load(target) == partial
    with menu("hit"):
        retry = check_cli(root, DAY)
        recovered = run_cli(root)
    assert retry.exit_code == recovered.exit_code == 0
    assert load(target)["streak"] == 1 and load(target)["applied_dates"] == [DAY]
    print(json.dumps({"case": "real scorer interrupt and midnight retry", "partial": partial,
                      "hold": held.stderr, "scorer_retry": retry.stdout, "replay_retry": recovered.stdout}))


if __name__ == "__main__":
    with offline():
        if len(sys.argv) > 1:
            scenario_runs(Path(sys.argv[1]))
        else:
            with scratch("delivery") as work:
                work = Path(work)
                for revision in (BASE, HEAD):
                    source = archive(revision, work / revision, "src") / "src"
                    print(revision, flush=True)
                    try:
                        proc = python(source, __file__, work / f"data-{revision}", timeout=20)
                        print(proc.stdout, end="")
                    except Exception as error:
                        if hasattr(error, "stdout"):
                            print(error.stdout, error.stderr, file=sys.stderr)
                        raise
                crash_marker(work)
