"""Trace real scheduler delivery calls, their selection identities and effects."""

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

from offline import forbid_network

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
forbid_network()

from click.testing import CliRunner
from pytest import MonkeyPatch

from bts import cli, contest_state, dm, orchestrator, picks, scheduler, strategy
from bts.daily_decision import load_decision
from bts.health import alert
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, record, set_dated
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world


def check_route(root, monkeypatch, route, reuse, matches):
    config, directory, _, statuses = world.__wrapped__(root, monkeypatch)
    config["scheduler"]["cluster_min"] = 30
    set_dated(directory)
    prepare_run_day(monkeypatch)
    daily = daily_pick()
    daily.chosen_on = record()
    daily.pick.projected_lineup = route == "fallback"
    path = picks.save_pick(daily, directory)
    real_delivery = scheduler._deliver_and_lock_pick
    real_refresh = scheduler._refresh_pick_at_fallback_decision
    frame, tier = orchestrator.run_cascade.return_value
    seen = []

    def deliver(candidate, cfg, pd, state, date, label, *, selection):
        state_file = pd / date / "scheduler_state.json"
        before = (path.read_bytes(), state_file.read_bytes())
        if label == route:
            assert selection.decision is None and selection.pick_result.daily is candidate
            assert selection.pick_result.locked is (reuse != "unavailable")
            assert candidate.chosen_on == record()
        result = real_delivery(candidate, cfg, pd, state, date, label, selection=selection)
        seen.append((label, result))
        if not result:
            assert (path.read_bytes(), state_file.read_bytes()) == before
            assert load_decision(DAY, pd) is None
        return result

    def refresh(*args, **kwargs):
        if not matches:
            set_dated(directory, best=10)
        if reuse == "locked":
            statuses[101] = {"abstract": "L", "detailed": "In Progress", "code": "I"}
        elif reuse == "coarse":
            # Fail the first detailed snapshot in run_and_pick, then allow
            # actual selection to classify the cached pick on the next read.
            detailed = Mock(side_effect=[OSError("offline detailed status"), *([statuses] * 10)])
            monkeypatch.setattr(picks, "get_game_statuses_detailed", detailed)
            monkeypatch.setattr(strategy, "get_game_statuses", Mock(side_effect=OSError("offline coarse status")))
        orchestrator.run_cascade.side_effect = None
        orchestrator.run_cascade.return_value = (frame.iloc[1:].copy(), tier)
        result = real_refresh(*args, **kwargs)
        assert result.selection.decision is None and result.selection.pick_result.daily is result.daily
        return result

    statuses[102] = {"abstract": "P", "detailed": "Warmup", "code": "PW"}
    monkeypatch.setattr(scheduler, "_deliver_and_lock_pick", deliver)
    if route == "lineup":
        if not matches:
            set_dated(directory, best=10)
        orchestrator.run_cascade.return_value = (frame.iloc[1:].copy(), tier)
        monkeypatch.setattr(scheduler, "_refresh_pick_at_fallback_decision", refresh)
    else:
        monkeypatch.setattr(scheduler, "_refresh_pick_at_fallback_decision", refresh)
        if route == "final fallback":
            orchestrator.run_cascade.return_value = (None, None)
        else:
            # The selectable slate contains only a Warmup candidate. The
            # earlier-stage result remains unposted through its projection gate.
            orchestrator.run_cascade.return_value = (frame.iloc[1:].copy(), tier)
    scheduler.run_day(date=DAY, config=config)
    assert (route, matches) in seen, (route, reuse, matches, seen)
    assert dm.send_dm.call_count == int(matches)
    assert scheduler.load_state(DAY, directory).pick_locked is matches
    assert bool(load_decision(DAY, directory)) is matches
    if not matches:
        assert all(not sent for _, sent in seen)
        messages = [c.args[1] for c in alert.send_dm.call_args_list]
        assert sum("[contest_state]" in msg for msg in messages) == 1
        assert sum("[missed_pick]" in msg for msg in messages) == 1


def out_of_cycle(root, monkeypatch):
    config, directory, _, _ = world.__wrapped__(root, monkeypatch)
    set_dated(directory)
    _, first, _ = orchestrator.run_and_pick(config, DAY)
    assert first.decision is not None
    path = picks.save_pick(first.pick_result.daily, directory)
    set_dated(directory, best=10)
    reread = picks.load_pick(DAY, directory)
    state = scheduler_state()
    assert not scheduler._deliver_and_lock_pick(reread, config, directory, state, DAY, "later cycle", selection=first)
    assert not state.pick_locked and not dm.send_dm.called
    assert path.exists() and load_decision(DAY, directory) is None
    # Deliberately violate the internal caller precondition and supply the exact
    # earlier-cycle memory object as well. Production reloads or forwards its own
    # check or refresh. This call records the limit of that caller discipline.
    assert scheduler._deliver_and_lock_pick(first.pick_result.daily, config, directory, state, DAY,
                                           "artificial stale object", selection=first)
    assert dm.send_dm.called and state.pick_locked


def entry_next_day(root, monkeypatch):
    config, directory, now, _ = world.__wrapped__(root, monkeypatch)
    set_dated(directory, best=10)
    prepare_run_day(monkeypatch)
    config["scheduler"]["entry_intent"] = "enter"
    daily = daily_pick()
    daily.chosen_on = record()
    picks.save_pick(daily, directory)
    orchestrator.run_cascade.return_value = (None, None)
    scheduler.run_day(DAY, config=config)
    entry = CliRunner().invoke(cli.cli, [
        "check-pick-entered", "--picks-dir", str(directory), "--now-et", f"{DAY}T18:46:00",
        "--expected-username", "offline", "--dm-recipient", "offline.test",
    ], catch_exceptions=False)
    assert entry.exit_code == 0 and "not committed/locked" in entry.stdout and not dm.send_dm.called
    now[0] = (NOW + timedelta(days=1)).replace(hour=8)
    config_file = root / "intent.toml"
    config_file.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
    outcome = CliRunner().invoke(cli.cli, ["day-outcome", "--date", DAY, "--config", str(config_file),
                                "--picks-dir", str(directory), "--expected-username", "offline",
                                "--dm-recipient", "offline.test"], catch_exceptions=False)
    assert outcome.exit_code == 1 and json.loads(outcome.stdout)["state"] == "no_decision"
    assert dm.send_dm.call_count == 1 and "Steven Kwan" not in dm.send_dm.call_args.args[1]


def missed_pick_gates(root, monkeypatch, required, minutes):
    config, directory, now, _ = world.__wrapped__(root, monkeypatch)
    daily = daily_pick()
    picks.save_pick(daily, directory)
    now[0] = picks.earliest_pick_game_et(daily) - timedelta(minutes=minutes)
    config["scheduler"]["contest_state_required"] = required == "scheduler"
    config["health_checks"]["contest_state_expected"] = required == "health"
    state = scheduler_state()
    for _ in range(2):
        scheduler._maybe_alert_missed_pick(config, DAY, directory, 10, None, state)
    alert.send_dm.assert_called_once()
    msg = alert.send_dm.call_args.args[1]
    if required:
        timing = f"~{minutes} min to first pitch" if minutes >= 0 else "first pitch has passed"
        assert "NO PICK DELIVERED" in msg and timing in msg and "bts season-contract status" in msg
        assert daily.pick.batter_name not in msg and "Deliver manually" not in msg
    else:
        assert f"~{minutes} min to first pitch" in msg and daily.pick.batter_name in msg
        assert "Deliver manually" in msg


def prove_boolean_mutation():
    with tempfile.TemporaryDirectory(dir=REVIEW) as workdir:
        scratch = Path(workdir) / "src"
        shutil.copytree(ROOT / "src", scratch, ignore=shutil.ignore_patterns("__pycache__"))
        target = scratch / "bts" / "picks.py"
        original = target.read_text()
        target.write_text(original.replace("type(streak) is not int", "not isinstance(streak, int)", 1))
        mutant = subprocess.run([sys.executable, str(REVIEW / "oracle.py"), str(scratch)], capture_output=True, text=True)
        assert mutant.returncode == 1 and "AssertionError: automatic boolean False" in mutant.stderr, mutant.stderr
        print("Independent oracle detects the Boolean-count mutation at automatic streak false")


if __name__ == "__main__":
    for label, reuses in [("lineup", ["unavailable"]), ("fallback", ["locked", "coarse", "unavailable"]),
                         ("final fallback", ["locked", "coarse", "unavailable"])]:
        for reuse in reuses:
            for matches in [True, False]:
                with tempfile.TemporaryDirectory(dir=REVIEW) as temp, MonkeyPatch.context() as patch:
                    trace = io.StringIO()
                    try:
                        with contextlib.redirect_stderr(trace):
                            check_route(Path(temp), patch, label, reuse, matches)
                    except Exception:
                        print(trace.getvalue(), file=sys.stderr)
                        raise
                    print(f"{label}/{reuse}/matches={matches} passed")
    for probe in (out_of_cycle, entry_next_day):
        with tempfile.TemporaryDirectory(dir=REVIEW) as temp, MonkeyPatch.context() as patch:
            probe(Path(temp), patch)
            print(f"{probe.__name__} passed")
    for setting in (None, "scheduler", "health"):
        for minutes in (9, 0, -3):
            with tempfile.TemporaryDirectory(dir=REVIEW) as temp, MonkeyPatch.context() as patch:
                missed_pick_gates(Path(temp), patch, setting, minutes)
    print("Missed-pick timing and dedup checked through the firing gate in nine scenarios")
    prove_boolean_mutation()
