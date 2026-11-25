"""Offline base-code probes. The only replacement dependencies are clock and I/O."""
import contextlib
import json
import socket
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from pytest import MonkeyPatch

from bts import dm, orchestrator, picks, scheduler, strategy, web
from bts.health import alert
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick
from tests.test_saved_pick_paths_cs5a import scheduler_state, world
from tests.test_stale_delivery_cs4c import seed_automatic


def offline(*args, **kwargs):
    raise AssertionError("network is forbidden")


def run_case(config):
    return scheduler.run_single_check(date=DAY, all_game_pks=[101, 102], confirmed_sides=set(),
                                      config=config, early_lock_gap=0.03)


def show(label, result):
    selection = result["selection"]
    decision = selection.decision if selection else None
    summary = result.get("skip_summary")
    print(label, "action", getattr(decision, "action", None), "policy", getattr(decision, "policy_action", None),
          "source", getattr(decision, "source", None), "reason", getattr(decision, "reason", None),
          "bar", getattr(decision, "pick_bar", None), "candidate", getattr(selection, "primary_candidate", None),
          "summary", summary)


def probe(root, mp):
    config, directory, clock, statuses = world.__wrapped__(root, mp)
    cascade = orchestrator.run_cascade
    frame, tier = cascade.return_value
    mp.setattr(scheduler, "count_new_confirmations", lambda *a, **k: 0)
    frame.loc[:, "p_game_hit"] = [0.999, 0.75]
    statuses[101] = {"abstract": "F", "detailed": "Postponed", "code": "D"}
    seed_automatic(directory)
    mdp = strategy._load_mdp()
    print("shipped policy", bool(mdp), "tail", bool(mdp and mdp.get("tail")))
    result = run_case(config)
    show("status removed top", result)
    summary = result["skip_summary"]
    print(scheduler.format_skip_dm(DAY, summary), web.render_skip_banner(summary))
    with patch.object(strategy, "_load_mdp", return_value=None):
        show("heuristic below .80", run_case(config))
    statuses.clear()
    show("no eligible", run_case(config))
    with patch.object(orchestrator, "run_and_pick", return_value=(frame, None, tier)), \
         patch("bts.contest_state.load_decision_streak_state", side_effect=ValueError("no state")):
        show("synthetic selection None, streak None", run_case(config))

    if mdp:
        for streak, day, best, best_status, primary, saver, allow in [
            (56, "2027-08-30", 56, "valid", .99, False, True),
            (56, DAY, 56, "valid", .99, False, True),
            (0, "2027-09-19", 18, "valid", .72, False, True),
            (0, DAY, 0, "valid", .90, True, False),
        ]:
            ctx = strategy.DecisionContext(primary, .88, True, day, allow, mdp, best, best_status)
            print("policy", day, strategy.resolve_policy_decision(ctx, streak, saver))
    for value in ["0.8", [], {}, None, float("nan"), 10**330, True]:
        raw = dict(summary, best_p=value, bar=value, streak="<b>8</b>")
        persisted = scheduler_state()
        persisted.skip_summary = raw
        scheduler.save_state(persisted, directory)
        saved = scheduler.load_state(DAY, directory).skip_summary
        parts = []
        for render in [lambda s: scheduler.format_skip_dm(DAY, s), web.render_skip_banner]:
            try:
                parts.append(render(saved))
            except Exception as exc:
                parts.append(type(exc).__name__)
        print("legacy", type(value).__name__, parts)
    odd = pd.concat([frame, frame], ignore_index=False)
    print("duplicate index", scheduler.build_skip_summary(odd, 7),
          "missing columns", scheduler.build_skip_summary(pd.DataFrame(), 7),
          "finite fallback", scheduler.build_skip_summary(frame, None, decision=SimpleNamespace(reason=None)))

    first = clock[0] + timedelta(minutes=30)
    scheduler._morning_state_check(config, directory, DAY, clock[0], first)
    record = {"stale": True, "reason": "old", "recorded_at": "2027-07-14T22:35:00-04:00", "age_hours": 20.0}
    scheduler._warn_stale_contest_reading(record, config, directory)
    print("morning + same-day delivery warning envelopes", alert.send_dm.call_args_list)
    clock[0] = datetime.fromisoformat("2027-07-16T00:05:00-04:00")
    scheduler._warn_stale_contest_reading(record, config, directory)
    print("after midnight warning", alert.send_dm.call_args)
    for guard in ("late", "void"):
        daily = daily_pick()
        picks.save_pick(daily, directory)
        state = scheduler_state()
        alert.send_dm.reset_mock()
        if guard == "late":
            scheduler._refuse_late_delivery(daily, config, directory, state, DAY, guard, NOW, NOW + timedelta(minutes=3))
        else:
            scheduler._refuse_void_game_delivery(daily, config, directory, state, DAY, guard, {101: "Postponed"})
        print("refusal", guard, "pick", picks.load_pick(DAY, directory), "refusals", state.delivery_refusals,
              "dm", alert.send_dm.call_args)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory, MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", offline)
        mp.setattr(socket, "create_connection", offline)
        probe(Path(directory), mp)
