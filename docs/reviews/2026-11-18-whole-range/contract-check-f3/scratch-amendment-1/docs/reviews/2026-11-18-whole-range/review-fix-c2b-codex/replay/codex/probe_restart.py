"""Compare real run_day restarts on the parent and the C2 commit."""

import argparse
from contextlib import redirect_stderr
from hashlib import sha256
from importlib import import_module
from io import StringIO
import json
from pathlib import Path
from unittest.mock import Mock

from probe_support import (
    day_pick, deadline, fixtures, prepare_schedules, raw, receipt, status_body, world_at,
)


def unknown(bodies):
    return [body for body in bodies if "DELIVERY OUTCOME UNKNOWN" in body]


def digest(value):
    return sha256(value).hexdigest() if value is not None else None


def replay(root, patch, opening, phase, head):
    from bts import scheduler
    from bts.daily_decision import decision_path

    edges = import_module("opening_edges")
    root.mkdir()
    season = opening.Season(root)
    opening.write_json(season.account_dir / "contest_streak.json", {
        "schema_version": "bts_contest_streak_auto_v1", "season": 2026,
        "active_streak": 17, "best_streak": 26, "saver_available": False,
        "source_date": opening.OLD_DAY, "recorded_at": "2026-10-21T14:00:00Z",
        "source": "manual_cli",
    })
    finish = edges.finish
    trace = {"returns": [], "polling": []}
    run_day = scheduler.run_day
    poll = scheduler.run_result_polling

    def observe_run(*args, **kwargs):
        result = run_day(*args, **kwargs)
        trace["returns"].append(result)
        return result

    def observe_poll(game_pk, date, picks_dir, **kwargs):
        result = poll(game_pk, date, picks_dir, **kwargs)
        trace["polling"].append([game_pk, date, result])
        return result

    def repeat_then_finish(season, day, outcome_exit):
        path = decision_path(day, season.picks)
        original = path.read_bytes()
        trace["first_lock"] = season.snapshot(day)["scheduler"]["pick_locked"]
        trace["first_committed"] = season.snapshot(day)["scheduler"]["committed_pick_written"]
        trace["polls_before_repeat"] = len(trace["polling"])
        season.command("schedule", "--config", season.config, "--date", day)
        assert path.read_bytes() == original
        return finish(season, day, outcome_exit)

    with season.run_offline(), patch.context() as scoped:
        scoped.setattr(edges, "finish", repeat_then_finish)
        scoped.setattr(scheduler, "run_day", observe_run)
        scoped.setattr(scheduler, "run_result_polling", observe_poll)
        season.restore()
        result = edges.restart_at_send(season, phase)
        state = result["snapshot"]
        day = state["day"]
        trace.update(
            pick_messages=len(edges.picks_sent(season)),
            pick_bytes=digest(raw(season.picks / f"{day}.json")),
            decision_bytes=digest(raw(decision_path(day, season.picks))),
            delivery_status=state["decision"]["delivery_status"],
            scoreable=state["decision"]["scoreable"],
            locked=state["scheduler"]["pick_locked"],
            committed=state["scheduler"]["committed_pick_written"],
            next_morning=result["next_morning"]["state"],
        )
        alerts = unknown(message["body"] for message in season.messages)
        assert alerts == ([status_body(day)] if head else [])
        assert trace["returns"] == [None, None]
        assert trace["locked"] and trace["committed"] and trace["scoreable"]
        assert trace["polls_before_repeat"] == len(trace["polling"]) == 1
        assert trace["pick_messages"] == int(phase == "after_remote_accept")
        assert trace["next_morning"] == "not_entered"
        assert any("Delivery of this pick was NOT confirmed" in m["body"] for m in season.messages)
        if phase == "corrected_after_marker":
            assert state["pick"]["chosen_on"]["streak"] == 1 and state["pick"]["double_down"]
            assert state["account"]["contest_streak.manual.json"]["active_streak"] == 46
    print(f"{phase}: recovery={len(alerts)}, {trace}", flush=True)
    return trace


def classification(root, patch, opening, case, head):
    from bts import daily_decision, dm, picks, scheduler
    from bts.health import alert

    day = "2027-03-25"
    with world_at(root, patch, opening, opening.at(day, "18:20")) as (config, directory, _, _, scoped):
        daily = day_pick(day)
        daily.delivery_attempted = True
        if case == "terminal":
            daily.result = "hit"
        prepare_schedules(daily)
        pick_path = picks.save_pick(daily, directory)
        decision_path = daily_decision.decision_path(day, directory)
        if case in {"prior_locked", "prior_delivered", "prior_private", "terminal"}:
            label = {
                "prior_locked": "locked_unconfirmed", "prior_delivered": "delivered",
                "prior_private": "private_locked", "terminal": "locked_unconfirmed",
            }[case]
            daily_decision.write_decision(
                day, directory, action="single", source="mdp", delivery_status=label,
                primary={"batter_id": daily.pick.batter_id, "batter_name": daily.pick.batter_name},
                scoreable=True,
            )
        elif case == "writer_none":
            scoped.setattr(daily_decision, "write_decision", Mock(return_value=None))

        pick_before = pick_path.read_bytes()
        decision_before = raw(decision_path)
        checkpoints = []

        def transport(_, body):
            if unknown([body]):
                state = scheduler.load_state(day, directory)
                checkpoints.append((state.pick_locked, state.committed_pick_written, raw(decision_path)))
            return "review-ack"

        alert.send_dm.side_effect = transport
        calls = 2 if case == "fresh" else 1
        results = [scheduler.run_day(day, config) for _ in range(calls)]
        decision_after = raw(decision_path)
        state = scheduler.load_state(day, directory)
        assert results == [None] * calls and state.pick_locked
        assert state.committed_pick_written is (case != "writer_none")
        assert pick_path.read_bytes() == pick_before
        assert decision_after == decision_before or decision_before is None
        dm.send_dm.assert_not_called()
        bodies = [call.args[1] for call in alert.send_dm.call_args_list]
        assert unknown(bodies) == ([status_body(day)] if head else [])
        assert receipt(directory, day).exists() is head
        assert checkpoints == ([(True, case != "writer_none", decision_after)] if head else [])
        assert scheduler.run_result_polling.call_count == (0 if case in {"terminal", "writer_none"} else calls)
        missed_pick = [body for body in bodies if "[missed_pick]" in body]
        assert bool(missed_pick) is (case == "writer_none")
        result = dict(
            returns=results, locked=state.pick_locked, committed=state.committed_pick_written,
            pick_bytes=digest(pick_before), decision_bytes=digest(decision_after),
            polls=scheduler.run_result_polling.call_count, missed_pick=missed_pick,
            result_status=state.result_status,
        )
    print(f"{case}: recovery={len(checkpoints)}, {result}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=["base", "head"])
    parser.add_argument("--comparison", type=Path)
    args = parser.parse_args()
    comparison = args.comparison.resolve() if args.comparison else None
    head = args.version == "head"
    with deadline(50), fixtures("restart") as (root, patch, opening), redirect_stderr(StringIO()):
        results = {
            phase: replay(root / phase, patch, opening, phase, head)
            for phase in ("before_remote_accept", "after_remote_accept", "corrected_after_marker")
        }
        for case in ("fresh", "prior_locked", "prior_delivered", "prior_private", "terminal", "writer_none"):
            results[case] = classification(root / case, patch, opening, case, head)
    if comparison:
        if head:
            baseline = json.loads(comparison.read_text())
            assert baseline == results, {key: (baseline[key], results[key]) for key in results
                                         if baseline[key] != results[key]}
            print("HEAD and base traces and persisted byte hashes match.")
        else:
            comparison.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
