"""Exercise startup gates and feed HEAD-written state to the archived CLI."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from harness import IS_BASE, Clock, World, contest_state, picks, scheduler
from opening import D1, restore


def startup(root, snapshot, live, required, intent):
    w = World(root, D1, live=live, required=required, intent=intent)
    restore(w, snapshot)
    with w.switched_on():
        w.slate(D1)
        w.invoke("schedule", "--config", w.config_path)
    return {"live": live, "required": required, "intent": intent,
            "events": w.events, "snapshot": w.snapshot()}


def prepare_rollback(root, snapshot, kind):
    day = "2027-09-26"
    w = World(root, day, "14:00")
    restore(w, snapshot)
    with w.switched_on():
        w.slate(day, ("19:05", "19:35"))
        w.invoke("season-contract", "enable", "--season", 2027, "--picks-dir", w.picks)
        result = w.fetch(57, 58, "2027-09-25")
        assert result.exit_code == 0, result.output
        if kind == "dated":
            w.dated(58, 58, "used")
        w.invoke("saver-state", "--init", "used", "--season", 2027, "--picks-dir", w.picks)
        w.run()
    return w.snapshot()


def rollback(root, snapshot):
    from bts import daily_decision
    day = "2027-09-26"
    w = World(root, day, "20:50")
    restore(w, snapshot)
    loaded = {}
    with w.switched_on():
        w.slate(day, ("19:05", "19:35"))
        for label, read in (
        ("pick", lambda: picks.load_pick(day, w.picks)),
        ("decision", lambda: daily_decision.load_decision(day, w.picks)),
        ("scheduler", lambda: scheduler.load_state(day, w.picks)),
        ("streak", lambda: picks.load_streak(w.picks)),
        ("contest", lambda: contest_state.load_decision_streak_state(w.picks, require_contest_state=True))):
            try:
                loaded[label] = repr(read())
            except Exception as error:
                loaded[label] = repr(error)
        w.invoke("schedule", "--date", day, "--config", w.config_path)
        w.fetch(59, 59, day)
        w.set_day("2027-09-27", "02:00")
        w.invoke("reconcile", "--picks-dir", w.picks)
    return {"loaded": loaded, "events": w.events, "snapshot": w.snapshot()}


def main():
    target = Path(sys.argv[1]).resolve().parent
    if IS_BASE:
        source = json.loads((target / "rollback-input.json").read_text())
        rows = {}
        for kind, snapshot in source.items():
            with TemporaryDirectory(prefix="wrblind-rollback-") as temp:
                rows[kind] = rollback(Path(temp), snapshot)
        (target / "rollback-results.json").write_text(json.dumps(rows, indent=2))
        return
    snapshot = json.loads((target / "restored.json").read_text())
    rows = []
    for live, required, intent in ((False, False, "research"), (True, False, "research"),
                                   (True, False, "enter"), (True, True, "enter")):
        with TemporaryDirectory(prefix="wrblind-startup-") as temp:
            rows.append(startup(Path(temp), snapshot, live, required, intent))
    (target / "startup-results.json").write_text(json.dumps(rows, indent=2))
    sources = {}
    for kind in ("automatic", "dated"):
        with TemporaryDirectory(prefix="wrblind-rollback-input-") as temp:
            sources[kind] = prepare_rollback(Path(temp), snapshot, kind)
    (target / "rollback-input.json").write_text(json.dumps(sources, indent=2))


if __name__ == "__main__":
    main()
