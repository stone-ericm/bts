"""Extra offline probes for job ordering, template size, and shipped policy decisions."""
import json
import shutil
import socket
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

from pytest import MonkeyPatch

from bts import contest_state, dm, orchestrator, scheduler, strategy
from bts.daily_decision import load_decision
from bts.health import alert
from probe_state import ET, invoke, observation
from probe_text import offline
from tests.test_pick_chosen_on_cs5a import DAY, NOW, set_automatic
from tests.test_saved_pick_paths_cs5a import prepare_run_day, world

REPO = Path(__file__).resolve().parents[2]


def first_boot(root, mp):
    config, directory, clock, _ = world.__wrapped__(root, mp)
    config["scheduler"].update(pick_delivery="private", entry_intent="research")
    config["health_checks"].update(contest_state_expected=True)
    prepare_run_day(mp)
    mp.setattr(scheduler, "notify_ready", lambda: None)
    mp.setattr(scheduler, "notify_watchdog", lambda: None)
    scheduler.run_day(date="2027-07-14", config=config)
    print("no game", orchestrator.run_cascade.call_count, dm.send_dm.call_count)
    scheduler.run_day(date=DAY, config=config)
    print("game with absent switch", orchestrator.run_cascade.call_count, load_decision(DAY, directory),
          [c.args[1].splitlines()[0] for c in alert.send_dm.call_args_list])
    set_automatic(directory, streak=7)
    scheduler.run_day(date=DAY, config=config)
    before = load_decision(DAY, directory)
    assert before and before["delivery_status"] == "private_locked", before
    print("game with valid observation", before["delivery_status"], dm.send_dm.call_count)
    config["scheduler"].update(pick_delivery="dm", entry_intent="enter")
    scheduler.run_day(date=DAY, config=config)
    print("switched same day", load_decision(DAY, directory)["delivery_status"], dm.send_dm.call_count)
    cfg = root / "drill.toml"
    cfg.write_text('[scheduler]\npick_delivery="dm"\nentry_intent="enter"\ncontest_state_required=true\n')
    copied = root / "copied"
    shutil.copytree(directory, copied)
    (copied / DAY / "decision.json").unlink()
    for mode, intent in (("dm", "enter"), ("private", "research")):
        cfg.write_text(f'[scheduler]\npick_delivery="{mode}"\nentry_intent="{intent}"\ncontest_state_required=true\n')
        for selected_date, pdir in ((DAY, directory), (DAY, copied), ("2027-07-14", copied)):
            result = invoke("day-outcome", "--date", selected_date, "--config", str(cfg), "--dm-recipient",
                            "offline.test", directory=pdir)
            print("day-outcome", mode, selected_date, pdir.name, result.exit_code, json.loads(result.output)["state"])


def text_lengths(root, mp):
    cfg, directory, clock, _ = world.__wrapped__(root, mp)
    clock[0] = datetime(2027, 12, 31, 23, 59, tzinfo=ET)
    set_automatic(directory, now=clock[0])
    prefix = "0002-01-01T00:00:00."
    timestamp = prefix + "1" * (128 - len(prefix) - 6) + "+00:00"
    path = directory / "account_state/contest_streak.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), recorded_at=timestamp)))
    state = contest_state.load_decision_streak_state(directory, require_contest_state=True, now=clock[0])
    record = contest_state.automatic_reading_record(state, clock[0])
    actual_day = clock[0].date().isoformat()
    scheduler._warn_stale_contest_reading(record, cfg, directory)
    clock[0] += timedelta(minutes=2)
    scheduler._warn_stale_contest_reading(record, cfg, directory)
    print("year warning dates", [entry.args[1].split("--date ")[1][:10] for entry in alert.send_dm.call_args_list])
    rows = {}
    text = (REPO / "docs/reviews/2026-11-18-whole-range/kickoff-fix-e.md").read_text()
    for line in text.splitlines():
        if line.startswith("| "):
            cells = line.split(" | ")
            if len(cells) > 3 and cells[0][2:] in {"S1", "S2", "S3", "S4", "S5", "S6", "M1", "M2", "M3", "M4", "M5"}:
                new = cells[2] if cells[2].startswith("unchanged") else cells[3]
                if cells[3].startswith("unchanged"):
                    new = cells[2]
                rows[cells[0][2:]] = new.strip("`")
    detail = f"last accepted fetch {timestamp}, about {record['age_hours']:.2f} hours ago"
    envelope = "BTS health WARN attention:\n- [contest_state] "
    for seq in ("S1 S2 S3 S4 S6 S5", "M1 M2 M3 M4 S6 M5"):
        body = " ".join(rows[k] for k in seq.split()).format(D=actual_day, detail=detail)
        print("proposed length", seq[0], len(envelope + body), "max timestamp", len(timestamp), "detail", len(detail))


def policy_edges():
    mdp = strategy._load_mdp()
    for d, s, b in (("2027-09-19", 0, 18), ("2027-09-19", 0, 15), ("2027-08-30", 56, 56)):
        ctx = strategy.DecisionContext(.99, .88, True, d, True, mdp, b, "trusted")
        print(d, "bar", strategy.effective_pick_bar(s, d, False, mdp=mdp, best_streak=b, best_status="trusted"),
              strategy.resolve_policy_decision(ctx, s, False))
    for d in ("2027-07-16", "2027-09-20", "2027-09-26"):
        hits = []
        for s in range(16):
            for p in (.75, .79, .80, .81, .82, .83, .90, .96, .99):
                ctx = strategy.DecisionContext(p, p, True, d, True, mdp, s, "trusted")
                actions = [strategy.resolve_policy_decision(ctx, s, flag).action for flag in (True, False)]
                if actions[0] != actions[1]:
                    hits.append((s, p, *actions))
        print("policy with/without saver", d, hits[:25])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as name, MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", offline)
        mp.setattr(socket, "create_connection", offline)
        if sys.argv[1] == "jobs":
            first_boot(Path(name), mp)
        elif sys.argv[1] == "length":
            text_lengths(Path(name), mp)
        else:
            policy_edges()
