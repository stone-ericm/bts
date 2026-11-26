"""Rehearse restored opening slates, changes in observation, and restarts."""
import gzip
import json
import sys
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from harness import ACCOUNT, IS_BASE, RECIPIENT, Clock, PowerLoss, World, cli, picks, wall

D1, D2, D3 = "2027-03-24", "2027-03-25", "2027-03-26"
OLD = "2026-09-27"


def baseline_snapshot(root):
    assert IS_BASE
    w = World(root, OLD, live=False, required=False)
    with w.switched_on():
        w.slate(OLD)
        for _ in range(16):
            picks.update_streak([True], w.picks)
        assert w.fetch(16, 16, OLD).exit_code == 0
        result = w.invoke("saver-state", "--init", "active", "--season", 2026, "--picks-dir", w.picks)
        assert result.exit_code == 0, result.output
        w.run()
    return w.snapshot()


def restore(w, snapshot):
    from harness import write_json
    for name, contents in snapshot.items():
        write_json(w.root / name, contents)


def verify_entry(w, *, enter):
    from bts.entry_receipt import units_snapshot_dir
    daily = picks.load_pick(w.day, w.picks)
    w.pending = []
    w.crosswalk = {}
    if daily is not None and enter:
        legs = [slot for slot in (daily.pick, daily.double_down) if slot is not None]
        units = []
        for index, slot in enumerate(legs, 10):
            w.pending.append({"playerId": index, "unitId": index, "roundId": 2})
            w.crosswalk[index] = slot.batter_id
            units.append({"id": index, "feedId": slot.game_pk, "roundId": 2})
        folder = units_snapshot_dir(w.picks)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{w.day}T010000Z.json.gz").write_bytes(
            gzip.compress(json.dumps({"units": units}).encode()))
        w.event("fixture_operator_entered", legs=len(legs))
    with patch.object(cli, "_fetch_rounds", lambda: {2: date.fromisoformat(w.day)}):
        result = w.invoke("check-pick-entered", "--cron", "--picks-dir", w.picks,
                          "--expected-username", ACCOUNT, "--dm-recipient", RECIPIENT)
    assert result.exit_code in (0, 1), result.output


def open_case(root, snapshot, name, restart):
    w = World(root, D1, {"early": "01:40", "early_regular_wake": "05:10:30"}.get(name, "10:00"))
    restore(w, snapshot)
    with w.switched_on():
        result = w.invoke("season-contract", "revoke", "--season", 2027,
                          "--reason", "offline restore rehearsal", "--picks-dir", w.picks)
        assert result.exit_code == 0, result.output
        is_early = name.startswith("early")
        start = "06:10" if is_early else "22:10" if name == "midnight" else "19:05"
        w.slate(D1, (start,), projected_until="02:30" if is_early else "16:00")
        w.slate(D2, ("13:05", "19:05"))
        w.slate(D3, ("19:05", "19:35", "20:05"))
        if name == "midnight":
            w.finish[D1, 700] = wall(D2, "00:15")
        if name not in {"forgot", "late_before", "late_after", "corrected"}:
            w.dated()
        if name == "late_before":
            w.at("15:00", w.dated)
        if name == "late_after":
            w.at("18:25", w.dated)
            # No saved pick is available to keep the scheduler awake after 18:20.
        if name == "corrected":
            w.dated(2, 2)
            w.at("15:45", w.dated)
        w.fetch(0, 0)
        if restart == "midday":
            def power():
                raise PowerLoss("scheduled restart during sleep")
            if name == "early":
                w.at("02:45", power)
            else:
                w.at("15:30", power)
        if restart == "mid_send":
            w.send_failure = PowerLoss("remote accepted before local acknowledgement")
        if restart == "transport_timeout":
            w.send_failure = TimeoutError("remote accepted then reply timed out")
        def early_check():
            result = w.invoke("check-pick-entered", "--cron", "--picks-dir", w.picks,
                              "--expected-username", ACCOUNT, "--dm-recipient", RECIPIENT)
            assert result.exit_code in (0, 1), result.output
        if name == "early":
            for hm in ("05:00", "05:15", "05:30", "05:45", "06:00"):
                w.at(hm, early_check)
        else:
            entry_check_at = "05:15" if is_early else "21:00" if name == "midnight" else "18:00"
            w.at(entry_check_at, lambda: verify_entry(w, enter=True))
        try:
            w.run()
        except PowerLoss as error:
            w.event("power_loss", error=str(error))
            w.advance(30)
            w.run()
        w.send_failure = None
        if name == "late_after" and w.actions:
            w.advance((wall(D1, "18:26") - Clock.at).total_seconds())
        if is_early:
            w.set_day(D1, "10:00")
            early_check()
        first = w.snapshot()
        entered_first = sum(e["legs"] for e in w.events if e["type"] == "fixture_operator_entered")
        w.set_day(D2, "08:30")
        w.invoke("day-outcome", "--date", D1, "--config", w.config_path, "--picks-dir", w.picks,
                 "--expected-username", ACCOUNT, "--dm-recipient", RECIPIENT)
        w.dated(entered_first, entered_first)
        w.fetch(entered_first, entered_first, D1 if entered_first else None)
        w.at("12:00", lambda: verify_entry(w, enter=True))
        w.run()
        second = w.snapshot()
        w.set_day(D3, "08:30")
        w.invoke("day-outcome", "--date", D2, "--config", w.config_path, "--picks-dir", w.picks,
                 "--expected-username", ACCOUNT, "--dm-recipient", RECIPIENT)
        entered_total = sum(e["legs"] for e in w.events if e["type"] == "fixture_operator_entered")
        if name in ("auto", "auto_no_fetch", "auto_previous_year_source"):
            w.invoke("season-contract", "status", "--picks-dir", w.picks)
            w.invoke("season-contract", "enable", "--season", 2027, "--picks-dir", w.picks)
            if name == "auto":
                w.fetch(entered_total, entered_total, D2)
            elif name == "auto_previous_year_source":
                w.fetch(16, 16, OLD)
            w.invoke("season-contract", "status", "--picks-dir", w.picks)
        else:
            w.dated(entered_total, entered_total)
        w.at("18:25", lambda: verify_entry(w, enter=True))
        w.run()
        return {"case": name, "restart": restart,
                "events": w.events, "first": first, "second": second, "third": w.snapshot()}


def main():
    target = Path(sys.argv[1]).resolve()
    if IS_BASE:
        with TemporaryDirectory(prefix="wrblind-base-") as temp:
            target.write_text(json.dumps(baseline_snapshot(Path(temp)), indent=2))
        return
    snapshot = json.loads(target.read_text())
    cases = ["correct", "forgot", "late_before", "late_after", "corrected", "auto", "auto_no_fetch",
             "midnight", "early"]
    records = []
    for case in cases:
        for restart in ("none", "midday", "mid_send"):
            with TemporaryDirectory(prefix="wrblind-opening-") as temp:
                records.append(open_case(Path(temp), snapshot, case, restart))
    with TemporaryDirectory(prefix="wrblind-opening-") as temp:
        records.append(open_case(Path(temp), snapshot, "correct", "transport_timeout"))
    for case in ("early_regular_wake", "auto_previous_year_source"):
        with TemporaryDirectory(prefix="wrblind-opening-") as temp:
            records.append(open_case(Path(temp), snapshot, case, "none"))
    target.with_name("opening-results.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
