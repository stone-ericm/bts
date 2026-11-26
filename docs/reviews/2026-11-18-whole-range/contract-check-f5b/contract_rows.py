"""Replays from both F5 rounds, with assertions for the two-marker design."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from replay_support import DAY, RIGHT, WRONG, World, alert_probe, compact, generate, health, row, source
import fallback_recovery
import mutation_witness
import state_replay
import surviving_mutations


def revised(root, v, replacement=False, missing=False, manual=False):
    w = World(root, v)
    daily = w.commit_at("10:05", at="09:40")
    early = w.cron("09:45", pending=[] if missing else [row()])
    if replacement:
        w.commit_at("10:50", at="09:52", game=102)
    else:
        w.edit_pitch(daily, "10:06" if missing else "10:50")
    corrected = WRONG if not replacement else RIGHT
    w.capture([corrected], f"{DAY}T140000Z")
    middle = w.invoke("09:55", manual=True, pending=[row()], crosswalk={}) if missing and manual else None
    manual_at_ten = manual and not missing
    runs = [w.invoke("10:00", manual=True, pending=[row()])] if manual_at_ten else [
        w.cron(at, pending=[] if missing else [row()]) for at in ("10:00", "10:15", "10:30")]
    morning = w.morning()
    print(root.name, compact(early), compact(middle), list(map(compact, runs)), morning)
    assert runs[0]["calls"] and runs[0]["sent"]
    if missing:
        assert "1 min to submit" in runs[0]["sent"][0]
        assert morning[1]["state"] == "not_entered"
    else:
        assert all(run["calls"] and run["sent"] for run in runs)
        assert "recorded in game " + str(corrected["feedId"]) in runs[0]["sent"][0]
        assert morning[1]["state"] == "entry_unverifiable" and len(morning[2]) == 1
    if v == "f5":
        assert early["receipt"]["early_cron_run"] is True and not early["marker_changed"]


def gate(root):
    for pitch in ("10:06", "10:15", "10:20"):
        w = World(root / pitch)
        w.commit_at(pitch, at="09:40")
        early = w.cron("09:45", pending=[row()])
        w.capture([WRONG], f"{DAY}T140000Z")
        ten = w.cron("10:00", pending=[row()])
        print("R5/R6", pitch, compact(early), compact(ten))
        assert early["receipt"]["outcome"] == "base_slot_available" and not early["calls"]
        assert ten["calls"] and ten["sent"]


def clear(root, v):
    # Reuse the archived replay, check the messages independently of its old two-clear assertion.
    calls = []
    old = state_replay.display

    def display(label, run):
        old(label, run)
        if run is not None:
            calls.append((label, run["sent"]))

    with patch.object(state_replay, "display", display):
        try:
            state_replay.duplicate_clear(root, v)
        except AssertionError:
            if v == "base" or len(calls) != 4:
                raise
    messages = [message for _, sent in calls for message in sent]
    print("R9 messages", calls)
    if v == "base":
        assert not messages
    else:
        assert len(messages) == 2 and "recorded in game 102" in messages[0] and messages[1].startswith("✅")


def live_precedence(root):
    w = World(root)
    w.commit_at("10:05", at="09:40")
    w.cron("09:45", dm="failed")
    assert w.early_state()["status"] == "dm_failed"
    w.set_marker({"date": w.day, "status": "confirmed"})
    assert health(w) == []
    print("R13 live confirmed beats early dm_failed")


def year_boundary(root, v):
    d, jan = "2027-12-31", "2028-01-01"
    w = World(root, v, d)
    w.commit_at("19:10", at="18:00")
    assert w.cron("18:00", dm="failed")["marker"]["status"] == "dm_failed"
    j = World(root, v, jan)
    j.commit_at("06:10", at="04:55")
    j.cron("05:00", pending=[row()])
    w.capture([RIGHT], f"{d}T220000Z")
    assert w.invoke("18:15", pending=[row()], manual=True)["marker"]["status"] == "confirmed"
    saved = w.marker.read_bytes()
    j.cron("05:15", pending=[row()])
    assert saved == w.marker.read_bytes() and health(w) == []
    levels, sent = fallback_recovery.dispatch(w, d)
    assert not levels and not sent
    print("R12", levels, sent)


if __name__ == "__main__":
    mode = sys.argv[1]
    with TemporaryDirectory() as temp:
        root = Path(temp)
        if mode == "install":
            generate(root)
        elif mode == "gate":
            gate(root)
        elif mode == "manual":
            revised(root / "R8-R13", "f5", manual=True)
        elif mode == "precedence":
            live_precedence(root)
        else:
            source(mode)
            alert_probe.closing_fixture(root, mode)
            for variant in ("bind-wrong", "remove-entry", "late-commit"):
                alert_probe.shared_window(root, mode, variant)
            for name, options in (
                ("R1", {}), ("R2", {"replacement": True}), ("R3", {"missing": True}),
                ("R7", {"missing": True, "manual": True}), ("R8-R13", {"manual": True}),
            ):
                revised(root / name, mode, **options)
            if mode == "f5":
                gate(root)
                mutation_witness.replay("candidate")
                live_precedence(root / "R13-health")
            try:
                alert_probe.previous_date_health(root, mode)
            except AssertionError:
                if mode == "base":
                    raise
                w = World(root / "prior-health", mode)
                assert health(w, "2027-03-23") == ["CRITICAL"] and w.early_state()["status"] != "dm_failed"
                print("R4: real March 23 CRITICAL after March 24 early run")
            clear(root / "R9", mode)
            year_boundary(root / "R12", mode)
            try:
                fallback_recovery.decode_failure(root / "R11", mode)
            except AssertionError:
                w = World(root / "R11", mode, fallback_recovery.CURRENT)
                levels, sent = fallback_recovery.dispatch(w, w.day)
                assert levels == ["CRITICAL"] and "UnicodeDecodeError" in sent[0]
                print("R11", levels, sent)
